import uproot
import numpy as np
import awkward as ak
from kvikio.nvcomp_codec import NvCompBatchCodec

def GPU_basket_to_array(data: bytes, dtype: np.dtype, expected_entries: int):
    format1 = uproot.models.TBasket._tbasket_format1
    (
        fNbytes,
        key_version,
        fObjlen,
        fDatime,
        fKeylen,
        fCycle,
    ) = format1.unpack(data[: format1.size])
    assert fNbytes == len(data)
    assert fNbytes - fKeylen != fObjlen  # this is only true for uncompressed baskets

    format2 = uproot.models.TBasket._tbasket_format2
    (
        fVersion,
        fBufferSize,
        fNevBufSize,
        fNevBuf,
        fLast,
    ) = format2.unpack(data[fKeylen - format2.size - 1: fKeylen - 1])
    border = fLast - fKeylen  # border between the contents and offsets in uncompressed data
    assert border <= fObjlen, f"{border} <= {fObjlen}"
    assert expected_entries == border // dtype.itemsize

    format3 = uproot.compression._decompress_header_format
    algo, method, c1, c2, c3, u1, u2, u3 = format3.unpack(
        data[fKeylen : fKeylen + format3.size]
    )
    assert algo == b"ZS", f"Unsupported algorithm: {algo}"  # zstd, we can support more later
    block_compressed_bytes = c1 + (c2 << 8) + (c3 << 16)
    block_uncompressed_bytes = u1 + (u2 << 8) + (u3 << 16)
    assert fObjlen == block_uncompressed_bytes
    assert len(data) == fKeylen + format3.size + block_compressed_bytes  # may not be true for baskets larger than 16 MiB
    compressed_content = data[fKeylen + format3.size :]

    #############################
    #Decompress on CPU
    # raw_content = cramjam.zstd.decompress(compressed_content, output_len=block_uncompressed_bytes)
    # content = np.frombuffer(
    # raw_content, dtype=dtype, count=border // dtype.itemsize
    # )
    
    return (compressed_content, border)

def GPU_all_baskets_to_array(filehandle, branch):
    basket_starts = branch.all_members["fBasketSeek"]
    basket_lengths = branch.all_members["fBasketBytes"]
    basket_entryoffsets = branch.all_members["fBasketEntry"]
    assert basket_starts[-1] == 0  # offsets array is one larger than the number of baskets
    assert basket_lengths[-1] == 0

    if isinstance(branch.interpretation, uproot.interpretation.jagged.AsJagged):
        counts = GPU_all_baskets_to_array(filehandle, branch.count_branch)
        assert len(counts) == basket_entryoffsets[-1]
        offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(counts)
        dtype = branch.interpretation.content.from_dtype
        content = np.empty(offsets[-1], dtype=dtype)

        #####
        N_baskets = len(basket_starts) - 1
        compressed_contents = []
        borders = []
        put_starts = []
        put_stops = []

        
        for i in range(N_baskets):
            filehandle.seek(basket_starts[i])
            data = filehandle.read(basket_lengths[i])
            # print(f"Reading basket {i} with {len(data)} bytes at entry offset {basket_entryoffsets[i]}:{basket_entryoffsets[i+1]}")
            put_starts.append(offsets[basket_entryoffsets[i]])
            put_stops.append(offsets[basket_entryoffsets[i+1]])
            # print(f"Destination: {put_start}:{put_stop}")
            compressed_content, border = GPU_basket_to_array(data, dtype, put_stops[i] - put_starts[i])
            compressed_contents.append(compressed_content)
            borders.append(border)
            
        #Decompress batch of compressed_content    
        codec = NvCompBatchCodec("zstd")
        decompressed_contents = codec.decode_batch(compressed_contents)

        for i in range(N_baskets):
            content[put_starts[i]:put_stops[i]] = decompressed_contents[i].view(dtype)[:borders[i] // dtype.itemsize]
        
        return ak.unflatten(content.astype(branch.interpretation.content.to_dtype), counts)
        
    elif isinstance(branch.interpretation, uproot.interpretation.numerical.AsDtype):
        dtype = branch.interpretation.from_dtype
        content = np.empty(basket_entryoffsets[-1], dtype=dtype)

        #####
        N_baskets = len(basket_starts) - 1
        compressed_contents = []
        borders = []
        put_starts = []
        put_stops = []
        
        
        for i in range(N_baskets):
            filehandle.seek(basket_starts[i])
            data = filehandle.read(basket_lengths[i])

            put_starts.append(basket_entryoffsets[i])
            put_stops.append(basket_entryoffsets[i+1])
            # print(f"Reading basket {i} with {len(data)} bytes at entry offset {basket_entryoffsets[i]}:{basket_entryoffsets[i+1]}")
            compressed_content, border = GPU_basket_to_array(data, dtype, put_stops[i] - put_starts[i])
            compressed_contents.append(compressed_content)
            borders.append(border)
            
        #Decompress batch of compressed_content    
        codec = NvCompBatchCodec("zstd")
        decompressed_contents = codec.decode_batch(compressed_contents)

        for i in range(N_baskets):
            content[put_starts[i]:put_stops[i]] = decompressed_contents[i].view(dtype)[:borders[i] // dtype.itemsize]
        return content.astype(branch.interpretation.to_dtype)
    raise NotImplementedError("Only AsJagged and AsDtype are supported")


# /store/user/IDAP/RunIISummer20UL18NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/40000/BCB3E2FC-D575-0341-A211-5C9A8D8798B9.root
filename = "/home/fstrug/uscmshome/nobackup/GPU/kvikio_playground/TTToSemiLeptonic_UL18JMENanoAOD-zstd.root"
file = uproot.open(filename)
tree = file["Events"]
branch = tree["Muon_pt"]

with open(filename, "rb") as rawfile:
    myarray = GPU_all_baskets_to_array(rawfile, branch)

assert ak.all(myarray == branch.array(library="ak"))