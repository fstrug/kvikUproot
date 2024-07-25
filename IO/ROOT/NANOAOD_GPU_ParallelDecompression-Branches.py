import uproot
import awkward as ak
from kvikio.nvcomp_codec import NvCompBatchCodec
import cupy as cp

# Returns a basket's compressed data, border in the file, and cp array output buffer
def basket_to_compcont_border(data: bytes, dtype: cp.dtype, expected_entries: int):
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
    output_buffer = cp.empty(block_uncompressed_bytes // dtype.itemsize, dtype = dtype)
    
    return (compressed_content, border, output_buffer)


# Returns branch's compressed data, empty cp array output buffers for kvikio, branch metadata for array building
def get_branch_buffer_metadata(filehandle, branch, counts_):
    def get_data(i):
        filehandle.seek(basket_starts[i])
        data = filehandle.read(basket_lengths[i])
        return(data)
        
    basket_starts = branch.all_members["fBasketSeek"]
    basket_lengths = branch.all_members["fBasketBytes"]
    basket_entryoffsets = branch.all_members["fBasketEntry"]
    assert basket_starts[-1] == 0  # offsets array is one larger than the number of baskets
    assert basket_lengths[-1] == 0
    output_buffers = []
    N_baskets = len(basket_starts) - 1
    compressed_contents = []
    borders = []
    put_starts = []
    put_stops = []
    
    if isinstance(branch.interpretation, uproot.interpretation.jagged.AsJagged):
        counts = counts_[branch.count_branch.name]
        assert len(counts) == basket_entryoffsets[-1]
        offsets = cp.zeros(len(counts) + 1, dtype=cp.int64)
        offsets[1:] = cp.cumsum(counts)
        dtype = branch.interpretation.content.from_dtype
        content = cp.empty(offsets[-1].get(), dtype=dtype)
    
        for i in range(N_baskets):
            # Grab each basket's metadata and store for later
            data = get_data(i)
            # print(f"Reading basket {i} with {len(data)} bytes at entry offset {basket_entryoffsets[i]}:{basket_entryoffsets[i+1]}")
            put_starts.append(offsets[basket_entryoffsets[i]])
            put_stops.append(offsets[basket_entryoffsets[i+1]])
            # print(f"Destination: {put_start}:{put_stop}")
            compressed_content, border, output_buffer = basket_to_compcont_border(data, dtype, put_stops[i] - put_starts[i])
            compressed_contents.append(compressed_content)
            borders.append(border)
            output_buffers.append(output_buffer)
        
    elif isinstance(branch.interpretation, uproot.interpretation.numerical.AsDtype): 
        dtype = branch.interpretation.from_dtype
        content = cp.empty(basket_entryoffsets[-1], dtype=dtype)

        for i in range(N_baskets):
            data = get_data(i)
            put_starts.append(basket_entryoffsets[i])
            put_stops.append(basket_entryoffsets[i+1])
            # print(f"Reading basket {i} with {len(data)} bytes at entry offset {basket_entryoffsets[i]}:{basket_entryoffsets[i+1]}")
            compressed_content, border, output_buffer = basket_to_compcont_border(data, dtype, put_stops[i] - put_starts[i])
            compressed_contents.append(compressed_content)
            borders.append(border)
            output_buffers.append(output_buffer)
            
    branch_metadata = {
                        "N_baskets": N_baskets,
                        "content": content,
                        "compressed_contents": compressed_contents,
                        "borders": borders,
                        "put_starts": put_starts,
                        "put_stops": put_stops,
                    }

    return(compressed_contents, output_buffers, branch_metadata)

# Convenience function - avoids redudant count branch decompression
def get_count_branches(branches):
    count_branches = {}
    for branch in branches:
        if branch.count_branch.name not in count_branches.keys():
            count_branches[branch.count_branch.name] = branch.count_branch
    return list(count_branches.values())

# Convert cp array from big-endian to little-endian dtype
def byteswap(bigend):
    littleend = cp.empty_like(bigend)
    for i in range(4):
        littleend.view("u1")[i::4] = bigend.view("u1")[3-i::4]
    return littleend

# Main function - retrieves compressed bytes and needed metadatas to create array from decompressed baskets.
# Then decompresses on GPU, structures the arrays, and keeps them on the GPU.
def GPU_all_baskets_to_array(filehandle, branches, counts_call = False):
    arrays = {}
    branches_metadatas = {}
    output_buffers = []
    all_compressed_content = []
    counts = None
    if counts_call == False:
        count_branches = get_count_branches(branches)
        counts = GPU_all_baskets_to_array(filehandle, count_branches, counts_call = True)
    
    for branch in branches:
        compressed_content, output_buffers_temp, branch_metadata = get_branch_buffer_metadata(filehandle, branch, counts)
        all_compressed_content.extend(compressed_content)
        output_buffers.extend(output_buffers_temp)
        branches_metadatas[branch.name] = branch_metadata
        
    
    
    # Decompress the data on the GPU
    codec = NvCompBatchCodec("zstd")
    all_decompressed_content = codec.decode_batch(all_compressed_content, output_buffers)

    # Assemble the branches as awkward arrays
    for i, branch in enumerate(branches):
        # Get metadata
        branch_metadata = branches_metadatas[branch.name]
        N_baskets = branch_metadata["N_baskets"]
        content = branch_metadata["content"]
        put_starts = branch_metadata["put_starts"]
        put_stops = branch_metadata["put_stops"]
        borders = branch_metadata["borders"]
        
        if isinstance(branch.interpretation, uproot.interpretation.jagged.AsJagged):
            dtype = branch.interpretation.content.from_dtype
            branch_decompressed_contents = all_decompressed_content[:N_baskets] #Grab chunks of data

            for i in range(N_baskets):
                content[put_starts[i]:put_stops[i]] = branch_decompressed_contents[i].view(dtype)[:borders[i] // dtype.itemsize]
            # Root is big-endian, but nvidia only supports little-endian
            content = byteswap(content)
            arrays[branch.name] = ak.unflatten(content.astype(cp.float32), counts[branch.count_branch.name])
            
            # Slice out used chunks
            all_decompressed_content = all_decompressed_content[N_baskets:]
            
        elif isinstance(branch.interpretation, uproot.interpretation.numerical.AsDtype):
            dtype = branch.interpretation.from_dtype
            branch_decompressed_contents = all_decompressed_content[:N_baskets]
            for i in range(N_baskets):
                content[put_starts[i]:put_stops[i]] = branch_decompressed_contents[i].view(dtype)[:borders[i] // dtype.itemsize]
            # Root is big-endian, but nvidia only support little-endian
            content = byteswap(content)
            arrays[branch.name] = content.astype(branch.interpretation.to_dtype)
            
            # Slice out used chunks
            all_decompressed_content = all_decompressed_content[N_baskets:]
    
    return(arrays)


# /store/user/IDAP/RunIISummer20UL18NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/40000/BCB3E2FC-D575-0341-A211-5C9A8D8798B9.root
filename = "/home/fstrug/uscmshome/nobackup/GPU/kvikio_playground/TTToSemiLeptonic_UL18JMENanoAOD-zstd.root"
tree_name = "Events"
tree = uproot.open(filename+":"+"Events")
branches = [tree["Muon_pt"], tree["Muon_eta"], tree["Muon_phi"], tree["Jet_pt"], tree["Jet_eta"], tree["Jet_phi"], tree["FatJet_pt"], tree["FatJet_eta"], tree["FatJet_phi"], tree["Electron_pt"], tree["Electron_eta"], tree["Electron_phi"]]
with open(filename, "rb") as rawfile:
    myarrays = GPU_all_baskets_to_array(rawfile, branches)