from uproot.models.TBasket import _tbasket_format1 as format1
from uproot.models.TBasket import _tbasket_format2 as format2
from uproot.compression import _decompress_header_format as format3
import uproot
import awkward as ak
import kvikio
from kvikio import defaults, CuFile
from kvikio.nvcomp_codec import NvCompBatchCodec
import cupy as cp
import concurrent
import time

def byteswap(bigend, profile = False):
    with cp.cuda.Device() as d:
        littleend = cp.empty_like(bigend)
        for i in range(4):
            littleend.view("u1")[i::4] = bigend.view("u1")[3-i::4]
        if profile: d.synchronize()
    return littleend

def read_branch(filepath, branch, profile):
    if profile:
        time_s = time.monotonic()
    with open(filepath, "rb") as filehandle:

        basket_starts = branch.all_members["fBasketSeek"]
        basket_lengths = branch.all_members["fBasketBytes"]
        basket_entryoffsets = branch.all_members["fBasketEntry"]
        N_baskets = len(basket_starts) - 1
        assert basket_starts[-1] == 0 and basket_lengths[-1] == 0

        output_buffers, compressed_contents, borders = [], [], []
        compressed_buffers, compressed_buffer_sizes = [], []
        put_starts, put_stops = [None]*N_baskets, [None]*N_baskets
        fKeylens_list = []
        fLasts_list = []
        offset_remainders = []
        buffer_end_cut = []
        content = None
        

        is_jagged = False
        if isinstance(branch.interpretation, uproot.interpretation.jagged.AsJagged):
            dtype  = branch.interpretation.content.from_dtype
            is_jagged = True
            
        elif isinstance(branch.interpretation, uproot.interpretation.numerical.AsDtype): 
            dtype = branch.interpretation.from_dtype
            with cp.cuda.Device() as d:
                content = cp.empty(basket_entryoffsets[-1], dtype=dtype)
                if profile: d.synchronize()
        else:
            raise TypeError("Unsupported interpretation type")

        futures = []
        with CuFile(filepath, "rb") as f:
            for i in range(N_baskets):
                if is_jagged:
                    pass
                else:
                    put_start = basket_entryoffsets[i]
                    put_stop = basket_entryoffsets[i+1]
                
                ################
                # Read uproot format1 header
                filehandle.seek(basket_starts[i])
                data = filehandle.read(basket_lengths[i])
                (
                    fNbytes,
                    key_version,
                    fObjlen,
                    fDatime,
                    fKeylen,
                    fCycle,
                ) = format1.unpack(data[: format1.size])
                assert fNbytes - fKeylen != fObjlen # this is only true for uncompressed baskets
                fKeylens_list.append(fKeylen)
                
                ################
                # Read uproot format2 header
                (
                    fVersion,
                    fBufferSize,
                    fNevBufSize,
                    fNevBuf,
                    fLast,
                ) = format2.unpack(data[fKeylen - format2.size - 1: fKeylen - 1])
                fLasts_list.append(fLast)
                border = fLast - fKeylen
    
                # Check that border and expected entries agree
                assert border <= fObjlen, f"{border} <= {fObjlen}"
                
                ################
                #Read uproot format3 header
                algo, method, c1, c2, c3, u1, u2, u3 = format3.unpack(data[fKeylen : fKeylen + format3.size])
                assert algo == b"ZS", f"Unsupported algorithm: {algo}"  # zstd, we can support more later
                block_compressed_bytes = c1 + (c2 << 8) + (c3 << 16)
                block_uncompressed_bytes = u1 + (u2 << 8) + (u3 << 16)
                with cp.cuda.Device() as d:
                    output_buffer = cp.empty(block_uncompressed_bytes // dtype.itemsize, dtype = dtype)
                    if profile: d.synchronize()                
                
                borders.append(border)
                output_buffers.append(output_buffer)
    

                #################
                # Align offset with multiple of 4096 bytes
                # The GPU page size is 4kB, so all reads should be at an offset that is a multiple of 4096 bytes.
                offset = (int(basket_starts[i] + fKeylen + format3.size) // 4096) * 4096
                offset_remainder = int(basket_starts[i] + fKeylen + format3.size) % 4096
                offset_remainders.append(offset_remainder)
                
                # Round buffer size up to multiple of 4096 bytes
                size_buffer = int(block_compressed_bytes) + offset_remainder
                buffer_end_cut.append(size_buffer) # store to truncate post read
                if (size_buffer % 4096) > 0:
                    size_buffer = ((size_buffer // 4096) + 1) * 4096
                assert (offset % 4096) == 0
                assert (size_buffer % 4096) == 0
                with cp.cuda.Device() as d:
                    buffer = cp.empty(size_buffer, dtype = "b")
                    if profile: d.synchronize()

                #################
                # Read data into buffer
                if profile & (i == 0):
                    start_gpu = cp.cuda.Event()
                    end_gpu = cp.cuda.Event()
                    start_gpu.record()

                #Start async read
                future = f.pread(buffer,
                                 size = int(size_buffer), 
                                 file_offset = int(offset))
                
                compressed_contents.append(buffer)
                futures.append(future)

            ### Ensure all reads complete before accessing memory
            # Profiling file read
            for future in futures:
                future.get() # Wait for all threads to complete before returning branch compressed contents

            if profile == True:
                end_gpu.record()
                end_gpu.synchronize()
                t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
                print("")
                print(branch.name, t_gpu, "ms")
                print("Baskets :", N_baskets, "\n Basket_lengths :", basket_lengths)
        
        # Truncate buffers
        for i, buffer in enumerate(compressed_contents):
            preamble_length = offset_remainders[i]
            buffer = buffer[preamble_length:buffer_end_cut[i]]
            compressed_contents[i] = buffer

        branch_metadata = {
            "N_baskets": N_baskets,
            "content": content,
            "compressed_contents": compressed_contents,
            "borders": borders,
            "put_starts": put_starts,
            "put_stops": put_stops,
            "fKeylens": fKeylens_list,
            "fLasts": fLasts_list
        }

    return (compressed_contents, output_buffers, branch_metadata) 

def assemble_arrays(branches, branches_metadatas, decompressed_content, profile = False):
    # Assemble the branch arrays
    arrays = {}
    for i, branch in enumerate(branches):
        # Get metadata
        branch_metadata = branches_metadatas[branch.name]
        N_baskets = branch_metadata["N_baskets"]
        content = branch_metadata["content"]
        put_starts = branch_metadata["put_starts"]
        put_stops = branch_metadata["put_stops"]
        borders = branch_metadata["borders"]
        fKeylens_list = branch_metadata["fKeylens"]
        fLasts_list = branch_metadata["fLasts"]
        basket_entryoffsets = branch.all_members["fBasketEntry"]
        
        if isinstance(branch.interpretation, uproot.interpretation.jagged.AsJagged):
            dtype = branch.interpretation.content.from_dtype
            branch_decompressed_contents = decompressed_content[:N_baskets] #Grab chunks of data
            assert (N_baskets) == (len(branch_decompressed_contents))
            
            flattened_contents = cp.concatenate([contents.view(dtype)[:border // dtype.itemsize] 
                                                 for contents, border in zip(branch_decompressed_contents, borders)])
            
            ####################################
            # Get counts from the decompressed buffer
            offsets = [(byteswap(contents[border//dtype.itemsize:]).view(cp.int32)[1:] - fKeylen) 
                                                 for contents, border, fKeylen in zip(branch_decompressed_contents, borders, fKeylens_list)]
            for i, offset in enumerate(offsets):
                if i == 0:
                    offset_temp = offset
                    offset_temp[0] = 0
                    offset_temp[-1] = borders[i]
                    offsets[i] = offset_temp
                else:
                    offset_temp = offset[1:len(offset)]
                    offset_temp[-1] = borders[i]
                    offset_temp = offset_temp + offsets[i-1][-1]
                    offsets[i] = offset_temp
            
            
            flattened_offsets = cp.concatenate(offsets)>>2
            for i in range(N_baskets):
                put_starts[i] = flattened_offsets[basket_entryoffsets[i]]
                put_stops[i] = flattened_offsets[basket_entryoffsets[i+1]]
            
            counts = cp.diff(flattened_offsets)
            content = cp.empty(flattened_offsets[-1].get(), dtype=dtype)          
            #########################################################
            
            content[put_starts[0]:put_stops[-1]] = flattened_contents

            # Root is big-endian, nvidia only supports little-endian
            content = byteswap(content, profile)
            arrays[branch.name] = ak.unflatten(content.astype(cp.float32), counts)
            # Slice out used chunks
            decompressed_content = decompressed_content[N_baskets:]
            
        elif isinstance(branch.interpretation, uproot.interpretation.numerical.AsDtype):
            dtype = branch.interpretation.from_dtype
            branch_decompressed_contents = decompressed_content[:N_baskets]
            assert (N_baskets) == (len(branch_decompressed_contents))
            
            flattened_contents = cp.concatenate([contents.view(dtype)[:border // dtype.itemsize] 
                                                 for contents, border in zip(branch_decompressed_contents, borders)])
            content[put_starts[0]:put_stops[-1]] = flattened_contents
            
            # Root is big-endian, but nvidia only support little-endian
            content = byteswap(content, profile)
            arrays[branch.name] = content.astype(branch.interpretation.to_dtype)
            # Slice out used chunks
            decompressed_content = decompressed_content[N_baskets:]
    
    return(arrays)

def GPU_CuFile_all_baskets_to_array(filepath, branches, profile):
    #kvikio.defaults.num_threads_reset(1)          # default = 1
    #kvikio.defaults.gds_threshold_reset(1048576) # default = 1048576
    
    #if profile == True:
        #print("KvikIO is using {} threads.".format(kvikio.libkvikio.thread_pool_nthreads()))
    branches_metadatas = {}
    output_buffers = []
    all_compressed_content = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, branch in enumerate(branches):
            futures.append(
                executor.submit(
                    read_branch, filepath, branch, profile
                )
            )
        for i, future in enumerate(futures):
            compressed_content, output_buffers_temp, branch_metadata = future.result()
            all_compressed_content.extend(compressed_content)
            output_buffers.extend(output_buffers_temp)
            branches_metadatas[branches[i].name] = branch_metadata
    
    # Decompress the data on the GPU
    codec = NvCompBatchCodec("zstd")
    all_decompressed_content = codec.decode_batch(all_compressed_content, output_buffers)

    # Assemble the branch arrays
    arrays = assemble_arrays(branches, branches_metadatas, all_decompressed_content, profile)

    return(arrays)

def kvikuproot_open(filepath, branches, TTree_name, profile = False):
    TTree = uproot.open(filepath + ":" + TTree_name)
    branches = [TTree[branch] for branch in branches]
    events = ak.zip(GPU_CuFile_all_baskets_to_array(filepath, branches, profile), depth_limit = 1)
    return(events)

path = "/home/fstrug/uscmshome/nobackup/GPU/kvikio_playground/TTToSemiLeptonic_UL18JMENanoAOD-zstd.root"
branches = ["Electron_pt", "Electron_eta", "Electron_phi",
            "Muon_pt", "Muon_eta", "Muon_phi",
            "FatJet_pt", "FatJet_eta", "FatJet_phi",
            "Jet_pt", "Jet_eta", "Jet_phi",
            "MET_pt"]
TTree_name = "Events"
events_kvikIO = kvikuproot_open(path, branches, TTree_name)