from kvikio.nvcomp_codec import NvCompBatchCodec
from kvikio import defaults, CuFile
import uproot
import numpy as np
import numpy
import cupy as cp
import awkward as ak

def _recursive_find(form, res):
    ak = uproot.extras.awkward()

    if hasattr(form, "form_key"):
        if form.form_key not in res:
            res.append(form.form_key)
    if hasattr(form, "contents"):
        for c in form.contents:
            _recursive_find(c, res)
    if hasattr(form, "content") and issubclass(type(form.content), ak.forms.Form):
        _recursive_find(form.content, res)

def GPU_read_col_cluster_pages(in_ntuple, ncol, cluster_i, filehandle):
    linklist = in_ntuple.page_list_envelopes.pagelinklist[cluster_i]
    pagelist = linklist[ncol].pages if ncol < len(linklist) else []
    dtype_byte = in_ntuple.column_records[ncol].type
    dtype_str = uproot.const.rntuple_col_num_to_dtype_dict[dtype_byte]
    total_len = np.sum([desc.num_elements for desc in pagelist], dtype=int)
    if dtype_str == "switch":
        dtype = np.dtype([("index", "int64"), ("tag", "int32")])
    elif dtype_str == "bit":
        dtype = np.dtype("bool")
    else:
        dtype = np.dtype(dtype_str)
    full_output_buffer = cp.empty(total_len, dtype = dtype)    
    split = dtype_byte in uproot.const.rntuple_split_types
    

    n_pages = len(pagelist)
    output_buffers = []
    compressed_buffers = []
    futures = []

    tracker = 0
    nelements_tracker = 0
    for page_desc in pagelist:
        n_elements = page_desc.num_elements
        loc = page_desc.locator
        n_bytes = loc.num_bytes
        isbit = dtype_str == "bit"
        len_divider = 8 if isbit else 1
        num_elements = n_elements
        if isbit:
            num_elements_toread = int(numpy.ceil(num_elements / 8))
        elif dtype_str in ("real32trunc", "real32quant"):
            num_elements_toread = int(numpy.ceil((num_elements * 4 * nbits) / 32))
            dtype = numpy.dtype("uint8")
        else:
            num_elements_toread = num_elements
        nelements_tracker += num_elements_toread
        uncomp_size = num_elements_toread * dtype.itemsize
        tracker_end = tracker + num_elements_toread
        out_buff = full_output_buffer[tracker:tracker_end]
        # print("Page {} base buffer ID: ".format(i) ,id(out_buff.base))
        # Use locator to read page
        comp_buff = cp.empty(n_bytes - 9, dtype = "b")
        fut = filehandle.pread(comp_buff,
                              size = int(n_bytes - 9),
                              file_offset = int(loc.offset + 9))


        output_buffers.append(out_buff)
        compressed_buffers.append(comp_buff)
        futures.append(fut)
        tracker = tracker_end
    # Because some columns contain extra bits in the compressed buffer that 
    # get 'chopped' off, need to check that total_len == sum(num_elements_to_read)
    # for prototype for now. Only relevant for cols with (isBit = True)
    # assert(total_len == nelements_tracker)
    
    #for future in futures:
    #    future.get()
            
    return (compressed_buffers, output_buffers, futures, full_output_buffer)
    
        
    
def GPU_read_col_clusters(in_ntuple, ncol, cluster_range, dtype_byte):
    filepath = in_ntuple.file.source.file_path
    compressed_buffers = []
    output_buffers = []
    futures = []
    cluster_output_buffers = []
    #print("Reading column {}".format(ncol))
    with CuFile(filepath, "rb") as filehandle:
        for i, cluster_i in enumerate(cluster_range):
            (cluster_compressed_buffers,
             page_output_buffers,
             cluster_futures,
             cluster_full_output_buffer) = GPU_read_col_cluster_pages(in_ntuple,
                                                                ncol,
                                                                cluster_i,
                                                                filehandle)
            # print("Cluster {} output buffer ID after return".format(cluster_i), id(cluster_full_output_buffer))
            # print("Page base output buffer ID after return", id(page_output_buffers[0].base))
            # Aggregate results
            compressed_buffers.extend(cluster_compressed_buffers)
            output_buffers.extend(page_output_buffers)
            futures.extend(cluster_futures)
            cluster_output_buffers.append(cluster_full_output_buffer)
            # print("Cluster {} output buffer ID after append into aggregate list".format(cluster_i), id(cluster_output_buffers[0]))
            # print("Page base output buffer ID after extend into aggregate list", id(output_buffers[0].base))
        for future in futures:
            future.get()
    return (compressed_buffers, output_buffers, cluster_output_buffers)
    

def GPU_read_cols(in_ntuple, columns, start_cluster_idx, stop_cluster_idx):
    compressed_buffers = []
    all_page_output_buffers = []
    all_cluster_output_buffers = []
    for key in columns:
        if "column" in key and "union" not in key:
            key_nr = int(key.split("-")[1])
            dtype_byte = in_ntuple.ntuple.column_records[key_nr].type
            cluster_range = range(start_cluster_idx, stop_cluster_idx)
            
            (compressed_buffers_col,
             page_output_buffers_col,
             cluster_output_buffers)= GPU_read_col_clusters(in_ntuple,
                                                                       key_nr,
                                                                       cluster_range,
                                                                       dtype_byte)
            compressed_buffers.extend(compressed_buffers_col)
            all_page_output_buffers.extend(page_output_buffers_col)
            all_cluster_output_buffers.extend(cluster_output_buffers)
            
    #print(len(compressed_buffers), len(output_buffers))
    #print(compressed_buffers, output_buffers)
    return(compressed_buffers, all_page_output_buffers, all_cluster_output_buffers)


def process_page_decompressed_buffer(destination, desc, dtype_str, dtype, nbits, split):
    context = {}
    # bool in RNTuple is always stored as bits
    isbit = dtype_str == "bit"
    num_elements = len(destination)
    if split:
        content = cp.copy(destination).view(cp.uint8)

        if nbits == 16:
            # AAAAABBBBB needs to become
            # ABABABABAB
            res = cp.empty(len(content), cp.uint8)
            res[0::2] = content[len(res) * 0 // 2 : len(res) * 1 // 2]
            res[1::2] = content[len(res) * 1 // 2 : len(res) * 2 // 2]

        elif nbits == 32:
            # AAAAABBBBBCCCCCDDDDD needs to become
            # ABCDABCDABCDABCDABCD
            res = cp.empty(len(content), cp.uint8)
            res[0::4] = content[len(res) * 0 // 4 : len(res) * 1 // 4]
            res[1::4] = content[len(res) * 1 // 4 : len(res) * 2 // 4]
            res[2::4] = content[len(res) * 2 // 4 : len(res) * 3 // 4]
            res[3::4] = content[len(res) * 3 // 4 : len(res) * 4 // 4]

        elif nbits == 64:
            # AAAAABBBBBCCCCCDDDDDEEEEEFFFFFGGGGGHHHHH needs to become
            # ABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGH
            res = cp.empty(len(content), cp.uint8)
            res[0::8] = content[len(res) * 0 // 8 : len(res) * 1 // 8]
            res[1::8] = content[len(res) * 1 // 8 : len(res) * 2 // 8]
            res[2::8] = content[len(res) * 2 // 8 : len(res) * 3 // 8]
            res[3::8] = content[len(res) * 3 // 8 : len(res) * 4 // 8]
            res[4::8] = content[len(res) * 4 // 8 : len(res) * 5 // 8]
            res[5::8] = content[len(res) * 5 // 8 : len(res) * 6 // 8]
            res[6::8] = content[len(res) * 6 // 8 : len(res) * 7 // 8]
            res[7::8] = content[len(res) * 7 // 8 : len(res) * 8 // 8]

        content = res.view(dtype)

    if isbit:
        content = (
            cp.unpackbits(content.view(dtype=cp.uint8))
            .reshape(-1, 8)[:, ::-1]
            .reshape(-1)
        )
    elif dtype_str in ("real32trunc", "real32quant"):
        if nbits == 32:
            content = content.view(cp.uint32)
        elif nbits % 8 == 0:
            new_content = cp.zeros((num_elements, 4), cp.uint8)
            nbytes = nbits // 8
            new_content[:, :nbytes] = content.reshape(-1, nbytes)
            content = new_content.view(cp.uint32).reshape(-1)
        else:
            ak = uproot.extras.awkward()
            vm = ak.forth.ForthMachine32(
                f"""input x output y uint32 {num_elements} x #{nbits}bit-> y"""
            )
            vm.run({"x": content})
            content = vm["y"]
        if dtype_str == "real32trunc":
            content <<= 32 - nbits

    # needed to chop off extra bits incase we used `unpackbits`
    destination[:] = content[:num_elements]



def Process_decompressed_content(in_ntuple, columns,
                                 start_cluster_idx, stop_cluster_idx,
                                 all_cluster_decompressed_buffers):

    cluster_range = range(start_cluster_idx, stop_cluster_idx)
    n_clusters = stop_cluster_idx - start_cluster_idx
    col_arrays = {} # collect content for each col
    j = 0
    for key in columns:
        if "column" in key and "union" not in key:
            # Get decompressed array for key for all clusters
            col_decompressed_buffers = all_cluster_decompressed_buffers[j*(n_clusters):(j+1)*n_clusters]
            j += 1
            key_nr = int(key.split("-")[1])
            #print("Processing key {} buffers".format(key_nr))
            dtype_byte = in_ntuple.ntuple.column_records[key_nr].type
            arrays = []
            ncol = key_nr
            
            for i in cluster_range:
                # Get decompressed buffer corresponding to cluster i
                cluster_buffer = col_decompressed_buffers[i]
                
                # Get pagelist and metadatas
                linklist = in_ntuple.page_list_envelopes.pagelinklist[i]
                pagelist = linklist[ncol].pages if ncol < len(linklist) else []
                dtype_byte = in_ntuple.column_records[ncol].type
                dtype_str = uproot.const.rntuple_col_num_to_dtype_dict[dtype_byte]
                total_len = np.sum([desc.num_elements for desc in pagelist], dtype=int)
                if dtype_str == "switch":
                    dtype = cp.dtype([("index", "int64"), ("tag", "int32")])
                elif dtype_str == "bit":
                    dtype = cp.dtype("bool")
                else:
                    dtype = cp.dtype(dtype_str)
                split = dtype_byte in uproot.const.rntuple_split_types
                zigzag = dtype_byte in uproot.const.rntuple_zigzag_types
                delta = dtype_byte in uproot.const.rntuple_delta_types
                index = dtype_byte in uproot.const.rntuple_index_types
                nbits = uproot.const.rntuple_col_num_to_size_dict[dtype_byte]
                
                # Begin looping through pages
                tracker = 0
                cumsum = 0
                for page_desc in pagelist:
                    n_elements = page_desc.num_elements
                    tracker_end = tracker + n_elements
                    
                    # Get content associated with page
                    page_buffer = cluster_buffer[tracker:tracker_end]
                    process_page_decompressed_buffer(page_buffer,
                                                    page_desc,
                                                    dtype_str,
                                                    dtype,
                                                    nbits,
                                                    split)

                    if delta:
                        cluster_buffer[tracker] -= cumsum
                        cumsum += cp.sum(cluster_buffer[tracker:tracker_end])
                    tracker = tracker_end

                if index:
                    cluster_buffer = cupy_insert0(cluster_buffer)  # for offsets
                if zigzag:
                    cluster_buffer = _from_zigzag(cluster_buffer)
                elif delta:
                    cluster_buffer = cp.cumsum(cluster_buffer)
                elif dtype_str == "real32trunc":
                    cluster_buffer = cluster_buffer.view(cp.float32)
                elif dtype_str == "real32quant" and ncol < len(self.column_records):
                    min_value = self.column_records[ncol].min_value
                    max_value = self.column_records[ncol].max_value
                    cluster_content = min_value + cluster_content.astype(cp.float32) * (max_value - min_value) / (
                        (1 << nbits) - 1
                    )
                    cluster_buffer = cluster_buffer.astype(cp.float32)
                arrays.append(cluster_buffer)

            if dtype_byte in uproot.const.rntuple_delta_types:
                # Extract the last offset values:
                last_elements = [
                    arr[-1].get() for arr in arrays[:-1]
                ]  # First value always zero, therefore skip first arr.
                # Compute cumulative sum using itertools.accumulate:
                last_offsets = np.cumsum(last_elements)
                
                # Add the offsets to each array
                for i in range(1, len(arrays)):
                    arrays[i] += last_offsets[i - 1]
                # Remove the first element from every sub-array except for the first one:
                arrays = [arrays[0]] + [arr[1:] for arr in arrays[1:]]
    
            res = cp.concatenate(arrays, axis=0)
    
            if True:
                first_element_index = in_ntuple.column_records[ncol].first_element_index
                res = cp.pad(res, (first_element_index, 0))
            
            col_arrays[key_nr] = res
    
    return col_arrays

def cupy_insert0(arr):
    #Intended for flat cupy arrays
    array_len = arr.shape[0]
    array_dtype = arr.dtype
    out_arr = cp.empty(array_len + 1, dtype = array_dtype)
    cp.copyto(out_arr[1:], arr)
    out_arr[0] = 0
    return(out_arr)
                    

            
def kvikuproot_open_RNTuple(in_ntuple_path, columns, classname, prototype = True, entry_start = 0, entry_stop = None):
    in_ntuple = uproot.open(in_ntuple_path)[classname]
    entry_stop = entry_stop or in_ntuple.ntuple.num_entries
    
    # Find clusters to read that contain data from entry_start to entry_stop
    clusters = in_ntuple.ntuple.cluster_summaries
    cluster_starts = np.array([c.num_first_entry for c in clusters])

    start_cluster_idx = (
        np.searchsorted(cluster_starts, entry_start, side="right") - 1
    )
    stop_cluster_idx = np.searchsorted(cluster_starts, entry_stop, side="right")
    cluster_num_entries = np.sum(
        [c.num_entries for c in clusters[start_cluster_idx:stop_cluster_idx]]
    )

    # Get form for requested columns
    form = in_ntuple.to_akform().select_columns(
        columns, prune_unions_and_records=False
    )

    # Only read columns mentioned in the awkward form
    target_cols = []
    container_dict = {}
    _recursive_find(form, target_cols)


    #print("Begin reading")
    # Read all columns 'compressed' data
    (all_compressed_buffers,
     all_output_buffers,
     all_cluster_output_buffers) = GPU_read_cols(in_ntuple,
                                                 target_cols,
                                                 start_cluster_idx,
                                                 stop_cluster_idx)
    #print("Reading complete")

    # Decompression GPU
    #print("\nGPU decompression")
    codec = NvCompBatchCodec("zstd")
    codec.decode_batch(all_compressed_buffers, all_output_buffers)
    #print("GPU decompression complete")

    #print("Process decompressed data")
    content_dict = Process_decompressed_content(in_ntuple,
                                          target_cols,
                                          start_cluster_idx,
                                          stop_cluster_idx,
                                          all_cluster_output_buffers)
    #if prototype:
    #    return(content_dict, form)

    # Pick back up here
    container_dict = {}


    # return(ak.unflatten(content, cp.diff(offsets))) 
    
    # Debugging
    for key in target_cols:
        if "column" in key and "union" not in key:
            key_nr = int(key.split("-")[1])
            dtype_byte = in_ntuple.ntuple.column_records[key_nr].type
            content = content_dict[key_nr].view(dtype = 'b')

            if "cardinality" in key:
                content = numpy.diff(content)
            if dtype_byte == uproot.const.rntuple_col_type_to_num_dict["switch"]:
                kindex, tags = _split_switch_bits(content)
                # Find invalid variants and adjust buffers accordingly
                invalid = numpy.flatnonzero(tags == -1)
                if len(invalid) > 0:
                    kindex = numpy.delete(kindex, invalid)
                    tags = numpy.delete(tags, invalid)
                    invalid -= numpy.arange(len(invalid))
                    optional_index = numpy.insert(
                        numpy.arange(len(kindex), dtype=numpy.int64), invalid, -1
                    )
                else:
                    optional_index = numpy.arange(len(kindex), dtype=numpy.int64)
                container_dict[f"{key}-index"] = optional_index
                container_dict[f"{key}-union-index"] = kindex
                container_dict[f"{key}-union-tags"] = tags
            else:
                # don't distinguish data and offsets
                container_dict[f"{key}-data"] = content
                container_dict[f"{key}-offsets"] = content
    cluster_offset = cluster_starts[start_cluster_idx]
    entry_start -= cluster_offset
    entry_stop -= cluster_offset

    return ak.from_buffers(
        form, cluster_num_entries, container_dict, allow_noncanonical_form=True,
        backend = "cuda"
    )[entry_start:entry_stop]

def GPU_akzip_RNTuple(container_dict, form):
    # Associate content/offsets with their keys
    arrays = {}

    form_contents = form.contents
    for i, field in enumerate(form.fields):
        field_content = form_contents[i]

        # if field_content is instance(ak.forms.NumpyForm):
        
        if isinstance(field_content, ak.forms.ListOffsetForm):
            offsets_key = int(field_content.form_key.split("-")[1])
            content_key = int(field_content.content.form_key.split("-")[1])
            array_ = ak.unflatten(container_dict[content_key], cp.diff(container_dict[offsets_key]))
            arrays[field] = array_
        elif isinstance(field_content, ak.forms.NumpyForm):
            content_key = int(field_content.form_key.split("-")[1])
            arrays[field] = ak.Array(container_dict[content_key])
    
    return(ak.zip(arrays, depth_limit = 1))

def kvikUproot_openGPU(in_ntuple_path, columns, classname):
    events_RNTuple_GPU = kvikuproot_open_RNTuple(in_ntuple_path, columns, classname)
    # events_RNTuple_GPU = GPU_akzip_RNTuple(content_dict, form)

    return(events_RNTuple_GPU)