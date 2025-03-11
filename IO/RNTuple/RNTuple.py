from kvikio.nvcomp_codec import NvCompBatchCodec
from kvikio import defaults, CuFile
from dataclasses_RNTuple import *
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

def cupy_insert0(arr):
    #Intended for flat cupy arrays
    array_len = arr.shape[0]
    array_dtype = arr.dtype
    out_arr = cp.empty(array_len + 1, dtype = array_dtype)
    cp.copyto(out_arr[1:], arr)
    out_arr[0] = 0
    return(out_arr)


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
            length = len(res)
            res[0::2] = content[length * 0 // 2 : length * 1 // 2]
            res[1::2] = content[length * 1 // 2 : length * 2 // 2]

        elif nbits == 32:
            # AAAAABBBBBCCCCCDDDDD needs to become
            # ABCDABCDABCDABCDABCD
            res = cp.empty(len(content), cp.uint8)
            length = len(res)
            res[0::4] = content[length * 0 // 4 : length * 1 // 4]
            res[1::4] = content[length * 1 // 4 : length * 2 // 4]
            res[2::4] = content[length * 2 // 4 : length * 3 // 4]
            res[3::4] = content[length * 3 // 4 : length * 4 // 4]

        elif nbits == 64:
            # AAAAABBBBBCCCCCDDDDDEEEEEFFFFFGGGGGHHHHH needs to become
            # ABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGH
            res = cp.empty(len(content), cp.uint8)
            length = len(res)
            res[0::8] = content[length * 0 // 8 : length * 1 // 8]
            res[1::8] = content[length * 1 // 8 : length * 2 // 8]
            res[2::8] = content[length * 2 // 8 : length * 3 // 8]
            res[3::8] = content[length * 3 // 8 : length * 4 // 8]
            res[4::8] = content[length * 4 // 8 : length * 5 // 8]
            res[5::8] = content[length * 5 // 8 : length * 6 // 8]
            res[6::8] = content[length * 6 // 8 : length * 7 // 8]
            res[7::8] = content[length * 7 // 8 : length * 8 // 8]

        content = res.view(dtype)

    if isbit:
        content = cp.unpackbits(
            destination.view(dtype=cp.uint8), bitorder="little"
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
                                 clusters_datas):

    cluster_range = range(start_cluster_idx, stop_cluster_idx)
    n_clusters = stop_cluster_idx - start_cluster_idx
    col_arrays = {} # collect content for each col
    j = 0
    for key in columns:
        if "column" in key and "union" not in key:
            # Get uncompressed array for key for all clusters
            j += 1
            key_nr = int(key.split("-")[1])
            col_decompressed_buffers = clusters_datas.grab_ColOutput(key_nr)
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
                nbits = (
                    in_ntuple.column_records[ncol].nbits
                    if ncol < len(in_ntuple.column_records)
                    else uproot.const.rntuple_col_num_to_size_dict[dtype_byte]
                    )
                
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

def GPU_read_col_cluster_pages(in_ntuple, ncol, cluster_i, filehandle):
    # Get cluster and pages metadatas
    linklist = in_ntuple.page_list_envelopes.pagelinklist[cluster_i]
    pagelist = linklist[ncol].pages if ncol < len(linklist) else []
    dtype_byte = in_ntuple.column_records[ncol].type
    split = dtype_byte in uproot.const.rntuple_split_types
    dtype_str = uproot.const.rntuple_col_num_to_dtype_dict[dtype_byte]
    isbit = dtype_str == "bit"
    # Prepare full output buffer
    total_len = np.sum([desc.num_elements for desc in pagelist], dtype=int)
    if dtype_str == "switch":
        dtype = np.dtype([("index", "int64"), ("tag", "int32")])
    elif dtype_str == "bit":
        dtype = np.dtype("bool")
    else:
        dtype = np.dtype(dtype_str)
    full_output_buffer = cp.empty(total_len, dtype = dtype)    

    # Check if col compressed/decompressed
    if isbit: # Need to correct length when dtype = bit
        total_len = int(numpy.ceil(total_len / 8))    
    total_bytes = np.sum([desc.locator.num_bytes for desc in pagelist])
    if (total_bytes != total_len * dtype.itemsize):
        isCompressed = True
    else:
        isCompressed = False
    Cluster_Contents = ColBuffers_Cluster(ncol,
                                          full_output_buffer,
                                          isCompressed)
    
    tracker = 0
    futures = []
    for page_desc in pagelist:
        num_elements = page_desc.num_elements
        loc = page_desc.locator
        n_bytes = loc.num_bytes
        tracker_end = tracker + num_elements
        out_buff = full_output_buffer[tracker:tracker_end]
        # If compressed, skip 9 byte header
        if isCompressed:
            comp_buff = cp.empty(n_bytes - 9, dtype = "b")
            fut = filehandle.pread(comp_buff,
                                  size = int(n_bytes - 9),
                                  file_offset = int(loc.offset+9))
        
        # If uncompressed, read directly into out_buff
        else:
            comp_buff = None
            fut = filehandle.pread(out_buff,
                                  size = int(n_bytes),
                                  file_offset = int(loc.offset))

        Cluster_Contents.add_page(comp_buff)
        Cluster_Contents.add_output(out_buff)
        futures.append(fut)
        tracker = tracker_end
            
    return (Cluster_Contents, futures)

def GPU_read_clusters(in_ntuple, columns, start_cluster_idx, stop_cluster_idx):
    cluster_range = range(start_cluster_idx, stop_cluster_idx)
    clusters_datas = Cluster_Refs()
    # Iterate through each cluster
    for cluster_i in cluster_range:
        with CuFile(in_ntuple.file.source.file_path, "rb") as filehandle:
            futures = []
            cluster_colrefs = Cluster_ColRefs(cluster_i)
            #Open filehandle and read columns for cluster
        
            for key in columns:
                if "column" in key and "union" not in key:
                    key_nr = int(key.split("-")[1])
    
                    (Col_ClusterBuffers,
                     future)           = GPU_read_col_cluster_pages(in_ntuple,
                                                                    key_nr,
                                                                    cluster_i,
                                                                    filehandle)
                    futures.extend(future)
                    cluster_colrefs.add_Col(Col_ClusterBuffers)
            for future in futures:
                future.get()

        cluster_colrefs.decompress()
        clusters_datas.add_cluster(cluster_colrefs)
    
    return(clusters_datas)    



def kvikUproot_openGPU(in_ntuple_path, columns, classname, prototype = True, entry_start = 0, entry_stop = None):
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

    # Read all columns 'compressed' data
    clusters_datas = GPU_read_clusters(in_ntuple,
                                       target_cols,
                                       start_cluster_idx,
                                       stop_cluster_idx)

    content_dict = Process_decompressed_content(in_ntuple,
                                          target_cols,
                                          start_cluster_idx,
                                          stop_cluster_idx,
                                          clusters_datas)

    container_dict = {}
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