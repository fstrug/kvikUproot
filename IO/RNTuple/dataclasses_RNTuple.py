from dataclasses import dataclass, field
from kvikio.nvcomp_codec import NvCompBatchCodec
import cupy as cp
import functools
import operator

@dataclass
class ColBuffers_Cluster:
    """
    A Cluster_ColBuffers is a cupy ndarray that contains the compressed and 
    decompression output buffers for a particular column in a particular cluster
    of all pages. It contains pointers to portions of the cluster data
    which correspond to the different pages of that cluster. 
    """

    key: str
    data: cp.ndarray
    isCompressed: bool
    pages: list[cp.ndarray] = field(default_factory=list)
    output: list[cp.ndarray] = field(default_factory=list)

    def add_page(self, page: cp.ndarray):
        self.pages.append(page)

    def add_output(self, buffer: cp.ndarray):
        self.output.append(buffer)

@dataclass
class Cluster_ColRefs:
    """
    A Cluster_ColRefs is a set of dictionaries containing the Cluster_ColBuffers
    for all requested columns in a given cluster. Columns are separated by 
    whether they are compressed or uncompressed. Compressed columns can be
    decompressed. 
    """
    cluster_i: int
    data_dict: dict[str: list[cp.ndarray]] = field(default_factory=dict)
    data_dict_comp: dict[str: list[cp.ndarray]] = field(default_factory=dict)
    data_dict_uncomp: dict[str: list[cp.ndarray]] = field(default_factory=dict)

    def add_Col(self, ColBuffers_Cluster):
        self.data_dict[ColBuffers_Cluster.key] = ColBuffers_Cluster
        if ColBuffers_Cluster.isCompressed == True:
            self.data_dict_comp[ColBuffers_Cluster.key] = ColBuffers_Cluster
        else:
            self.data_dict_uncomp[ColBuffers_Cluster.key] = ColBuffers_Cluster

    def decompress(self, alg = "zstd"):
        # Combine comp and output buffers into two flattened lists
        list_ColBuffers = list(self.data_dict_comp.values())
        list_pagebuffers = [buffers.pages for buffers in list_ColBuffers]
        list_outputbuffers = [buffers.output for buffers in list_ColBuffers]

        list_pagebuffers = functools.reduce(operator.iconcat, list_pagebuffers, [])
        list_outputbuffers = functools.reduce(operator.iconcat, list_outputbuffers, [])

        # Decompress
        if len(list_outputbuffers) == 0:
            print("No output buffers provided for decompression")
        assert len(list_pagebuffers) != 0, "No page buffers to decompress"

        codec = NvCompBatchCodec(alg)
        codec.decode_batch(list_pagebuffers, list_outputbuffers)

@dataclass        
class Cluster_Refs:
    """"
    A Cluster_refs is a dictionary containing the Cluster_ColRefs for multiple
    clusters.
    """
    clusters: [int] = field(default_factory=list)
    refs: dict[int: Cluster_ColRefs] = field(default_factory=dict)

    def add_cluster(self, Cluster):
        cluster_i = Cluster.cluster_i
        self.clusters.append(cluster_i)
        self.refs[cluster_i] = Cluster

    def grab_ColOutput(self, nCol):
        output_list = []
        for cluster in self.refs.values():
            colbuffer = cluster.data_dict[nCol].data
            output_list.append(colbuffer)
        
        return output_list
            

            