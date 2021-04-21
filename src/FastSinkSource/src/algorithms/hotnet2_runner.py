from .alg_utils import str_
from scipy import sparse
import numpy as np
import scipy.io
from pathlib import Path

def setupInputs(run_obj):
    print("inside hotnet2_runner: setupInputs")
    print("checking run_obj.net_obj ",run_obj.net_obj.sparse_netx_graphs)
    #print("checking run_obj.ann_obj ",run_obj.ann_obj.ann_matrix)
    #this are the uniprot IDs
    #print("checking prots ",run_obj.ann_obj.prots)
    #this maps the uniprot IDs with index
    #print("checking node to index ",run_obj.ann_obj.node2idx)
    '''input_net_path = Path("../../../../fss_inputs/networks/stringv11/400/sparse-nets/") 
    input_net_file = input_net_path / 'c400-combined_score-sparse-nets.mat'
    f = open(input_net_file)
    mat = scipy.io.loadmat(f)
    print('printing mattrix ',mat)'''


def run(run_obj):
    #print("inside hotnet2_runner: run")
    return

def setupOutputs(run_obj):
    #print("inside hotnet2_runner: setupOutputs")
    return

def setup_params_str(weight_str, params, name):
    #print("inside hotnet2_runner: setup_params_str")
    return

def get_alg_type():
    return 'hotnet2'