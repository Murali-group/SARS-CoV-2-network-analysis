from .alg_utils import str_
from scipy import sparse
import numpy as np
import scipy.io
from pathlib import Path
import os

def setupInputs(run_obj):
    print("inside hotnet2_runner: setupInputs")
    #print('checking network ', run_obj.net_obj.sparse_networks[0].shape)
    '''row_nums = run_obj.net_obj.sparse_networks[0].shape[0]
    col_nums = run_obj.net_obj.sparse_networks[0].shape[1]
    net_arr = run_obj.net_obj.sparse_networks[0].toarray()
    print(net_arr)'''
    '''converting to input file for makeNetwork'''
    '''with open("edgeList.txt", "w") as file2:
        for i in range(row_nums):
            for j in range(col_nums):
                if(net_arr[i][j]):
                    file2.write(str(i)+' '+str(j)+' '+str(net_arr[i][j])+'\n')'''
    #print('checking dictionary size ',len(run_obj.ann_obj.node2idx))
    #print(run_obj.ann_obj.node2idx)

    '''with open("indexGene.txt", "w") as file3:
        for (key, value) in run_obj.ann_obj.node2idx.items():
            file3.write(str(value)+' '+str(key)+' '+'0\n')'''

    #print('checking network ', run_obj.net_obj.sparse_networks)
    #print('checking shape ', run_obj.ann_obj.ann_matrix.shape)
    #print('checking value ',run_obj.ann_obj.ann_matrix[0,0:20])
    #print("checking run_obj.ann_obj ",run_obj.ann_obj.ann_matrix)
    '''y = run_obj.ann_obj.ann_matrix[0, :].toarray()'''
    #positives = (y > 0).nonzero()[1]
    #print("checking positives ", positives)
    #writing input of the makeHeatFile
    '''with open("proteinToFreq.txt", "w") as file1:
        ind = 0
        for val in y[0]:
            if(val==-1):
                val = 0
            #prot = run_obj.ann_obj.node2idx.keys()[run_obj.ann_obj.node2idx.values().index(ind)]
            prot = list(run_obj.ann_obj.node2idx.keys())[list(run_obj.ann_obj.node2idx.values()).index(ind)]
            file1.write(str(prot)+'\t'+str(val)+'\n')
            ind += 1'''
        
    #print(y[0])
    #print('checking size of converted array ',len(y[0]))

    #this are the uniprot IDs
    #print("checking prots ",run_obj.ann_obj.prots)
    #this maps the uniprot IDs with index
    #print("checking node to index ",run_obj.ann_obj.node2idx)

def run(run_obj):
    print("inside hotnet2_runner: run")
    #scripts to run makeHeatFile, makeNetworkFile and HotNet2

    myHeatFile = Path('data/heats/pan12.gene2freq.json')
    myNetworkDir = Path('data/networks/hint+hi2012')
    myResultsDir = Path('results')
    if (myHeatFile.is_file()==False):
        print('heat file does not exist')
        os.system('python makeHeatFile.py scores -hf data/heats/pan12.gene2freq.txt -o  data/heats/pan12.gene2freq.json -n  pan12.freq')
    else:
        print('heat file already exists')
    if (myNetworkDir.is_dir()==False):
        print('network file does not exist')
        os.system('python makeNetworkFiles.py -e  data/networks/hint+hi2012/hint+hi2012_edge_list -i  data/networks/hint+hi2012/hint+hi2012_index_gene -nn hint+hi2012 -p  hint+hi2012 -b  0.4 -o  data/networks/hint+hi2012 -np $num_network_permutations -c  $num_cores')
    else:
        print('network folder already exists')
    if (myResultsDir.is_dir()==False):
        os.system('python src/FastSinkSource/src/algorithms/HotNet2.py -nf  data/networks/hint+hi2012/hint+hi2012_ppr_0.4.h5 -pnp data/networks/hint+hi2012/permuted/hint+hi2012_ppr_0.4_1.h5 -hf  data/heats/pan12.gene2freq.json -np 2 -hp 3 -o results -c -1')
    else:
        print('Results directory already exists')
    return

def setupOutputs(run_obj):
    print("inside hotnet2_runner: setupOutputs")
    return

def setup_params_str(weight_str, params, name):
    print("inside hotnet2_runner: setup_params_str")
    return

def get_alg_type():
    return 'hotnet2'