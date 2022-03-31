# script to find the cutoff to use to reduce the size of the string network until it matches the other networks

from scipy import sparse as sp
from scipy.io import savemat, loadmat
import numpy as np
#import networkx as nx


# first read string
#combined_string_file = "stringv11/900//sparse-nets/c900-combined_score-sparse-nets.mat"
combined_string_file = "stringv11/400//sparse-nets/c400-combined_score-sparse-nets.mat"
sparse_networks = list(loadmat(combined_string_file)['Networks'][0])
W = sparse_networks[0]
#print(W)
print(len(W.data))
#print(W.data)

# now for each other network file, increase the cutoff on the string network until we get the right size
net_files = [
        "HI-union/HI-union.npz", 
        "biogrid-no-genetic/biogrid-9606-no-genetic.npz", 
        "biogrid-y2h/biogrid-9606-two-hybrid.npz"
        ]


for net_file in net_files:
    W2 = sp.load_npz(net_file)
    num_edges = len(W2.data)
    print(num_edges)

    #cutoff_to_use = 900
    cutoff_to_use = 400
    while True:
        num_string_edges = len((W > cutoff_to_use).data)
        if num_string_edges <= num_edges:
            break
        cutoff_to_use += 1 
    print(f"cutoff to use: {cutoff_to_use}. {num_string_edges} <= {num_edges} for {net_file}")

