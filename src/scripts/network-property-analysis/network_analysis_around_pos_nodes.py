import os, sys
import yaml
import argparse
import itertools
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib
if __name__ == "__main__":
    # Use this to save files remotely.
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns
import networkx as nx

sys.path.insert(0,"/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/")
from src.FastSinkSource.src.main import setup_dataset
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils
from src.FastSinkSource.src.evaluate import stat_sig
#from FastSinkSource.src.algorithms import rl_genemania_runner as gm_runner
from src.FastSinkSource.src.algorithms import rl_genemania as gm
from src.FastSinkSource.src.algorithms import sinksource_bounds as ss
from src.FastSinkSource.src.evaluate import stat_sig


def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        #config_map = yaml.load(conf, Loader=yaml.FullLoader)
        config_map = yaml.load(conf)
    # TODO check to make sure the inputs are correct in config_map

    return config_map, kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="Script to analyze the top contributors of each prediction's diffusion score, as well as the effective diffusion (i.e., fraction of diffusion received from non-neighbors)")

    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, default = "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                                                       "fss_inputs/config_files/provenance/"
                                                        "provenance_biogrid_y2hsept22_s12.yaml" ,
                       help="Configuration file used when running FSS. ")

    return parser

def get_total_neighbour(pos_nodes_idx, W):
    mask = np.zeros(W.shape[0], dtype=bool)
    mask[pos_nodes_idx] = True
    Z = W[mask, :] #take rows for positive nodes only
    Z = Z.A

    # print(Z.shape[1])

    Z = Z[:, ~np.all(Z == 0, axis=0)]

    print('Neighbours: ', Z.shape[1], '\n')


def get_connectivity_btn_pos_nodes(pos_nodes_idx, W):
    print(type(W), W.shape)
    mask = np.zeros(W.shape[0], dtype=bool)
    mask[pos_nodes_idx] = True
    Z = W[mask, :]  # take rows and columns for positive nodes only
    Z = Z[:, mask]
    Z = Z.A

    print('shape Z: ', Z.shape[0], Z.shape[1])
    # print(Z)
    Z = Z[:, ~np.all(Z == 0, axis=0)]
    print('#Seed node without any seed node as neighbor: ', Z.shape[0]-Z.shape[1],'/', Z.shape[0])

def main(config_map, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """
    # extract the general variables from the config map
    input_settings, input_dir, output_dir, alg_settings, kwargs \
        = config_utils.setup_config_variables(config_map, **kwargs)

    for dataset in input_settings['datasets']:

        dataset_name = config_utils.get_dataset_name(dataset)

        # load the network and the positive examples for each term
        net_obj, ann_obj, eval_ann_obj = setup_dataset(
            dataset, input_dir, **kwargs)
        prots, node2idx = net_obj.nodes, net_obj.node2idx
        #
        # G = nx.from_scipy_sparse_matrix(net_obj.W, create_using=nx.DiGraph())
        # dim = nx.diameter(G)
        # print(dim)
        # TODO using this for the SARS-CoV-2 project,
        # but this should really be a general purpose script
        # and to work on any number of terms
        orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, 0)
        orig_pos = [prots[p] for p in orig_pos_idx]


        # convert the krogan nodes and drugs to ids
        # drug_nodes_idx = [node2idx[d] for d in drug_nodes if d in node2idx]
        pos_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]

        print("**********************\n")
        print("Loading data for %s" % (dataset['net_version']))
        print(dataset['exp_name'])
        #
        print("\n total prots: %d" % (len(prots)))
        print("\n pos prots: %d" % (len(orig_pos_idx)))

        print( '\npos prots in net: ', len(pos_nodes_idx))
        get_total_neighbour(pos_nodes_idx, net_obj.W)

        get_connectivity_btn_pos_nodes(pos_nodes_idx, net_obj.W)
        print("**********************\n")




if __name__ == "__main__":
    config_map, kwargs = parse_args()
    for k in kwargs.get('k_to_test', [332]):
        main(config_map, k=k, **kwargs)