
import argparse
import yaml
from collections import defaultdict
import os
import sys
#from tqdm import tqdm
import copy
import time
import itertools
import numpy as np
from scipy import sparse as sp
import pandas as pd
# add this file's directory to the path so these imports work from anywhere
sys.path.insert(0,os.path.dirname(__file__))
from FastSinkSource.run_eval_algs import setup_net
from FastSinkSource.src.plot import plot_utils
from FastSinkSource.src.algorithms import runner
from setup_datasets import parse_mapping_file


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
    parser = argparse.ArgumentParser(description="Script to compute the shortest path to each viral protein.")

    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, default="fss_inputs/config_files/stringv11/400.yaml",
                       help="Configuration file used when running FastSinkSource")
    group.add_argument('--pred-table', type=str, 
                       help="Table of predictions output by write_pred_table.py")
    group.add_argument('--virus-human-ppis', type=str, default="datasets/protein-networks/2020-03-biorxiv-krogan-sars-cov-2-human-ppi.tsv",
                       help="Viral proteins and the human proteins they interact with. PPIs should be the first two columns." +
                       "Default=datasets/protein-networks/2020-03-biorxiv-krogan-sars-cov-2-human-ppi.tsv")

#    # additional parameters
#    group = parser.add_argument_group('Additional options')
#    group.add_argument('--forcealg', action="store_true", default=False,
#            help="Force re-running algorithms if the output files already exist")
#    group.add_argument('--forcenet', action="store_true", default=False,
#            help="Force re-building network matrix from scratch")
#    group.add_argument('--verbose', action="store_true", default=False,
#            help="Print additional info about running times and such")

    return parser


def main(config_map, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """
    input_settings = config_map['input_settings']
    input_dir = input_settings['input_dir']
    alg_settings = config_map['algs']
    output_settings = config_map['output_settings']
    if config_map.get('eval_settings'):
        kwargs.update(config_map['eval_settings'])

    print("Reading %s" % (kwargs['pred_table']))
    pred_df = pd.read_csv(kwargs['pred_table'], sep='\t', header=[0,1,2], index_col=0)
    print(pred_df.head())
    pred_prots = list(pred_df.index)
    #for p in pred_prots:
    #    print(pred_df.loc([p]))

    print("Reading %s" % (kwargs['virus_human_ppis']))
    virus_df = pd.read_csv(kwargs['virus_human_ppis'], sep='\t')
    print(virus_df.head())
    virus_df["#Bait"] = virus_df["#Bait"].apply(lambda x: x.replace('SARS-CoV2 ',''))
    # the edges should be the first two columns
    virus_edges = list(zip(virus_df[virus_df.columns[0]], virus_df[virus_df.columns[1]]))
    virus_nodes = sorted(set(u for u,v in virus_edges))
    print("\t%d virus prots, %d virus - host ppis" % (len(virus_nodes), len(virus_edges)))
    #print(virus_edges)

    # for each dataset, compute the shortest paths from each gene to each viral gene
    dataset_pred_closest_virus_prots = {}
    for dataset in input_settings['datasets']:
        dataset_name = dataset['plot_exp_name'] 
        # load the network
        net_obj = setup_net(input_dir, dataset, **kwargs)
        W, nodes = net_obj.W, net_obj.nodes
        print("Adding the virus - host PPIs to the network")

        # first resize the network to account for the new nodes
        nodes = list(nodes) + virus_nodes
        node2idx = {n: i for i, n in enumerate(nodes)}
        W.resize(len(nodes), len(nodes))
        # now add the virus edges to the matrix
        # some human prots will likely not be in the network
        num_edges_added = 0
        virus_nodes_not_skipped = set()
        for u,v in virus_edges:
            i, j = node2idx[u], node2idx.get(v)
            if j is None:
                continue
            num_edges_added += 1
            virus_nodes_not_skipped.add(u)
            # and add the edge to the network
            W[i,j] = 1
            W[j,i] = 1
        virus_nodes_skipped = set(virus_nodes) - virus_nodes_not_skipped
        print("\t%d/%d PPIs added, %d virus prots have no PPIs" % (
            num_edges_added, len(virus_edges), len(virus_nodes_skipped)))
        if len(virus_nodes_skipped) > 0:
            print("\t\t" + str(list(virus_nodes_skipped)))

        # now compute the shortest paths
        print("Computing the unweighted shortest paths from the virus proteins to all host proteins")
        unweighted = False if W.data[0] != 1 else True
        if unweighted is False:
            # convert the weights to -log
            max_weight = max(W.data.max(), 1)
            W.data = -np.log(W.data / (max_weight+1))
            print(W.data)
        shortest_paths = sp.csgraph.dijkstra(W, directed=False, unweighted=unweighted, indices=[node2idx[p] for p in virus_nodes])
        #print(shortest_paths.shape)
        #print(shortest_paths)

        # now for each prediction, get the closest virus prot(s)
        pred_closest_virus_prots = {} 
        for p in pred_prots:
            idx = node2idx.get(p)
            if idx is None:
                continue
            dist = shortest_paths[:,idx]
            if min(dist) == np.inf:
                continue
            if unweighted:
                for path_length in range(2,10):
                    curr_dist_nodes = np.nonzero(dist <= path_length)[0]
                    if len(curr_dist_nodes) > 0:
                        #print(curr_dist_nodes) 
                        break
            else:
                curr_dist_nodes = [np.argsort(dist)[0]]
                path_length = dist[curr_dist_nodes[0]]
                #print(curr_dist_nodes, path_length) 
            closest_virus_prots = ','.join(set(virus_nodes[int(i)] for i in curr_dist_nodes))
            pred_closest_virus_prots[p] = (closest_virus_prots, path_length)
        dataset_pred_closest_virus_prots[dataset_name] = pred_closest_virus_prots
    #df = pd.DataFrame(dataset_pred_closest_virus_prots)
    #print(df)
    # now for each prediction, figure out the network with the lowest rank, and use its virus prots
    rank_df = pred_df[[col for col in pred_df.columns if 'Rank' in col]]
    best_rank_col = rank_df.idxmin(axis=1)
    pred_virus_prots = {}
    pred_virus_dist = {}
    for p, d in best_rank_col.iteritems():
        try:
            prots, dist = dataset_pred_closest_virus_prots[d[0]][p]
        except KeyError:
            print("no viral proteins found for %s, %s. Skipping" % (str(d), p))
            continue
        pred_virus_prots[p] = prots 
        pred_virus_dist[p] = dist 
    #best_rank_col['closest-sarscov2-prot'] = pd.DataFrame(best_rank_col).apply(
    #    lambda x: dataset_pred_closest_virus_prots[x[0][0]][x.name], axis=1)
    pred_virus_df = pd.DataFrame(best_rank_col, columns=['best-col'])
    pred_virus_df['closest-virus-prot'] = pd.Series(pred_virus_prots)
    pred_virus_df['dist'] = pd.Series(pred_virus_dist)

    out_file = kwargs['pred_table'].replace('.tsv','-closest-virus-prots.tsv')
    print("Writing %s" % (out_file))
    pred_virus_df.to_csv(out_file, sep='\t')


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)
