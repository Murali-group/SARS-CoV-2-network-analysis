
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
#sys.path.insert(0,os.path.dirname(__file__))
from src.FastSinkSource.src.main import setup_net
from src.FastSinkSource.src.plot import plot_utils
from src.FastSinkSource.src.algorithms import runner
from src.setup_datasets import parse_mapping_file


def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    config_maps = []
    for config in args.config:
        with open(config, 'r') as conf:
            #config_map = yaml.load(conf, Loader=yaml.FullLoader)
            config_map = yaml.load(conf)
            config_maps.append(config_map)

    return config_maps, kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="Script to compute the shortest path to each viral protein.")

    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', action="append",
                       help="Configuration file used when running FastSinkSource")
    group.add_argument('--pred-table', type=str, 
                       help="Table of predictions output by write_pred_table.py")
    group.add_argument('--virus-human-ppis', type=str, default="datasets/protein-networks/2020-03-biorxiv-krogan-sars-cov-2-human-ppi-ace2.tsv",
                       help="Viral proteins and the human proteins they interact with. PPIs should be the first two columns." +
                       "Default=datasets/protein-networks/2020-03-biorxiv-krogan-sars-cov-2-human-ppi-ace2.tsv")
    group.add_argument('--id-mapping-file', type=str, default="datasets/mappings/human/uniprot-reviewed-status.tab.gz",
                       help="Table downloaded from UniProt to map to gene names. Expected columns: 'Entry' and 'Gene names'")

#    # additional parameters
#    group = parser.add_argument_group('Additional options')
#    group.add_argument('--forcealg', action="store_true", default=False,
#            help="Force re-running algorithms if the output files already exist")
#    group.add_argument('--forcenet', action="store_true", default=False,
#            help="Force re-building network matrix from scratch")
#    group.add_argument('--verbose', action="store_true", default=False,
#            help="Print additional info about running times and such")

    return parser


def main(config_maps, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """
    print("Reading %s" % (kwargs['pred_table']))
    pred_df = pd.read_csv(kwargs['pred_table'], sep='\t', header=[0,1,2], index_col=0)
    print(pred_df.head())
    pred_prots = list(pred_df.index)
    if kwargs.get('id_mapping_file'):
        print("Reading %s" % (kwargs['id_mapping_file']))
        df = pd.read_csv(kwargs['id_mapping_file'], sep='\t', header=0) 
        ## keep only the first gene for each UniProt ID
        uniprot_to_gene = {p: genes.split(' ')[0] for p, genes in zip(df['Entry'], df['Gene names'].astype(str))}

    print("Reading %s" % (kwargs['virus_human_ppis']))
    virus_df = pd.read_csv(kwargs['virus_human_ppis'], sep='\t')
    print(virus_df.head())
    virus_df["#Bait"] = virus_df["#Bait"].apply(lambda x: x.replace('SARS-CoV2 ',''))
    # the edges should be the first two columns
    virus_edges = list(zip(virus_df[virus_df.columns[0]], virus_df[virus_df.columns[1]]))
    virus_nodes = sorted(set(u for u,v in virus_edges))
    krogan_nodes = sorted(set(v for u,v in virus_edges))
    krogan_edges = {n: v for v,n in virus_edges}
    print("\t%d virus prots, %d virus - host ppis" % (len(virus_nodes), len(virus_edges)))
    #print(virus_edges)

    # for each dataset, compute the shortest paths from each gene to each viral gene
    dataset_pred_closest_virus_prots = {}
    for config_map in config_maps:
        input_settings = config_map['input_settings']
        input_dir = input_settings['input_dir']
        #alg_settings = config_map['algs']
        #output_settings = config_map['output_settings']
        if config_map.get('eval_settings'):
            kwargs.update(config_map['eval_settings'])

        for dataset in input_settings['datasets']:
            dataset_name = dataset['plot_exp_name'] 
            # load the network
            net_obj = setup_net(input_dir, dataset, **kwargs)
            W, nodes, node2idx = net_obj.W, net_obj.nodes, net_obj.node2idx
            # make sure the krogan node is in the network, and the prediction table
            curr_krogan_nodes = [n for n in krogan_nodes if n in nodes]
            print("%d krogan nodes in the network" % (len(curr_krogan_nodes)))
            # UPDATE: Instead of computing the paths to the virus node, compute the shortest paths to the krogan nodes
            # and then list the virus node that goes with it
#            #print("Adding the virus - host PPIs to the network")
#
#            # first resize the network to account for the new nodes
#            nodes = list(nodes) + virus_nodes
#            node2idx = {n: i for i, n in enumerate(nodes)}
#            W.resize(len(nodes), len(nodes))
#            # now add the virus edges to the matrix
#            # some human prots will likely not be in the network
#            num_edges_added = 0
#            virus_nodes_not_skipped = set()
#            for u,v in virus_edges:
#                i, j = node2idx[u], node2idx.get(v)
#                if j is None:
#                    continue
#                num_edges_added += 1
#                virus_nodes_not_skipped.add(u)
#                # and add the edge to the network
#                W[i,j] = 1
#                W[j,i] = 1
#            virus_nodes_skipped = set(virus_nodes) - virus_nodes_not_skipped
#            print("\t%d/%d PPIs added, %d virus prots have no PPIs" % (
#                num_edges_added, len(virus_edges), len(virus_nodes_skipped)))
#            if len(virus_nodes_skipped) > 0:
#                print("\t\t" + str(list(virus_nodes_skipped)))

            # now compute the shortest paths
            print("Computing the shortest paths from the krogan proteins to all host proteins")
            unweighted = False if W.data[0] != 1 else True
            if unweighted is False:
                # convert the weights to -log
                max_weight = max(W.data.max(), 1)
                W.data = -np.log(W.data / (max_weight+1))
                print(W.data)
            shortest_paths = sp.csgraph.dijkstra(W, directed=False, unweighted=unweighted, indices=[node2idx[p] for p in curr_krogan_nodes])
            if unweighted is False:
                # also find the path lengths
                path_lengths = sp.csgraph.dijkstra(W, directed=False, unweighted=True, indices=[node2idx[p] for p in curr_krogan_nodes])
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
                        closest_nodes = np.nonzero(dist <= path_length)[0]
                        if len(closest_nodes) > 0:
                            #print(closest_nodes) 
                            break
                else:
                    closest_nodes = [np.argsort(dist)[0]]
                    #path_length = dist[closest_nodes[0]]
                    # UPDATE: use the unweighted distance to get the path length
                    path_length = path_lengths[:,idx][closest_nodes[0]]
                    #print(closest_nodes, path_length) 
                closest_virus_prots = [curr_krogan_nodes[int(i)] for i in closest_nodes]
                # also list the virus node
                closest_virus_prots = ', '.join(["(%s, %s)" % (
                    uniprot_to_gene[n], krogan_edges[n]) for n in closest_virus_prots])
                pred_closest_virus_prots[p] = (closest_virus_prots, path_length)
            dataset_pred_closest_virus_prots[dataset_name] = pred_closest_virus_prots
    #df = pd.DataFrame(dataset_pred_closest_virus_prots)
    #print(df)
    # now for each prediction, figure out the network with the lowest rank, and use its virus prots
    rank_df = pred_df[[col for col in pred_df.columns if 'Rank' in col]]
    # UPDATE: also limit the rank_df to the columns with STRING
    rank_df = rank_df[[col for col in rank_df.columns if 'STRING-400' in col]]
    print(rank_df.head())
    rank_df.dropna(how='all', inplace=True)
    print(rank_df.head())
    best_rank_col = rank_df.idxmin(axis=1)
    pred_virus_prots = {}
    pred_virus_dist = {}
    for p, d in best_rank_col.iteritems():
        try:
            prots, dist = dataset_pred_closest_virus_prots[d[0]][p]
        except KeyError:
            print("WARNING: no krogan proteins found for %s, %s. Skipping" % (str(d), p))
            #sys.exit()
            continue
        pred_virus_prots[p] = prots 
        pred_virus_dist[p] = dist 
    #best_rank_col['closest-sarscov2-prot'] = pd.DataFrame(best_rank_col).apply(
    #    lambda x: dataset_pred_closest_virus_prots[x[0][0]][x.name], axis=1)
    pred_virus_df = pd.DataFrame(best_rank_col, columns=['best-col'])
    pred_virus_df['closest-krogan-virus-prot'] = pd.Series(pred_virus_prots)
    #pred_virus_df['closest-krogan-virus-prot'] = pred_virus_df['closest-krogan-virus-prot'].apply(lambda x: "(%s, %s)" % (x, ))
    pred_virus_df['dist'] = pd.Series(pred_virus_dist)
    # before concatenating, add the MultiIndex columns to maintain the 3 levels of columns
    pred_virus_df.columns = pd.MultiIndex.from_tuples([(col,"","") for col in pred_virus_df.columns])
    out_df = pd.concat([pred_virus_df, pred_df], axis=1)

    out_file = kwargs['pred_table'].replace('.tsv','-closest-virus-prots.tsv')
    print("Writing %s" % (out_file))
    out_df.to_csv(out_file, sep='\t')


if __name__ == "__main__":
    config_maps, kwargs = parse_args()
    main(config_maps, **kwargs)
