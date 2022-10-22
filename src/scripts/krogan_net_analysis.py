
import os, sys
import yaml
import argparse
import itertools
from tqdm import tqdm
from collections import defaultdict
# TODO give the path to this repo
#from graphspace_python.api.client import GraphSpace
#from graphspace_python.graphs.classes.gsgraph import GSGraph
import pandas as pd
import numpy as np
import scipy
from scipy import sparse as sp
# GSGraph already implements networkx
import networkx as nx
import matplotlib
if __name__ == "__main__":
    # Use this to save files remotely. 
    matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns


# local imports
from src import setup_datasets
from src.FastSinkSource.src import main as run_eval_algs
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils
from src.FastSinkSource.src.evaluate import stat_sig
from src.FastSinkSource.src.algorithms import genemania_runner as gm_runner
from src.FastSinkSource.src.algorithms import genemania as gm
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
    parser = argparse.ArgumentParser(description="")

    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, required=True,
                       help="Configuration file used when running the ann_pred scripts. ")
    group.add_argument('--sarscov2-human-ppis', default='datasets/protein-networks/2020-03-biorxiv-krogan-sars-cov-2-human-ppi-ace2.tsv',
                       help="Table of virus and human ppis. Default: datasets/protein-networks/2020-03-biorxiv-krogan-sars-cov-2-human-ppi-ace2.tsv")
    group.add_argument('--id-mapping-file', type=str, default="datasets/mappings/human/uniprot-reviewed-status.tab.gz",
                       help="Table downloaded from UniProt to map to gene names. Expected columns: 'Entry', 'Gene names', 'Protein names'")
    group.add_argument('--analysis-type', type=str, default="diffusion_analysis",
                       help="Type of network analysis to perform. Options: 'diffusion_analysis', 'shortest_paths', 'degrees'. Default: 'diffusion_analysis")
    #group.add_argument('--drug-id-mapping-file', type=str, 
    #                   help="Table parsed from DrugBank xml with drug names and other info")
    #group.add_argument('--node', type=str, action="append",
    #                   help="Check the distance of the given node")
    group.add_argument('--edge-weight-cutoff', type=float, 
                       help="Cutoff to apply to the edges to view (e.g., 900 for STRING)")
    group.add_argument('--unweighted', type=float, 
                       help="Don't use the edge weights when computing shortest paths")
#    group.add_argument('--k-to-test', '-k', type=int, action="append",
#                       help="k-value(s) for which to get the top-k predictions to test. " +
#                       "If not specified, will check the config file. Default=100")
#    group.add_argument('--range-k-to-test', '-K', type=int, nargs=3,
#                       help="Specify 3 integers: starting k, ending k, and step size. " +
#                       "If not specified, will check the config file.")
#
#    group = parser.add_argument_group('FastSinkSource Pipeline Options')
#    group.add_argument('--alg', '-A', dest='algs', type=str, action="append",
#                       help="Algorithms for which to get results. Must be in the config file. " +
#                       "If not specified, will get the list of algs with 'should_run' set to True in the config file")
#    group.add_argument('--num-reps', type=int, 
#                       help="Number of times negative sampling was repeated to compute the average scores. Default=1")
    group.add_argument('--sample-method', type=str, default='kmeans',
                       help="Approach used to sample random sets of positive examples. " + \
                       "Options: 'kmeans' (bin nodes by kmeans clustring on the degree distribution), 'simple' (uniformly at random)")
    group.add_argument('--num-random-sets', type=int, 
                       help="Factor/ratio of negatives to positives used when making predictions. Not used for methods which use only positive examples.")
    group.add_argument('--force-run', action='store_true', default=False,
                       help="Force re-running the path lengths for random sets, and re-writing the output files")

    return parser


def main(config_map, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """
    # extract the general variables from the config map
    input_settings, input_dir, output_dir, alg_settings, kwargs \
        = config_utils.setup_config_variables(config_map, **kwargs)

    uniprot_to_gene = {}

    # or we could get a distribution of distances for each virus node
    # load human-virus ppis
    df = pd.read_csv(kwargs['sarscov2_human_ppis'], sep='\t')
    edges = zip(df[df.columns[0]], df[df.columns[1]])
    edges = [(v.replace("SARS-CoV2 ",""), h) for v,h in edges]
    virus_nodes = [v for v,h in edges]
    krogan_nodes = [h for v,h in edges]
    virhost_edges = edges 

    # for each dataset, extract the path(s) to the prediction files,
    # read in the predictions, and test for the statistical significance of overlap 
    for dataset in input_settings['datasets']:
        dataset_name = config_utils.get_dataset_name(dataset)
        print("Loading data for %s" % (dataset['net_version']))
        # load the network and the positive examples for each term
        net_obj, ann_obj, eval_ann_obj = run_eval_algs.setup_dataset(
            dataset, input_dir, **kwargs) 
        prots, node2idx = net_obj.nodes, net_obj.node2idx
        print("\t%d total prots" % (len(prots)))
        # TODO using this for the SARS-CoV-2 project,
        # but this should really be a general purpose script
        # and to work on any number of terms 
        orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, 0)
        orig_pos = [prots[p] for p in orig_pos_idx]
        print("\t%d original positive examples" % (len(orig_pos)))

        # convert the krogan nodes and drugs to ids
        #drug_nodes_idx = [node2idx[d] for d in drug_nodes if d in node2idx]
        krogan_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]

        # TODO add the diffusion option as well
        if kwargs.get('analysis_type') == 'diffusion_analysis':
            eval_diffusion(dataset, net_obj, krogan_nodes_idx, **kwargs)

        elif kwargs.get('analysis_type') == 'shortest_paths':
            eval_shortest_paths(dataset, net_obj, krogan_nodes_idx, **kwargs) 

        elif kwargs.get('analysis_type') == 'degrees':
            plot_degrees(net_obj.W, krogan_nodes, dataset_name, **kwargs)


def eval_diffusion(dataset, net_obj, krogan_nodes_idx, **kwargs):
    num_random_sets = kwargs.get('num_random_sets', 10)
    plot_file = "outputs/viz/%s/%s/diffusion-comp/%s-%s-rand-set-diffusion-comp.pdf" % (
        dataset['net_version'], dataset['exp_name'],
        num_random_sets, kwargs.get('sample_method', 'kmeans'))
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)

    W = net_obj.W

    M_inv = get_diffusion_matrix(W, alpha=1.0, diff_mat_file=None)
    # Now get the diffusion values between the pairs of Krogan nodes
    krogan_diff = M_inv[krogan_nodes_idx,:][:,krogan_nodes_idx]
    all_diff = -np.log10(np.ravel(krogan_diff))
    s1 = pd.Series(all_diff)
    #krogan_diff
    #statistic, pval = scipy.stats.ks_2samp(s1, s2, alternative='greater')

    # TODO just get the degrees directly from the matrix
    print("converting network to networkx graph")
    G = nx.from_scipy_sparse_matrix(W)

    # now compare with the diffusion of random pairs of nodes
    if kwargs.get('sample_method', 'kmeans') == 'kmeans':
        # bin nodes by their degree, sample positive examples from the matching bins,
        # and then repeat the shortest paths computations for random sets of positive examples
        num_bins = 10
        bin_method = "kmeans"
        node_bins, nodes_per_bin, degrees = stat_sig.split_nodes_to_degree_bins(G, nbins=num_bins, method=bin_method, **kwargs)
        stat_sig.print_bin_table(nodes_per_bin, degrees, bin_method)
        random_sets = stat_sig.sample_pos_examples_from_bins(krogan_nodes_idx, node_bins, nodes_per_bin, num_random_sets=num_random_sets)
        print(len(random_sets))
    elif kwargs.get('sample_method') == 'simple':
        nodes = list(range(len(net_obj.nodes)))
        random_sets = [np.random.choice(nodes, size=len(krogan_nodes_idx), replace=False) for i in range(num_random_sets)]

    print("getting the diffusion in %d random sets" % (num_random_sets))
    rand_diffusions = []
    rand_pvals = []
    for i, rand_idx in enumerate(tqdm(random_sets)):                           
        # Now get the diffusion values between the pairs of Krogan nodes
    #     rand_idx = np.asarray([node2idx[n] for n in random_set if n in node2idx])
        rand_pair_diff = np.ravel(M_inv[rand_idx,:][:,rand_idx])
        rand_diffusions.append(rand_pair_diff)    
        # and compute the pval
        statistic, pval = scipy.stats.ks_2samp(np.ravel(krogan_diff), rand_pair_diff, alternative='less')
        rand_pvals.append(pval)
    if kwargs['sample_method'] == 'kmeans':
        df_rand_kmeans = pd.DataFrame(rand_diffusions).T 
        df_rand = df_rand_kmeans
    else:
        df_rand_simple = pd.DataFrame(rand_diffusions).T 
        df_rand = df_rand_simple

    # df_rand
    s2 = pd.Series(-np.log10(np.ravel(df_rand.values)))
    min_val = min([s1.min(), s2.min()])
    max_val = max([s1.max(), s2.max()]) 
    f, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    s1.hist(range=(min_val, max_val), weights=np.ones(len(s1)) / len(s1), bins=20, ax=ax1)
    orange = "#ffb521"
    s2.hist(range=(min_val, max_val), weights=np.ones(len(s2)) / len(s2), bins=20, color=orange, ax=ax2)
    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.set_xlabel("-log10(diffusion values)")
    ax1.set_title("Krogan node pairs")
    ax2.set_title("%d sets of random nodes, %s sampling" % (num_random_sets, kwargs.get('sample_method','kmeans')))
    # ax1.set_ylabel("Percentage of Node Pairs")
    # ax2.set_ylabel("Percentage of Node Pairs")
    plt.tight_layout()
    print("writing %s" % (plot_file))
    plt.savefig(plot_file, bbox_inches='tight') 
    plt.close()
    return


def eval_shortest_paths(dataset, net_obj, krogan_nodes_idx, **kwargs):
    W = net_obj.W
    num_random_sets = kwargs.get('num_random_sets', 10)
    # store the rand path lengths to be able to compute more statistics later
    out_file = "outputs/viz/%s/%s/path-lengths/%s-%s-rand-set-path-lengths.tsv.gz" % (
        dataset['net_version'], dataset['exp_name'],
        num_random_sets, kwargs.get('sample_method', 'kmeans'))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    if not kwargs.get('force_run') and os.path.isfile(out_file):
        print("Reading %s" % (out_file))
        df_rand_lengths = pd.read_csv(out_file, sep='\t')
        col1 = df_rand_lengths.columns[0]
        krogan_pair_lengths = df_rand_lengths[col1]
        df_rand_lengths.drop(columns=col1, inplace=True) 

    else:
        if kwargs.get('edge_weight_cutoff'):
            print("\tapplying edge weight cutoff %s" % (kwargs['edge_weight_cutoff']))
            W = W.multiply((W > kwargs['edge_weight_cutoff']).astype(int))

        print("converting network to networkx graph")
        G = nx.from_scipy_sparse_matrix(W)

        if not kwargs.get('unweighted'):
            # convert the weights to a cost by shifting the weights between 0 and 1 and then taking the negative log
            # Set the max weight at 0.7 so that an edge with confidence 1000 is twice as likely to be used as an edge with 700, and about 4 times as likely as an edge with 400.
            print("\tconverting edge weight to cost by taking -log10(weight/max_weight) * 0.7")
            W.data = -np.log10((W.data / W.data.max()) * 0.7)
            print("\tcost stats: min: %s, median: %s max: %s" % (W.data.min(), np.median(W.data), W.data.max()))

        print("computing the shortest paths from the krogan nodes to all other nodes")
        krogan_pair_lengths = compute_pairwise_shortest_paths(W, krogan_nodes_idx, **kwargs)

        # plot a histogram of the path lengths
        plot_file = "outputs/viz/%s/%s/path-lengths/krogan-path-lengths.pdf" % (
            dataset['net_version'], dataset['exp_name'])
        s = pd.Series(krogan_pair_lengths)
        s = s.replace(0.0,np.nan).dropna()
        s.hist()
        print("writing %s" % (plot_file))
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()

        if kwargs.get('sample_method', 'kmeans') == 'kmeans':
            # bin nodes by their degree, sample positive examples from the matching bins,
            # and then repeat the shortest paths computations for random sets of positive examples
            num_bins = 10
            bin_method = "kmeans"
            node_bins, nodes_per_bin, degrees = stat_sig.split_nodes_to_degree_bins(G, nbins=num_bins, method=bin_method, **kwargs)
            stat_sig.print_bin_table(nodes_per_bin, degrees, bin_method)
            random_sets = stat_sig.sample_pos_examples_from_bins(
                krogan_nodes_idx, node_bins, nodes_per_bin, num_random_sets=num_random_sets)
            print(len(random_sets))

        elif kwargs.get('sample_method') == 'simple':
            nodes = list(range(len(net_obj.nodes)))
            random_sets = [np.random.choice(nodes, size=len(krogan_nodes_idx), replace=False) for i in range(num_random_sets)]

        print("computing the shortest paths in %d random sets" % (num_random_sets))
        rand_lengths = [krogan_pair_lengths]
        for i, random_set in enumerate(tqdm(random_sets)):
            pair_lengths = compute_pairwise_shortest_paths(W, random_set, **kwargs) 
            rand_lengths.append(pair_lengths)
        df_rand_lengths = pd.DataFrame(rand_lengths).T
        # drop the rows with path lengths of 0 (self comparison)
        df_rand_lengths = df_rand_lengths.replace(0.0,np.nan).dropna()
        print("Writing to %s" % (out_file))
        df_rand_lengths.to_csv(out_file, sep='\t', compression='gzip', index=False) 
        # and drop teh krogan_pair_lengths from the df
        df_rand_lengths.drop(columns=0, inplace=True)
    print(df_rand_lengths.head())
    print(df_rand_lengths.shape)
    # now make a histogram of the randomly sampled 

    plot_file = "outputs/viz/%s/%s/path-lengths/%s-%s-rand-set-path-lengths.pdf" % (
        dataset['net_version'], dataset['exp_name'], num_random_sets, kwargs['sample_method'])
    f, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    s = pd.Series(krogan_pair_lengths)
    s1 = s.replace(0.0,np.nan).dropna()

    s = pd.Series(np.ravel(df_rand_lengths.values))
    s2 = s.replace(0.0,np.nan).dropna()
    min_val = min([s1.min(), s2.min()])
    max_val = max([s1.max(), s2.max()])
    s1.hist(
        range=(min_val, max_val),
        weights=np.ones(len(s1)) / len(s1), bins=20, ax=ax1)
    orange = "#ffb521"
    s2.hist(
        range=(min_val, max_val),
        weights=np.ones(len(s2)) / len(s2), bins=20, color=orange, ax=ax2)

    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.set_xlabel("Shortest Path Length")
    ax1.set_title("Krogan node pairs")
    ax2.set_title("%d sets of random nodes, %s sampling" % (
        num_random_sets, kwargs.get('sample_method','kmeans')))
    ax1.set_ylabel("Percentage of Node Pairs")
    ax2.set_ylabel("Percentage of Node Pairs")
    plt.tight_layout()
    print("writing %s" % (plot_file))
    plt.savefig(plot_file, bbox_inches='tight')


def get_diffusion_matrix(W, alpha=1.0, diff_mat_file=None):
    """
    Generate the diffusion matrix of a scipy sparse matrix
    *Note that the result is a dense matrix*
    """
    if diff_mat_file is not None and os.path.isfile(diff_mat_file):
        # read in the diffusion mat file
        print("Reading %s" % (diff_mat_file))
        return np.load(diff_mat_file)

    # now get the laplacian
    L = gm.setup_laplacian(W)
    # the equation is (I + a*L)s = y
    # we want to solve for (I + a*L)^-1
    M = sp.eye(L.shape[0]) + alpha*L 
    print("computing the inverse of (I + a*L) as the diffusion matrix, for alpha=%s" % (alpha))
    # first convert the scipy sparse matrix to a numpy matrix
    M_full = M.A
    # now try and take the inverse
    M_inv = scipy.linalg.inv(M_full)

    # write to file so this doesn't have to be recomputed
    if diff_mat_file is not None:
        print("Writing to %s" % (diff_mat_file))
        np.save(diff_mat_file, M_inv)

    return M_inv


def compute_pairwise_shortest_paths(W, indices, **kwargs):
    shortest_paths = sp.csgraph.dijkstra(W, directed=False, unweighted=kwargs.get('unweighted'), indices=indices)
    pair_lengths = np.ravel(shortest_paths[:,indices])
    return pair_lengths


def plot_degrees(W, krogan_nodes, dataset_name, **kwargs):
    if kwargs.get('edge_weight_cutoff'):
        print("\tapplying edge weight cutoff %s" % (kwargs['edge_weight_cutoff']))
        W = W.multiply((W > kwargs['edge_weight_cutoff']).astype(int))

    print("converting network to networkx graph")
    G = nx.from_scipy_sparse_matrix(W)
    # now convert the edges back to prots
    G = nx.relabel_nodes(G, {i: p for i, p in enumerate(prots)})
    print("\t%d nodes, %d edges" % (G.number_of_nodes(), G.number_of_edges()))

    #virhost_edges = [(v,node2idx[h]) for v,h in virhost_edges if h in node2idx]
    #print("adding %d virus edges" % (len(virhost_edges)))
    #G.add_edges_from(virhost_edges)
    curr_pos = [n for n in krogan_nodes if G.has_node(n) and G.degree[n] != 0]

    # get the degree distribution of the drug nodes
    if kwargs.get('unweighted'):
        degrees = {n: G.degree[n] for n in curr_pos}
    else:
        degrees = {n: G.degree(n, weight='weight') for n in curr_pos}
    s = pd.Series(degrees)
    #print(s.value_counts())
    f, ax = plt.subplots()
    s.hist(bins=20)
    ax.set_yscale('log')

    ax.set_title("%s - Krogan degrees" % (dataset_name))
    out_file = "outputs/viz/networks/%s-krogan-degree.png" % (dataset_name)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    print("writing %s" % (out_file))
    plt.savefig(out_file, bbox_inches='tight')


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)
