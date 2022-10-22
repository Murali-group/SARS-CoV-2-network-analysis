"""
Functions to evaluate the statistical significance of each node's score.
"""

import os
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
# for clustering the degrees
#from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.cluster import KMeans
import networkx as nx
import random
# plotting imports
import matplotlib
#if __name__ == "__main__":
matplotlib.use('Agg')  # Use this to save files remotely. 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import pandas as pd

# my local imports
from .. import main
from ..algorithms import alg_utils as alg_utils


def eval_stat_sig_nodes_runners(
        alg_runners, net_obj, ann_obj, num_random_sets=10,
        num_bins=10, bin_method='kmeans', k=5000, **kwargs):
    """
    *k*: number of nodes for which to store p-values, sorted by prediction score
    """
    if kwargs.get('stat_sig_drug_nodes') is not None:
        # load the drug nodes to evaluate them separately
        print("getting drug nodes from %s" % (kwargs['stat_sig_drug_nodes']))
        df = pd.read_csv(kwargs['stat_sig_drug_nodes'], sep='\t', header=None)
        kwargs['drug_nodes'] = set(list(df[1].values))
    alg_runners = get_pred_and_rand_scores(
        alg_runners, net_obj, ann_obj, num_random_sets=num_random_sets,
        num_bins=num_bins, bin_method=bin_method, k=k, **kwargs)

    for run_obj in alg_runners:
        df = run_obj.scores_df

        out_str = "%srand-%s%s" % (num_random_sets, num_bins, bin_method)
        title = "%s - %s" % (run_obj.name, kwargs.get('dataset_name', ""))
        title += "\n%s random sets, %s degree bins using '%s'" % (num_random_sets, num_bins, bin_method)
        out_file = run_obj.out_file.replace("networks/", "viz/networks/") \
                   .replace(".txt", "-%s-pvals.tsv" % (out_str))
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        compute_and_plot_pvals(df, out_file, k=1500, title=title)


def get_pred_and_rand_scores(
        alg_runners, net_obj, ann_obj, num_random_sets=10,
        num_bins=10, bin_method='kmeans', k=None, **kwargs):
    """ For every algorithm, rerun it *num_random_sets* times, 
    with a different set of carefully selected "random" positive examples each time
    """
    print("converting network to networkx graph")
    G = nx.from_scipy_sparse_matrix(net_obj.W)
    # now convert the edges back to prots
    #G = nx.relabel_nodes(G, {i: p for i, p in enumerate(net_obj.nodes)})
    print("\t%d nodes, %d edges" % (G.number_of_nodes(), G.number_of_edges()))
    drug_nodes = kwargs.get('drug_nodes')
    if drug_nodes is not None:
        # map the drug nodes to the indexes
        kwargs['drug_nodes'] = set(net_obj.node2idx[d] for d in drug_nodes if d in net_obj.node2idx)
    # compare to uniform
    node_bins, nodes_per_bin, degrees = split_nodes_to_degree_bins(G, nbins=num_bins, method='uniform', **kwargs) 
    print_bin_table(nodes_per_bin, degrees, 'uniform')
    node_bins, nodes_per_bin, degrees = split_nodes_to_degree_bins(G, nbins=num_bins, method=bin_method, **kwargs) 
    # clear up the RAM the networkx object could be using
    del G
    print_bin_table(nodes_per_bin, degrees, bin_method)
    orig_pos, neg = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, 0)
    pos = [p for p in orig_pos if p in node_bins]
    print("%d/%d pos examples in network" % (len(pos), len(orig_pos)))
    orig_pos = [ann_obj.prots[p] for p in orig_pos]

    random_sets = sample_pos_examples_from_bins(
        pos, node_bins, nodes_per_bin, num_random_sets=num_random_sets)
    #print("%d sets. # nodes per set: %s" % (len(random_sets), str([len(nodes) for nodes in random_sets])))
    for run_obj in alg_runners:
        main.run_algs([run_obj], **kwargs) 
        out_file = run_obj.out_file.replace('.txt','-%drand-%s%s.tsv.gz' % (
            num_random_sets, num_bins, bin_method))
        if os.path.isfile(out_file) and not kwargs.get('forcealg'):
            print("reading %s" % (out_file))
            run_obj.scores_df = pd.read_csv(out_file, sep='\t', index_col=0)
            continue

        # now load the original prediction scores
        pred_scores = pd.read_csv(run_obj.out_file, sep='\t')
        pred_scores.set_index('prot', inplace=True)
        # and generate the scores for the random subsets
        rand_pred_scores, nodes = generate_scores_rand_sets(run_obj, random_sets, **kwargs)
        # write to file(?)
        df = pd.DataFrame(rand_pred_scores).T
        df.index = nodes
        df.insert(0, 'pred_scores', pred_scores['score'])
        df = df[~df.index.isin(orig_pos)]
        df.sort_values(by='pred_scores', ascending=False, inplace=True)
        # if specified, store only the scores for the drug nodes
        if drug_nodes is not None:
            df = df[df.index.isin(drug_nodes)]
        if k is not None:
            df = df[:k]
        # TODO only write for the top k predictions
        print("writing %s%s" % (
            "top k=%s scores for random sets to "%k if k is not None else "",
            out_file))
        df.to_csv(out_file, sep='\t', compression='gzip')
        run_obj.scores_df = df

    return alg_runners


def generate_scores_rand_sets(run_obj, random_sets, **kwargs):
    # make sure to not write the individual score files
    factor_pred_to_write = kwargs.get('factor_pred_to_write')
    kwargs['factor_pred_to_write'] = None
    num_pred_to_write = kwargs['num_pred_to_write']
    kwargs['num_pred_to_write'] = 0
    # TODO update the drug targets such that the edges connected to positive examples are removed
    # and the edges connected to the original positive examples are kept
    if kwargs.get('stat_sig_drug_nodes') is not None and 'curr_dataset' in kwargs:
        # first reload the network with all drug targets (including those to the krogan nodes, since those won't be positive examples)
        # TODO this is a hack that doesn't work if the name of the drug-target network changes
        #kwargs['curr_dataset']['net_files'] = [os.path.basename(kwargs['stat_sig_drug_nodes'])]
        new_net_files = []
        for net_file in kwargs['curr_dataset']['net_files']:
            if net_file == "prot-drug-itxs.tsv":
                new_net_files.append(os.path.basename(kwargs['stat_sig_drug_nodes']))
            else:
                new_net_files.append(net_file)
        kwargs['curr_dataset']['net_files'] = new_net_files
        net_obj, ann_obj, _ = main.setup_dataset(kwargs['curr_dataset'], kwargs['curr_input_dir'], **kwargs) 
        run_obj.ann_obj = ann_obj
        # remap the rand pos since the indexes have changed
        random_sets = [[net_obj.node2idx[run_obj.net_obj.nodes[n]] \
                        for n in random_set] for random_set in random_sets]
        kwargs['drug_nodes'] = [run_obj.net_obj.nodes[d] for d in kwargs['drug_nodes']]

    ann_mat = run_obj.ann_obj.ann_matrix.copy()
    rand_pred_scores = []
    for i, rand_set in enumerate(tqdm(random_sets)):
        pos_vec = np.zeros(ann_mat.shape[1])
        pos_vec[rand_set] = 1
        ann_mat[0] = pos_vec
        if kwargs.get('stat_sig_drug_nodes') is not None and 'curr_dataset' in kwargs:
            # remove the drug-target edges to the new positive examples
            # since we removed the krogan drug targets when computing the prediction scores
            curr_W = remove_drug_target_pos_edges(pos_vec, net_obj, kwargs['drug_nodes'])
            print("removing %d drug-pos edges" % ((len(net_obj.W.data) - len(curr_W.data))/2))
            run_obj.net_obj.W = curr_W
            run_obj.net_obj.nodes = net_obj.nodes

        #ann_obj.ann_matrix = new_ann_matrix 
        ## make a copy of the run obj with the new ann_obj
        #curr_run_obj = runner.Runner(
        #    run_obj.name, run_obj.net_obj, ann_obj, run_obj.out_dir, run_obj.params, **run_obj.kwargs)
        run_obj.ann_obj.ann_matrix = ann_mat
        # re-run the methods
        curr_run_obj = main.run_algs([run_obj], **kwargs)[0]
        # get the scores for the single term
        term_scores = curr_run_obj.term_scores[0].toarray().flatten()
        rand_pred_scores.append(term_scores)

    kwargs['factor_pred_to_write'] = factor_pred_to_write 
    kwargs['num_pred_to_write'] = num_pred_to_write 
    return rand_pred_scores, run_obj.net_obj.nodes


def remove_drug_target_pos_edges(pos_vec, net_obj, drug_nodes):
    W = net_obj.W
    # first extract the drug 
    drug_nodes = [net_obj.node2idx[d] for d in drug_nodes]
    drug_vec = np.zeros(W.shape[0])
    drug_vec[drug_nodes] = 1
    # get a matrix with the edges from drug to pos nodes
    # get a matrix with 1s across the diagonal for the drug nodes
    drug_I = sp.diags(drug_vec)
    pos_I = sp.diags(pos_vec)
    # the first dot product will get the drug rows of W
    # the second dot product will get the edges/columns of the pos examples from the drug rows
    drug_to_pos_edges = drug_I.dot(W).dot(pos_I)
    drug_pos_edges = drug_to_pos_edges + drug_to_pos_edges.T
    curr_W = W - drug_pos_edges
    return curr_W


def split_nodes_to_degree_bins(G, nbins=10, method="kmeans", unweighted=False, **kwargs):
    """
    *method*: the method used to create the bins. 
        Options are: 'uniform', 'kmeans'
    *unweighted*: whether or not to use edge weights when computing the bins. Default is to use weights
    """
    # only keep the nodes that have at least 1 edge
    degrees = {n: G.degree(n, weight='weight') for n in G.nodes() if G.degree[n] != 0}
    if unweighted:
        degrees = {n: G.degree[n] for n in G.nodes() if G.degree[n] != 0}
    # also remove the drug nodes from consideration if specified
    if kwargs.get('drug_nodes'):
        print("removing %d drug nodes" % (len(kwargs['drug_nodes'])))
        degrees = {n: d for n,d in degrees.items() if n not in kwargs['drug_nodes']}
        print("\t%d nodes" % (len(degrees)))
    bin_per_node = {}
    nodes_per_bin = defaultdict(set)
    #bin_degree_vals = defaultdict(list)
    curr_bin = 0
    if method == 'uniform':
        num_nodes_per_bin = len(degrees) / float(nbins)
        # make sure all nodes are put in a bin
        num_nodes_per_bin = np.ceil(num_nodes_per_bin) 
        # make sure nodes of the same degree aren't put in different bins
        curr_degree = 0
        # now split the nodes into bins
        for i, n in enumerate(sorted(degrees, key=degrees.get, reverse=True)):
            if i >= curr_bin*num_nodes_per_bin and degrees[n] != curr_degree:
                curr_bin += 1
                curr_degree = degrees[n] 
            bin_per_node[n] = curr_bin
            nodes_per_bin[curr_bin].add(n)
            #bin_degree_vals[curr_bin].append(degrees[n])
    elif method == 'kmeans':
        nodes = sorted(degrees.keys())
        deg_arr = np.asarray([degrees[n] for n in nodes], dtype=float)
        # centroids, distance = kmeans(deg_arr, nbins, iter=1000, thresh=1e-05)
        # min_degree_per_bin = {}
        # for i, centroid in enumerate(sorted(centroids, reverse=True)):
        #     min_degree = max([centroid - distance, 1])
        #     min_degree_per_bin[i] = min_degree
        # print(len(centroids))
        # print(min_degree_per_bin)
        # # now assign the nodes to the bins 
        # for i, n in enumerate(sorted(degrees, key=degrees.get, reverse=True)):
        #     # figure out which bin this node goes in
        #     bin_min = min_degree_per_bin[curr_bin]
        #     if degrees[n] < bin_min:
        #         curr_bin += 1
        #     bin_per_node[n] = curr_bin
        #     nodes_per_bin[curr_bin].add(n)
        #     bin_degree_vals[curr_bin].append(degrees[n])
        kmeans = KMeans(nbins).fit(deg_arr.reshape(-1,1))
        for i, curr_bin in enumerate(kmeans.labels_):
            n = nodes[i]
            bin_per_node[n] = curr_bin
            nodes_per_bin[curr_bin].add(n)
            #bin_degree_vals[curr_bin].append(degrees[n])

    # convert to list for simpler handling later
    nodes_per_bin = {b: list(nodes) for b, nodes in nodes_per_bin.items()} 
    #print("%d bins using %s. Max degree per bin: %s" % (
    #    len(nodes_per_bin), method,
    #    str({b: max(bin_degree_vals[b]) for b in sorted(nodes_per_bin)})))
    #    #str({b: "%s - %s" % (max(bin_degree_vals[b]), min(bin_degree_vals[b])) for b in range(sorted(nodes_per_bin))})))
    return bin_per_node, nodes_per_bin, degrees


def print_bin_table(nodes_per_bin, degrees, method):
    bin_degree_vals = {}
    for b, nodes in nodes_per_bin.items():
        bin_degree_vals[b] = [degrees[n] for n in nodes]
    max_degree_per_bin = {b: max(bin_degree_vals[b]) for b in sorted(nodes_per_bin)}
    # print out a table of the bin size and max degree
    print("%s bin\tsize\tmax-degre" % (method))
    for b in sorted(max_degree_per_bin, key=max_degree_per_bin.get, reverse=True):
        print("%s\t%s\t%s" % (b, len(nodes_per_bin[b]), max_degree_per_bin[b]))
    #print("# nodes per bin: %s" % (str([len(nodes) for b, nodes in nodes_per_bin.items()])))


def sample_pos_examples_from_bins(
        positives, node_bins, nodes_per_bin, num_random_sets=10):
    """ For a given positive example p, sample a node uniformly at random 
    from the bin in which p resides
    """
    random_sets = []
    for i in range(num_random_sets):
        random_set = []
        for p in positives:
            curr_nodes = nodes_per_bin[node_bins[p]]
            rand_p = random.choice(curr_nodes)
            random_set.append(rand_p)
        random_sets.append(random_set) 
    return random_sets


# now compute the p-value per node (fraction of scores for rand nodes with a larger score)
def compute_frac_greater_score(row):
    pred_score = row['pred_scores']
#     print(row)
    rand_scores = row[1:]
    count = len([r for r in rand_scores if r >= pred_score])
    frac = count / float(len(row)-1)
    return frac


def compute_and_plot_pvals(df, out_file, k=1000, title=""):
    # now for each protein, get the pvalue
    df['pval'] = df.apply(compute_frac_greater_score, axis=1)
    df_pval = df[['pred_scores', 'pval']]
    df_pval.sort_values('pred_scores', ascending=False, inplace=True)
    print(df_pval.head())
    print("writing %s" % (out_file))
    df_pval.to_csv(out_file, sep='\t')
    out_file = out_file.replace('.tsv','.pdf')
    plot_pvals(df_pval, out_file=out_file, k=k, title=title)


def plot_pvals(df, out_file=None, k=500, title="", ax=None):
    #k=100
    labels = [str(i) if not i%int(k/10) else "" for i in range(k)]
    # make a bar plot, and make the bars full width so they will be visible
    # also need to make the borders of the bars very small for k > 100
    curr_df = df.reset_index()
    if k is not None:
        curr_df = curr_df.iloc[:k]
    data = curr_df['pval']
    # make the figure bigger to see all the values
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,6))
    ax = data.plot.bar(width=1, linewidth=0, rot=0, color="#2a419c", ax=ax)
    #ax.bar(data.index, data.values, width=1, linewidth=0)
    #ax = sns.barplot(y='pval', x='level_0', data=curr_df, ax=ax)
    ax.set_xticklabels(labels)
    ax.axhline(0.05, color='r', linestyle='--', linewidth=1)
    #ax.set_xlim(-k/100.0, k+(k/100.0))

    ax.set_title(title)
    ax.set_ylabel("p-value (fraction of rand scores >= predicted score)")
    ax.set_xlabel("Top %s nodes (sorted by prediction score)" % (k))

    if out_file is not None:
        print("writing %s" % (out_file))
        plt.savefig(out_file, bbox_inches='tight')
        #print("writing %s" % (out_file.replace('.pdf','.svg')))
        #plt.savefig(out_file.replace('.pdf','.svg'), bbox_inches='tight')
        #print("writing %s" % (out_file.replace('.pdf','.png')))
        #plt.savefig(out_file.replace('.pdf','.png'), dpi=300, bbox_inches='tight')
    return ax


