# Script to analyze the amount of score combing from non-neighboring nodes through network diffusion.
# here's an example call to this script: 
#python src/scripts/effective_diffusion_analysis.py --config fss_inputs/config_files/params-testing/400-cv5-nf5-nr100-ace2.yaml --cutoff 0.01 --k-to-test=332 --stat-sig-cutoff 0.05

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


# local imports
from src import setup_datasets
from src.annotation_prediction.src import main as run_eval_algs
from src.annotation_prediction.src.utils import config_utils
from src.annotation_prediction.src.algorithms import alg_utils
from src.annotation_prediction.src.evaluate import stat_sig_node
#from src.annotation_prediction.src.algorithms import rl_genemania_runner as gm_runner
from src.annotation_prediction.src.algorithms import rl_genemania as rl


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
    group.add_argument('--config', type=str, required=True,
                       help="Configuration file used when running annotation_prediction. ")
    #group.add_argument('--sarscov2-human-ppis', default='datasets/protein-networks/2020-03-biorxiv-krogan-sars-cov-2-human-ppi-ace2.tsv',
    #                   help="Table of virus and human ppis. Default: datasets/protein-networks/2020-03-biorxiv-krogan-sars-cov-2-human-ppi-ace2.tsv")
    group.add_argument('--analysis-type', type=str, default="diffusion_analysis",
                       help="Type of network analysis to perform. Options: 'diffusion_analysis', 'shortest_paths', 'degrees'. Default: 'diffusion_analysis")
    #group.add_argument('--node', type=str, action="append",
    #                   help="Check the distance of the given node")
    group.add_argument('--cutoff', type=float, default=0.05,
                       help="Cutoff of fraction of diffusion recieved to use to choose the main contributors.")
    group.add_argument('--k-to-test', '-k', type=int, action="append",
                       help="k-value(s) for which to get the top-k predictions to test. " +
                       "If not specified, will check the config file. Default=332")
    group.add_argument('--stat-sig-cutoff', type=float, 
                       help="Cutoff on the node p-value for a node to be considered in the topk. " + \
                       "The p-values should already have been computed with run_eval_algs.py")
    group.add_argument('--terms-file', type=str, 
                       help="Plot the effective diffusion values per term.")
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
    group.add_argument('--num-random-sets', type=int, default=1000,
                       help="Number of random sets used when computing pvals. Default: 1000")
    group.add_argument('--force-run', action='store_true', default=False,
                       help="Force re-running the path lengths for random sets, and re-writing the output files")
    group.add_argument('--id-mapping-file', type=str, default="datasets/mappings/human/uniprot-reviewed-status.tab.gz",
                       help="Table downloaded from UniProt to map to gene names. Expected columns: 'Entry' and 'Gene names'")

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
    if kwargs.get('id_mapping_file'):
        print("Reading %s" % (kwargs['id_mapping_file']))
        df = pd.read_csv(kwargs['id_mapping_file'], sep='\t', header=0) 
        ## keep only the first gene for each UniProt ID
        uniprot_to_gene = {p: genes.split(' ')[0] for p, genes in zip(df['Entry'], df['Gene names'].astype(str))}

    # # or we could get a distribution of distances for each virus node
    # # load human-virus ppis
    # df = pd.read_csv(kwargs['sarscov2_human_ppis'], sep='\t')
    # edges = zip(df[df.columns[0]], df[df.columns[1]])
    # edges = [(v.replace("SARS-CoV2 ",""), h) for v,h in edges]
    # virus_nodes = [v for v,h in edges]
    # krogan_nodes = [h for v,h in edges]
    # virhost_edges = edges 

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

        # load the top predictions
        alg_pred_files = config_utils.get_dataset_alg_prediction_files(
            output_dir, dataset, alg_settings, ['rl'], **kwargs)

        k = kwargs.get('k',332)
        cutoff = kwargs.get('cutoff', 0.01)
        # get the alpha values to use 
        alphas = alg_settings['rl']['alpha']
        alpha_frac_main_contr_nonnbrs = {} 
        alpha_nodes_pos_nbr_dfsn = {} 
        alpha_dfs = {} 
        for alpha, alg in zip(alphas, alg_pred_files):
            print("Setting alpha=%s" % (alpha))
            pred_file = alg_pred_files[alg]
            if not os.path.isfile(pred_file):
                print("Warning: %s not found. skipping" % (pred_file))
                continue
            print("reading %s for alpha=%s" % (pred_file, alpha))
            df = pd.read_csv(pred_file, sep='\t')
            # remove the original positives
            df = df[~df['prot'].isin(orig_pos)]
            df.reset_index(inplace=True, drop=True)

            sig_cutoff = kwargs.get('stat_sig_cutoff')
            if sig_cutoff:
                df = config_utils.get_pvals_apply_cutoff(df, pred_file, **kwargs)

            if k > len(df['prot']):
                print("ERROR: k %s > num predictions %s. Quitting" % (k, len(df['prot'])))
                sys.exit()

            diff_mat_file = "%sdiffusion-mat-a%s.npy" % (net_obj.out_pref, str(alpha).replace('.','_'))

            pred_scores = np.zeros(len(net_obj.nodes))
            df = df[:k]
            top_k_pred = df['prot']
            top_k_pred_idx = [net_obj.node2idx[n] for n in top_k_pred]
            pred_scores[top_k_pred_idx] = df['score'].values

            sig_str = "-sig%s" % (str(sig_cutoff).replace('.','_')) if sig_cutoff else ""
            out_pref = "outputs/viz/%s/%s/diffusion-analysis/k%s-a%s%s" % (
                dataset['net_version'], dataset['exp_name'], k, alpha, sig_str)
            os.makedirs(os.path.dirname(out_pref), exist_ok=True)

            M_inv = rl.get_diffusion_matrix(net_obj.W, alpha=alpha, diff_mat_file=diff_mat_file)

            # first write the percentage contributed from the krogan proteins to the top predictions  
            out_file = "%s-fraction-krogan-diffusion.tsv" % (out_pref)
            # get the diffusion values from the positive nodes to the top predictions 
            pos_to_top_dfsn = M_inv[top_k_pred_idx,:][:,krogan_nodes_idx]
            # normalize by the column to get the fraction of diffusion from each pos node
            pos_to_k_dfsn_norm = (pos_to_top_dfsn*np.divide(1,pos_to_top_dfsn.sum(axis=0)))
            vir_host_ppi_nodes = [prots[idx] for idx in krogan_nodes_idx]
            if kwargs.get('id_mapping_file'):
                vir_host_ppi_nodes = [uniprot_to_gene[p] for p in vir_host_ppi_nodes]
                top_k_pred = [uniprot_to_gene[p] for p in top_k_pred]
            # map to gene names as well
            df_dfsn = pd.DataFrame(pos_to_k_dfsn_norm, columns=vir_host_ppi_nodes, index=top_k_pred)
            # sort the columns by gene names
            df_dfsn = df_dfsn.T.sort_index().T
            print("Writing %s" % (out_file))
            df_dfsn.applymap(lambda x: "%0.2e"%x).to_csv(out_file, sep='\t')

            frac_main_contr_nonnbrs, nodes_pos_nbr_dfsn = get_effective_diffusion_score(
                pred_scores, M_inv, net_obj, krogan_nodes_idx, alpha=alpha,
                diff_mat_file=diff_mat_file, out_pref=out_pref, **kwargs)
            # # exploratory analysis:
            # dist_main_contributors(pred_scores, M_inv, krogan_nodes_idx, k, net_obj.W, prots, uniprot_to_gene)

            alpha_frac_main_contr_nonnbrs[alpha] = frac_main_contr_nonnbrs
            # convert the node IDs to protein IDs
            nodes_pos_nbr_dfsn = {prots[n]: val for n, val in nodes_pos_nbr_dfsn.items()}
            alpha_nodes_pos_nbr_dfsn[alpha] = nodes_pos_nbr_dfsn
            alpha_dfs[alpha] = df

        #out_pref = "outputs/viz/%s/%s/diffusion-comp/%s-%s-rand-set-diffusion-comp.pdf" % (
        out_pref = "outputs/viz/%s/%s/diffusion-analysis/cutoff%s-k%s%s" % (
            dataset['net_version'], dataset['exp_name'], cutoff, k, sig_str)
        os.makedirs(os.path.dirname(out_pref), exist_ok=True)

        # now plot
        f, axes = plt.subplots(ncols=len(alphas), figsize=(max([6, len(alphas)*3]),6), sharey=True)
        if len(alphas) == 1:
            axes = [axes]
        for alpha, ax in zip(alphas, axes):
            plot_fracs(alpha_frac_main_contr_nonnbrs[alpha].values(), alpha, ax=ax, cutoff=kwargs.get('cutoff',0.5))
        #for i, alpha in enumerate(alphas):
        #    frac_main_contr_nonnbrs = frac_main_contr_nonnbrs_list
        out_file = "%s-effective-diffusion.pdf" % (out_pref)
        #print(out_file)
        # plt.savefig(out_file, bbox_inches='tight')
        plt.close()

        out_pref = "outputs/viz/%s/%s/diffusion-analysis/k%s%s" % (
            dataset['net_version'], dataset['exp_name'], k, sig_str)
        out_file = "%s-frac-nonnbr-dfsn.pdf" % (out_pref)

        df = pd.DataFrame(alpha_nodes_pos_nbr_dfsn)
        #print(df)
        print("median effective diffusion values:")
        print(df.median())
        ylabel = 'Effective Diffusion'
        plot_effective_diffusion(df, out_file, xlabel=r"$\alpha$", ylabel=ylabel, title="") 

        ## also make a scatterplot of the nodes rank with the effective diffusion
        for alpha in alphas:
            pred_df = alpha_dfs[alpha]
            pred_df.set_index('prot', inplace=True)
            #alpha_diff_df = df[alpha]
            df['score'] = pred_df[pred_df.index.isin(df.index)]['score']
            #print(df.head())

            # out_file = "%s-alpha%s-eff-diff-by-score.pdf" % (out_pref, alpha)
            # plot_eff_diff_by_rank(df, alpha, out_file, xlabel="Score", ylabel="Effective Diffusion", title="Alpha=%s, k=%s" % (alpha, k))

            if kwargs.get('terms_file'):
                out_file = "%s-alpha%s-eff-diff-per-term.pdf" % (out_pref, alpha)
                plot_eff_diff_per_term(df[alpha], kwargs['terms_file'], out_file, title="Alpha=%s, k=%s" % (alpha, k))

            # TODO count the distance away of the main contributors
            # count the # krogan proteins each prediction is connected to(?)
            out_file = "%s-alpha%s-num-pos-nbrs.pdf" % (out_pref, alpha)
            plot_num_krogan_nbrs(df, alpha, krogan_nodes_idx, net_obj, out_file) 


def plot_num_krogan_nbrs(df, alpha, krogan_nodes_idx, net_obj, out_file):
    num_krogan_nbrs = {}
    frac_krogan_nbrs = {}
    node2idx = net_obj.node2idx
    W = net_obj.W
    for p in df.index:
        idx = node2idx[p]
        # now get the edges of this node and see if they overlap with the top pos node influencers
        # extract the row of network to get the neighbors
        row = W[idx,:]
        nbrs = (row > 0).nonzero()[1]
        num_krogan_nbrs[p] = len(set(nbrs) & set(krogan_nodes_idx))
        frac_krogan_nbrs[p] = num_krogan_nbrs[p] / float(len(nbrs))

    # now make a scatterplot of the two
    df['frac_krogan_nbrs'] = pd.Series(frac_krogan_nbrs)
    df['num_krogan_nbrs'] = pd.Series(num_krogan_nbrs)
    print(df.head())
    g = sns.jointplot(data=df, x=alpha, y='frac_krogan_nbrs')
    g.ax_joint.set_xlabel("Effective Diffusion (a=%s)" %alpha)
    g.ax_joint.set_ylabel("Fraction of Nbrs that are Krogan nodes")
    # g.ax_joint.set_title(title)
    #plt.suptitle(title)
    print(out_file)
    plt.savefig(out_file)
    plt.close()

    g = sns.jointplot(data=df, x='num_krogan_nbrs', y='frac_krogan_nbrs')
    out_file = out_file.replace('.pdf','-by-num.pdf')
    print(out_file)
    plt.savefig(out_file)
    #sns.pairplot(data=df)
    plt.close()


def plot_eff_diff_per_term(eff_diff_values, terms_file, out_file, xlabel="Effective Diffusion", title=""):
    df = pd.read_csv(terms_file, sep=',')
    # convert the string version of a set to actual python sets
    df['geneID'] = df['geneID'].apply(lambda x: eval(str(x)))
    print(df)
    #df['geneID'] = df['geneID'].astype(set) 
    # print(df['geneID'])
    term_prots = dict(zip(df['Description'], df['geneID']))
    prot_eff_diff = eff_diff_values.to_dict()
    term_eff_diff = defaultdict(dict)
    # print(term_prots)
    for term, prots in term_prots.items():
        for p in prots:
            if p in prot_eff_diff:
                term_eff_diff[term][p] = prot_eff_diff[p]
    df2 = pd.DataFrame(term_eff_diff)
    print(df2)

    ax = sns.boxplot(data=df2, orient='h', whis=np.inf)
    ax = sns.swarmplot(data=df2, orient='h', size=3.25, color="0.25")

    ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    ax.set_title(title)

    print(out_file)
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()


def plot_eff_diff_by_rank(df, alpha, out_file, xlabel="Alpha", ylabel="", title=""):
    g = sns.jointplot(x='score', y=alpha, data=df)

    g.ax_joint.set_xlabel(xlabel)
    g.ax_joint.set_ylabel(ylabel)
    # g.ax_joint.set_title(title)
    plt.suptitle(title)

    print(out_file)
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()


def plot_effective_diffusion(df, out_file, xlabel="Alpha", ylabel="", title=""):
    f, ax = plt.subplots(figsize=(6,3))
    ax = sns.boxplot(data=df)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    print(out_file)
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()


def plot_fracs(fracs_top_nbrs, alpha, ax=None, cutoff=0.1):
    if ax is None:
        f, ax = plt.subplots()
    ax.hist(fracs_top_nbrs, bins=25)
    ax.set_xlabel("Fraction of non-neighboring \nmain contributor krogan nodes (cutoff %s)" % (cutoff))
    ax.set_title("alpha = %s" % (alpha))
    ax.set_yscale('log')


def get_effective_diffusion_score(
        pred_scores, M_inv, net_obj, krogan_nodes_idx, 
        alpha=1.0, k=332, out_pref=None, **kwargs):
    """
    For each of the top k predictions, get the "effective diffusion score" 
        which is the fraction of non-neighboring nodes that contribute to the score
    *k*: number of top predictions to test
    """
    W = net_obj.W

    #plot_file = "%s-dfsn-curves.pdf" % (out_pref)
    plot_file = None

    main_contributors, fracs_top_nbrs, nodes_pos_nbr_dfsn = rl.get_pred_main_contributors(
        pred_scores, M_inv, krogan_nodes_idx, cutoff=kwargs.get('cutoff',0.05), k=k,
        W=W, plot_file=plot_file, alpha=alpha)

    fracs_main_contr_nonnbrs = {n: 1-x for n, x in fracs_top_nbrs.items()}
    nodes_pos_nbr_dfsn = {n: 1-x for n, x in nodes_pos_nbr_dfsn.items()}
    return fracs_main_contr_nonnbrs, nodes_pos_nbr_dfsn
    #return fracs_top_nbrs


def dist_main_contributors(pred_scores, M_inv, pos_idx, k, W, prots, uniprot_to_gene):
    pred_scores[pos_idx] = 0
    top_k_pred_idx = np.argsort(pred_scores)[::-1][:k]
    # get the diffusion values from the positive nodes to the top predictions 
    pos_to_top_dfsn = M_inv[pos_idx,:][:,top_k_pred_idx] 
    # normalize by the column to get the fraction of diffusion from each pos node
    pos_to_k_dfsn_norm = (pos_to_top_dfsn*np.divide(1,pos_to_top_dfsn.sum(axis=0)))

    for i, n in enumerate(top_k_pred_idx):
        # get the biggest contributor
        diff_to_i = pos_to_k_dfsn_norm[:,i]
        top_contributor = np.argsort(diff_to_i)[::-1][0]
        if W is not None:
            # now get the edges of this node and see if they overlap with the top pos node influencers
            # extract the row of network to get the neighbors
            row = W[n,:]
            nbrs = (row > 0).nonzero()[1]
            top_pos_node_idx = pos_idx[top_contributor]
            g = uniprot_to_gene[prots[n]]
            g_p = uniprot_to_gene[prots[top_pos_node_idx]]
            if top_pos_node_idx not in nbrs:
                print("%s is a top contributor, non-neighbor for %s (dfsn_val: %s, rank: %s)" % (g_p, g, diff_to_i[top_contributor], i))
            else:
                print("%s is a top contributor, neighbor for %s (dfsn_val: %s, rank: %s)" % (g_p, g, diff_to_i[top_contributor], i))


def run_GM_get_top_pred(L, y, alpha, k=332):
    f, process_time, wall_time, num_iters = rl.runGeneMANIA(L, y, alpha=alpha)
#     non_krogan_nodes = np.ones(L.shape[0])
#     non_krogan_nodes[krogan_node_idx] = 0
    f[y.nonzero()[0]] = 0
    top_pred = np.argsort(f)[::-1][:k]
    return top_pred


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    for k in kwargs.get('k_to_test', [332]):
        main(config_map, k=k, **kwargs)
