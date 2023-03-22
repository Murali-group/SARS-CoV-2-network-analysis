# here's an example call to this script: 
#python src/scripts/node_based_effective_diffusion.py --config fss_inputs/config_files/params-testing/400-cv5-nf5-nr100-ace2.yaml --cutoff 0.01 --k-to-test=332 --stat-sig-cutoff 0.05

import os, sys
import yaml
import argparse
from collections import defaultdict
import numpy as np
import matplotlib
import copy
if __name__ == "__main__":
    # Use this to save files remotely. 
    matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0,"/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/")
from src.FastSinkSource.src.main import setup_dataset
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils
#from FastSinkSource.src.algorithms import rl_genemania_runner as gm_runner
from src.FastSinkSource.src.algorithms import rl_genemania as gm
from src.FastSinkSource.src.algorithms import PageRank_Matrix as rwr

import src.scripts.utils as script_utils
from src.scripts.plot_utils import *


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
    group.add_argument('--config', type=str,  default = "/data/tasnina/Provenance-Tracing/"
                    "SARS-CoV-2-network-analysis/fss_inputs/config_files/provenance/biogrid_y2h_s12.yaml" ,
                       help="Configuration file used when running FSS. ")

    group.add_argument('--analysis-type', type=str, default="diffusion_analysis",
                       help="Type of network analysis to perform. Options: 'diffusion_analysis', 'shortest_paths', 'degrees'. Default: 'diffusion_analysis")

    group.add_argument('--cutoff', type=float, default=0.01,
                       help="Cutoff of fraction of diffusion recieved to use to choose the main contributors.")

    group.add_argument('--k-to-test', '-k', type=int, action="append", default=[],
                       help="k-value(s) for which to get the top-k predictions to test. " +
                       "If not specified, will check the config file.")

    group.add_argument('--sub-k', type=int, default=10,
                       help="k-value(s) for which to get the top-k predictions to plot the individual contribution from"
                            "neighboring nodes " +
                            "If not specified, will check the config file. Default=10")
    group.add_argument('--stat-sig-cutoff', type=float,
                       help="Cutoff on the node p-value for a node to be considered in the topk. " + \
                       "The p-values should already have been computed with run_eval_algs.py")
    group.add_argument('--terms-file', type=str, 
                       help="Plot the effective diffusion values per term.")
#
#    group = parser.add_argument_group('FastSinkSource Pipeline Options')
    group.add_argument('--run-algs', type=str, action='append', default=[])
#    group.add_argument('--num-reps', type=int, 
#                       help="Number of times negative sampling was repeated to compute the average scores. Default=1")
    group.add_argument('--sample-method', type=str, default='kmeans',
                       help="Approach used to sample random sets of positive examples. " + \
                       "Options: 'kmeans' (bin nodes by kmeans clustring on the degree distribution), 'simple' (uniformly at random)")
    group.add_argument('--num-random-sets', type=int, default=1000,
                       help="Number of random sets used when computing pvals. Default: 1000")
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


    # for each dataset, extract the path(s) to the prediction files,
    # read in the predictions, and test for the statistical significance of overlap 
    for dataset in input_settings['datasets']:
        dataset_name = config_utils.get_dataset_name(dataset)
        print("Loading data for %s" % (dataset['net_version']))
        # load the network and the positive examples for each term
        net_obj, ann_obj, eval_ann_obj = setup_dataset(
            dataset, input_dir, **kwargs)
        prots, node2idx = net_obj.nodes, net_obj.node2idx

        for term in ann_obj.terms:
            term_idx = ann_obj.term2idx[term]
            orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, term_idx)
            orig_pos = [prots[p] for p in orig_pos_idx]
            pos_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]

            for alg_name in alg_settings:
                if (alg_settings[alg_name]['should_run'][0]==True) or (alg_name in kwargs.get('run_algs')):
                    # load the top predictions

                    alg_pred_files = config_utils.get_dataset_alg_prediction_files(
                        output_dir, dataset, alg_settings, [alg_name], **kwargs)
                    cutoff = kwargs.get('cutoff', 0.01)

                    # get the alpha values to use
                    alphas = alg_settings[alg_name]['alpha']
                    alpha_frac_main_contr_nonnbrs = {}
                    alpha_nodes_pos_nbr_dfsn = {}
                    beta_nodes_pos_nbr_dfsn = {}
                    alpha_dfs = {}

                    for alpha, alg in zip(alphas, alg_pred_files):
                        pred_file = alg_pred_files[alg]

                        #Now find term_spec prediction file
                        # pred_file = pred_file_prefix.split('.')[0] + '-'+term.replace(':','-')\
                        #             + '.' + pred_file_prefix.split('.')[-1]
                        pred_file = script_utils.term_based_pred_file(pred_file, term)


                        if not os.path.isfile(pred_file):
                            print("Warning: %s not found. skipping" % (pred_file))
                            continue
                        print("reading %s for alpha=%s" % (pred_file, alpha))
                        df = pd.read_csv(pred_file, sep='\t')

                        # remove the original positives
                        df = df[~df['prot'].isin(orig_pos)]
                        df.reset_index(inplace=True, drop=True)

                        sig_cutoff = kwargs.get('stat_sig_cutoff')
                        sig_str = "-sig%s" % (str(sig_cutoff).replace('.','_')) if sig_cutoff else ""
                        if sig_cutoff:
                            df = config_utils.get_pvals_apply_cutoff(df, pred_file, **kwargs)

                        if k > len(df['prot']):
                            print("ERROR: k %s > num predictions %s. Quitting" % (k, len(df['prot'])))
                            sys.exit()

                        #Nure: add alg_name to diff_mat_file
                        diff_mat_file = "%s/diffusion-mat-%s-a%s.npy" % \
                                (net_obj.out_pref,alg_name, str(alpha).replace('.','_'))

                        pred_scores = np.zeros(len(net_obj.nodes))
                        df = df[:k]
                        top_k_pred = df['prot']
                        top_k_pred_idx = [net_obj.node2idx[n] for n in top_k_pred]
                        pred_scores[top_k_pred_idx] = df['score'].values

                        # print('pred_scores: ', pred_scores[0:10])


                        # Nure: add alg_name to out_pref
                        out_pref = config_map['output_settings']['output_dir']+"/viz/%s/%s/diffusion-node-analysis/%s/cutoff%s-k%s-a%s%s" % (
                            dataset['net_version'], dataset['exp_name'], alg_name, cutoff, k, alpha, sig_str)

                        if alg_name == 'genemaniaplus':
                            M_inv = gm.get_diffusion_matrix(net_obj.W, alpha=alpha, diff_mat_file=diff_mat_file,
                                                            force_run=False)
                        elif alg_name == 'rwr':
                            M_inv = rwr.get_diffusion_matrix(net_obj.W, alpha=alpha, diff_mat_file=diff_mat_file,
                                                             force_run=False)
                            M_inv = M_inv * (alpha / float(len(pos_nodes_idx)))

                        frac_main_contr_nonnbrs, nodes_pos_nonnbr_dfsn = get_effective_diffusion_score(
                            pred_scores, M_inv, net_obj, pos_nodes_idx, alpha=alpha,
                            diff_mat_file=diff_mat_file, out_pref=out_pref, **kwargs)


                        # # exploratory analysis:
                        nodes_pos_nonnbr_dfsn_idx = copy.deepcopy(nodes_pos_nonnbr_dfsn)

                        # convert the node IDs to protein IDs
                        nodes_pos_nonnbr_dfsn = {prots[n]: val for n, val in nodes_pos_nonnbr_dfsn.items()}
                        alpha_nodes_pos_nbr_dfsn[alpha] = nodes_pos_nonnbr_dfsn

                        if alg_name=='genemaniaplus':
                            #Nure: introduced new param beta = 1/(1+alpha). this beta variabel is comparable with alpha in rwr.
                            beta_nodes_pos_nbr_dfsn[round(float(1/(1+alpha)),2)] = nodes_pos_nonnbr_dfsn


                        #save the effective diffusion values
                        effective_diff_file = config_map['output_settings'][
                                         'output_dir'] + "/viz/%s/%s/diffusion-node-analysis/%s/node-effective-diff-k%s-a%s%s.tsv" % (
                                         dataset['net_version'], term, alg_name, k, alpha, sig_str)
                        ed_df = pd.DataFrame({'target': list(nodes_pos_nonnbr_dfsn_idx.keys()),
                                              'node_effective_diffusion': list(nodes_pos_nonnbr_dfsn_idx.values())})
                        os.makedirs(os.path.dirname(effective_diff_file),exist_ok=True)
                        ed_df.to_csv(effective_diff_file, sep='\t', index=False)


                        alpha_frac_main_contr_nonnbrs[alpha] = frac_main_contr_nonnbrs
                        alpha_dfs[alpha] = df

                        ####### Analysis for per source node contribution
                        sub_k = kwargs.get('sub_k')
                        per_source_out_file_path = config_map['output_settings']['output_dir'] + "/viz/%s/%s/diffusion-node-analysis/%s/per_source_node_contr/%s-subk%s-a-%s%s.pdf" % (
                            dataset['net_version'], dataset['exp_name'], alg_name, term.replace(':','-'), sub_k, alpha, sig_str)
                        os.makedirs(os.path.dirname(per_source_out_file_path), exist_ok=True)
                        # if not os.path.exists(out_file_path):
                        plot_contribution_from_individual_sources(top_k_pred_idx[0:sub_k], pred_scores,
                                                                  pos_nodes_idx, M_inv, per_source_out_file_path)


                    out_pref = config_map['output_settings']['output_dir']+\
                        "/viz/%s/%s/diffusion-node-analysis/%s/%s-k%s%s" % (
                        dataset['net_version'], dataset['exp_name'],alg_name, term.replace(':','-'), k, sig_str)
                    out_file = "%s-frac-nonnbr-dfsn.pdf" % (out_pref)

                    df = pd.DataFrame(alpha_nodes_pos_nbr_dfsn)

                    # ylabel = 'Fraction Non-Nbr Diffusion'
                    ylabel = 'Node Based Effective Diffusion'
                    title = dataset['plot_exp_name'] + '_' + term + '_' + get_plot_alg_name(alg_name)

                    plot_effective_diffusion(df, out_file, xlabel="Alpha", ylabel=ylabel, title=title)

                    #in RL/genemaniaplus I introduced a new variable beta=1/(1+alpha) which makes the effective diffusion plots between
                    #RWR and RL comparable. as beta increases effective diffusion decreases, same as in RWR, as alpha increases
                    # effective diffusion decreases.
                    if alg_name == 'genemaniaplus':
                        out_file = "%s-frac-nonnbr-dfsn-beta.pdf" % (out_pref)
                        df_with_beta = pd.DataFrame(beta_nodes_pos_nbr_dfsn)
                        df_with_beta = df_with_beta.reindex(sorted(df_with_beta.columns), axis=1)
                        plot_effective_diffusion(df_with_beta, out_file, xlabel="Beta", ylabel=ylabel, title=title)


                    ## also make a scatterplot of the nodes rank with the effective diffusion
                    for alpha in alphas:
                        pred_df = alpha_dfs[alpha]
                        pred_df.set_index('prot', inplace=True)
                        #alpha_diff_df = df[alpha]
                        df['score'] = pred_df[pred_df.index.isin(df.index)]['score']

                        if kwargs.get('terms_file'):
                            out_file = "%s-alpha%s-eff-diff-per-term.pdf" % (out_pref, alpha)
                            plot_eff_diff_per_term(df[alpha], kwargs['terms_file'], out_file, title="Alpha=%s, k=%s" % (alpha, k))

                        # TODO count the distance away of the main contributors
                        # count the # krogan proteins each prediction is connected to(?)
                        out_file = "%s-alpha%s-num-pos-nbrs.pdf" % (out_pref, alpha)
                        plot_num_krogan_nbrs(df, alpha, pos_nodes_idx, net_obj, out_file)


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
    # print(df.head())
    g = sns.jointplot(data=df, x=alpha, y='frac_krogan_nbrs')
    g.ax_joint.set_xlabel("Effective Diffusion (a=%s)" %alpha)
    g.ax_joint.set_ylabel("Fraction of Nbrs that are Krogan nodes")
    # g.ax_joint.set_title(title)
    #plt.suptitle(title)
    print(out_file)
    plt.savefig(out_file)
    plt.show()
    plt.close()

    g = sns.jointplot(data=df, x='num_krogan_nbrs', y='frac_krogan_nbrs')
    out_file = out_file.replace('.pdf','-by-num.pdf')
    print(out_file)
    plt.savefig(out_file)
    plt.show()
    #sns.pairplot(data=df)
    plt.close()


def plot_eff_diff_per_term(eff_diff_values, terms_file, out_file, xlabel="Effective Diffusion", title=""):
    df = pd.read_csv(terms_file, sep=',')
    # convert the string version of a set to actual python sets
    df['geneID'] = df['geneID'].apply(lambda x: eval(str(x)))
    # print(df)
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
    # print(df2)

    ax = sns.boxplot(data=df2, orient='h', whis=np.inf)
    ax = sns.swarmplot(data=df2, orient='h', size=3.25, color="0.25")

    ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    ax.set_title(title)

    print(out_file)
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_eff_diff_by_rank(df, alpha, out_file, xlabel="Alpha", ylabel="", title=""):
    g = sns.jointplot(x='score', y=alpha, data=df)

    g.ax_joint.set_xlabel(xlabel)
    g.ax_joint.set_ylabel(ylabel)
    # g.ax_joint.set_title(title)
    plt.suptitle(title)

    print(out_file)
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_effective_diffusion(df, out_file, xlabel="Alpha", ylabel="", title=""):
    ax = sns.boxplot(data=df)
    ax.set(ylim=(0, 1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    print(out_file)
    plt.savefig(out_file, bbox_inches='tight')
    plt.savefig(out_file.replace('.pdf','.png'), bbox_inches='tight')

    plt.show()
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

    main_contributors, fracs_top_nbrs, nodes_pos_nbr_dfsn = gm.get_pred_main_contributors(
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
    f, process_time, wall_time, num_iters = gm.runGeneMANIA(L, y, alpha=alpha)
#     non_krogan_nodes = np.ones(L.shape[0])
#     non_krogan_nodes[krogan_node_idx] = 0
    f[y.nonzero()[0]] = 0
    top_pred = np.argsort(f)[::-1][:k]
    return top_pred

def plot_contribution_from_individual_sources( top_preds,pred_scores, pos_nodes_idx, M_inv, out_file):
    #applicable for any M_inv where sources -> columns, targets -> rows.
    mask = np.zeros(M_inv.shape[1], dtype=bool)
    mask[pos_nodes_idx] = True
    M_inv_new = M_inv[:, mask] #keep the columns with source/pos_nodes only
    print(M_inv.shape, M_inv_new.shape)
    for prot_idx in top_preds:
        # get contribution from source nodes for each top predictions and sort.
        contr = list(M_inv_new[prot_idx])
        dif = np.abs(pred_scores[prot_idx] - np.sum(np.array(contr)))
        assert dif < 0.001, print('big_diff: ', dif)
        contr = [x/pred_scores[prot_idx] for x in contr]
        contr.sort(reverse=True)
        # print('prot_idx: ',prot_idx)
        # print('M_inv_sub', M_inv_new[prot_idx])
        # print('contr', contr)
        plt.plot(contr, linewidth=1)
        plt.xticks(range(0, len(pos_nodes_idx), 10), fontsize=6, rotation=90)
        plt.xlabel('Source Proteins ')
        plt.ylabel('Fraction of Contibution to Diffusion Score')
        # plt.ylim([0, contr[0]])
    plt.savefig(out_file)
    plt.show()
    print(out_file)
    plt.close()

if __name__ == "__main__":
    config_map, kwargs = parse_args()
    for k in kwargs.get('k_to_test', [332]):
        main(config_map, k=k, **kwargs)
