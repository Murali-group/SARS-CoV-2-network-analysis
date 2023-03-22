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


    group.add_argument('--cutoff', type=float, default=0.01,
                       help="Cutoff of fraction of diffusion recieved to use to choose the main contributors.")

    group.add_argument('--k-to-test', '-k', type=int, action="append", default=[332],
                       help="k-value(s) for which to get the top-k predictions to test. " +
                       "If not specified, will check the config file.")
    group.add_argument('--pos-k', action='store_true', default=False,
                       help="if true get the top-k predictions to test is equal to the number of positive annotations")

    group.add_argument('--terms-file', type=str, 
                       help="Plot the effective diffusion values per term.")
#
#    group = parser.add_argument_group('FastSinkSource Pipeline Options')
    group.add_argument('--run-algs', type=str, action='append', default=[])


    group.add_argument('--force-run', action='store_true', default=False,
                       help="Force re-running the path lengths for random sets, and re-writing the output files")


    group.add_argument('--balancing-alpha-only', action='store_true', default=True,
                       help="Ignore alpha from config file rather take the alpha value\
                            that balanced the two loss terms in quad loss function for the corresponding\
                            network-term-alg")

    return parser


def main(config_map, k , **kwargs):
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

        for alg_name in alg_settings:
            #read info for Balancing alpha
            alpha_summary_filename = config_map['output_settings']['output_dir'] + \
                "/viz/%s/%s/param_select/" % (dataset['net_version'], dataset['exp_name']) + '/' + alg_name + '/alpha_summary.tsv'
            alpha_summary_df = pd.read_csv(alpha_summary_filename, sep='\t', index_col=None)[['term', 'balancing_alpha']]
            term_2_balancing_alpha_dict = dict(zip(alpha_summary_df['term'], alpha_summary_df['balancing_alpha']))

            if (alg_settings[alg_name]['should_run'][0]==True) or (alg_name in kwargs.get('run_algs')):
                node_based_effective_diffsuion_all_term_dict = {}
                for term in ann_obj.terms:

                    term_idx = ann_obj.term2idx[term]
                    orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, term_idx)
                    orig_pos = [prots[p] for p in orig_pos_idx]
                    pos_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]
                    n_pos = len(pos_nodes_idx)
                    if kwargs.get('pos_k'):
                        k = n_pos
                    #find balancing alpha for the param
                    balancing_alpha = term_2_balancing_alpha_dict[term]
                    alg_settings[alg_name]['alpha'] = [balancing_alpha]

                    alg_pred_files = config_utils.get_dataset_alg_prediction_files(
                        output_dir, dataset, alg_settings, [alg_name], **kwargs)
                    cutoff = kwargs.get('cutoff', 0.01)

                    # get the alpha values to use
                    alphas = alg_settings[alg_name]['alpha']

                    for alpha, alg in zip(alphas, alg_pred_files):
                        pred_file = alg_pred_files[alg]
                        pred_file = script_utils.term_based_pred_file(pred_file, term)
                        if not os.path.isfile(pred_file):
                            print("Warning: %s not found. skipping" % (pred_file))
                            continue
                        print("reading %s for alpha=%s" % (pred_file, alpha))
                        df = pd.read_csv(pred_file, sep='\t')

                        # remove the original positives
                        df = df[~df['prot'].isin(orig_pos)]
                        df.reset_index(inplace=True, drop=True)


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
                        out_pref = config_map['output_settings']['output_dir']+"/viz/%s/%s/diffusion-node-analysis/%s/cutoff%s-k%s-a%s" % (
                            dataset['net_version'], dataset['exp_name'], alg_name, cutoff, k, alpha)

                        if alg_name == 'genemaniaplus':
                            M_inv = gm.get_diffusion_matrix(net_obj.W, alpha=alpha, diff_mat_file=diff_mat_file,
                                                            force_run=False)
                        elif alg_name == 'rwr':
                            M_inv = rwr.get_diffusion_matrix(net_obj.W, alpha=alpha, diff_mat_file=diff_mat_file,
                                                             force_run=False)
                            M_inv = M_inv * (alpha / float(len(pos_nodes_idx)))

                        frac_main_contr_nonnbrs, nodes_pos_nonnbr_dfsn = get_effective_diffusion_score(
                            pred_scores, M_inv, net_obj, pos_nodes_idx, alpha=alpha,k=k,
                            diff_mat_file=diff_mat_file, out_pref=out_pref, **kwargs)


                        nodes_pos_nonnbr_dfsn_idx = copy.deepcopy(nodes_pos_nonnbr_dfsn)
                        node_based_effective_diffsuion_all_term_dict[term] = list(nodes_pos_nonnbr_dfsn_idx.values())

                        #save the effective diffusion values
                        effective_diff_file = config_map['output_settings'][
                                         'output_dir'] + "/viz/%s/%s/diffusion-node-analysis/%s/node-effective-diff-k%s-a%s.tsv" % (
                                         dataset['net_version'], term, alg_name, k, alpha)
                        ed_df = pd.DataFrame({'target': list(nodes_pos_nonnbr_dfsn_idx.keys()),
                                              'node_effective_diffusion': list(nodes_pos_nonnbr_dfsn_idx.values())})
                        os.makedirs(os.path.dirname(effective_diff_file),exist_ok=True)
                        ed_df.to_csv(effective_diff_file, sep='\t', index=False)

                # ********************plot*******************
                plot_dir = config_map['output_settings']['output_dir'] + \
                           "/viz/%s/%s/diffusion-node-analysis/%s/" % (
                           dataset['net_version'], dataset['exp_name'], alg_name)
                os.makedirs(plot_dir, exist_ok=True)
                title = dataset['plot_exp_name'] + '_' + dataset['exp_name'] + '_' + get_plot_alg_name(alg_name)

                # plot diffusion coming beyond pathlength 1 across alphas
                boxplot_dfsn_across_terms(node_based_effective_diffsuion_all_term_dict, 'Terms',
                                          'Node Based Effective Diffusion', \
                                          title, plot_dir + 'node_effective_diffusion_terms' + '.pdf')


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

if __name__ == "__main__":
    config_map, kwargs = parse_args()
    for k in kwargs.get('k_to_test', [332]):
        main(config_map, k=k, **kwargs)
