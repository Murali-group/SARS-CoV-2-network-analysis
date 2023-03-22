import os, sys
import yaml
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu, describe

sys.path.insert(1, "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/")
from src.FastSinkSource.src.main import setup_dataset
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils


def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        # config_map = yaml.load(conf, Loader=yaml.FullLoader)
        config_map = yaml.load(conf)
    return config_map, kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="Script to analyze the top contributors of each prediction's "
                                                 "diffusion score, as well as the effective diffusion (i.e.,"
                                                 " fraction of diffusion received from non-neighbors)")
    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                        "fss_inputs/config_files/provenance/string700_s12.yaml"
                       , help="Configuration file used when running FSS. ")

    group.add_argument('--run-algs', type=str, action='append', default=[])
    group.add_argument('--k-to-test', '-k', type=int, action='append', default=[332],
                       help="k-value(s) for which to get the top-k predictions to test. " +
                            "If not specified, will check the config file.")

    group.add_argument('--n-sp', '-n', type=int, default=200,
                       help="n-sp is the number of shortest paths to be considered" +
                            "If not specified, will check the config file. Default=20")
    group.add_argument('--m', type=int, default=20,
                       help="for each top prediction, for how many top contributing sources we wanna analyse the path" +
                            "Default=20")


    group.add_argument('--stat-sig-cutoff', type=float,
                       help="Cutoff on the node p-value for a node to be considered in the topk. " + \
                            "The p-values should already have been computed with run_eval_algs.py")

    return parser


def main(config_map, k, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """

    # extract the general variables from the config map
    input_settings, input_dir, output_dir, alg_settings, kwargs \
        = config_utils.setup_config_variables(config_map, **kwargs)

    sig_cutoff = kwargs.get('stat_sig_cutoff')
    sig_str = "-sig%s" % (str(sig_cutoff).replace('.', '_')) if sig_cutoff else ""

    for dataset in input_settings['datasets']:

        print("Loading data for %s" % (dataset['net_version']))
        # load the network and the positive examples for each term
        net_obj, ann_obj, _ = setup_dataset(
            dataset, input_dir, **kwargs)
        prots, node2idx = net_obj.nodes, net_obj.node2idx

        for term in ann_obj.terms:
            term_idx = ann_obj.term2idx[term]
            orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, term_idx)
            orig_pos = [prots[p] for p in orig_pos_idx]
            pos_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]

            for alg_name in alg_settings:
                print('\n\nAlgorithm Name: ', alg_name)
                if (alg_settings[alg_name]['should_run'][0] == True) or (alg_name in kwargs.get('run_algs')):
                    # get the alpha values to use
                    alphas = alg_settings[alg_name]['alpha']

                    #dictionaries to pass to plotting functions
                    path_based_ed_all_alpha_dict={alpha:[] for alpha in alphas}
                    path_based_ed_all_beta_dict={}

                    node_based_ed_all_alpha_dict = {alpha: [] for alpha in alphas}
                    node_based_ed_all_beta_dict = {}


                    for alpha in alphas:

                        path_based_net_alg_settings = config_map['output_settings']['output_dir'] + \
                                        "/viz/%s/%s/diffusion-path-analysis/%s/" % (dataset['net_version'], term, alg_name)
                        path_based_run_settings = "k%s-nsp%s-m%s-a%s%s" \
                                       % (k, kwargs.get('n_sp'), kwargs.get('m'), alpha, sig_str)
                        path_based_ed_file = "%s/path-length-wise-effective-diff-%s.tsv" \
                                                    % (path_based_net_alg_settings, path_based_run_settings)

                        node_based_net_alg_settings = config_map['output_settings']['output_dir'] + \
                                                      "/viz/%s/%s/diffusion-node-analysis/%s/" % (
                                                      dataset['net_version'], term, alg_name)
                        node_based_run_settings = "k%s-a%s%s" % (k, alpha, sig_str)
                        node_based_ed_file = "%s/node-effective-diff-%s.tsv" \
                                                                     % (node_based_net_alg_settings,
                                                                        node_based_run_settings)

                        path_based_ed_df = pd.read_csv(path_based_ed_file, sep='\t', index_col=None).set_index('target').sort_index()
                        node_based_ed_df = pd.read_csv(node_based_ed_file, sep='\t', index_col=None).set_index('target').sort_index()

                        path_based_ed_all_alpha_dict[alpha] = list(path_based_ed_df['effective_diffusion'])
                        node_based_ed_all_alpha_dict[alpha] = list(node_based_ed_df['node_effective_diffusion'])

                        if alg_name == 'genemaniaplus':
                            # Nure: introduced new param beta = 1/(1+alpha). this beta variabel is comparable with alpha in rwr.
                            path_based_ed_all_beta_dict[round(float(1 / (1 + alpha)), 2)] = \
                                path_based_ed_all_alpha_dict[alpha]
                            node_based_ed_all_beta_dict[round(float(1 / (1 + alpha)), 2)] = \
                                node_based_ed_all_alpha_dict[alpha]


                    #******************** significance analysis *******************
                    #see if the difference between node based and path based diffusion values is significant
                    net_alg_settings = config_map['output_settings']['output_dir'] + \
                                       "/viz/%s/%s/node-vs-path-analysis/%s/" % (dataset['net_version'], term, alg_name)
                    os.makedirs(net_alg_settings, exist_ok=True)
                    pvalue_file = net_alg_settings + "pval-k%s-nsp%s-m%s%s.tsv" \
                                        % ( k, kwargs.get('n_sp'),kwargs.get('m'),sig_str)

                    if alg_name=='genemaniaplus':
                        for beta in node_based_ed_all_beta_dict:
                            _, ttest_pval = ttest_ind(node_based_ed_all_beta_dict[beta], path_based_ed_all_beta_dict[beta],\
                                            alternative = 'less')
                            print('beta: ', beta, 'ttest_pval: ', ttest_pval)

                            _, wc_pval = wilcoxon(node_based_ed_all_beta_dict[beta],
                                                      path_based_ed_all_beta_dict[beta], \
                                                      alternative='less')
                            print('beta: ', beta, 'wc_pval: ', wc_pval)

                            _, mw_pval = mannwhitneyu(node_based_ed_all_beta_dict[beta],
                                                      path_based_ed_all_beta_dict[beta], \
                                                      alternative='less')
                            print('beta: ', beta, 'mw_pval: ', mw_pval)

                    elif alg_name=='rwr':
                        for alpha in node_based_ed_all_alpha_dict:
                            _, ttest_pval = ttest_ind(node_based_ed_all_alpha_dict[alpha],
                                            path_based_ed_all_alpha_dict[alpha],\
                                            alternative = 'less')
                            print('alpha: ', alpha, 'ttest_pval: ', ttest_pval)
                            _, wc_pval = wilcoxon(node_based_ed_all_alpha_dict[alpha],
                                                      path_based_ed_all_alpha_dict[alpha], \
                                                      alternative='less')
                            print('alpha: ', alpha, 'wc_pval: ', wc_pval)
                            _, mw_pval = mannwhitneyu(node_based_ed_all_alpha_dict[alpha],
                                                      path_based_ed_all_alpha_dict[alpha], \
                                                      alternative='less')
                            print('alpha: ', alpha, 'mw_pval: ', mw_pval)

if __name__ == "__main__":
    config_map, kwargs = parse_args()
    for k in kwargs.get('k_to_test'):
        main(config_map, k=k, **kwargs)
