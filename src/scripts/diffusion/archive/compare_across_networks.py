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
                        "fss_inputs/config_files/provenance/string700_biogrid_y2h_signor_s1.yaml"
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


def diff_in_ed_across_networks(ed_all_alpha_dict, sig_test, alternative='two-sided'):
    '''
    ed_all_alpha_dict: dict of dict. first key = alpha, second key = (network_name, term), value = list containing
    nodebased/pathbased effective diffusion values for top k predictions.

    Output: For a term, for each alpha values . we will have difference in effective diffusion
    values between two networks.
    '''
    ed_alphas={}
    for alpha in ed_all_alpha_dict:
        ed_alphas[alpha] ={}
        termwise_nets = {}
        for (net,term) in ed_all_alpha_dict[alpha]:
            if term not in termwise_nets:
                termwise_nets[term] = [ed_all_alpha_dict[alpha][(net,term)]]
            else:
                termwise_nets[term].append(ed_all_alpha_dict[alpha][(net,term)])

        for term in termwise_nets:
            ed_network1 = termwise_nets[term][0]
            ed_network2 = termwise_nets[term][1]

            if sig_test=='wc':
                _, pval = wilcoxon(ed_network1,ed_network2, alternative=alternative)
            elif sig_test=='mw':
                _, pval = mannwhitneyu(ed_network1,ed_network2, alternative=alternative)
            elif sig_test=='t':
                _, pval = ttest_ind(ed_network1, ed_network2, alternative=alternative)

            ed_alphas[alpha][term] = pval
            print('term: ', term,  ' alpha: ', alpha, sig_test,': ', pval )

    return ed_alphas

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



    for alg_name in alg_settings:
        if (alg_settings[alg_name]['should_run'][0] == True) or (alg_name in kwargs.get('run_algs')):
            # get the alpha values to use
            alphas = alg_settings[alg_name]['alpha']
            betas = [ round(float(1 / (1 + alpha)), 2) for alpha in alphas]
            #dictionaries to pass to plotting functions
            path_based_ed_all_alpha_dict={alpha:{} for alpha in alphas}
            path_based_ed_all_beta_dict={beta:{} for beta in betas}

            node_based_ed_all_alpha_dict = {alpha: {} for alpha in alphas}
            node_based_ed_all_beta_dict = {beta:{} for beta in betas}

            network_names=[]
            for alpha in alphas:
                for dataset in input_settings['datasets']:
                    print("Loading data for %s" % (dataset['net_version']))
                    network_name = dataset['net_version'].replace('networks/','')
                    network_names.append(network_name)
                    # load the network and the positive examples for each term
                    net_obj, ann_obj, _ = setup_dataset(
                        dataset, input_dir, **kwargs)
                    prots, node2idx = net_obj.nodes, net_obj.node2idx

                    for term in ann_obj.terms:  # TODO for GO analysis get terms from a list of chosen terms.
                        term_idx = ann_obj.term2idx[term]
                        orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, term_idx)
                        orig_pos = [prots[p] for p in orig_pos_idx]
                        pos_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]
                        path_based_net_alg_settings = config_map['output_settings']['output_dir'] + \
                                        "/viz/%s/%s/diffusion-path-analysis/%s/" % (dataset['net_version'], term, alg_name)
                        path_based_run_settings = "k%s-nsp%s-m%s-a%s%s" \
                                       % (k, kwargs.get('n_sp'), kwargs.get('m'), alpha, sig_str)
                        path_based_ed_file = "%s/path-length-wise-effective-diff-ss-%s.tsv" \
                                                    % (path_based_net_alg_settings, path_based_run_settings)

                        node_based_net_alg_settings = config_map['output_settings']['output_dir'] + \
                                                      "/viz/%s/%s/diffusion-node-analysis/%s/" % (
                                                      dataset['net_version'], term, alg_name)
                        node_based_run_settings = "k%s-a%s%s" % (k, alpha, sig_str)
                        node_based_ed_file = "%s/node-effective-diff-%s.tsv" \
                                                                     % (node_based_net_alg_settings,
                                                                        node_based_run_settings)
                        # ###TODO Remove the following 4 lines when you are prepared to run this analysis on fixed version
                        # ## of STRING and BioGRIDY2H i.e. the ed analysis is done on them.
                        # path_based_ed_file = path_based_ed_file.replace("sept22","sept22-MV")
                        # node_based_ed_file = node_based_ed_file.replace("sept22","sept22-MV")
                        #
                        # path_based_ed_file = path_based_ed_file.replace("11-5", "11-5-oldmapfile")
                        # node_based_ed_file = node_based_ed_file.replace("11-5", "11-5-oldmapfile")

                        path_based_ed_df = pd.read_csv(path_based_ed_file, sep='\t', index_col=None).set_index('target').sort_index()
                        node_based_ed_df = pd.read_csv(node_based_ed_file, sep='\t', index_col=None).set_index('target').sort_index()

                        path_based_ed_all_alpha_dict[alpha][(network_name, term)] = list(path_based_ed_df['effective_diffusion'])
                        node_based_ed_all_alpha_dict[alpha][(network_name, term)] = list(node_based_ed_df['node_effective_diffusion'])

                        if alg_name == 'genemaniaplus':
                            # Nure: introduced new param beta = 1/(1+alpha). this beta variabel is comparable with alpha in rwr.
                            path_based_ed_all_beta_dict[round(float(1 / (1 + alpha)), 2)][(network_name, term)] = \
                                path_based_ed_all_alpha_dict[alpha][(network_name, term)]
                            node_based_ed_all_beta_dict[round(float(1 / (1 + alpha)), 2)][(network_name, term)] = \
                                node_based_ed_all_alpha_dict[alpha][(network_name, term)]

            #
            # #******************** significance analysis *******************
            #see the difference between node based(or path based effective diffusion) values across multiple networks
            alphas_string = '-'.join(str(alpha) for alpha in alphas)
            network_names = '-'.join(network_names)
            # out_dir_prefix = config_map['output_settings']['output_dir'] + \
            #                    "/viz/compare-networks/%s/%s/%s/" % (alg_name, alphas_string, network_names)
            # os.makedirs(out_dir_prefix, exist_ok=True)

            # diff_in_ed_across_networks(node_based_ed_all_alpha_dict, sig_test='t', alternative='two-sided')
            # diff_in_ed_across_networks(path_based_ed_all_alpha_dict, sig_test='t', alternative='two-sided')

            print('\nAlg name: ', alg_name)
            print('\n Network Names: ', network_names)
            alternative = 'less'
            if alg_name == 'genemaniaplus':
                print('\nNode based diffusion:\n')
                diff_in_ed_across_networks(node_based_ed_all_beta_dict, sig_test='mw', alternative=alternative)
                print('\nPath based diffusion:\n')
                diff_in_ed_across_networks(path_based_ed_all_beta_dict, sig_test='mw', alternative=alternative)
            elif alg_name == 'rwr':
                print('\nNode based diffusion:\n')
                diff_in_ed_across_networks(node_based_ed_all_alpha_dict, sig_test='mw', alternative=alternative)
                print('\nPath based diffusion:\n')
                diff_in_ed_across_networks(path_based_ed_all_alpha_dict, sig_test='mw', alternative=alternative)

            print('comparing across networks done')

if __name__ == "__main__":
    config_map, kwargs = parse_args()
    for k in kwargs.get('k_to_test'):
        main(config_map, k=k, **kwargs)
