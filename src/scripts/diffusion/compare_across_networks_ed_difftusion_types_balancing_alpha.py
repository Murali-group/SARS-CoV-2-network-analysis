import os, sys
import yaml
import argparse
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
import math
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu, describe

sys.path.insert(1, "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/")
from src.FastSinkSource.src.main import setup_dataset
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils

net_name_alias = {'stringv11-5/700': 'STRING', 'biogrid-physical-sept22-single-uniprot':'BioGRID-Physical',
                 'biogrid-y2h-sept22-single-uniprot':'BioGRID-Y2H',
                  'HI-union': 'HI-union'}
def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        config_map = yaml.load(conf, Loader=yaml.FullLoader)
        # config_map = yaml.load(conf)
    return config_map, kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="Script to analyze the top contributors of each prediction's "
                                                 "diffusion score, as well as the effective diffusion (i.e.,"
                                                 " fraction of diffusion received from non-neighbors)")
    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                        "fss_inputs/config_files/provenance/string700_biogrid_physical_biogrid_y2h_hi_union_s1.yaml"
                       , help="Configuration file used when running FSS. ")

    # group.add_argument('--config', type=str, default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
    #                     "fss_inputs/config_files/provenance/"
    #                     "string700_biogrid_y2h_signor_s1.yaml"
    #                    , help="Configuration file used when running FSS. ")

    group.add_argument('--run-algs', type=str, action='append', default=[])
    group.add_argument('--k-to-test', '-k', type=int, action='append', default=[332],
                       help="k-value(s) for which to get the top-k predictions to test. " +
                            "If not specified, will check the config file.")
    group.add_argument('--pos-k', action='store_true', default=False,
                       help="if true get the top-k predictions to test is equal to the number of positive annotations")

    group.add_argument('--common', action='store_true', default=False,
                       help="if true then while comparing across networks, only compare the effective diffusion of"
                       "overlapping top-predicted proteins i.e. proteins that ranked top both in STRING"
                       "and BioGRID")

    group.add_argument('--n-sp', '-n', type=int, default=1000,
                       help="n-sp is the number of shortest paths to be considered" +
                            "If not specified, will check the config file. Default=20")

    group.add_argument('--stat-sig-cutoff', type=float,
                       help="Cutoff on the node p-value for a node to be considered in the topk. " + \
                            "The p-values should already have been computed with run_eval_algs.py")
    # group.add_argument('--path-types', action='store_true', default=False,
    #                    help="if true then plot the diffusion across different types of paths")

    return parser


def get_pval_sig_test(sig_test, ed_network1, ed_network2, alternative):
    if sig_test == 'wc':
        _, pval = wilcoxon(ed_network1, ed_network2, alternative=alternative)
    elif sig_test == 'mw':
        _, pval = mannwhitneyu(ed_network1, ed_network2, alternative=alternative)
    elif sig_test == 't':
        _, pval = ttest_ind(ed_network1, ed_network2, alternative=alternative)
    return pval

def diff_in_ed_across_networks(ed_dict, sig_test, network_names, alternative='two-sided'):
    '''
    ed_dict: dict of dict. key1= term,key2=network, value = pandas series containing
    nodebased/pathbased effective diffusion values for top k predictions.

    Output: A dictionary. key = term, value=pvalue of significant test for difference between effective
    diffusion between two networks(ordered as in networks_names)
    '''

    pval_dict = {}
    for term in ed_dict:
        # check if for a certain term effective diffusion from both networks are present, otherwise exclude that term.
        if len(ed_dict[term].keys())!=2:
            continue
        ed_network1 = list(ed_dict[term][network_names[0]])
        ed_network2 = list(ed_dict[term][network_names[1]])
        pval = get_pval_sig_test(sig_test, ed_network1, ed_network2, alternative)

        pval_dict[term] = pval
        # print('term: ', term, sig_test,': ', pval )

    return pval_dict


def process_for_overlap(ed_dict, network_names):
    '''
    Input: ed_dict: dict of dict. key1= term,key2=network, value = pandas series containing
    nodebased/pathbased effective diffusion values for top k predictions.

    Output: a dict of same type as ed_dict but only containing overlapping proteins between two nets
    '''

    #TODO Check the generalized  function for any number of networks, instead of just 2

    for term in ed_dict:
        common_targets = set()
        for net_name in network_names:
            net = ed_dict[term][net_name]
            common_targets = set(net.index).intersection(set(common_targets)) if len(common_targets)>0 else set(net.index)
        for net_name in network_names:
            ed_dict[term][net_name] = ed_dict[term][net_name][ed_dict[term][net_name].index.isin(common_targets)]

        print('keep common preds only', len(common_targets))


def pval_node_vs_path(node_based_ed_dict, path_based_ed_dict, network_names, pval_node_vs_path_file):
    os.makedirs(os.path.dirname(pval_node_vs_path_file), exist_ok=True)
    f = open(pval_node_vs_path_file, 'w')
    f.write('term\tnetwork\tmw_pval\n')
    for term in node_based_ed_dict:
        # check if for a certain term effective diffusion from all networks are present, otherwise exclude that term.
        if len(node_based_ed_dict[term].keys())!=len(network_names) | len(path_based_ed_dict[term].keys())!=len(network_names):
            continue
        for net_name in network_names:
            node_ed = list(node_based_ed_dict[term][net_name])
            path_ed = list(path_based_ed_dict[term][net_name])
            _,pval = mannwhitneyu(node_ed, path_ed, alternative='less')
            f.write(term + '\t'+ net_name + '\t' + str(pval) + '\n')
    f.close()
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
            path_based_ed_dict={}
            node_based_ed_dict = {}
            network_names=[]
            network_wise_prots = {}

            for dataset in input_settings['datasets']:
                print("Loading data for %s" % (dataset['net_version']))
                network_name = dataset['net_version'].replace('networks/','')
                network_names.append(network_name)
                # load the network and the positive examples for each term
                net_obj, ann_obj, _ = setup_dataset(
                    dataset, input_dir, **kwargs)
                prots, node2idx = net_obj.nodes, net_obj.node2idx
                network_wise_prots[network_name] = set(prots)
                alpha_summary_filename = config_map['output_settings']['output_dir'] + "/viz/%s/%s/param_select/" \
                    % (dataset['net_version'], dataset['exp_name']) + '/' + alg_name + '/alpha_summary.tsv'
                alpha_summary_df = pd.read_csv(alpha_summary_filename, sep='\t', index_col=None)[['term', 'balancing_alpha']]
                term_2_balancing_alpha_dict = dict(zip(alpha_summary_df['term'], alpha_summary_df['balancing_alpha']))

                for term in ann_obj.terms:  # TODO for GO analysis get terms from a list of chosen terms.
                    term_idx = ann_obj.term2idx[term]
                    orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, term_idx)
                    orig_pos = [prots[p] for p in orig_pos_idx]
                    pos_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]
                    n_pos = len(pos_nodes_idx)
                    if kwargs.get('pos_k'):
                        k = n_pos

                    balancing_alpha = term_2_balancing_alpha_dict[term]
                    print('network:', network_name, ' alg: ', alg_name, ' term:', term, ' balancing_alpha:', balancing_alpha )

                    net_alg_settings = config_map['output_settings']['output_dir'] + \
                                        "/viz/%s/%s/diffusion-analysis/%s/" % (dataset['net_version'], term, alg_name)
                    run_settings = "k%s-nsp%s-a%s%s" % (k, kwargs.get('n_sp'), balancing_alpha, sig_str)
                    ed_file = "%s/effective-diff-ss-%s.tsv" % (net_alg_settings, run_settings)

                    if (not os.path.exists(ed_file)):
                        print('File not found: ', ed_file)
                        print('Continuing')
                        continue
                    ed_df = pd.read_csv(ed_file, sep='\t', index_col=None).set_index('prot')


                    if term not in path_based_ed_dict:
                        path_based_ed_dict[term] = {}
                        node_based_ed_dict[term] = {}

                    path_based_ed_dict[term][network_name] = ed_df['path_based_ed']
                    node_based_ed_dict[term][network_name] = ed_df['node_based_ed']

            ######## Process the ed_dict. If in config, common==True, then do all the downstream analysis
            ## on only overlapping too ranked proteins across two nets.
            if kwargs.get('common'):
                #the process_for_overlap() function changes the input dict inplace
                print(alg_name)
                process_for_overlap(path_based_ed_dict, network_names)
                process_for_overlap(node_based_ed_dict, network_names)
            common_str = '-common' if kwargs.get('common') else ''

            # #******************** significance analysis *******************
            # print('\nAlg name: ', alg_name)
            # print('\n Network Names: ', network_names)
            alias_alt = {'greater': 'g', 'less': 'l', 'two-sided': '2'}
            network_names_str = '_'.join(network_names)
            network_names_str = network_names_str.replace('/', '-')

            diff_in_ed_prefix = config_map['output_settings']['output_dir'] + \
                                "/viz/compare-across-networks/%s/" % (alg_name)

            ed_across_nets_plot_file = "%s/ed%s-%s.png" % (
                diff_in_ed_prefix, common_str, network_names_str)
            pval_node_vs_path_file = "%s/node_vs_path_ed_pval%s-%s.tsv" % (
                diff_in_ed_prefix, common_str, network_names_str)

            # TODO: uncomment when compare across networks
            # plot diffusion along y axis, and along x axis two clusters: 1. node-based-ed across all nets
            # 2. path-based-ed across all nets
            # boxplot_ed_across_networks(node_based_ed_dict, path_based_ed_dict, network_names, ed_across_nets_plot_file)

            # plot diffusion along y axis, and along x axis, pack node and path ed from a net together. plot such
            #pack for each net
            pval_node_vs_path(node_based_ed_dict, path_based_ed_dict, network_names, pval_node_vs_path_file)
            boxplot_ed_node_vs_path(node_based_ed_dict, path_based_ed_dict, network_names, ed_across_nets_plot_file)

            # individual_net_ed_boxplot(node_based_ed_dict, path_based_ed_dict, ed_indv_nets_plot_file)


            #TODO: The following code currently work for comparison btn only two networks. Modify it to work for pairwise
            # comparison from any number of networks

            # for alt_node_based, alt_path_based in [('greater','greater'), ('less','less'),
            #             ('greater','less'), ('less','greater'), ('two-sided','two-sided')]:
            #
            #     print('\nNode based diffusion:\n')
            #     node_ed_termwise_pval = diff_in_ed_across_networks(node_based_ed_dict, sig_test='mw', network_names= network_names, alternative=alt_node_based)
            #     node_ed_termwise_pval = OrderedDict(sorted(node_ed_termwise_pval.items()))
            #     print('\nPath based diffusion:\n')
            #     path_ed_termwise_pval = diff_in_ed_across_networks(path_based_ed_dict, sig_test='mw', network_names= network_names, alternative=alt_path_based)
            #     path_ed_termwise_pval = OrderedDict(sorted(path_ed_termwise_pval.items()))
            #
            #     #define a dataframe with terms as index, pval of difference in node-based effective diffusion across two
            #     # networks in one column, pval of difference in path-based effective diffusion across two
            #     # networks in another column
            #     ed_df = pd.DataFrame({'term': list(node_ed_termwise_pval.keys()), 'node-based': list(node_ed_termwise_pval.values()),
            #                           'path-based':list(path_ed_termwise_pval.values())})
            #
            #     assert list(node_ed_termwise_pval.keys()) ==  list(path_ed_termwise_pval.keys()), print('terms in both lists are not'
            #                                                                                              'identical')
            #
            #     ## save effective diffusion significant test results and plot
            #
            #     #save the pvalue of differences in effective diffusion across networks
            #     diff_in_ed_file = "%s/difference-in-effective-diff%s-%s-%s-%s.tsv" % \
            #                       (diff_in_ed_prefix,common_str, network_names_str,
            #                       alias_alt[alt_node_based], alias_alt[alt_path_based])
            #     ed_df.to_csv(diff_in_ed_file, sep='\t', index=False)
            #
            #     # print('total terms: ', len(ed_df))
            #
            #     ################ PLOT a Heatmap showing pvalue of differences in effective diffusion across networks
            #     diff_in_ed_plot_file = "%s/difference-in-effective-diff%s-%s-%s-%s.png" % \
            #                            (diff_in_ed_prefix, common_str, network_names_str,
            #                             alias_alt[alt_node_based], alias_alt[alt_path_based])
            #
            #     heatmap_diff_in_ed_across_networks(ed_df, diff_in_ed_plot_file)
            #
            #     # print(alt_node_based+' '+ alt_path_based + ' comparing across networks done')
            #

def heatmap_diff_in_ed_across_networks(ed_df, diff_in_ed_plot_file ):
    os.makedirs(os.path.dirname(diff_in_ed_plot_file), exist_ok=True)
    ed_df.set_index('term', inplace=True)

    ed_df_temp = copy.deepcopy(ed_df)
    ed_df_temp['node-based'] = ed_df_temp['node-based'].astype(float).apply(lambda x: -math.log10(x))
    ed_df_temp['path-based'] = ed_df_temp['path-based'].astype(float).apply(lambda x: -math.log10(x))

    p_sig_log = -math.log10(0.05)
    # ed_df_temp['node-based'] = ed_df_temp['node-based'].astype(float).apply(lambda x: x if x > p_sig_log else 0)
    # ed_df_temp['path-based'] = ed_df_temp['path-based'].astype(float).apply(lambda x: x if x > p_sig_log else 0)
    #
    my_cmap = copy.copy(plt.cm.YlGnBu)
    my_cmap.set_over("red")
    my_cmap.set_under("red")

    fig, ax = plt.subplots(figsize=(5, 15))
    sns.heatmap(ed_df_temp, cmap=my_cmap, vmin=p_sig_log)
    # plt.yticks(rotation=90)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(diff_in_ed_plot_file)
    plt.savefig(diff_in_ed_plot_file.replace('.png','.pdf'))
    plt.show()
    plt.close()


def boxplot_ed_across_networks(node_based_ed_dict, path_based_ed_dict, network_names, ed_across_nets_plot_file):
    for term in node_based_ed_dict:
        # check if for a certain term effective diffusion from all networks are present, otherwise exclude that term.
        if len(node_based_ed_dict[term].keys())!=len(network_names) | len(path_based_ed_dict[term].keys())!=len(network_names):
            continue
        df = pd.DataFrame()
        #convert each network into a df
        for net_name in network_names:
            node_ed = list(node_based_ed_dict[term][net_name])
            path_ed = list(path_based_ed_dict[term][net_name])
            n = len(node_ed)

            #use the network name I want to see in plot. Make it short.
            temp_df = pd.DataFrame({'network':[net_name_alias[net_name]]*2*n,
                           'effective diffusion type' : ['node-based']*n + ['path-based']*n,
                           'effective diffusion': node_ed + path_ed
                            })
            df = pd.concat([df,temp_df],axis=0)

        sns.boxplot(x = df['effective diffusion type'],
                    y = df['effective diffusion'],
                    hue = df['network'])
        plt.ylim([0,1])
        plt.tight_layout()
        plt.savefig(ed_across_nets_plot_file.replace('.png','-'+term+'.png'))
        plt.savefig(ed_across_nets_plot_file.replace('.png','-'+term+'.pdf'))
        plt.close()

def boxplot_ed_node_vs_path(node_based_ed_dict, path_based_ed_dict, network_names, ed_across_nets_plot_file):
    for term in node_based_ed_dict:
        # check if for a certain term effective diffusion from all networks are present, otherwise exclude that term.
        if len(node_based_ed_dict[term].keys())!=len(network_names) | len(path_based_ed_dict[term].keys())!=len(network_names):
            continue
        df = pd.DataFrame()
        #convert each network into a df
        for net_name in network_names:
            node_ed = list(node_based_ed_dict[term][net_name])
            path_ed = list(path_based_ed_dict[term][net_name])
            n = len(node_ed)
            temp_df = pd.DataFrame({'network':[net_name_alias[net_name]]*2*n,
                           'effective diffusion type' : ['node-based']*n + ['path-based']*n,
                           'effective diffusion': node_ed + path_ed
                            })
            df = pd.concat([df,temp_df],axis=0)

        sns.boxplot(x = df['network'],
                    y = df['effective diffusion'],
                    hue = df['effective diffusion type'] )
        plt.ylim([0,1])
        plt.tight_layout()
        plt.savefig(ed_across_nets_plot_file.replace('.png','-'+term+'.png'))
        plt.savefig(ed_across_nets_plot_file.replace('.png','-'+term+'.pdf'))
        plt.close()
def boxplot_stst_loop_across_networks(stst_looped_paths_dict, network_names, stst_across_nets_plot_file):
    for term in stst_looped_paths_dict:
        # check if for a certain term effective diffusion from both networks are present, otherwise exclude that term.
        if len(stst_looped_paths_dict[term].keys())!=2:
            continue

        stst_loop_network1 = list(stst_looped_paths_dict[term][network_names[0]])
        stst_loop_network2 = list(stst_looped_paths_dict[term][network_names[1]])

        df_len = len(stst_loop_network1) + len(stst_loop_network2)

        df = pd.DataFrame({'network': [network_names[0]]*len(stst_loop_network1) +
                                      [network_names[1]]*len(stst_loop_network2),
                           'looped_path_type' : ['stst']*df_len,
                           'contribution': stst_loop_network1 + stst_loop_network2
        })

        sns.boxplot(x = df['looped_path_type'],
                    y = df['contribution'],
                    hue = df['network'])
        plt.ylim([0,1])
        plt.tight_layout()
        plt.savefig(stst_across_nets_plot_file.replace('.png','-'+term+'.png'))
        plt.savefig(stst_across_nets_plot_file.replace('.png','-'+term+'.pdf'))
        plt.close()



def individual_net_ed_boxplot(node_based_ed_dict, path_based_ed_dict, ed_indv_nets_plot_file):
    for term in node_based_ed_dict:
        for network_name in node_based_ed_dict[term]:
            bionet_node_based_ed = list(node_based_ed_dict[term][network_name])
            bionet_path_based_ed = list(path_based_ed_dict[term][network_name])

            plt.boxplot(x = np.column_stack((bionet_node_based_ed, bionet_path_based_ed)))
            plt.ylim([0,1])
            plt.tight_layout()

            network_name_str = network_name.replace('/','-')
            ed_indv_nets_plot_file_temp = ed_indv_nets_plot_file.replace('.png', '-'+network_name_str+'-' + term +'.png')

            plt.savefig(ed_indv_nets_plot_file_temp)
            # also save fig as pdf file
            plt.savefig(ed_indv_nets_plot_file_temp.replace('.png','.pdf'))
            plt.close()


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    for k in kwargs.get('k_to_test'):
        main(config_map, k=k, **kwargs)
