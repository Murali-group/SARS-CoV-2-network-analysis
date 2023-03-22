import os, sys
import yaml
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.insert(1, "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/")
from src.FastSinkSource.src.main import setup_dataset
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils

from src.scripts.plot_utils import *

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
    # group.add_argument('--config', type=str, default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
    #                     "fss_inputs/config_files/provenance/string700_s12.yaml"
    #                    , help="Configuration file used when running FSS. ")
    group.add_argument('--config', type=str, default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                        "fss_inputs/config_files/provenance/string700_s12.yaml"
                       , help="Configuration file used when running FSS. ")

    group.add_argument('--run-algs', type=str, action='append', default=[])
    group.add_argument('--k-to-test', '-k', type=int, action='append', default=[332],
                       help="k-value(s) for which to get the top-k predictions to test. " +
                            "If not specified, will check the config file.")
    group.add_argument('--pos-k', action='store_true', default=False,
                       help="if true get the top-k predictions to test is equal to the number of positive annotations")
    group.add_argument('--n-sp', '-n', type=int, default=1000,
                       help="n-sp is the number of shortest paths to be considered" +
                            "If not specified, will check the config file. Default=20")
    # group.add_argument('--m', type=int, default=20,
    #                    help="for each top prediction, for how many top contributing sources we wanna analyse the path" +
    #                         "Default=20")
    group.add_argument('--stat-sig-cutoff', type=float,
                       help="Cutoff on the node p-value for a node to be considered in the topk. " + \
                            "The p-values should already have been computed with run_eval_algs.py")
    group.add_argument('--balancing-alpha-only', action='store_true', default=True,
                       help="Ignore alpha from config file rather take the alpha value\
                            that balanced the two loss terms in quad loss function for the corresponding\
                            network-term-alg")


    return parser


def boxplot_dfsn(diffusion_all_alpha_dict, xlabel, ylabel,title, filename):
    '''
    diffusion_all_alpha_dict: dict. A dictionary with (key,value) where key= an alpha, value= a list, that contains
    some measure (e.g. contribution via pathlength 1 for a target) for each target.
    filename = string. Filename to save the boxplot.
    '''

    #sort dict according to ascending order of alpha values/keys
    alpha_dict = {key:diffusion_all_alpha_dict[key] \
                                for key in sorted(diffusion_all_alpha_dict.keys())}

    plt.boxplot(alpha_dict.values(),\
                labels= list(alpha_dict.keys()))
    plt.ylim([0,1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.savefig(filename.replace('.pdf','.png')) #save plot in .png format as well
    # plt.show()
    plt.close()
    print('Save fig to ', filename)


def boxplot_diffusion_along_path_types(diffusion_path_types,title, top_or_bottom, term, param_str, plot_dir,
    cols = ['target','frac_pathlen_1','frac_pathlen_2','frac_pathlen_3','frac_pathlen_4','frac_beyond_pathlen4']):
    df = diffusion_path_types[cols]
    sns.boxplot(data = df.set_index('target'))

    plt.ylim([0, 1])
    plt.xlabel('Different types of paths')
    plt.ylabel('Fraction of diffusion')
    plt.xticks(rotation=45)
    plt.title(title+'_'+top_or_bottom)
    plt.tight_layout()

    filename = plot_dir+term+'_'+top_or_bottom+'_path_type_based_diffusion'+param_str+'.pdf'
    plt.savefig(filename)
    plt.savefig(filename.replace('.pdf', '.png'))  # save plot in .png format as well
    plt.show()
    plt.close()
    print('Save fig to ', filename)

def get_plot_dir(config_map,dataset, alg_name):
    plot_dir = config_map['output_settings']['output_dir'] + \
               "/viz/%s/%s/diffusion-path-analysis/plot/%s/"\
               % (dataset['net_version'], dataset['exp_name'], alg_name)
    os.makedirs(os.path.dirname(plot_dir), exist_ok=True)
    return plot_dir



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

        for alg_name in alg_settings:
            plot_dir = get_plot_dir(config_map, dataset, alg_name)

            if (alg_settings[alg_name]['should_run'][0] == True) or (alg_name in kwargs.get('run_algs')):
                # Figure out the balancing alpha for term-alg-network combo
                alpha_summary_filename = config_map['output_settings']['output_dir'] + \
                    "/viz/%s/%s/param_select/" % (dataset['net_version'], dataset[
                    'exp_name']) + '/' + alg_name + '/alpha_summary.tsv'

                #in the following file for go terms, 'term_name' is the short descriptive go term and
                #'term' is just the GO term ID.
                alpha_summary_df = pd.read_csv(alpha_summary_filename, sep='\t', index_col=None)[
                    ['term','term_name' ,'balancing_alpha']]
                term_2_balancing_alpha_dict = dict(
                    zip(alpha_summary_df['term'], alpha_summary_df['balancing_alpha']))

                term_2_term_name_dict = dict(
                    zip(alpha_summary_df['term'], alpha_summary_df['term_name']))

                # dictionaries to pass to plotting functions
                path_based_effective_diffusion_all_term_dict = {}
                node_based_effective_diffusion_all_term_dict = {}

                for term in ann_obj.terms:
                    term_idx = ann_obj.term2idx[term]
                    orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, term_idx)
                    orig_pos = [prots[p] for p in orig_pos_idx]
                    pos_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]
                    n_pos = len(pos_nodes_idx)
                    if kwargs.get('pos_k'):
                        k= n_pos
                    # get the alpha values to use
                    balancing_alpha = term_2_balancing_alpha_dict[term]
                    #*********************************
                   

                    #Read effective diffusion file for the balancing alpha


                    ed_file = config_map['output_settings']['output_dir'] +\
                              "/viz/%s/%s/diffusion-analysis/%s/effective-diff-ss-k%s-nsp%s-a%s%s.tsv" \
                            % (dataset['net_version'], term, alg_name, k, kwargs.get('n_sp'), balancing_alpha, sig_str)

                   
                    #The following contains diffusion across paths of different types: st, sut, stst,
                    # stut, sust, suvt, paths_of_len_beyond_3
                    paths_of_different_types_stat_file = config_map['output_settings'][
                        'output_dir'] + "/viz/%s/%s/diffusion-path-analysis/%s/type_based_path_stat-a%s%s.tsv"\
                        % (dataset['net_version'], term, alg_name, balancing_alpha, sig_str)

                    if not os.path.isfile(ed_file):
                        continue
                    effective_diffusion_df = \
                        pd.read_csv(ed_file, sep='\t', index_col=None)
                    top_preds_diffusion_across_different_types_of_paths_df =\
                        pd.read_csv(paths_of_different_types_stat_file.replace('stat','top'), sep='\t', index_col=None)

                    bot_preds_diffusion_across_different_types_of_paths_df = \
                        pd.read_csv(paths_of_different_types_stat_file.replace('stat', 'bot'), sep='\t', index_col=None)

                    ##plot diffusion coming along different types of paths
                    title = dataset['plot_exp_name'] + '_' + dataset['exp_name'] + '_' + \
                            get_plot_alg_name(alg_name)
                    param_str = 'a-'+str(balancing_alpha) +sig_str

                    boxplot_diffusion_along_path_types(top_preds_diffusion_across_different_types_of_paths_df,\
                                                       title, 'top_preds', term,param_str, plot_dir)
                    boxplot_diffusion_along_path_types(bot_preds_diffusion_across_different_types_of_paths_df, \
                                                       title,'selected_bottom_preds', term,param_str, plot_dir)
                    #**************************
                    #use alias for term name in plot
                    term_name = term_2_term_name_dict[term]
                    #Extract path-based effective diffusion
                    path_based_effective_diffusion_all_term_dict[get_plot_term_name(term_name)] = list(effective_diffusion_df\
                                                                      ['path_based_ed'])
                    # #Extract node-based effective diffusion
                    # node_based_effective_diffusion_all_term_dict[get_plot_term_name(term_name)] = list(effective_diffusion_df\
                    #                                                   ['node_based_ed'])


                #********************plot*******************
                os.makedirs(plot_dir, exist_ok=True)
                title = dataset['plot_exp_name'] + '_' + dataset['exp_name'] + '_' + get_plot_alg_name(alg_name)

                #boxplot effective diffusion across alphas
                boxplot_dfsn_across_terms(path_based_effective_diffusion_all_term_dict, 'Terms', 'Path Based Effective Diffusion', \
                                          title, plot_dir +'path_based_effective_diffusion' +sig_str+'.pdf')
                # boxplot_dfsn_across_terms(node_based_effective_diffusion_all_term_dict, 'Terms',
                #                           'Node Based Effective Diffusion', \
                #                           title, plot_dir + 'node_based_effective_diffusion' + sig_str + '.pdf')


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    for k in kwargs.get('k_to_test'):
        main(config_map, k=k, **kwargs)
