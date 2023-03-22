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
                                                     "fss_inputs/config_files/provenance/string700_go.yaml"
                       , help="Configuration file used when running FSS. ")

    group.add_argument('--run-algs', type=str, action='append', default=[])
    group.add_argument('--k-to-test', '-k', type=int, action='append', default=[332],
                       help="k-value(s) for which to get the top-k predictions to test. " +
                            "If not specified, will check the config file.")

    group.add_argument('--n-sp', '-n', type=int, default=1000,
                       help="n-sp is the number of shortest paths to be considered" +
                            "If not specified, will check the config file. Default=20")

    group.add_argument('--stat-sig-cutoff', type=float,
                       help="Cutoff on the node p-value for a node to be considered in the topk. " + \
                            "The p-values should already have been computed with run_eval_algs.py")
    # group.add_argument('--balancing-alpha-only', action='store_true', default=False,
    #                    help="Ignore alpha from config file rather take the alpha value\
    #                         that balanced the two loss terms in quad loss function for the corresponding\
    #                         network-term-alg")
    #

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
                if (alg_settings[alg_name]['should_run'][0] == True) or (alg_name in kwargs.get('run_algs')):
                    # get the alpha values to use

                    alphas = alg_settings[alg_name]['alpha']

                    #dictionaries to pass to plotting functions
                    diffusion_beyond_pathlen_1_all_alpha_dict={alpha:[] for alpha in alphas}
                    diffusion_beyond_pathlen_1_all_beta_dict={}

                    diffusion_via_pathlen_1_all_alpha_dict = {alpha: [] for alpha in alphas}
                    diffusion_via_pathlen_1_all_beta_dict = {}

                    diffusion_via_pathlen_2_all_alpha_dict = {alpha: [] for alpha in alphas}
                    diffusion_via_pathlen_2_all_beta_dict = {}

                    # diffusion_via_pathlen_3_all_alpha_dict = {alpha: [] for alpha in alphas}
                    # diffusion_via_pathlen_3_all_beta_dict = {}


                    for alpha in alphas:
                        net_alg_settings = config_map['output_settings']['output_dir'] + \
                                        "/viz/%s/%s/diffusion-path-analysis/%s/" % (dataset['net_version'], term, alg_name)
                        run_settings = "k%s-nsp%s-a%s%s" \
                                        % ( k, kwargs.get('n_sp'), alpha, sig_str)
                        path_length_based_effective_diffusion_file =\
                                        "%s/path-length-wise-effective-diff-ss-%s.tsv"%(net_alg_settings,run_settings)

                        path_length_based_effective_diffusion_df = \
                            pd.read_csv(path_length_based_effective_diffusion_file, sep='\t', index_col=None)

                        diffusion_beyond_pathlen_1_all_alpha_dict[alpha] = list(path_length_based_effective_diffusion_df\
                                                                          ['effective_diffusion'])
                        diffusion_via_pathlen_1_all_alpha_dict[alpha] = list(path_length_based_effective_diffusion_df \
                                                                                    ['frac_total_score_via_len_1'])
                        diffusion_via_pathlen_2_all_alpha_dict[alpha] = list(path_length_based_effective_diffusion_df \
                                                                                    ['frac_total_score_via_len_2'])
                        # diffusion_via_pathlen_3_all_alpha_dict[alpha] = list(path_length_based_effective_diffusion_df \
                        #                                                          ['frac_total_score_via_len_3'])
                        if alg_name == 'genemaniaplus':
                            # Nure: introduced new param beta = 1/(1+alpha). this beta variabel is comparable with alpha in rwr.
                            diffusion_beyond_pathlen_1_all_beta_dict[round(float(1 / (1 + alpha)), 2)] = \
                                diffusion_beyond_pathlen_1_all_alpha_dict[alpha]
                            diffusion_via_pathlen_1_all_beta_dict[round(float(1 / (1 + alpha)), 2)] = \
                                diffusion_via_pathlen_1_all_alpha_dict[alpha]
                            diffusion_via_pathlen_2_all_beta_dict[round(float(1 / (1 + alpha)), 2)] = \
                                diffusion_via_pathlen_2_all_alpha_dict[alpha]
                            # diffusion_via_pathlen_3_all_beta_dict[round(float(1 / (1 + alpha)), 2)] = \
                            #     diffusion_via_pathlen_3_all_alpha_dict[alpha]

                    #********************plot*******************
                    plot_dir = net_alg_settings + '/plot/ss/'
                    os.makedirs(plot_dir, exist_ok=True)
                    title = dataset['plot_exp_name'] + '_' + term + '_' + get_plot_alg_name(alg_name)
                    common_plot_file_substring =  "k%s-nsp%s%s" \
                                        % ( k, kwargs.get('n_sp'),sig_str)

                    #plot diffusion coming beyond pathlength 1 across alphas
                    boxplot_dfsn(diffusion_beyond_pathlen_1_all_alpha_dict, 'Alpha', 'Path Based Effective Diffusion',\
                                 title, plot_dir+'path_effective_diffusion_alphas_'+common_plot_file_substring+'.pdf')

                    #plot diffusion coming via only pathlength 1,( pathlength 2, pathlength 3) across alphas
                    boxplot_dfsn(diffusion_via_pathlen_1_all_alpha_dict, 'Alpha', 'diffusion via path length 1', \
                                 title, plot_dir + 'contribution_via_path_length_1_alphas_' + common_plot_file_substring + '.pdf')

                    boxplot_dfsn(diffusion_via_pathlen_2_all_alpha_dict, 'Alpha', 'diffusion via path length 2', \
                                 title, plot_dir + 'contribution_via_path_length 2_alphas_' + common_plot_file_substring + '.pdf')

                    # boxplot_dfsn(diffusion_via_pathlen_3_all_alpha_dict, 'Alpha', 'diffusion via path length 3', \
                    #              title,plot_dir + 'contribution_via_path_length 3_alphas_' + common_plot_file_substring + '.pdf')


                    if alg_name=='genemaniaplus':
                        boxplot_dfsn(diffusion_beyond_pathlen_1_all_beta_dict, 'Beta', 'Path Based Effective Diffusion', \
                                     title,plot_dir + 'path_effective_diffusion_betas_' + common_plot_file_substring + '.pdf')
                        boxplot_dfsn(diffusion_via_pathlen_1_all_beta_dict, 'Beta', 'diffusion via path length 1', \
                                     title,plot_dir + 'contribution_via_path_length 1_betas_' + common_plot_file_substring + '.pdf')
                        boxplot_dfsn(diffusion_via_pathlen_2_all_beta_dict, 'Beta', 'diffusion via path length 2', \
                                     title,plot_dir + 'contribution_via_path_length 2_betas_' + common_plot_file_substring + '.pdf')
                        # boxplot_dfsn(diffusion_via_pathlen_3_all_beta_dict, 'Beta', 'diffusion via path length 3', \
                        #              title,plot_dir + 'contribution_via_path_length 3_betas_' + common_plot_file_substring + '.pdf')

if __name__ == "__main__":
    config_map, kwargs = parse_args()
    for k in kwargs.get('k_to_test'):
        main(config_map, k=k, **kwargs)
