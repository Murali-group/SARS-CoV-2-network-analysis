import sys
import numpy as np
import yaml
import argparse
import pandas as pd
import os

sys.path.insert(1, "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/")
from src.FastSinkSource.src.main import setup_dataset
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils
import src.scripts.betweenness.betweenness_utils as btns_utils
import src.scripts.betweenness.plot_utils  as btns_plot_utils
import src.scripts.utils  as script_utils
alg_plot_name = {'rwr': 'RWR', 'genemaniaplus': 'RL'}

def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        config_map = yaml.load(conf, Loader=yaml.FullLoader)
        # config_map = yaml.load(conf)
    with open(args.master_config, 'r') as conf:
        master_config_map = yaml.load(conf, Loader=yaml.FullLoader)
    # TODO check to make sure the inputs are correct in config_map
    return config_map, master_config_map, kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="Compute betweenness score for each node in the network")
    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                        "fss_inputs/config_files/provenance/string700_s12.yaml"
                       , help="Configuration file used when running FSS. ")
    group.add_argument('--master-config', type=str, default="/data/tasnina/Provenance-Tracing/"
                        "SARS-CoV-2-network-analysis/config-files/master-config.yaml"
                       , help="Configuration file used to do mappings")

    # group.add_argument('--config', type=str, default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
    #                                                  "fss_inputs/config_files/provenance/yeast/provenance_string900v11.5_s1.yaml"
    #                    , help="Configuration file used when running FSS. ")
    # group.add_argument('--master-config', type=str, default="/data/tasnina/Provenance-Tracing/"
    #                     "SARS-CoV-2-network-analysis/config-files/master-config-yeast.yaml"
    #                    , help="Configuration file used to do mappings")
    group.add_argument('--essential-prot-file', type=str,
                       default ="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                        "fss_inputs/essential-prots/deg_annotation_e.csv",
                       help="This file should contain the essential genes for corresponding species")

    group.add_argument('--viral-prot-file', type=str,
                       default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                               "fss_inputs/viral-prots/HVIDB.csv",
                       help="This file should contain the essential genes for corresponding species")
    # group.add_argument('--pleiotropic-prot-file', type=str,
    #                    default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
    #                     "fss_inputs/pleiotropic-prots/pleiotropic_gene_human_ra.xls",
    #                    help="This file should contain the pleiotropic genes for corresponding species")
    # group.add_argument('--pleiotropic-corr', action='store_true', default=False,
    #                    help='If true we will find correlation between pleoitrophy and betweenness.')
    group.add_argument('--ks', type=str, action='append', default=[200,400,600,800,1000, 2000, 3000,4000, 5000, 10000])
    group.add_argument('--run-algs', type=str, action='append', default=[])

    group.add_argument('--force-download', action='store_true', default=False,
                       help="Force re-downloading and parsing of the input files")
    group.add_argument('--balancing-alpha-only', action='store_true', default=True,
                       help="Ignore alpha from config file rather take the alpha value\
                               that balanced the two loss terms in quad loss function for the corresponding\
                               network-term-alg")
    return parser


def main(config_map, master_config_map, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """
    ks = kwargs.get('ks')
    # extract the general variables from the config map
    input_settings, input_dir, output_dir, alg_settings, kwargs \
        = config_utils.setup_config_variables(config_map, **kwargs)

    for dataset in input_settings['datasets']:
        print("Loading data for %s" % (dataset['net_version']))
        dataset_name = dataset['plot_exp_name']
        # load the network and the positive examples for each term
        net_obj, ann_obj, _ = setup_dataset(dataset, input_dir, **kwargs)
        prots, node2idx = net_obj.nodes, net_obj.node2idx

        # get essential protein list.
        # map essential prots to uniprot ids
        ess_types = ['org', 'cell']
        viral_types = ['sars2--']
        ess_uniprots_dict = btns_utils.handle_essential_uniprot_mapping(master_config_map, **kwargs)
        #get human proteins that interact with viral prots
        viral_prot_file = kwargs.get('viral_prot_file')
        viral_uniprots_dict =  btns_utils.parse_viral_prot_file(viral_prot_file)

        ##Directory for saving any betweenness related analysis result
        pred_overlap_out_dir = output_dir + '/biological_significance_of_preds/' + dataset['net_version'] + '/'

        for alg_name in alg_settings:
            if (alg_settings[alg_name]['should_run'][0] == True) or (alg_name in kwargs.get('run_algs')):
                print(alg_name)

                for term in ann_obj.terms:
                    alg_term_spec_btns_out_dir = pred_overlap_out_dir + alg_name + '/' + dataset['exp_name'] + '/'

                    term_idx = ann_obj.term2idx[term]
                    orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, term_idx)
                    orig_pos = [prots[p] for p in orig_pos_idx]
                    pos_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]
                    assert len(orig_pos) == len(pos_nodes_idx);
                    print('not all source present in net')

                    # remove any positive protein present in viral interactors
                    for viral_type in viral_uniprots_dict:
                        viral_uniprots_dict[viral_type] = viral_uniprots_dict[viral_type].difference(set(orig_pos))

                    if kwargs.get('balancing_alpha_only'):  # in alg_setting[alg_name]['alpha'] put the balancing alpha

                        balancing_alpha = script_utils.get_balancing_alpha(config_map, dataset,alg_name,term)
                        alg_settings[alg_name]['alpha'] = [balancing_alpha]
                    #Get prediction files
                    alg_pred_files = config_utils.get_dataset_alg_prediction_files(
                        output_dir, dataset, alg_settings, [alg_name], **kwargs)
                    # get the alpha values to use
                    alphas = alg_settings[alg_name]['alpha']
                    count=0
                    for alpha, alg in zip(alphas, alg_pred_files):
                        count+=1

                        #Now get the top predicted proteins
                        #here k= len(orig_pos)
                        pred_file = alg_pred_files[alg]
                        pred_file = script_utils.term_based_pred_file(pred_file, term)
                        sorted_src_spec_preds_df = pd.read_csv(pred_file, sep='\t')
                        top_k_predictions_df = script_utils.get_top_k_predictions(pred_file, alpha, len(orig_pos) , orig_pos)

                        ##filter the sorted_src_spec_btns_df to contain only the non-source, non-top predictions
                        sorted_filtered_src_spec_preds_df, sorted_df_pos, sorted_df_top_k = \
                            btns_utils.filter_sorted_src_spec_btns(sorted_src_spec_preds_df, top_k_predictions_df,
                                                                   orig_pos)

                        ###Do analysis to find relationship between betweenness score and  ESSENTAIL PROT



                        #*********************************  ESSENTIAL PROTS *****************************
                        print('\n\n ESSENTAIL PROTS ANALYSIS\n')
                        for ess_type in ess_types :
                            print(ess_type)
                            alg_term_ess_btns_corr_file = alg_term_spec_btns_out_dir + 'corr_ess_'+ess_type+'.tsv'
                            os.makedirs(os.path.dirname(alg_term_ess_btns_corr_file), exist_ok=True)

                            #Hypergeometric test
                            #compute overlap between top_ranked_prots and essential_prots. The ranking was done using
                            # paths of len 2,3,4 separately and all together.
                            all_criteria_overlap_pvals_topks = btns_utils.handle_Fishers_exact_test_in_topks\
                                            (sorted_filtered_src_spec_preds_df, ess_uniprots_dict[ess_type], ks,
                                             ranking_criteria = ['score'])

                            #now compute  1. frac of src_nodes are essential 2. frac of predicted nodes by algorithms are essential
                            # 3. frac of nodes in netwokrs excluding src and predicted_nodes are essential.
                            ess_in_pos = btns_utils.compute_frac_interesting_prot(sorted_df_pos['prot'], ess_uniprots_dict[ess_type])
                            ess_in_top = btns_utils.compute_frac_interesting_prot(sorted_df_top_k['prot'], ess_uniprots_dict[ess_type])
                            ess_in_net = btns_utils.compute_frac_interesting_prot(sorted_filtered_src_spec_preds_df['prot'], ess_uniprots_dict[ess_type])


                            title = alg_plot_name[alg_name] + '_a_' + str(alpha) + '_' + term + '_' + dataset_name
                            overlap_pval_plt_file = alg_term_spec_btns_out_dir + 'overlap_ess_'+ess_type+'_a' + str(alpha) + '.pdf'
                            #Plot for hypergeometric test/Fisher's exact test
                            btns_plot_utils.plot_hypergeom_pval(all_criteria_overlap_pvals_topks, ess_in_pos, ess_in_top,
                                                                ess_in_net, title,overlap_pval_plt_file)
                            #Compute correlation between rank_percentiles of bins and percentage of essential prots in bins


                        #*********************************  VIRAL PROTS *****************************
                        print('\n\nVIRAL INTERACTOR ANALYSIS\n')
                        for viral_type in viral_types:
                            print(viral_type)

                            # Hypergeometric test
                            all_criteria_overlap_pvals_topks = btns_utils.handle_Fishers_exact_test_in_topks \
                                (sorted_filtered_src_spec_preds_df, viral_uniprots_dict[viral_type], kwargs.get('ks'),
                                 ranking_criteria = ['score'])

                            viral_in_pos = btns_utils.compute_frac_interesting_prot(sorted_df_pos['prot'],
                                                                                    viral_uniprots_dict[viral_type])
                            viral_in_top = btns_utils.compute_frac_interesting_prot(sorted_df_top_k['prot'],
                                                                                    viral_uniprots_dict[viral_type])
                            viral_in_net = btns_utils.compute_frac_interesting_prot(
                                sorted_filtered_src_spec_preds_df['prot'], viral_uniprots_dict[viral_type])

                            title = alg_plot_name[alg_name] + '_a_' + str(alpha) + '_' + term + '_' + dataset_name
                            overlap_pval_plt_file = alg_term_spec_btns_out_dir + 'overlap_viral_' + viral_type + '_a' + str(alpha) + '.pdf'

                            # Plot for hypergeometric test/Fisher's exact test
                            btns_plot_utils.plot_hypergeom_pval(all_criteria_overlap_pvals_topks, viral_in_pos,
                                                                viral_in_top, viral_in_net, title, overlap_pval_plt_file)





if __name__ == "__main__":
    config_map, master_config_map, kwargs = parse_args()
    main(config_map,master_config_map, **kwargs)
