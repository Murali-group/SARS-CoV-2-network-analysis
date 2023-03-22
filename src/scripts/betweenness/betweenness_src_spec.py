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
    group.add_argument('--ks', type=str, action='append', default=[200,400,600,800,1000, 2000, 3000,4000, 5000,
                                                                   6000, 7000, 8000, 9000, 10000])
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
    # extract the general variables from the config map

    # sep_pos_top = kwargs.get('sep_pos_top')
    ks = kwargs.get('ks')
    input_settings, input_dir, output_dir, alg_settings, kwargs \
        = config_utils.setup_config_variables(config_map, **kwargs)

    #keep track of how many prots with non-zero betweenness appear as we consider contribution via
    #paths of len 2,3,4
    n_prots_appearing_at_each_pathlens={'network':[],'term':[],'alg':[],'alpha':[], 'pathlen_2':[],  'pathlen_3':[],  'pathlen_4':[]}
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
        btns_out_dir = output_dir + '/betweenness/' + dataset['net_version'] + '/'

        for alg_name in alg_settings:
            if (alg_settings[alg_name]['should_run'][0] == True) or (alg_name in kwargs.get('run_algs')):
                print(alg_name)
                if kwargs.get('balancing_alpha_only'):  # in alg_setting[alg_name]['alpha'] put the balancing alpha
                    # get the balancing alpha for this network - alg - term
                    alpha_summary_filename = config_map['output_settings']['output_dir'] + \
                                             "/viz/%s/%s/param_select/" % (dataset['net_version'], dataset[
                        'exp_name']) + '/' + alg_name + '/alpha_summary.tsv'
                    alpha_summary_df = pd.read_csv(alpha_summary_filename, sep='\t', index_col=None)[
                        ['term', 'balancing_alpha']]
                    term_2_balancing_alpha_dict = dict(
                        zip(alpha_summary_df['term'], alpha_summary_df['balancing_alpha']))

                    balancing_alpha = term_2_balancing_alpha_dict[term]
                    alg_settings[alg_name]['alpha'] = [balancing_alpha]

                #Get prediction file
                alg_pred_files = config_utils.get_dataset_alg_prediction_files(
                    output_dir, dataset, alg_settings, [alg_name], **kwargs)
                # get the alpha values to use
                alphas = alg_settings[alg_name]['alpha']

                count=0
                for alpha, alg in zip(alphas, alg_pred_files):
                    count+=1
                    beta = alpha
                    if alg_name == 'genemaniaplus':
                        beta = round(1.0 / (1 + alpha), 2)
                    M_pathmtx_loginv = script_utils.get_M_pathmtx_loginv(net_obj, alg_name, alpha)

                    for term in ann_obj.terms:
                        alg_term_spec_btns_out_dir = btns_out_dir + alg_name +'/'+dataset['exp_name']+'/'

                        term_idx = ann_obj.term2idx[term]
                        orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, term_idx)
                        orig_pos = [prots[p] for p in orig_pos_idx]
                        pos_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]
                        assert len(orig_pos)==len(pos_nodes_idx); print('not all source present in net')
                        #remove any positive protein present in viral interactors
                        for viral_type in viral_uniprots_dict:
                            viral_uniprots_dict[viral_type] = viral_uniprots_dict[viral_type].difference(set(orig_pos))


                        #file to save the betweenness score and percent_rank(according to betweenness score ) of each gene
                        src_spec_btns_file = alg_term_spec_btns_out_dir +'btns_a' + str(alpha) + '.tsv'
                        #Compute BETWEENNESS score for each protein  and get a dataframe with proteins sorted in ascending order of btns score
                        sorted_src_spec_btns_df = btns_utils.handle_src_spec_btns\
                            (M_pathmtx_loginv,prots,src_spec_btns_file, pos_nodes_idx, force_run=False)

                        #compute and plot how many new prots appear with nonzero betweenness
                        # as we consider path lens of 2, 3, and 4
                        new_prots_appearing_at_each_pathlens = \
                            btns_utils.find_new_prots_appearing_at_each_pathlens(sorted_src_spec_btns_df)
                        n_prots_appearing_at_each_pathlens['network'].append(dataset_name)
                        n_prots_appearing_at_each_pathlens['term'].append(term)
                        n_prots_appearing_at_each_pathlens['alg'].append(alg_name)
                        n_prots_appearing_at_each_pathlens['alpha'].append(alpha)
                        n_prots_appearing_at_each_pathlens['pathlen_2'].append(len(new_prots_appearing_at_each_pathlens['pathlen_2']))
                        n_prots_appearing_at_each_pathlens['pathlen_3'].append(len(new_prots_appearing_at_each_pathlens['pathlen_3']))
                        n_prots_appearing_at_each_pathlens['pathlen_4'].append(len(new_prots_appearing_at_each_pathlens['pathlen_4']))

                        # btns_plot_utils.plot_prots_appearing_at_each_pathlens(n_prots_appearing_at_each_pathlens,
                        #                                                       filename=output_dir + '/betweenness/' + 'new_appering_prots.pdf')


                        #Now get the top predicted proteins
                        #here k= len(orig_pos)
                        pred_file = alg_pred_files[alg]
                        pred_file = script_utils.term_based_pred_file(pred_file, term)
                        top_k_predictions_df = script_utils.get_top_k_predictions(pred_file, alpha, len(orig_pos) , orig_pos)

                        # filter the sorted_src_spec_btns_df to contain only the non-source, non-top predictions
                        sorted_filtered_src_spec_btns_df, sorted_df_pos, sorted_df_top_k = \
                            btns_utils.filter_sorted_src_spec_btns(sorted_src_spec_btns_df, top_k_predictions_df, orig_pos)


                        ###Do analysis to find relationship between betweenness score and  ESSENTAIL PROT
                        # Kolmogorov-Smirnov test to see if essential proteins show significantly high
                        # btns score

                        ks_file = alg_term_spec_btns_out_dir + 'KS_pvals' + '_a' + str(alpha) + '.tsv'
                        KS_dict = btns_utils.handle_Kolmogorov_Smirnov_test(sorted_filtered_src_spec_btns_df, ess_uniprots_dict, viral_uniprots_dict)
                        script_utils.save_dict(KS_dict, ks_file)

                        marker= 'rank'
                        frac_prots_ge_btns_marker = btns_utils.prepare_plotdata_for_Kolmogorov_Smirnov\
                                                    (sorted_filtered_src_spec_btns_df, ess_uniprots_dict, viral_uniprots_dict,
                                                     marker =marker, ks=ks)
                        title = alg_plot_name[alg_name] + '_a_' + str(alpha) + '_' + term + '_' + dataset_name
                        ks_plt_file = alg_term_spec_btns_out_dir + 'KS_' +marker+ '_a' + str(alpha) + '.pdf'
                        # Plot for KS
                        btns_plot_utils.plot_KS(frac_prots_ge_btns_marker, marker, title, ks_plt_file)

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
                                            (sorted_filtered_src_spec_btns_df, ess_uniprots_dict[ess_type], ks)

                            #now compute  1. frac of src_nodes are essential 2. frac of predicted nodes by algorithms are essential
                            # 3. frac of nodes in netwokrs excluding src and predicted_nodes are essential.
                            ess_in_pos = btns_utils.compute_frac_interesting_prot(sorted_df_pos['prot'], ess_uniprots_dict[ess_type])
                            ess_in_top = btns_utils.compute_frac_interesting_prot(sorted_df_top_k['prot'], ess_uniprots_dict[ess_type])
                            ess_in_net = btns_utils.compute_frac_interesting_prot(sorted_filtered_src_spec_btns_df['prot'], ess_uniprots_dict[ess_type])


                            title = alg_plot_name[alg_name] + '_a_' + str(alpha) + '_' + term + '_' + dataset_name
                            overlap_pval_plt_file = alg_term_spec_btns_out_dir + 'overlap_ess_'+ess_type+'_a' + str(alpha) + '.pdf'
                            #Plot for hypergeometric test/Fisher's exact test
                            btns_plot_utils.plot_hypergeom_pval(all_criteria_overlap_pvals_topks, ess_in_pos, ess_in_top, ess_in_net, title,overlap_pval_plt_file)
                            #Compute correlation between rank_percentiles of bins and percentage of essential prots in bins
                            pc_ess, pval_ess, mw_ess, prcntl_ess, prcnt_ess =\
                                btns_utils.handle_percentile_percent_corr(sorted_filtered_src_spec_btns_df, ess_uniprots_dict[ess_type])
                            # SAVE correlations.
                            btns_utils.save_btns_corr(alg_term_ess_btns_corr_file, count, beta, pc_ess, pval_ess,mw_ess)


                            # Scatter plot for rank percentile and percentage of essential protein in each bin
                            ext_prcntl_ess, ext_prcnt_ess = btns_utils.find_interesting_prot_in_src_top(
                                sorted_df_pos, sorted_df_top_k, ess_uniprots_dict[ess_type], prcntl_ess, prcnt_ess)
                            prcntl_prcnt_btns_ess_plt_file = alg_term_spec_btns_out_dir + 'scatter_ess_'+ess_type+'_a' + str(alpha) + '.pdf'
                            title = alg_plot_name[alg_name] + '_a_' + str(alpha) + '_' + term + '_' + dataset_name
                            x_label = 'percentile rank'
                            btns_plot_utils.scatter_plot(ext_prcntl_ess, ext_prcnt_ess, x_label=x_label,
                                                         y_label='percentage of essential prot : '+ess_type,
                                                         title=title, filename=prcntl_prcnt_btns_ess_plt_file)

                        #*********************************  VIRAL PROTS *****************************
                        print('\n\nVIRAL INTERACTOR ANALYSIS\n')
                        for viral_type in viral_types:
                            print(viral_type)

                            # Hypergeometric test
                            all_criteria_overlap_pvals_topks = btns_utils.handle_Fishers_exact_test_in_topks \
                                (sorted_filtered_src_spec_btns_df, viral_uniprots_dict[viral_type], ks)

                            viral_in_pos = btns_utils.compute_frac_interesting_prot(sorted_df_pos['prot'],
                                                                                    viral_uniprots_dict[viral_type])
                            viral_in_top = btns_utils.compute_frac_interesting_prot(sorted_df_top_k['prot'],
                                                                                    viral_uniprots_dict[viral_type])
                            viral_in_net = btns_utils.compute_frac_interesting_prot(
                                sorted_filtered_src_spec_btns_df['prot'], viral_uniprots_dict[viral_type])

                            title = alg_plot_name[alg_name] + '_a_' + str(alpha) + '_' + term + '_' + dataset_name
                            overlap_pval_plt_file = alg_term_spec_btns_out_dir + 'overlap_viral_' + viral_type + '_a' + str(alpha) + '.pdf'

                            # Plot for hypergeometric test/Fisher's exact test
                            btns_plot_utils.plot_hypergeom_pval(all_criteria_overlap_pvals_topks, viral_in_pos,
                                                                viral_in_top, viral_in_net, title, overlap_pval_plt_file)

                            #Binwise Pearsons correlations
                            pc_viral, pval_viral, mw_viral, prcntl_viral, prcnt_viral = \
                                btns_utils.handle_percentile_percent_corr(sorted_filtered_src_spec_btns_df, viral_uniprots_dict[viral_type])
                            alg_term_viral_btns_corr_file = alg_term_spec_btns_out_dir + 'corr_viral_' + viral_type + '.tsv'
                            os.makedirs(os.path.dirname(alg_term_viral_btns_corr_file), exist_ok=True)
                            btns_utils.save_btns_corr(alg_term_viral_btns_corr_file, count, beta, pc_viral, pval_viral, mw_viral)

                            #Plot bin percentile-percentage scatter plot
                            ext_prcntl_viral, ext_prcnt_viral = btns_utils.find_interesting_prot_in_src_top(sorted_df_pos,
                                                                sorted_df_top_k, viral_uniprots_dict[viral_type], prcntl_viral, prcnt_viral)
                            # Plot a scatter plot of percentile-score and percentage of viral prot per bin
                            prcntl_prcnt_btns_viral_plt_file = alg_term_spec_btns_out_dir + 'scatter_viral_'+viral_type+'_a' + str(alpha) + '.pdf'
                            btns_plot_utils.scatter_plot(ext_prcntl_viral, ext_prcnt_viral, x_label=x_label,
                                y_label='percentage of viral interactors : '+viral_type,
                                title=title, filename=prcntl_prcnt_btns_viral_plt_file)


                        #Plot binwise percentile rank for source proteins and top_k_predictions
                        src_top_rank_df = pd.DataFrame({'percent_rank': list(sorted_df_pos['percent_rank']) +
                            (list(sorted_df_top_k['percent_rank'])),'node_spec': (['pos'] * len(sorted_df_pos)) +
                            (['top_pred'] * len(sorted_df_top_k))})

                        src_top_rank_plot_file = alg_term_spec_btns_out_dir + 'percentile_rank_pos_top_a' + str(alpha) + '.pdf'
                        btns_plot_utils.box_plot(src_top_rank_df,x = 'node_spec', y='percent_rank', ymin=0, ymax=1,
                                 title=title, filename=src_top_rank_plot_file)

    btns_plot_utils.plot_prots_appearing_at_each_pathlens(n_prots_appearing_at_each_pathlens,
                                          filename= output_dir + '/betweenness/'+'new_appering_prots.pdf')

if __name__ == "__main__":
    config_map, master_config_map, kwargs = parse_args()
    main(config_map,master_config_map, **kwargs)
