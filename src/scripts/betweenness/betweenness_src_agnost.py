# here's an example call to this script:
# python src/scripts/effective_diffusion_node_path.py --config fss_inputs/config_files/provenance/provenance_biogrid_y2h_go.yaml
# --run-algs genemaniaplus --k 500 --m 20 --n-sp 500

import sys
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
                        "fss_inputs/config_files/provenance/biogrid_y2h_s12.yaml"
                       , help="Configuration file used when running FSS. ")
    group.add_argument('--master-config', type=str, default="/data/tasnina/Provenance-Tracing/"
                        "SARS-CoV-2-network-analysis/config-files/master-config.yaml"
                       , help="Configuration file used to do mappings")
    group.add_argument('--essential-prot-file', type=str,
                       default ="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                        "fss_inputs/essential-prots/deg_annotation_e.csv",
                       help="This file should contain the essential genes for corresponding species")
    group.add_argument('--viral-prot-file', type=str,
                       default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                               "fss_inputs/viral-prots/HVIDB.csv",
                       help="This file should contain the essential genes for corresponding species")
    group.add_argument('--run-algs', type=str, action='append', default=[])
    group.add_argument('--force-download', action='store_true', default=False,
                       help="Force re-downloading and parsing of the input files")
    return parser



def main(config_map, master_config_map, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """
    # extract the general variables from the config map
    input_settings, input_dir, output_dir, alg_settings, kwargs \
        = config_utils.setup_config_variables(config_map, **kwargs)

    for dataset in input_settings['datasets']:
        print("Loading data for %s" % (dataset['net_version']))
        # load the network and the positive examples for each term
        net_obj, ann_obj, _ = setup_dataset(dataset, input_dir, **kwargs)
        prots, node2idx = net_obj.nodes, net_obj.node2idx
        ks = [100, 200, 300, 400, 500, 1000, 2000] #the top k betweenness proteins to consider

        # get essential protein list.
        # map essential prots to uniprot ids
        essential_uniprots = btns_utils.handle_essential_uniprot_mapping(master_config_map, **kwargs)
        #get human proteins that interact with viral prots
        viral_prot_file = kwargs.get('viral_prot_file')
        viral_uniprots =  btns_utils.parse_viral_prot_file(viral_prot_file)

        ###Directory for saving any betweenness related analysis result
        btns_out_dir = config_map['output_settings']['output_dir'] + '/betweenness/' + \
                       dataset['net_version'] + '/'

        # #*************************************************************************************************
        # ********************** NETWORK SPECIFIC NODE PROPERTIES AND ESSENTIAL PROTS *********************
        # all types of centrality (betweenness, degree) for nodes which is a graph node property
        # will be saved in the centrality.tsv file
        centr_file = "%s/net_centrality.tsv" % (btns_out_dir)
        sorted_cntr_df = btns_utils.compute_network_centrality_scores(net_obj.W, prots, centr_file)
        sorted_cntr_df['percent_rank'] = sorted_cntr_df['betweenness'].rank(pct=True)


        # compute correlation between network centrality and essential prots and save
        centr_corr_file = "%s/net_centrality_corr.txt" % (btns_out_dir)
        os.makedirs(os.path.dirname(centr_corr_file), exist_ok=True)
        cntr_corr_out = open(centr_corr_file, 'w')
        cntr_corr_out.write('pc_essential' + '\t' + 'pval_essential' + '\t'+'mw_essential' +
                            '\t'+'pc_viral' + '\t' + 'pval_viral'+'\t'+'mw_viral'+'\n')

        #compute pearson's correlation with centrality and essetiality
        #also output the lowest_percentile_of_centrality_score in each bin and corresponding percentage_of_essential_proteins
        #(or viral_interactors) in that bin.
        pc_cntr_ess, pval_cntr_ess, mw_pval_ess, prcntl_ess, prcnt_ess = btns_utils.handle_percentile_percent_corr(sorted_cntr_df, essential_uniprots,
                                                            score_name='betweenness', interesting_prot='essential prots')
        pc_cntr_viral, pval_cntr_viral, mw_pval_viral, prcntl_viral, prcnt_viral  = btns_utils.handle_percentile_percent_corr(sorted_cntr_df, viral_uniprots,
                                                            score_name='betweenness', interesting_prot='viral interactors')
        cntr_corr_out.write(str(pc_cntr_ess) + '\t' + str(pval_cntr_ess) + '\t'+str(mw_pval_ess)+
                            '\t'+str(pc_cntr_viral) + '\t' + str(pval_cntr_viral)+'\t'+str(mw_pval_viral))
        cntr_corr_out.close()

        ### PLOT essential prots in top ks central prots #################
        plot_ess_in_topk_cntr_file = btns_out_dir + 'topk_central_essential_a' + '.pdf'
        n_ess_prots_per_topk_cntr = btns_utils.interesting_prot_in_topks(sorted_cntr_df, essential_uniprots, ks)
        btns_plot_utils.barplot_from_dict(n_ess_prots_per_topk_cntr, x_label='top k network centrality',
                          y_label='number of essential prots', ymax=9000, title= dataset['plot_exp_name'],
                          filename=plot_ess_in_topk_cntr_file)
            
        ### PLOT viral prots in top ks central prots #################
        plot_viral_in_topk_cntr_file = btns_out_dir + 'topk_central_viral_a' + '.pdf'
        n_viral_prots_per_topk_cntr = btns_utils.interesting_prot_in_topks(sorted_cntr_df, viral_uniprots, ks)
        btns_plot_utils.barplot_from_dict(n_viral_prots_per_topk_cntr, x_label='top k network centrality',
                          y_label='number of viral interactors',ymax=8000, title=dataset['plot_exp_name'],
                          filename=plot_viral_in_topk_cntr_file)
        # ********************************************************************************************

        for alg_name in alg_settings:
            if (alg_settings[alg_name]['should_run'][0] == True) or (alg_name in kwargs.get('run_algs')):
                # load the top predictions
                print(alg_name)
                #alg specific betweenness scores and comparison of these btns
                # scores with other centrality measure of network
                # will be save in the following directory
                alg_spec_btns_out_dir = btns_out_dir + alg_name + '/'

                #across each alpha the correlation btn rwr(rl)_betweenness and essentiality is saved here
                alg_btns_corr_file = alg_spec_btns_out_dir + 'btns_corr.tsv'
                os.makedirs(os.path.dirname(alg_btns_corr_file), exist_ok=True)
                corr_fout = open(alg_btns_corr_file, 'w')
                corr_fout.write('alpha' + '\t' + 'pc_essential'+ '\t' + 'pc_pval_essential' +'\t'+'mw_essential'
                                +'\t'+ 'pc_viral'+ '\t' + 'pc_pval_viral'+'\t'+'mw_viral'
                                +'\t' + 'ktau_centrality' + '\t' + 'ktau_pval_centrality'  +'\n')

                plot_essential_in_topk_file = alg_spec_btns_out_dir + 'topk_btns_essential'+ '.pdf'
                plot_viral_in_topk_file = alg_spec_btns_out_dir + 'topk_btns_viral'+ '.pdf'


                n_interesting_prots_per_topk_df = pd.DataFrame()
                # n_interesting_prots_per_term_spec_topk_df = dict(pd.DataFrame())

                #Get prediction files
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

                    # betweenness_score_file contains rwr(rl)betweenness score of
                    # all proteins in a network given certain alpha
                    btns_file = alg_spec_btns_out_dir + 'btns_a' + str(alpha) + '.tsv'
                    sorted_btns_scores_df = btns_utils.handle_btns(M_pathmtx_loginv, prots, btns_file)
                    
                    #compute Pearsons correlation btn betweenness score and essentiality
                    pc_ess, pval_ess,mw_pval_ess, prcntl_ess,prcnt_ess = btns_utils.handle_percentile_percent_corr(sorted_btns_scores_df, essential_uniprots)
                    # compute Pearsons correlation btn betweenness score and viral prots
                    pc_viral, pval_viral, mw_pval_viral, prcntl_viral, prcnt_viral = btns_utils.handle_percentile_percent_corr(sorted_btns_scores_df, viral_uniprots,
                                                                        interesting_prot='viral interactors')

                    # also compute kendal_tau score between protein list sorted by betweenness score and network centrality score respectively
                    ktau_corr, ktau_pvalue  = \
                        script_utils.kendal_tau(list(sorted_cntr_df['prot']), list(sorted_btns_scores_df['prot']))
                    #SAVE correlation scores
                    corr_fout.write(str(beta) + '\t'+str(pc_ess)+'\t'+ str(pval_ess)+'\t'+ str(mw_pval_ess)+
                                    '\t'+str(pc_viral)+'\t'+ str(pval_viral)+'\t'
                                    +str(ktau_corr)+'\t'+str(ktau_pvalue)+'\t'+ str(mw_pval_viral)+'\n')

                    #Also plot a scatter plot of percentile-betweenness-score and percentage of essential prot per bin
                    percentile_percentage_btns_ess_plot_file = alg_spec_btns_out_dir + 'percentile_percentage_btns_essential_a' + str(alpha) + '.pdf'
                    btns_plot_utils.scatter_plot(prcntl_ess, prcnt_ess, x_label='percentile rank',
                                 y_label='percentage of essential prot', ymin=0, ymax=100,
                                 title=alg_plot_name[alg_name]+'_a_'+str(alpha)+'_'+dataset['plot_exp_name'],
                                 filename= percentile_percentage_btns_ess_plot_file )

                    # Also plot a scatter plot of percentile-betweenness-score and percentage of essential prot per bin
                    percentile_percentage_btns_viral_plot_file = alg_spec_btns_out_dir + 'percentile_percentage_btns_viral_a' + str( alpha) + '.pdf'
                    btns_plot_utils.scatter_plot(prcntl_viral, prcnt_viral, x_label='percentile rank',
                                 y_label='percentage of viral interactors', ymin=0, ymax=100,
                                 title=alg_plot_name[alg_name]+'_a_'+str(alpha)+'_'+dataset['plot_exp_name'],
                                 filename=percentile_percentage_btns_viral_plot_file)

                    #compute how many essential proteins present in different top scoring k prots
                    n_essential_prots_per_topk = btns_utils.interesting_prot_in_topks(sorted_btns_scores_df, essential_uniprots, ks)
                    n_viral_prots_per_topk = btns_utils.interesting_prot_in_topks(sorted_btns_scores_df, viral_uniprots, ks)


                    temp_df = pd.DataFrame({'alpha': [beta]*len(ks), 'top_k':ks,
                                        'n_essential_prot':list(n_essential_prots_per_topk.values()),
                                        'n_viral_interactor': list(n_viral_prots_per_topk.values())})

                    n_interesting_prots_per_topk_df = pd.concat([n_interesting_prots_per_topk_df, temp_df], axis=0)


                #*************************** PLOT ****************************
                # plot how many essential proteins present in different top scoring k prots
                btns_plot_utils.barplot_from_df(n_interesting_prots_per_topk_df, filename=plot_essential_in_topk_file, x='top_k',y='n_essential_prot',
                                ymax=9000, title= alg_plot_name[alg_name]+'_'+dataset['plot_exp_name'])
                #plot how many viral proteins present in different top scoring k prots
                btns_plot_utils.barplot_from_df(n_interesting_prots_per_topk_df, filename=plot_viral_in_topk_file,  x='top_k', y='n_viral_interactor',
                                ymax= 8000, title= alg_plot_name[alg_name]+'_'+dataset['plot_exp_name'])
                corr_fout.close()




if __name__ == "__main__":
    config_map, master_config_map, kwargs = parse_args()
    main(config_map,master_config_map, **kwargs)
