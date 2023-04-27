"""
Script to test for enrichment of FSS outputs
"""
import argparse
import yaml
from collections import defaultdict
import os
import sys
#from tqdm import tqdm
import copy
import time
import requests
#import numpy as np
#from scipy import sparse
import pandas as pd
from bs4 import BeautifulSoup


import itertools
import statistics
import scipy.stats as stats

#import subprocess
sys.path.insert(1, "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/")

# packages in this repo
import src.scripts.enrichment.enrichment_analysis as enrichment
# from src.utils import parse_utils as utils
from src.FastSinkSource.src import main as run_eval_algs
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.utils import go_prep_utils
from src.FastSinkSource.src.algorithms import alg_utils
import src.scripts.utils as script_utils

ont_revigo_name={'BP':'BiologicalProcess', 'MF':'MolecularFunction', 'CC':'CellularComponent'}
def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        config_map = yaml.load(conf, Loader=yaml.FullLoader)
        # config_map = yaml.load(conf)
    # TODO check to make sure the inputs are correct in config_map

    return config_map, kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="Script to test for enrichment of the top predictions among given genesets. " + \
                                     "Currently only tests for GO term enrichment")

    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str,
                       default = "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                     "fss_inputs/config_files/provenance/signor_s12.yaml",
                      help="Configuration file used when running RL and RWR algs")
    group.add_argument('--id-mapping-file', type=str,
                       default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                               "datasets/mappings/human/uniprot-reviewed-status.tab.gz",
                       help="Table downloaded from UniProt to map to gene names. Expected columns: 'Entry', 'Gene names', 'Protein names'")
    ##The goannotation file will help to filter the prots in prot universe and from top prot list such that
    ## any protein that does not have any go term(Corresponding BP, CC or MF ) annotated to it will be
    ## removed for downstream analysis
    parser.add_argument('-g', '--gaf-file', type=str, default =
        '/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/datasets/go/goa_human.gaf',
         help="File containing GO annotations in GAF format. Required")

    group.add_argument('--compare-krogan-terms',
            help="path/to/krogan-enrichment-dir with the enriched terms files (i.e., enrich-BP.csv)"
            " inside. Will be added to the combined table. can be = 'outputs/enrichment/krogan/p1_0/")

    group.add_argument('--out-pref',
                       help="Output prefix where final output file will be placed. " +
                       "Default is <outputs>/enrichement/combined/<config_file_name>")
    group.add_argument('--file-per-alg', action='store_true',
                       help="Make a separate summary file per algorithm")

    group = parser.add_argument_group('Enrichment Testing Options')
    group.add_argument('--balancing-alpha-only', action='store_true', default=True,
                       help="Ignore alpha from config file rather take the alpha value\
                            that balanced the two loss terms in quad loss function for the corresponding\
                            network-term-alg")
    group.add_argument('--enrichment-on', type=str, default='top_paths',
                       help="if 'top_preds' then do enrichment analysis on top k predicted nodes, "
                        "if 'top_paths' then do enrichment analysis on nodes present in top nsp paths.")
    group.add_argument('--n-sp', '-n', type=int, default=1000,
                       help="n-sp is the number of shortest paths to be considered" +
                            "If not specified, will check the config file. Default=1000")
    group.add_argument('--pos-k', action='store_true', default=True,
                       help="if true get the top-k predictions to test is equal to the number of positive annotations")
    group.add_argument('--k-to-test', '-k', type=int, default=332,
                       help="k-value(s) for which to get the top-k predictions to test. " +
                       "If not specified, will check the config file. Default=100")
    group.add_argument('--range-k-to-test', '-K', type=int, nargs=3,
                       help="Specify 3 integers: starting k, ending k, and step size. " +
                       "If not specified, will check the config file.")
    group.add_argument('--stat-sig-cutoff', type=float,
                       help="Cutoff on the node p-value for a node to be considered in the topk. " + \
                       "The p-values should already have been computed with run_eval_algs.py")
    group.add_argument('--add-orig-pos-to-prot-universe', action='store_true',
                       help="Add the positives listed in the pos_neg_file (e.g., Krogan nodes) to the prot universe ")

    #Paramters to ClusterProfiler
    group.add_argument('--pval-cutoff', type=float, default=0.01,
                       help="Cutoff on the corrected p-value for enrichment.")
    group.add_argument('--qval-cutoff', type=float, default=0.05,
                       help="Cutoff on the Benjamini-Hochberg q-value for enrichment.")
    group.add_argument('--fss-pval', type=float, default=0.01,
                       help="Cutoff on the Benjamini-Hochberg q-value for enrichment.")
    group.add_argument('--simplify', type=bool, default=True,
                       help="Remove redundant GO terms ")

    group = parser.add_argument_group('FastSinkSource Pipeline Options')

    group.add_argument('--force-run', action='store_true', default=False,
                       help="Force re-running the enrichment tests, and re-writing the output files")
    return parser


def main(config_map, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """
    # extract the general variables from the config map
    input_settings, input_dir, output_dir, alg_settings, kwargs \
        = config_utils.setup_config_variables(config_map, **kwargs)

    uniprot_to_gene = enrichment.load_gene_names(kwargs.get('id_mapping_file'))
    kwargs['uniprot_to_gene'] = uniprot_to_gene

    gene_to_uniprot = enrichment.load_uniprot(kwargs.get('id_mapping_file'))
    kwargs['gene_to_uniprot'] = gene_to_uniprot

    k = kwargs.get('k_to_test')
    sig_cutoff = kwargs.get('stat_sig_cutoff')
    sig_str = "-sig%s" % (str(sig_cutoff).replace('.', '_')) if sig_cutoff else ""

    enrichment_on = kwargs.get('enrichment_on')
    nsp = kwargs.get('n_sp')

    # for each dataset, extract the path(s) to the prediction files,
    # read in the predictions, and test for the statistical significance of overlap
    for dataset in input_settings['datasets']:
        print("Loading data for %s" % (dataset['net_version']))
        base_out_dir = "%s/enrichment/%s/%s" % (output_dir, dataset['net_version'], dataset['exp_name'])
        # load the network and the positive examples for each term
        net_obj, ann_obj, _ = run_eval_algs.setup_dataset(
            dataset, input_dir, **kwargs)
        prots, node2idx = net_obj.nodes, net_obj.node2idx
        prot_universe = set(prots) #TODO discuss with Murali about what prot set to use as universe
        dataset_name = config_utils.get_dataset_name(dataset)

        print("\t%d prots in universe" % (len(prot_universe)))

        for term in ann_obj.terms:
            term_idx = ann_obj.term2idx[term]
            orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, term_idx)
            orig_pos = [prots[p] for p in orig_pos_idx]
            pos_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]
            n_pos = len(pos_nodes_idx)
            #If 'pos_k'=True, then the number of top predictions is equal to the number of positively annotated nodes
            # for this certain term.
            if kwargs.get('pos_k'):
                k = n_pos
                print('k: ', k)
            for alg_name in alg_settings:
                if (alg_settings[alg_name]['should_run'][0] == True):
                    # load the top predictions
                    print(alg_name)

                    if kwargs.get('balancing_alpha_only'):
                        balancing_alpha = script_utils.get_balancing_alpha(config_map, dataset, alg_name, term)
                        alg_settings[alg_name]['alpha'] = [balancing_alpha]

                    alg_pred_files = config_utils.get_dataset_alg_prediction_files(
                        output_dir, dataset, alg_settings, [alg_name], **kwargs)
                    # get the alpha values to use
                    alphas = alg_settings[alg_name]['alpha']

                    for alpha, alg in zip(alphas, alg_pred_files):

                        #read prediction file
                        pred_file = alg_pred_files[alg]
                        pred_file = script_utils.term_based_pred_file(pred_file, term)
                        df_pred = pd.read_csv(pred_file, sep='\t')
                        # remove the original positives
                        df_pred = df_pred[~df_pred['prot'].isin(orig_pos)]
                        df_pred.reset_index(inplace=True, drop=True)
                        # df = df[['prot', 'score']]
                        df_pred.sort_values(by='score', ascending=False, inplace=True)

                        #read top contributing paths file
                        nsp_processed_paths_file = config_map['output_settings'][
                        'output_dir'] + "/viz/%s/%s/diffusion-path-analysis/%s/shortest_path_2ss/" \
                                        "processed_shortest-paths-2ss-k%s-nsp%s-a%s%s.tsv" % (
                        dataset['net_version'], term, alg_name,k, nsp, alpha, sig_str)
                        # reading 'path_prots' column value as list
                        df_path = pd.read_csv(nsp_processed_paths_file, sep='\t', index_col=None, converters={'path_prots': pd.eval})
                        # get prots on top nsp paths
                        path_prots = list(df_path['path_prots'])  # this is one nested list. flatten it.


                        if kwargs.get('enrichment_on') =='top_preds':
                            # for k in k_to_test:
                            query_prots = list(df_pred.iloc[:k]['prot'])
                            enrich_str = enrichment_on+'_'+str(k)

                        elif kwargs.get('enrichment_on') =='top_paths':
                            query_prots = set([element for innerList in path_prots for element in innerList])
                            #remove source nodes
                            query_prots = query_prots.difference(set(orig_pos))
                            enrich_str = enrichment_on + '_k'+str(k)+'_nsp' + str(nsp)

                            #check how different is proteins_in_top_contibuting_paths and top_predictions
                            #take top_m predicted proteins where m=len(proteins in top k contributing paths)
                            m = len(query_prots)
                            top_m_preds = list(df_pred['prot'])[0:m]
                            jaccard_idx = script_utils.compute_jaccard(query_prots, top_m_preds)
                            print('jaccard index between similar number of proteins in top preds and top nsp contributing paths: ', jaccard_idx)

                            prots_dif_in_top_paths =  set(query_prots).difference(set(top_m_preds))



                        # now run clusterProfiler from R
                        out_dir = "%s/%s/%s/a-%s/" % (
                            base_out_dir, alg, term,str(alpha) )

                        direct_prot_goids_by_c,_,_,_ = \
                            go_prep_utils.parse_gaf_file(kwargs.get('gaf_file'))
                        go_category = {'BP':'P', 'MF':'F', 'CC':'C'}

                        # # Now do KEGG enrichment analysis
                        # _, _ = enrichment.run_clusterProfiler_KEGG(
                        #     prot_universe, query_prots,
                        #     enrich_str, out_dir, forced=kwargs.get('force_run'), **kwargs)

                        for ont in ['BP', 'MF', 'CC']:
                            ##keep the prots that have atleast one go annotation
                            filtered_topk_predictions = go_prep_utils.keep_prots_min_1_ann(query_prots,
                                                                                           go_category[ont], direct_prot_goids_by_c)
                            filtered_prot_universe= go_prep_utils.keep_prots_min_1_ann(prot_universe,
                                                                                       go_category[ont], direct_prot_goids_by_c)

                            print('#prots in query: ', len(filtered_topk_predictions),
                                  '\n #prots in universe: ', len(filtered_prot_universe))
                            enrich_df, out_file = enrichment.run_clusterProfiler_GO(
                                filtered_prot_universe, filtered_topk_predictions, ont,
                                enrich_str, out_dir, forced=kwargs.get('force_run'), **kwargs)
                            print('number of enriched terms for ', ont, ': ', len(enrich_df))

                            #now simplify the go terms using REVIGO
                            if kwargs.get('simplify'):
                                simplified_file = out_file.replace('.csv','_revigo_simplified.csv')
                                revigo_simplify(enrich_df, simplified_file,ont_revigo_name[ont])


                        print('number of enriched terms for ', ont, ': ', len(enrich_df))
                            # #some extra analysis
                            # if kwargs.get('enrichment_on') == 'top_paths':
                            #     #enrichment on protein that are present only in top paths
                            #     filtered_query_diff = go_prep_utils.keep_prots_min_1_ann\
                            #         (prots_dif_in_top_paths,go_category[ont],direct_prot_goids_by_c)
                            #     enrich_df, out_file = enrichment.run_clusterProfiler_GO(
                            #         filtered_prot_universe, filtered_query_diff, ont,
                            #         'diff_'+enrich_str, out_dir, forced=kwargs.get('force_run'), **kwargs)
                        print('Running enrichgo done: ', term, ' ', alg)





def revigo_simplify(enrich_df, simplified_file, ont_revigo):
    # Read enrichments file
    # userData = open(enrich_go_file, 'r').read()
    df = copy.deepcopy(enrich_df)
    df['concat_str'] = df['ID'] + ' ' + df['p.adjust'].astype(str)
    df.set_index('ID', inplace=True)

    userData = '\n'.join(list(df['concat_str']))
    # Submit job to Revigo
    payload = {'cutoff': '0.7', 'valueType': 'pvalue', 'speciesTaxon': '9606', 'measure': 'SIMREL', 'goList': userData}
    r = requests.post("http://revigo.irb.hr/Revigo", data=payload)
    # Write results to a file - if file name is not provided the default is result.csv
    soup = BeautifulSoup(r.text, 'html.parser')
    tbl = soup.find("table", {"id": ont_revigo})

    simplified_terms_df = pd.read_html(str(tbl))[0]
    simplified_terms_df.rename(columns={'Term ID':'ID'}, inplace=True)

    simplified_terms_df.set_index('ID', inplace=True)
    simplified_terms_df = pd.concat([simplified_terms_df, df], axis=1)

    print('Initial: ', len(df))
    print('REVIGO info available: ', len(simplified_terms_df))

    #now keep the rows where 'Eliminated==False' i.e. Revigo Do not eliminate those terms.
    simplified_terms_df = simplified_terms_df[simplified_terms_df['Eliminated']==False]
    print('REVIGO simplified: ', len(simplified_terms_df))

    simplified_terms_df.reset_index(inplace=True)
    simplified_terms_df = simplified_terms_df[['ID','Description','p.adjust','Frequency','Uniqueness','Dispensability','geneID','geneName']]
    simplified_terms_df.to_csv(simplified_file)



if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map,**kwargs)