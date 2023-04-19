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
#import numpy as np
#from scipy import sparse
import pandas as pd
import numpy as np

import itertools
import statistics
import scipy.stats as stats

#import subprocess
sys.path.insert(1, "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/")

# packages in this repo
import src.scripts.enrichment.enrichment_analysis as enrichment
import src.scripts.enrichment.utils as enrichment_utils

from src.FastSinkSource.src import main as run_eval_algs
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.utils import go_prep_utils
from src.FastSinkSource.src.algorithms import alg_utils
import src.scripts.utils as script_utils

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
                      help="Configuration file used when running RWR algs")
    group.add_argument('--id-mapping-file', type=str,
                       default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                               "datasets/mappings/human/uniprot-reviewed-status.tab.gz",
                       help="Table downloaded from UniProt to map to gene names. Expected columns: 'Entry', 'Gene names', 'Protein names'")
    parser.add_argument('-g', '--gaf-file', type=str, default='/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/datasets/go/goa_human.gaf',
                        help="File containing GO annotations in GAF format. Required")
    group = parser.add_argument_group('Enrichment Testing Options')
    group.add_argument('--balancing-alpha-only', action='store_true', default=True,
                       help="Ignore alpha from config file rather take the alpha value\
                            that balanced the two loss terms in quad loss function for the corresponding\
                            network-term-alg")

    group.add_argument('--n-sp', '-n', type=int, default=1000,
                       help="n-sp is the number of shortest paths to be considered" +
                            "If not specified, will check the config file. Default=1000")
    # group.add_argument('--pos-k', action='store_true', default=True,
    #                    help="if true get the top-k predictions to test is equal to the number of positive annotations")
    group.add_argument('--k-to-test', '-k', type=int, default=332,
                       help="k-value(s) for which to get the top-k predictions to test. " +
                       "If not specified, will check the config file. Default=100")

    group.add_argument('--stat-sig-cutoff', type=float,
                       help="Cutoff on the node p-value for a node to be considered in the topk. " + \
                       "The p-values should already have been computed with run_eval_algs.py")
    group.add_argument('--force-run', action='store_true', default=False,
                       help="Force re-running the enrichment tests, and re-writing the output files")

    #Enrichment
    group.add_argument('--pval-cutoff', type=float, default=0.01,
                       help="Cutoff on the corrected p-value for enrichment.")
    group.add_argument('--qval-cutoff', type=float, default=0.05,
                       help="Cutoff on the Benjamini-Hochberg q-value for enrichment.")
    group.add_argument('--fss-pval', type=float, default=0.01,
                       help="Cutoff on the Benjamini-Hochberg q-value for enrichment.")
    return parser

def handle_simplify_go_terms(dataset_name, enrich_df_preds,enrich_df_paths, out_file, **kwargs):
    #combine enrichment result on multiple query protein sets
    terms_to_keep = set(list(enrich_df_preds[enrich_df_preds['p.adjust'] <= kwargs.get('fss_pval')]['ID']) + \
                    list(enrich_df_paths[enrich_df_paths['p.adjust'] <= kwargs.get('fss_pval')]['ID']))

    enrich_df_preds.set_index('ID', inplace=True)
    enrich_df_paths.set_index('ID', inplace=True)

    preds_tuple = [(dataset_name, 'top_preds', col) for col in enrich_df_preds.columns]
    index = pd.MultiIndex.from_tuples(preds_tuple)
    enrich_df_preds.columns = index
    paths_tuple = [(dataset_name, 'top_paths', col) for col in enrich_df_paths.columns]
    index = pd.MultiIndex.from_tuples(paths_tuple)
    enrich_df_paths.columns = index
    all_dfs = pd.concat([enrich_df_preds, enrich_df_paths], axis=1)

    all_dfs = all_dfs[all_dfs.index.isin(terms_to_keep)]

    #save the combined enrichment file before simplification
    out_file_1 = out_file.replace('.csv','before_simple.csv')
    all_dfs.to_csv(out_file_1)

    processed_df = enrichment_utils.process_df(all_dfs)
    #Simplify enrichment result
    greedy_simplified_df = \
        enrichment_utils.simplify_enrichment_greedy_algo_v3(processed_df.copy())
    greedy_simplified_df.to_csv(out_file)

    return greedy_simplified_df


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

    k_to_test = kwargs.get('k_to_test')
    sig_cutoff = kwargs.get('stat_sig_cutoff')
    sig_str = "-sig%s" % (str(sig_cutoff).replace('.', '_')) if sig_cutoff else ""


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
                        # ******************** Find prots in top contributing paths ************************
                        nsp_processed_paths_file = config_map['output_settings'][
                                    'output_dir'] + "/viz/%s/%s/diffusion-path-analysis/%s/shortest_path_2ss/processed_shortest-paths-2ss-nsp%s-a%s%s.tsv" % (
                                    dataset['net_version'], term, alg_name, nsp, alpha, sig_str)
                        # reading 'path_prots' column value as list
                        df = pd.read_csv(nsp_processed_paths_file, sep='\t', index_col=None,
                                         converters={'path_prots': pd.eval})

                        # get prots on top nsp paths
                        path_prots = list(df['path_prots'])  # this is one nested list. flatten it.
                        top_path_prots = set([element for innerList in path_prots for element in innerList])
                        # remove source nodes
                        top_path_prots = top_path_prots.difference(set(orig_pos))


                        #******************** Find prots with top prediction scores *********************
                        t1=time.time()
                        pred_file = alg_pred_files[alg]
                        pred_file = script_utils.term_based_pred_file(pred_file, term)

                        df = pd.read_csv(pred_file, sep='\t')
                        # remove the original positives
                        df = df[~df['prot'].isin(orig_pos)]
                        df.reset_index(inplace=True, drop=True)
                        #df = df[['prot', 'score']]
                        df.sort_values(by='score', ascending=False, inplace=True)

                        pred_filtered_file = "%s/%s/%s-filtered%s.tsv" % (
                            base_out_dir, alg, os.path.basename(pred_file).split('.')[0],
                            "-p%s"%sig_str)
                        os.makedirs(os.path.dirname(pred_filtered_file), exist_ok=True)
                        if kwargs.get('force_run') or not os.path.isfile(pred_filtered_file):
                            print("writing %s" % (pred_filtered_file))
                            df.to_csv(pred_filtered_file, sep='\t', index=None)

                        #take as many proteins as much present in top_path_prots
                        k_to_test = len(top_path_prots)
                        top_preds = set(list(df.iloc[:k_to_test ]['prot']))

                        #*************** COMPARE top preds and top paths prots
                        jaccard_idx = len(top_preds.intersection(top_path_prots))/len(top_preds.union(top_path_prots))
                        print('jaccard index: ', jaccard_idx)

                        # now run clusterProfiler from R
                        out_dir = "%s/%s/%s/a-%s/" % (
                            base_out_dir, alg, term, str(alpha))
                        direct_prot_goids_by_c, _, _, _ = \
                            go_prep_utils.parse_gaf_file(kwargs.get('gaf_file'))
                        go_category = {'BP': 'P', 'MF': 'F', 'CC': 'C'}

                        for ont in ['BP', 'MF', 'CC']:
                            ##keep the prots that have atleast one go annotation
                            filtered_prot_universe = go_prep_utils.keep_prots_having_atleast_one_go_ann(prot_universe,
                                                    go_category[ont], direct_prot_goids_by_c)

                            filtered_top_predictions = go_prep_utils.keep_prots_having_atleast_one_go_ann(top_preds,
                                                    go_category[ont], direct_prot_goids_by_c)
                            filtered_top_path_prots = go_prep_utils.keep_prots_having_atleast_one_go_ann(top_path_prots,
                                                    go_category[ont],direct_prot_goids_by_c)

                            print('#top predicted prots in query: ', len(filtered_top_predictions),
                                  '\n #prots in universe: ', len(filtered_prot_universe))
                            enrich_str = 'top_preds'
                            enrich_df_preds = enrichment.run_clusterProfiler_GO(filtered_prot_universe,
                                filtered_top_predictions, ont,enrich_str, out_dir, forced=kwargs.get('force_run'), **kwargs)


                            enrich_str = 'top_paths'
                            print('#top paths prots in query: ', len(filtered_top_path_prots),
                                  '\n #prots in universe: ', len(filtered_prot_universe))
                            enrich_df_paths = enrichment.run_clusterProfiler_GO(
                                filtered_prot_universe, filtered_top_path_prots, ont,
                                enrich_str, out_dir, forced=kwargs.get('force_run'), **kwargs)

                            # SIMPLIFY GO TERMS together from top_preds and top_paths anc compare those
                            compare_enrichment_out_file = "%s/enrich-%s-nsp%s-k%s-compare-%s.csv" % (out_dir, ont,str(nsp),
                                    str(k_to_test), str(kwargs.get('pval_cutoff')).replace('.', '_'))
                            handle_simplify_go_terms(dataset_name, enrich_df_preds, enrich_df_paths,
                                                     compare_enrichment_out_file, **kwargs)


                            #********* SIMPLIFY GO TERMS SEPERATELY from top_preds and top_paths and then compare those.
                            # simplify the list of GO terms
                            # go_terms_top_preds = set(enrichment_runner.simplify_enrichment_greedy_algo_v3(enrich_df_preds))
                            #simplify the list of GO terms
                            # go_terms_top_paths = set(enrichment_runner.simplify_enrichment_greedy_algo_v3(enrich_df_paths))
                            #compare enrich_df_preds and enrich_df_paths
                            # print(ont)
                            # jaccard(go_terms_top_preds, go_terms_top_paths )
                            # compare_enrichment_out_file = "%s/enrich-%s-%s-%s.csv" % (out_dir, ont,'compare',
                            #             str(kwargs.get('pval_cutoff')).replace('.', '_'))
                            # #find out the unique GO ID present in top_preds and top_paths
                            # unique_top_preds = go_terms_top_preds.difference(go_terms_top_paths)
                            # unique_top_paths = go_terms_top_paths.difference(go_terms_top_preds)
                            # common_ids = go_terms_top_preds.intersection(go_terms_top_paths)
                            # unique_df = pd.DataFrame({'unique_presence_in': ['top_preds']*len(unique_top_preds) +
                            #                          ['top_paths']*len(unique_top_paths),
                            #                         'GO_ID': list(unique_top_preds) + list(unique_top_paths)})
                            # common_df = pd.DataFrame({'unique_presence_in': ['common']*len(common_ids),
                            #                         'GO_ID': list(common_ids)})
                            # comp_df = pd.concat([unique_df,common_df],axis=0)
                            # comp_df.to_csv(compare_enrichment_out_file, index=False)

                        print('Running enrichgo done: ', term, ' ', alg)




def jaccard(set1, set2):
    jaccard_idx = len(set1.intersection(set2)) / len(set1.union(set2))
    print('jaccard index: ', jaccard_idx)
    return jaccard_idx

if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map,**kwargs)