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

# packages in this repo
import enrichment_analysis as enrichment
# from src.utils import parse_utils as utils
from src.FastSinkSource.src import main as run_eval_algs
from src.FastSinkSource.src.utils import config_utils
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
                     "fss_inputs/config_files/provenance/"
                     "string700_biogrid_y2h_signor_s1.yaml",
                      help="Configuration file used when running RL and RWR algs")
    group.add_argument('--id-mapping-file', type=str,
                       default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                               "datasets/mappings/human/uniprot-reviewed-status.tab.gz",
                       help="Table downloaded from UniProt to map to gene names. Expected columns: 'Entry', 'Gene names', 'Protein names'")


    group = parser.add_argument_group('Enrichment Testing Options')
    group.add_argument('--k-to-test', '-k', type=int, default=332,
                       help="k-value(s) for which to get the top-k predictions to test. " +
                       "If not specified, will check the config file. Default=100")

    group.add_argument('--add-orig-pos-to-prot-universe', action='store_true',
                       help="Add the positives listed in the pos_neg_file (e.g., Krogan nodes) "
                            "to the prot universe ")

    group.add_argument('--pval-cutoff', type=float, default=0.01,
                       help="Cutoff on the corrected p-value for enrichment.")
    group.add_argument('--qval-cutoff', type=float, default=0.05,
                       help="Cutoff on the Benjamini-Hochberg q-value for enrichment.")

    group = parser.add_argument_group('FastSinkSource Pipeline Options')


    group.add_argument('--force-run', action='store_true', default=False,
                       help="Force re-running the enrichment tests, and re-writing the output files")
    group.add_argument('--balancing-alpha-only', action='store_true', default=True,
                       help="Ignore alpha from config file rather take the alpha value\
                            that balanced the two loss terms in quad loss function for the corresponding\
                            network-term-alg")
    return parser

def main(config_map, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """
    # extract the general variables from the config map
    input_settings, input_dir, output_dir, alg_settings, kwargs \
        = config_utils.setup_config_variables(config_map, **kwargs)

    # for each dataset, extract the path(s) to the prediction files,
    # read in the predictions, and test for the statistical significance of overlap
    for alg_name in alg_settings:
        if (alg_settings[alg_name]['should_run'][0] == True):
            # load the top predictions
            print(alg_name)

            enriched_go_bp_terms = {}
            enriched_go_cc_terms = {}
            enriched_go_mf_terms = {}
            network_names = []
            for dataset in input_settings['datasets']:
                network_name = dataset['net_version'].replace('networks/','')
                network_names.append(network_name)

                print("Loading data for %s" % (dataset['net_version']))
                base_out_dir = "%s/enrichment/%s/%s" % (output_dir, dataset['net_version'], dataset['exp_name'])
                # load the network and the positive examples for each term
                net_obj, ann_obj, _ = run_eval_algs.setup_dataset(
                    dataset, input_dir, **kwargs)
                prots, node2idx = net_obj.nodes, net_obj.node2idx

                for term in ann_obj.terms:
                    term_idx = ann_obj.term2idx[term]
                    orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, term_idx)
                    orig_pos = [prots[p] for p in orig_pos_idx]
                    pos_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]
                    n_pos = len(pos_nodes_idx)

                    #If 'pos_k'=True, then the number of top predictions is equal to the number of positively annotated nodes
                    # for this certain term.
                    if kwargs.get('pos_k'):
                        k_to_test = n_pos
                        print('k: ', k_to_test)

                     # store all the enriched terms in a single dataframe

                    if kwargs.get('balancing_alpha_only'): #in alg_setting[alg_name]['alpha'] put the balancing alpha
                        # get the balancing alpha for this network - alg - term
                        alpha_summary_filename = config_map['output_settings']['output_dir'] + \
                            "/viz/%s/%s/param_select/" % (dataset['net_version'], dataset[
                            'exp_name']) + '/' + alg_name + '/alpha_summary.tsv'
                        alpha_summary_df = pd.read_csv(alpha_summary_filename, sep='\t', index_col=None)[['term','balancing_alpha']]
                        term_2_balancing_alpha_dict = dict(zip(alpha_summary_df['term'], alpha_summary_df['balancing_alpha']))

                        balancing_alpha = term_2_balancing_alpha_dict[term]
                        alg_settings[alg_name]['alpha'] = [balancing_alpha]


                    alg_pred_files = config_utils.get_dataset_alg_prediction_files(
                        output_dir, dataset, alg_settings, [alg_name], **kwargs)
                    # get the alpha values to use
                    alphas = alg_settings[alg_name]['alpha']
                    for alpha, alg in zip(alphas, alg_pred_files):
                        t1=time.time()
                        pred_file = alg_pred_files[alg]
                        pred_file = script_utils.term_based_pred_file(pred_file, term)

                        pred_filtered_file = "%s/%s/%s-filtered%s.tsv" % (
                            base_out_dir, alg, os.path.basename(pred_file).split('.')[0],
                            "-p%s"%str(kwargs['stat_sig_cutoff']).replace('.','_')
                            if kwargs.get('stat_sig_cutoff') else "")

                        # now run clusterProfiler from R
                        out_dir = pred_filtered_file.split('.')[0]

                        enrich_dfs = {}
                        for ont in ['BP', 'MF', 'CC']:
                            out_file = "%s/enrich-%s-%s.csv" % (out_dir, ont,
                                        str(kwargs.get('pval_cutoff')).replace('.', '_'))
                            enrich_dfs[ont] = pd.read_csv(out_file, sep=',', index_col=None)

                        if term not in enriched_go_bp_terms:
                            enriched_go_bp_terms[term] = {}
                            enriched_go_cc_terms[term] = {}
                            enriched_go_mf_terms[term] = {}
                        # enriched_go_bp_terms[term][network_name] = bp_df['ID']
                        # enriched_go_mf_terms[term][network_name] = mf_df['ID']
                        # enriched_go_cc_terms[term][network_name] = cc_df['ID']

                        if not enrich_dfs['BP'].empty:
                            enriched_go_bp_terms[term][network_name] = enrich_dfs['BP']['ID']
                        if not enrich_dfs['MF'].empty:
                            enriched_go_mf_terms[term][network_name] = enrich_dfs['MF']['ID']
                        if not enrich_dfs['CC'].empty:
                            enriched_go_cc_terms[term][network_name] = enrich_dfs['CC']['ID']


            #compare the enriched go terms for a particular source-node-set and particular
            # algorithm across networks
            print('Algorithm: ', alg_name)
            print('Enrichement: GO-BP')
            jaccard_idx_bp = handle_jaccard_enriched_terms(enriched_go_bp_terms, network_names)
            print('Enrichement: GO-MF')
            jaccard_idx_mf = handle_jaccard_enriched_terms(enriched_go_mf_terms, network_names)
            print('Enrichement: GO-CC')
            jaccard_idx_cc = handle_jaccard_enriched_terms(enriched_go_cc_terms, network_names)




def handle_jaccard_enriched_terms(enriched_go_terms, network_names):
    for term in enriched_go_terms:
        #any term-network pair may not have any enriched term at all
        enriched_go_terms_for_net_1  = set(enriched_go_terms[term][network_names[0]]) \
            if network_names[0] in enriched_go_terms[term] else set()
        enriched_go_terms_for_net_2  = set(enriched_go_terms[term][network_names[1]]) \
            if network_names[1] in enriched_go_terms[term] else set()
        jaccard_idx = script_utils.compute_jaccard(enriched_go_terms_for_net_1, enriched_go_terms_for_net_2)

        print('term: ', term, 'jaccard idx: ', jaccard_idx)
        print(network_names[0], ':',len(enriched_go_terms_for_net_1))
        print(network_names[1], ':',len(enriched_go_terms_for_net_2))

        return jaccard_idx
#
# def compute_jaccard(set1, set2):
#     '''
#     pass tow set or lists.
#     '''
#     return len(set(set1).intersection(set(set2)))/ len(set(set1).union(set(set2)))

if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)