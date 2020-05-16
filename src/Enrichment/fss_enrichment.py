"""
Script to test for enrichment of FSS outputs
"""
import sys
sys.path.insert(1, '/home/tasnina/SARS-CoV-2-network-analysis')
import argparse
import yaml
from collections import defaultdict
import os

#from tqdm import tqdm
import copy
import time
#import numpy as np
#from scipy import sparse
import pandas as pd
#import subprocess

# packages in this repo

import enrichment
from src.utils import parse_utils as utils
from src.FastSinkSource.src import main as run_eval_algs
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils


def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        #config_map = yaml.load(conf, Loader=yaml.FullLoader)
        config_map = yaml.load(conf)
    # TODO check to make sure the inputs are correct in config_map

    return config_map, kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="Script to test for enrichment of the top predictions among given genesets. " + \
                                     "Currently only tests for GO term enrichment")

    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, required=True,
                       help="Configuration file used when running FSS. " +
                       "Must have a 'genesets_to_test' section for this script. ")
    group.add_argument('--id-mapping-file', type=str, default="datasets/mappings/human/uniprot-reviewed-status.tab.gz",
                       help="Table downloaded from UniProt to map to gene names. Expected columns: 'Entry', 'Gene names', 'Protein names'")
    #group.add_argument('--compare-krogan-nodes',
    #                   help="Also test for enrichment of terms when using the Krogan nodes.")
    # Should be specified in the config file
    #group.add_argument('--gmt-file', append=True,
    #                   help="Test for enrichment using the genesets present in a GMT file.")
    #group.add_argument('--prot-list-file',
    #                   help="Test for enrichment of a list of proteins (UniProt IDs) (e.g., Krogan nodes).")
    #group.add_argument('--prot-universe-file',
    #                   help="Protein universe to use when testing for enrichment")
    #group.add_argument('--out-dir', type=str,
    #                   help="path/to/output directory for enrichemnt files")
    group.add_argument('--out-pref',
                       help="Output prefix where final output file will be placed. " +
                       "Default is <outputs>/enrichement/combined/<config_file_name>")
    group.add_argument('--file-per-alg', action='store_true',
                       help="Make a separate summary file per algorithm")

    group = parser.add_argument_group('Enrichment Testing Options')
    group.add_argument('--k-to-test', '-k', type=int, action="append",
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

    group = parser.add_argument_group('FastSinkSource Pipeline Options')
    group.add_argument('--alg', '-A', dest='algs', type=str, action="append",
                       help="Algorithms for which to get results. Must be in the config file. " +
                       "If not specified, will get the list of algs with 'should_run' set to True in the config file")
    group.add_argument('--num-reps', type=int,
                       help="Number of times negative sampling was repeated to compute the average scores. Default=1")
    group.add_argument('--sample-neg-examples-factor', type=float,
                       help="Factor/ratio of negatives to positives used when making predictions. " +
                       "Not used for methods which use only positive examples.")
    group.add_argument('--force-run', action='store_true', default=False,
                       help="Force re-running the enrichment tests, and re-writing the output files")

#    # additional parameters
#    group = parser.add_argument_group('Additional options')
#    group.add_argument('--forcealg', action="store_true", default=False,
#            help="Force re-running algorithms if the output files already exist")
#    group.add_argument('--forcenet', action="store_true", default=False,
#            help="Force re-building network matrix from scratch")
#    group.add_argument('--verbose', action="store_true", default=False,
#            help="Print additional info about running times and such")

    return parser


def main(config_map, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """
    # extract the general variables from the config map
    input_settings, input_dir, output_dir, alg_settings, kwargs \
        = config_utils.setup_config_variables(config_map, **kwargs)
    algs = config_utils.get_algs_to_run(alg_settings, **kwargs)
    print("algs: %s" % (str(algs)))
    del kwargs['algs']

    # load the namespace mappings
    uniprot_to_gene = None
    if kwargs.get('id_mapping_file'):
        uniprot_to_gene = enrichment.load_gene_names(kwargs.get('id_mapping_file'))
        kwargs['uniprot_to_gene'] = uniprot_to_gene

    # genesets_to_test = config_map.get('genesets_to_test')
    # if genesets_to_test is None or len(genesets_to_test) == 0:
    #     print("ERROR: no genesets specified to test for overlap. " +
    #           "Please add them under 'genesets_to_test'. \nQuitting")
    #     sys.exit()

    # # first load the gene sets
    # geneset_groups = {}
    # for geneset_to_test in genesets_to_test:
    #     name = geneset_to_test['name']
    #     gmt_file = "%s/genesets/%s/%s" % (
    #         input_dir, name, geneset_to_test['gmt_file'])
    #     if not os.path.isfile(gmt_file):
    #         print("WARNING: %s not found. skipping" % (gmt_file))
    #         sys.exit()

    #     geneset_groups[name] = utils.parse_gmt_file(gmt_file)

    # store all the enriched terms in a single dataframe
    all_dfs = {g: pd.DataFrame() for g in ['BP', 'CC', 'MF']}
    all_dfs_KEGG = pd.DataFrame()

    # for each dataset, extract the path(s) to the prediction files,
    # read in the predictions, and test for the statistical significance of overlap
    for dataset in input_settings['datasets']:
        print("Loading data for %s" % (dataset['net_version']))
        base_out_dir = "%s/enrichment/%s/%s" % (output_dir, dataset['net_version'], dataset['exp_name'])
        # load the network and the positive examples for each term
        net_obj, ann_obj, _ = run_eval_algs.setup_dataset(
            dataset, input_dir, **kwargs)
        prots = net_obj.nodes
        prot_universe = set(prots)
        print("\t%d prots in universe" % (len(prot_universe)))
        # TODO using this for the SARS-CoV-2 project,
        # but this should really be a general purpose script
        # and to work on any number of terms
        orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, 0)
        orig_pos = [prots[p] for p in orig_pos_idx]
        #print("\t%d original positive examples" % (len(orig_pos)))
        if kwargs.get('add_orig_pos_to_prot_universe'):
            pos_neg_file = "%s/%s" % (input_dir, dataset['pos_neg_file'])
            df = pd.read_csv(pos_neg_file, sep='\t')
            orig_pos = df[df['2020-03-sarscov2-human-ppi'] == 1]['prots']
            print("\t%d original positive examples" % (len(orig_pos)))
            prot_universe = set(prots) | set(orig_pos)
            print("\t%d prots in universe after adding them to the universe" % (len(prot_universe)))

        # now load the predictions, test at the various k values, and TODO plot
        k_to_test = enrichment.get_k_to_test(dataset, **kwargs)
        print("\ttesting %d k value(s): %s" % (len(k_to_test), ", ".join([str(k) for k in k_to_test])))

        # now load the prediction scores
        dataset_name = config_utils.get_dataset_name(dataset)
        alg_pred_files = config_utils.get_dataset_alg_prediction_files(
            output_dir, dataset, alg_settings, algs, **kwargs)
        for alg, pred_file in alg_pred_files.items():
            if not os.path.isfile(pred_file):
                print("Warning: %s not found. skipping" % (pred_file))
                continue
            print("reading: %s" % (pred_file))
            df = pd.read_csv(pred_file, sep='\t')
            # remove the original positives
            df = df[~df['prot'].isin(orig_pos)]
            df.reset_index(inplace=True, drop=True)
            #df = df[['prot', 'score']]
            df.sort_values(by='score', ascending=False, inplace=True)
            if kwargs.get('stat_sig_cutoff'):
                df = config_utils.get_pvals_apply_cutoff(df, pred_file, **kwargs)
            # write these results to file
            pred_filtered_file = "%s/%s/%s-filtered%s.tsv" % (
                base_out_dir, alg, os.path.basename(pred_file).split('.')[0],
                "-p%s"%str(kwargs['stat_sig_cutoff']).replace('.','_') if kwargs.get('stat_sig_cutoff') else "")
            os.makedirs(os.path.dirname(pred_filtered_file), exist_ok=True)
            if kwargs.get('force_run') or not os.path.isfile(pred_filtered_file):
                print("writing %s" % (pred_filtered_file))
                df.to_csv(pred_filtered_file, sep='\t', index=None)

            for k in k_to_test:
                topk_predictions = list(df.iloc[:k]['prot'])

                # now run clusterProfiler from R
                out_dir = pred_filtered_file.split('.')[0]
                bp_df, mf_df, cc_df = enrichment.run_clusterProfiler_GO(
                    topk_predictions, out_dir, prot_universe=prot_universe, forced=kwargs.get('force_run'), **kwargs)
                for ont, df in [('BP', bp_df), ('MF', mf_df), ('CC', cc_df)]:
                    # make it into a multi-column-level dataframe
                    tuples = [(dataset_name, alg, col) for col in df.columns]
                    index = pd.MultiIndex.from_tuples(tuples)
                    df.columns = index
                    all_dfs[ont] = pd.concat([all_dfs[ont], df], axis=1)

                # now run KEGG enrichment analysis

                KEGG_df = enrichment.run_clusterProfiler_KEGG(topk_predictions, out_dir, prot_universe=prot_universe, forced=kwargs.get('force_run'))
                tuples = [(dataset_name, alg, col) for col in KEGG_df.columns]
                index = pd.MultiIndex.from_tuples(tuples)
                KEGG_df.columns = index
                all_dfs_KEGG = pd.concat([all_dfs_KEGG, KEGG_df], axis=1)

    # now write the combined df to a file
    out_pref = kwargs.get('out_pref')
    if out_pref is None:
        out_pref = "%s/enrichment/combined/%s-" % (
            output_dir, os.path.basename(kwargs['config']).split('.')[0])
    for geneset, df in all_dfs.items():
        if kwargs.get('file_per_alg'):
            df = df.swaplevel(0,1,axis=1)
            for alg, df_alg in df.groupby(level=0, axis=1):
                df_alg.dropna(how='all', inplace=True)
                print(df_alg.head())
                out_file = "%s%s-k%s-%s.csv" % (out_pref, alg, k_to_test[0], geneset)
                write_combined_table(df_alg, out_file, dataset_level=1)
        else:
            out_file = "%sk%s-%s.csv" % (out_pref, k_to_test[0], geneset)
            write_combined_table(df, out_file, dataset_level=0)

    #write combined KEGG Enrichment

    if kwargs.get('file_per_alg'):
        all_dfs_KEGG = all_dfs_KEGG.swaplevel(0,1,axis=1)
        for alg, df_alg in all_dfs_KEGG.groupby(level=0, axis=1):
            df_alg.dropna(how='all', inplace=True)
            print(df_alg.head())
            out_file = "%s%s-k%s-KEGG.csv" % (out_pref, alg, k_to_test[0])
            write_combined_table(df_alg, out_file, dataset_level=1)
    else:
        out_file = "%sk%s-KEGG.csv" % (out_pref, k_to_test[0])
        write_combined_table(all_dfs_KEGG, out_file, dataset_level=0)


def write_combined_table(df, out_file, dataset_level=0):
    """
    """
    # for each term ID, store its name
    id_to_name = {}
    # also add the number of datasets/networks for which each term is enriched
    id_counts = defaultdict(int)
    for dataset, df_d in df.groupby(level=dataset_level, axis=1):
        #print(df_d.head())
        # get just the last level of columns
        #df_d.columns = df_d.columns.levels[-1]
        df_d.columns = df_d.columns.droplevel([0,1])
        df_d.dropna(how='all', inplace=True)
        print(df_d.head())
        #print(df_d.index)
        #print(df_d['Description'].head())
        id_to_name.update(dict(zip(df_d.index, df_d['Description'])))
        for geneset_id in df_d.index:
            id_counts[geneset_id] += 1
        #print(pd.Series(id_to_name).head())

    df.insert(0, 'Count', pd.Series(id_counts))
    df.insert(0, 'Description', pd.Series(id_to_name))
    # Drop ID and Description since those will be common for all columns
    # also drop pvalue since having pvalue, pvalue adjust, and qvalue is kind of redundant
    df.drop(['ID','Description', 'pvalue', 'p.adjust'], axis=1, level=2, inplace=True)
    print(df.head())

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    print("writing %s" % (out_file))
    df.to_csv(out_file, sep=',')


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)
