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
import seaborn as sns
from matplotlib import pyplot as plt
#import subprocess

# packages in this repo
sys.path.insert(1, '/home/tasnina/SARS-CoV-2-network-analysis')
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
    group.add_argument('--compare-krogan-terms',default = 'outputs/enrichment/krogan/p1_0/',
                       help="path/to/krogan-enrichment-dir with the enriched terms files (i.e., enrich-BP.csv) inside. Will be added to the combined table")
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

    group.add_argument('--pval-cutoff', type=float, default=0.01,
                       help="Cutoff on the corrected p-value for enrichment.")
    group.add_argument('--qval-cutoff', type=float, default=0.05,
                       help="Cutoff on the Benjamini-Hochberg q-value for enrichment.")
    group.add_argument('--fss-pval', type=float, default=0.01,
                       help="Cutoff on the Benjamini-Hochberg q-value for enrichment.")

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

    return parser

def include_Krogan_enrichment_result(krogan_dir, analysis_spec,g_df ):
    out_file = "%s/enrich-%s-1_0.csv" % (krogan_dir, analysis_spec)
    if not os.path.isfile(out_file):
        print("ERROR: %s not found. Quitting" % (out_file))
        sys.exit()
    print("\treading %s" % (out_file))
    df = pd.read_csv(out_file, index_col=0)
    # drop the terms that don't have a pval < 0.01 and aren't in the FSS results
    # terms_to_keep = set(list(g_df.index.values)) | set(list(df[df['p.adjust'] < kwargs.get('pval_cutoff')]['ID'].values))

    # drop the terms those aren't in the FSS results
    terms_to_keep = set(list(g_df.index.values))

    print("\t%d krogan terms to keep" % (len(terms_to_keep)))
    df = df[df['ID'].isin(terms_to_keep)]
    # also apply the
    tuples = [('Krogan', '-', col) for col in df.columns]
    index = pd.MultiIndex.from_tuples(tuples)
    df.columns = index
    return df


def add_qval_ratio(df,analysis_spec, krogan_dir,alg):
    krogan_file = "%s/enrich-%s-1_0.csv" % (krogan_dir, analysis_spec)
    if not os.path.isfile(krogan_file):
        print("ERROR: %s not found. Quitting" % (krogan_file))
        sys.exit()
    print("\treading %s" % (krogan_file))
    k_df = pd.read_csv(krogan_file, index_col=0)
    # drop the terms that don't have a pval < 0.01 and aren't in the FSS results
    # terms_to_keep = set(list(g_df.index.values)) | set(list(df[df['p.adjust'] < kwargs.get('pval_cutoff')]['ID'].values))

    # drop the terms those aren't in the FSS results
    terms_to_keep = set(list(df.index.values))

    print("\t%d krogan terms to keep" % (len(terms_to_keep)))
    k_df = k_df[k_df['ID'].isin(terms_to_keep)]
    k_df['k_qvalue'] = k_df['qvalue']
    k_df = k_df[['k_qvalue']]

    df = pd.concat([df,k_df], axis = 1)

    df['k_qvalue'] = df['k_qvalue'].fillna(1)

    df['-(log(qvalue '+alg+')- log(qvalue Krogan))'] = -(np.log10(df['qvalue']) - np.log10(df['k_qvalue']))
    df = df.drop('k_qvalue',axis=1)
    return df

def simplify_enrichment_result(df):
    description = df['Description']
    df.drop('Description', level = 0, axis = 1, inplace = True )
    parsed_df = pd.DataFrame({'Description':description})
    filtered_simplified_df = pd.DataFrame({'Description':description})

    for dataset, df_d in df.groupby(level = 0, axis = 1):

        for alg, df_a in df_d.groupby(level=1, axis = 1):

            df_a.columns = df_a.columns.droplevel([0,1])

            df_a['pvalue']=df_a['pvalue'].fillna(1)
            df_a['geneID'] = df_a['geneID'].fillna('/') #fix it later

            if 'geneID' not in parsed_df.columns:
                parsed_df['geneID'] = df_a['geneID']
                parsed_df['geneName'] = df_a['geneName']
            else:
                parsed_df['geneID'] =parsed_df['geneID'].astype(str) +'/'+ df_a['geneID']
                parsed_df['geneName'] =parsed_df['geneName'].astype(str) +'/'+ df_a['geneName']

            if 'weight' not in parsed_df.columns:
                parsed_df['weight'] = df_a['pvalue']

            else:
                parsed_df['weight'] =parsed_df['weight']*df_a['pvalue']


            if(alg !='-'):
                pval_col = alg+'_'+'pvalue'
                BgRatio_col = alg+'_'+'BgRatio'
                GeneRatio_col = alg+'_'+'GeneRatio'
                qvalRatio_col = alg+'_'+'-(log(qvalue '+alg+')- log(qvalue Krogan))'
                parsed_df[qvalRatio_col] = df_a['-(log(qvalue '+alg+')- log(qvalue Krogan))']

            else:
                pval_col = dataset+'_'+'pvalue'
                BgRatio_col = dataset+'_'+'BgRatio'
                GeneRatio_col = dataset+'_'+'GeneRatio'

            parsed_df[pval_col] = df_a['pvalue']
            parsed_df[BgRatio_col] = df_a['BgRatio']
            parsed_df[GeneRatio_col] = df_a['GeneRatio']

            filtered_simplified_df[pval_col] = df_a['pvalue']



    parsed_df['geneID'] = parsed_df['geneID'].astype(str).apply(lambda x: (set(x.split('/'))))

    # create protein universe
    prot_universe = set()
    for term, geneID_set in parsed_df['geneID'].items():
        prot_universe = prot_universe.union(geneID_set)


    uncovered_prot_universe = prot_universe
    terms_to_keep=[]
    simple_df = parsed_df.copy()

    while len(uncovered_prot_universe)!= 0:

        parsed_df['uncovered_protein_being_covered'] = parsed_df['geneID'].apply(lambda x: len(x.intersection(uncovered_prot_universe)))
        parsed_df['ratio_weight_uncovered_protein_being_covered'] = parsed_df['weight']/parsed_df['uncovered_protein_being_covered']
        min_idx = (parsed_df[['ratio_weight_uncovered_protein_being_covered']].idxmin()).iat[0]
        terms_to_keep.append(min_idx)
        uncovered_prot_universe = uncovered_prot_universe - parsed_df.at[min_idx,'geneID']
        parsed_df.drop(min_idx, axis = 0, inplace=True)
        # print('length: ', len(uncovered_prot_universe))

    simple_df = simple_df[simple_df.index.isin(terms_to_keep)]
    filtered_simplified_df = filtered_simplified_df[filtered_simplified_df.index.isin(terms_to_keep)]

    # order accrding to selection
    simple_df = simple_df.reindex(terms_to_keep)
    filtered_simplified_df = filtered_simplified_df.reindex(terms_to_keep)

    #correctedness checking
    # prot_universe_new=set()
    # for term, geneID_set in simple_df['geneID'].items():
    #     prot_universe_new = prot_universe_new.union(geneID_set)


    # print('simple_df: ',simple_df)
    # print('diff bewteen actual and calculated prot universe: ', len(prot_universe-prot_universe_new))
    # print('terms to keep',terms_to_keep)
    # print('simple_df index: ', simple_df.index)
    # print('terms to keep: ', terms_to_keep)
    # print('simple_df shape: ', simple_df.shape)
    return filtered_simplified_df, simple_df

def plot_heatmap(df,file_path):
    df.reset_index(inplace = True)
    df.set_index(['index','Description'],inplace=True)
    print(df)
    sns_plot = sns.heatmap(df)
    fig_num = plt.gcf().number
    figure = plt.figure(fig_num,figsize = (10,20))
    figure.savefig(file_path)


def main(config_map, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """
    # extract the general variables from the config map
    input_settings, input_dir, output_dir, alg_settings, kwargs \
        = config_utils.setup_config_variables(config_map, **kwargs)
    algs = config_utils.get_algs_to_run(alg_settings, **kwargs)
    pval_cutoff = kwargs.get('pval_cutoff')
    qval_cutoff = kwargs.get('qval_cutoff')


    if kwargs.get('compare_krogan_terms'):
        krogan_dir = kwargs['compare_krogan_terms']

    print("algs: %s" % (str(algs)))
    del kwargs['algs']

    # load the namespace mappings
    uniprot_to_gene = None
    gene_to_uniprot = None
    # if kwargs.get('id_mapping_file'):
    uniprot_to_gene = enrichment.load_gene_names(kwargs.get('id_mapping_file'))
    kwargs['uniprot_to_gene'] = uniprot_to_gene

    gene_to_uniprot = enrichment.load_uniprot(kwargs.get('id_mapping_file'))
    kwargs['gene_to_uniprot'] = gene_to_uniprot


    # store all the enriched terms in a single dataframe
    all_dfs = {g: pd.DataFrame() for g in ['BP', 'CC', 'MF']}
    all_dfs_KEGG = pd.DataFrame()
    all_dfs_reactome = pd.DataFrame()

    terms_to_keep_GO = {g: [] for g in ['BP', 'CC', 'MF']}
    pathways_to_keep_KEGG=[]
    pathways_to_keep_Reactome = []


    num_algs_with_results = 0
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
            num_algs_with_results += 1
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
                    # print('fss_pval: ' , kwargs.get('fss_pval'))
                    # print('fss_pval type : ', type(kwargs.get('fss_pval')))
                    terms_to_keep_GO[ont] = terms_to_keep_GO[ont] + list(df[df['pvalue']<=kwargs.get('fss_pval')]['ID'])
                    df = add_qval_ratio(df,ont, krogan_dir,alg)
                    tuples = [(dataset_name, alg, col) for col in df.columns]
                    index = pd.MultiIndex.from_tuples(tuples)
                    df.columns = index
                    all_dfs[ont] = pd.concat([all_dfs[ont], df], axis=1)


                KEGG_df = enrichment.run_clusterProfiler_KEGG(topk_predictions, out_dir, prot_universe=prot_universe, forced=kwargs.get('force_run'), **kwargs)
                KEGG_df = add_qval_ratio(KEGG_df,'KEGG',krogan_dir,alg)
                pathways_to_keep_KEGG = pathways_to_keep_KEGG + list(KEGG_df[KEGG_df['pvalue']<=kwargs.get('fss_pval')]['ID'])
                tuples = [(dataset_name, alg, col) for col in KEGG_df.columns]
                index = pd.MultiIndex.from_tuples(tuples)
                KEGG_df.columns = index
                all_dfs_KEGG = pd.concat([all_dfs_KEGG, KEGG_df], axis=1)



                reactome_df = enrichment.run_ReactomePA_Reactome(topk_predictions, out_dir, prot_universe=prot_universe, forced=kwargs.get('force_run'),**kwargs)
                reactome_df = add_qval_ratio(reactome_df,'Reactome',krogan_dir,alg)
                pathways_to_keep_Reactome = pathways_to_keep_Reactome + list(reactome_df[reactome_df['pvalue']<=kwargs.get('fss_pval')]['ID'])
                tuples = [(dataset_name, alg, col) for col in reactome_df.columns]
                index = pd.MultiIndex.from_tuples(tuples)
                reactome_df.columns = index
                all_dfs_reactome = pd.concat([all_dfs_reactome, reactome_df], axis=1)


    for geneset, g_df in all_dfs.items():
        all_dfs[geneset] = g_df[g_df.index.isin(terms_to_keep_GO[geneset])]
    all_dfs_KEGG = all_dfs_KEGG[all_dfs_KEGG.index.isin(pathways_to_keep_KEGG)]
    all_dfs_reactome = all_dfs_reactome[all_dfs_reactome.index.isin(pathways_to_keep_Reactome)]


    if num_algs_with_results == 0:
        print("No results found. Quitting")
        sys.exit()

    if kwargs.get('compare_krogan_terms'):
        for geneset, g_df in all_dfs.items():
            df = include_Krogan_enrichment_result(krogan_dir,geneset,g_df)
            all_dfs[geneset] = pd.concat([all_dfs[geneset], df], axis=1)

        kegg_df = include_Krogan_enrichment_result(krogan_dir,'KEGG',all_dfs_KEGG)
        all_dfs_KEGG = pd.concat([all_dfs_KEGG, kegg_df], axis=1)

        reactome_df = include_Krogan_enrichment_result(krogan_dir,'Reactome',all_dfs_reactome)
        all_dfs_reactome = pd.concat([all_dfs_reactome, reactome_df], axis=1)


    # now write the combined df to a file

    out_pref = kwargs.get('out_pref')
    if out_pref is None:
        pval_str = str(kwargs.get('pval_cutoff')).replace('.','_')
        out_pref = "%s/enrichment/combined%s-%s/%s-" % (
            output_dir, "-krogan" if kwargs.get('compare_krogan_terms') else "",
            pval_str, os.path.basename(kwargs['config']).split('.')[0])

    super_combined_file = "%s-k%s.xlsx" % (out_pref,k_to_test[0])
    super_combined_df = pd.DataFrame()
    combined_simplified_file = "%s-k%s-simplified.csv" % (out_pref,k_to_test[0])
    super_combined_simplified_file = "%s-k%s-super_combined_simplified.csv" % (out_pref,k_to_test[0])
    heatmap_file = "%s-k%s-super_combined_simplified.png" % (out_pref,k_to_test[0])
    combined_simplified_df = pd.DataFrame()

    #write combined KEGG Enrichment

    if kwargs.get('file_per_alg'):
        all_dfs_KEGG = all_dfs_KEGG.swaplevel(0,1,axis=1)
        for alg, df_alg in all_dfs_KEGG.groupby(level=0, axis=1):
            df_alg.dropna(how='all', inplace=True)
            # print('KEGG FILE PER ALGO')
            out_file = "%s%s-k%s-KEGG.csv" % (out_pref, alg, k_to_test[0])
            write_combined_table(df_alg, out_file, dataset_level=1)


    else:
        out_file = "%sk%s-KEGG.csv" % (out_pref, k_to_test[0])
        print('KEGG ALL')
        super_combined_df = pd.concat([super_combined_df, all_dfs_KEGG],axis=0)
        print('super_combined_df: ', super_combined_df.columns.values)

        processed_df = write_combined_table(all_dfs_KEGG, out_file, dataset_level=0)
        with pd.ExcelWriter(super_combined_file) as writer:
            # print('processed_df: ', processed_df.shape)
            processed_df.to_excel(writer, sheet_name = 'KEGG')
        filtered_simplified_df,simplified_df = simplify_enrichment_result(processed_df)
        combined_simplified_df = pd.concat([combined_simplified_df,simplified_df ], axis=0)


    #write combined Reactome Enrichment
    if kwargs.get('file_per_alg'):
        all_dfs_reactome = all_dfs_reactome.swaplevel(0,1,axis=1)
        for alg, df_alg in all_dfs_reactome.groupby(level=0, axis=1):
            df_alg.dropna(how='all', inplace=True)
            # print('REACTOME FILE PER ALGO')
            out_file = "%s%s-k%s-Reactome.csv" % (out_pref, alg, k_to_test[0])
            write_combined_table(df_alg, out_file,dataset_level=1)

    else:
        out_file = "%sk%s-Reactome.csv" % (out_pref, k_to_test[0])
        # print('REACTOME ALL')
        super_combined_df = pd.concat([super_combined_df, all_dfs_reactome],axis=0)
        processed_df = write_combined_table(all_dfs_reactome, out_file, dataset_level=0)
        with pd.ExcelWriter(super_combined_file, mode ='a') as writer:
            # print('processed_df: ', processed_df.shape)
            processed_df.to_excel(writer, sheet_name = 'Reactome')
        filtered_simplified_df,simplified_df = simplify_enrichment_result(processed_df)
        combined_simplified_df = pd.concat([combined_simplified_df,simplified_df ], axis=0)

    #write GO enrichment
    for geneset, df in all_dfs.items():
        if kwargs.get('file_per_alg'):
            df = df.swaplevel(0,1,axis=1)
            for alg, df_alg in df.groupby(level=0, axis=1):
                df_alg.dropna(how='all', inplace=True)
                # TODO add back the krogan terms
                #if kwargs.get('compare_krogan_terms') and :
                # print(df_alg.head())
                out_file = "%s%s-k%s-%s.csv" % (out_pref, alg, k_to_test[0], geneset)
                write_combined_table(df_alg, out_file, dataset_level=1)
        else:
            out_file = "%sk%s-%s.csv" % (out_pref, k_to_test[0], geneset)
            super_combined_df = pd.concat([super_combined_df, df],axis=0)
            processed_df= write_combined_table(df, out_file,dataset_level=0)
            with pd.ExcelWriter(super_combined_file, mode ='a') as writer:
                print('processed_df: ', processed_df.shape)
                processed_df.to_excel(writer, sheet_name = 'GO-'+ geneset)
            if geneset == 'MF':
                continue
            filtered_simplified_df,simplified_df = simplify_enrichment_result(processed_df)
            combined_simplified_df = pd.concat([combined_simplified_df,simplified_df ], axis=0)
    combined_simplified_df.to_csv(combined_simplified_file)

    filtered_super_combined_simplified_df, super_combined_simplified_df = simplify_enrichment_result(process_df(super_combined_df))
    # plot_heatmap(filtered_super_combined_simplified_df,heatmap_file)
    super_combined_simplified_df.to_csv(super_combined_simplified_file)


# Write each dataframe to a different worksheet.
# df1.to_excel(writer, sheet_name='Sheet1')
# df2.to_excel(writer, sheet_name='Sheet2')
def process_df(df, dataset_level=0):
    id_to_name = {}
    id_counts = defaultdict(int)
    keep_indices = []
    for dataset, df_d in df.groupby(level=dataset_level, axis=1):

        # print('DATASET:', dataset)
        df_d.columns = df_d.columns.droplevel([0,1])
        df_d.dropna(how='all', inplace=True)
        if isinstance ((df_d['Description']), pd.core.frame.DataFrame):
            description = df_d['Description'].iloc[:, 0]
            for i in range (1, len(df_d['Description'].columns), 1):
                description = description.fillna(df_d['Description'].iloc[:, i])
        else:
            description = df_d['Description']

        id_to_name.update(dict(zip(df_d.index, description)))

    df.insert(0, 'Description', pd.Series(id_to_name))
    df.drop(['ID','Description','p.adjust', 'Count'], axis=1, level=2, inplace=True)
    return df

def write_combined_table(df, out_file, dataset_level=0):
    """
    """
    # for each term ID, store its name
    # id_to_name = {}
    # id_counts = defaultdict(int)
    # keep_indices = []
    # for dataset, df_d in df.groupby(level=dataset_level, axis=1):
    #
    #     # print('DATASET:', dataset)
    #     df_d.columns = df_d.columns.droplevel([0,1])
    #     df_d.dropna(how='all', inplace=True)
    #     if isinstance ((df_d['Description']), pd.core.frame.DataFrame):
    #         description = df_d['Description'].iloc[:, 0]
    #         for i in range (1, len(df_d['Description'].columns), 1):
    #             description = description.fillna(df_d['Description'].iloc[:, i])
    #     else:
    #         description = df_d['Description']
    #
    #     id_to_name.update(dict(zip(df_d.index, description)))
    #
    # df.insert(0, 'Description', pd.Series(id_to_name))
    # df.drop(['ID','Description','p.adjust', 'Count'], axis=1, level=2, inplace=True)
    # # print(df.head())

    df = process_df(df,dataset_level)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    print("writing %s" % (out_file))
    df.to_csv(out_file, sep=',')
    return df


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)
