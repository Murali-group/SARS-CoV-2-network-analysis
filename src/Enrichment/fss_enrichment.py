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
from collections import Counter
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import hypergeom
import itertools
import statistics
import scipy.stats as stats

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
    parser = argparse.ArgumentParser(description="Script to test for enrichment of the top predictions among given genesets. ")

    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, required=True,
                       help="Configuration file used when running FSS. " +
                       "Must have a 'genesets_to_test' section for this script. ")
    group.add_argument('--id-mapping-file', type=str, default="datasets/mappings/human/uniprot-reviewed-status.tab.gz",
                       help="Table downloaded from UniProt to map to gene names. Expected columns: 'Entry', 'Gene names', 'Protein names'")
    group.add_argument('--compare-krogan-terms',type=str,
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


def reformat_enrichment_data(df):
    """
    Reformat the data frame containing enrichment results: for each term, get the gene set and the four numbers that are input to Fisher's exact test.
    """


    # convert gene ids into set of genes.
    df['GeneSet'] = df['geneID'].astype(str).apply(lambda x: (set(filter(None,x.split('/')))))
    # get numbers that go into the odds ratio. first make them a list.
    df['GeneRatioNumbers'] = df['GeneRatio'].astype(str).apply(lambda x: (x.split('/')))
    # Then get the first and second elements of the list.
    df['GeneRatioNumerator'] = df['GeneRatioNumbers'].apply(lambda x: int(x[0]))
    #    print(df['GeneRatioNumerator'])
    df['GeneRatioDenominator'] = df['GeneRatioNumbers'].apply(lambda x: int(x[1]))
    # repeat for BgRatio
    df['BgRatioNumbers'] = df['BgRatio'].astype(str).apply(lambda x: (x.split('/')))
    # Then get the first and second elements of the list.
    df['BgRatioNumerator'] = df['BgRatioNumbers'].apply(lambda x: int(x[0]))
    df['BgRatioDenominator'] = df['BgRatioNumbers'].apply(lambda x: int(x[1]))
    df['OddsRatio'] = (df['GeneRatioNumerator']/df['GeneRatioDenominator'])/(df['BgRatioNumerator']/df['BgRatioDenominator'])

    return  df

def simplify_enriched_terms_multiple(dfs, min_odds_ratio = 1):
    """
    dfs: a list of DataFrames containing the enrichment information for different sets of proteins.
    min_odds_ratio: Stop selecting additional terms once the maximum odds ratio falls below this threshold.

    Use a greedy-set-cover-like algorithm to compute a small set of terms that cover all the proteins annotated by the terms across the dfs.
    Here, we are considering only the proteins that also belong to the sets for which we ran enrichment, e.g., top-k GeneManiaPlus predictions and top-k SVM predictions.

    In each iteration, the idea is to pick the term that maximises a composite odds ratio. After picking the term,
    we update the GeneSet for each term in each df by deleting all the genes annotated to the selected term. We also update GeneRatioNumerator, OddsRatio, and CompositeOddsRatio.

    The odds ratio for a term in one df is (GeneRatioNumerator/GeneRatioDenominator)/(BgRatioNumerator/BgRatioDenominator).
    We define the composite odds ratio of a term to be the product across dfs of the odds ratios for the term.
    """

    selected_terms = []
    # compute product of OddsRatio values in dfs. Use concat followed by groupby.
    combined_df = pd.concat(dfs)
    combined_df = combined_df[['Description', 'OddsRatio']].groupby('Description', as_index=False).prod()

    prod_pvalue_for_selected_terms = []
    total_covered_prots = set()

    while (True):
        # pick the term with the largest odds ratio.
        max_index = combined_df['OddsRatio'].idxmax()
        # return if OddsRatio is < min_odds_ratio.
        if (combined_df['OddsRatio'][max_index] < min_odds_ratio):
            print('total_covered_prot: ',len(total_covered_prots))
            return len(total_covered_prots),selected_terms

        selected_term = combined_df['Description'][max_index]
        selected_terms.append(selected_term)
        print("Selected %s with odds ratio %f " % (selected_term, combined_df['OddsRatio'][max_index]))

        for df in dfs:
            # now update each term in each df. We have to find where seleced_term is located.
            term_row = df[df['Description'] == selected_term]

            if (term_row.shape[0] == 0):
                # a df may not contain a row for selected_term
                continue
            # print(term_row)
            gene_set_covered = term_row['GeneSet'].values[0]
            total_covered_prots = total_covered_prots.union(gene_set_covered)
            df['GeneSet'] = df['GeneSet'].apply(lambda x: x.difference(gene_set_covered))
            # Update GeneRatioNumerator. It is just the number of genes in GeneSet (since they have not been covered by any already selected term)
            df['GeneRatioNumerator'] = df['GeneSet'].apply(lambda x: len(x))
            # Update OddsRatio

            df['OddsRatio'] = (df['GeneRatioNumerator']/df['GeneRatioDenominator'])/(df['BgRatioNumerator']/df['BgRatioDenominator'])

            # print(df[df['Description']=='cellular respiration'][['GeneRatioNumerator','GeneSet']])

        # now update OddsRatio in combined_df. since the indices do not match in dfs and combined_df, the safest thing to do is to repeat the concat and groupby.
        combined_df = pd.concat(dfs)
        combined_df = combined_df[['Description', 'OddsRatio']].groupby('Description', as_index=False).prod()


def compare_selected_terms(df, terms):
    """
    For all pairs of terms, compare the overlap of their gene sets. The goal of this Python function is to verify that the terms are not too similar.
    """

    # protein sets separately and together
    protein_set_names = {}
    geneID_cols = [col for col in df.columns if 'geneID' in col ]
    for geneID_col in geneID_cols:
        protein_set_name = geneID_col.replace('_geneID','')
        # the column for all geneID's together is 'geneID' which whose name will not be changed by the above line. to do so:
        protein_set_name = protein_set_name.replace('geneID','all')
        protein_set_names[geneID_col]=protein_set_name
    # print('PROT SET NAME',protein_set_names)

    all_pairs_jaccard_coeffs_df_dict={protein_set_name:pd.DataFrame() for protein_set_name in protein_set_names.values()}
    all_pairs_jaccard_coeffs_dict = {protein_set_name:[] for protein_set_name in protein_set_names }

    for geneID_col in geneID_cols:
        description1_list = []
        description2_list = []
        all_pairs_jaccard_coeffs = []
        gs1_list =[]
        gs2_list=[]
        intersection_list = []
        t1_and_t2_by_t1_list = []

        protein_set_name = protein_set_names[geneID_col]
        all_pairs = itertools.combinations(terms, 2)
        # count = 0
        for (fn1, fn2) in all_pairs:

            if(geneID_col !='geneID'):
                p_adjust_col = geneID_col.replace('_geneID','_p.adjust')
                if(df.at[fn1,p_adjust_col]>0.01 or df.at[fn2,p_adjust_col]>0.01):
                    continue
            description1_list.append(df.at[fn1,'Description'])
            description2_list.append(df.at[fn2,'Description'])

            gs1 = df.at[fn1, geneID_col]
            gs2 = df.at[fn2 , geneID_col]

            if len(gs1.union(gs2))!=0:
                jc = len(gs1.intersection(gs2))*1.0/len(gs1.union(gs2))
            else:
                jc = 0

            all_pairs_jaccard_coeffs.append(jc)

            if len(gs1)!=0:
                t1_and_t2_by_t1_list.append(len(gs1.intersection(gs2))/len(gs1))
            else:
                t1_and_t2_by_t1_list.append(0)

            intersection_list.append(len(gs1.intersection(gs2)))
            gs1_list.append(len(gs1))
            gs2_list.append(len(gs2))
        # print(protein_set_name +'  COUNT:',count)
        all_pairs_jaccard_coeffs_dict[protein_set_name] = all_pairs_jaccard_coeffs

        all_pairs_jaccard_coeffs_df_dict[protein_set_name]['intersection'] = pd.Series(intersection_list)
        all_pairs_jaccard_coeffs_df_dict[protein_set_name]['t1_and_t2_by_t1_list'] = pd.Series(t1_and_t2_by_t1_list)

        all_pairs_jaccard_coeffs_df_dict[protein_set_name]['Jaccard'] = pd.Series(all_pairs_jaccard_coeffs)

        all_pairs_jaccard_coeffs_df_dict[protein_set_name]['description_1'] = pd.Series(description1_list)
        all_pairs_jaccard_coeffs_df_dict[protein_set_name]['gs1'] = pd.Series(gs1_list)

        all_pairs_jaccard_coeffs_df_dict[protein_set_name]['description_2'] = pd.Series(description2_list)
        all_pairs_jaccard_coeffs_df_dict[protein_set_name]['gs2'] = pd.Series(gs2_list)

        # print(all_pairs_jaccard_coeffs_df_dict[protein_set_name].columns)

    return all_pairs_jaccard_coeffs_df_dict,all_pairs_jaccard_coeffs_dict

def overlap_stat(jaccard_dict_before, jaccard_dict_after):

    df_dict={}
    for protein_set_name in jaccard_before_dict:
        jaccard_before = jaccard_before_dict[protein_set_name]
        jaccard_after = jaccard_after_dict[protein_set_name]

        jaccard_before.sort(reverse=True)
        jaccard_before = jaccard_before[0:len(jaccard_after)]

        max_list = [max(jaccard_before), max(jaccard_after)]
        median_list = [statistics.median(jaccard_before),statistics.median(jaccard_after) ]
        mean_list = [statistics.mean(jaccard_before),statistics.mean(jaccard_after) ]

        df = pd.DataFrame(list(zip(max_list, median_list,mean_list)),
                   columns =['Jaccard_Max', 'Jaccard_Median','Jaccard_Mean'], index = ['Before', 'After'])
        print('Overlap stat: ',df)
        df_dict[protein_set_name] = df
    return df_dict

# def compute_composite_p_adjust_mean(df):
#     p_adjust_col = [col for col in df.columns if 'p.adjust' in col]
#     df = df[p_adjust_col]
#     mult = df.prod(axis = 1)
#     return statistics.mean(list(mult))

def jaccard_histogram(jaccard_df_dict_before,jaccard_df_dict_after,out_file_base):

    for protein_set_name in jaccard_df_dict_before:

        jaccard_before = jaccard_df_dict_before[protein_set_name]
        jaccard_after = jaccard_df_dict_after[protein_set_name]


        # print(protein_set_name)
        # print(jaccard_before.columns)
        # print(jaccard_after.columns)

        hist_data_before = []
        hist_data_after= []
        hist_data_before = jaccard_before.groupby('description_1')['Jaccard'].max()
        hist_data_after = jaccard_after.groupby('description_1')['Jaccard'].max()
        #
        # print('hist data before:' , hist_data_before)
        # print('hist data after:' , hist_data_after)


        bin_seq = np.linspace(0,1,11)
        # print(bin_seq)

        print(protein_set_name,'\nNumber of terms BEFORE: ', len(hist_data_before))
        print(protein_set_name,'\nNumber of terms AFTER: ', len(hist_data_after))

        hist_data_before = np.asarray(hist_data_before)
        hist_data_after = np.asarray(hist_data_after)

        # print(hist_data_before)
        # print( hist_data_before.size)

        # print('JACCARD values for paper: BEFORE: ', np.zeros_like(hist_data_before) + 1. / hist_data_before.size)
        # print('JACCARD values for paper: AFTER: ', np.zeros_like(hist_data_after) + 1. / hist_data_after.size)

        n, bins,patches = plt.hist(hist_data_before, weights=np.zeros_like(hist_data_before) + 1. / hist_data_before.size, bins = bin_seq, label = 'Before', alpha = 0.7)
        print(protein_set_name,'\nJACCARD values for paper: BEFORE: ', '\nn: ', n, '\nbins: ', bins)
        n, bins,patches = plt.hist(hist_data_after,  weights=np.zeros_like(hist_data_after) + 1. / hist_data_after.size, bins = bin_seq, label = 'After', alpha = 0.5)
        print(protein_set_name,'\nJACCARD values for paper: AFTER: ', '\nn: ', n, '\nbins: ', bins)

        # plt.hist(hist_data_before,bins = 0.1)
        plt.xlabel('Maximum Jaccard index for a term')
        plt.ylabel('Fraction of terms')

        plt.legend(loc='upper right')
        plt.savefig(out_file_base+'_'+protein_set_name+'_Jaccard_hist.pdf',format='pdf')

        plt.close()

def prot_coverage_histogram(df_before, df_after,out_file_base):

    geneID_cols = [col for col in df_before if 'geneID' in col]

    for geneID_col in geneID_cols:

        hist_data_before = []
        hist_data_after=[]

        prot_count_before = {}
        prot_count_after = {}
        protein_set_name = geneID_col.replace('_geneID','')
        # the column for all geneID's together is 'geneID' which whose name will not be changed by the above line. to do so:
        protein_set_name = protein_set_name.replace('geneID','all')

        # set prot_universe here
        prot_universe = set()
        for term, geneID_set in df_before[geneID_col].items():
            prot_universe = prot_universe.union(geneID_set)

        for prot in prot_universe:
            prot_count_before[prot] = (df_before[geneID_col].apply(lambda x: 1 if prot in x else 0 )).sum()
            hist_data_before.append(prot_count_before[prot])

            prot_count_after[prot] = (df_after[geneID_col].apply(lambda x: 1 if prot in x else 0 )).sum()
            hist_data_after.append(prot_count_after[prot])

        bin_size = 5
        # bin_seq = [0,1,6,11,..] Added this 0 to 1 range to keep track of uncovered protein in simplified terms.
        bin_seq = [0]+list(range(1, max(hist_data_before)+1, bin_size))
        # bin_seq = list(range(1, max(hist_data_before)+1, bin_size))
        if(bin_seq[-1]<max(hist_data_before)):
            bin_seq.append(max(hist_data_before))
        print('BIN SEQ: ', bin_seq)
        # if(protein_set_name == 'RL'):
        #     print(hist_data_before, max(hist_data_before))

        print(protein_set_name, 'considered proteins BEFORE:  ',len(hist_data_before))
        print(protein_set_name, 'considered proteins AFTER:  ',len(hist_data_after))

        print(protein_set_name, 'Gene Coverage Distribution BEFORE:  ',Counter(hist_data_before))
        print(protein_set_name, 'Gene Coverage Distribution AFTER:  ',Counter(hist_data_after))

        hist_data_before = np.asarray(hist_data_before)
        hist_data_after = np.asarray(hist_data_after)

        # plt.bar([0], [hist_data_after[0]], color='#557f2d', width=0.25, edgecolor='white', label='0 bar')

        n,bins,patches = plt.hist(hist_data_before, weights=np.zeros_like(hist_data_before) + 1. / hist_data_before.size, bins = bin_seq, label = 'Before', alpha = 0.7)
        print(protein_set_name, '\nGENE COVERAGE values for paper: BEFORE: ', '\nn: ', n, '\nbins: ', bins)
        n,bins,patches = plt.hist(hist_data_after,  weights=np.zeros_like(hist_data_after) + 1. / hist_data_after.size,bins = bin_seq, label = 'After', alpha = 0.5)
        print(protein_set_name,'\nGENE COVERAGE values for paper: AFTER: ', '\nn: ', n, '\nbins: ', bins)

        # plt.hist(hist_data_before,bins = 0.1)
        plt.xlabel('Number of terms covering a protein')
        plt.ylabel('Fraction of proteins')

        plt.legend(loc='upper right')
        plt.savefig(out_file_base+'_'+protein_set_name+'_gene_coverage_hist.pdf',format='pdf')

        plt.close()

def write_uncovered_prot(df_before, df_after, out_file_base):
    geneID_cols = [col for col in df_before if 'geneID' in col]

    for geneID_col in geneID_cols:

        protein_set_name = geneID_col.replace('_geneID','')
        # the column for all geneID's together is 'geneID' which whose name will not be changed by the above line. to do so:
        protein_set_name = protein_set_name.replace('geneID','all')

        # set prot_universe here
        prot_universe_before = set()
        prot_universe_after = set()
        for term, geneID_set in df_before[geneID_col].items():
            prot_universe_before = prot_universe_before.union(geneID_set)
        for term, geneID_set in df_after[geneID_col].items():
            prot_universe_after = prot_universe_after.union(geneID_set)

        out_file= out_file_base+'_'+protein_set_name+'_'+'uncovered_prot.csv'
        uncovered_prot = prot_universe_before.difference(prot_universe_after)
        # print(uncovered_prot,'\n', list(uncovered_prot))
        uncovered_prot_df = pd.DataFrame(list(uncovered_prot))
        # print('Uncovered_prot_df', uncovered_prot_df.columns, uncovered_prot_df.shape)
        uncovered_prot_df.to_csv(out_file, sep = '\t',index=False,header=False)




def find_prot_universe(df):
    description = df['Description']
    df.drop('Description', level = 0, axis = 1, inplace = True)
    parsed_df = pd.DataFrame({'Description':description})
    # filtered_simplified_df = pd.DataFrame({'Description':description})

    for dataset, df_d in df.groupby(level = 0, axis = 1):

        for alg, df_a in df_d.groupby(level=1, axis = 1):

            df_a.columns = df_a.columns.droplevel([0,1])

            df_a['geneID'] = df_a['geneID'].fillna('/')
            if 'geneID' not in parsed_df.columns:
                parsed_df['geneID'] = df_a['geneID']
                parsed_df['geneName'] = df_a['geneName']
            else:
                parsed_df['geneID'] =parsed_df['geneID'].astype(str) +'/'+ df_a['geneID']
                parsed_df['geneName'] =parsed_df['geneName'].astype(str) +'/'+ df_a['geneName']


    parsed_df['geneID'] = parsed_df['geneID'].astype(str).apply(lambda x: (set(filter(None,x.split('/')))))

    # create protein universe
    prot_universe = set()
    for term, geneID_set in parsed_df['geneID'].items():
        prot_universe = prot_universe.union(geneID_set)

    return  prot_universe



def simplify_enrichment_greedy_algo(df,min_odds_ratio, out_file_base):

    prot_universe = find_prot_universe(df.copy())

    description = df['Description']
    df.drop('Description', level = 0, axis = 1, inplace = True )

    parsed_df = pd.DataFrame({'Description':description})


    df_a_dict={}

    for dataset, df_d in df.groupby(level = 0, axis = 1):

        for alg, df_a in df_d.groupby(level=1, axis = 1):

            df_a.columns = df_a.columns.droplevel([0,1])

            # add description column
            df_a['Description'] = description

            # fill nan values
            df_a['p.adjust']=df_a['p.adjust'].fillna(1)
            df_a['geneID'] = df_a['geneID'].fillna('/')

            # fill the nan values in GeneRatio and BgRatio columns
            df_temp = df_a.dropna()

            # print('inside simplified: ', df_temp, df_temp.iloc[0].at['GeneRatio'], type(df_temp.iloc[0].at['GeneRatio']))
            generatio_denominator = str(df_temp.iloc[0].at['GeneRatio']).split('/')[1]
            filler_generatio = '0'+'/'+ generatio_denominator


            # print('inside simplified: ', df_temp, df_temp.iloc[0].at['BgRatio'], type(df_temp.iloc[0].at['BgRatio']))
            bgratio_denominator = str(df_temp.iloc[0].at['BgRatio']).split('/')[1]
            # putting a dummy numerator=1 so that the odd ratio does not become nan. It does not affect the result as in such cases generatio will be 0 which makes oddratio=0
            filler_bgratio = '1'+'/'+bgratio_denominator

            df_a['GeneRatio'] = df_a['GeneRatio'].fillna(filler_generatio)
            df_a['BgRatio'] = df_a['BgRatio'].fillna(filler_bgratio)

            if 'geneID' not in parsed_df.columns:
                parsed_df['geneID'] = df_a['geneID']
                parsed_df['geneName'] = df_a['geneName']
            else:
                parsed_df['geneID'] =parsed_df['geneID'].astype(str) +'/'+ df_a['geneID']
                parsed_df['geneName'] =parsed_df['geneName'].astype(str) +'/'+ df_a['geneName']


            if(alg !='-'):
                adjust_pval_col = alg+'_'+'p.adjust'
                BgRatio_col = alg+'_'+'BgRatio'
                GeneRatio_col = alg+'_'+'GeneRatio'
                geneID_col = alg+'_'+'geneID'


            else:
                adjust_pval_col = dataset+'_'+'p.adjust'
                BgRatio_col = dataset+'_'+'BgRatio'
                GeneRatio_col = dataset+'_'+'GeneRatio'
                geneID_col = dataset+'_'+'geneID'

            parsed_df[adjust_pval_col] = df_a['p.adjust']
            parsed_df[BgRatio_col] = df_a['BgRatio']
            parsed_df[GeneRatio_col] = df_a['GeneRatio']
            parsed_df[geneID_col] =df_a['geneID'].astype(str).apply(lambda x: (set(filter(None,x.split('/')))))

            if(alg != '-'):
                df_a_dict[alg] = reformat_enrichment_data(df_a)
                print('\n\n df for simplification ', alg)
            else:
                df_a_dict[dataset] = reformat_enrichment_data(df_a)
                print('\n\n df for simplification ', dataset)

    parsed_df['geneID'] = parsed_df['geneID'].astype(str).apply(lambda x: (set(filter(None,x.split('/')))))

    simple_df = parsed_df.copy()

    number_of_prots_in_prediction_file = len(prot_universe)
    number_of_covered_prot, combined_selected_terms = simplify_enriched_terms_multiple(copy.deepcopy(list(df_a_dict.values())), min_odds_ratio)
    # print('Length of combined_selected_terms', len(combined_selected_terms))


    description_id_map = simple_df[['Description']].reset_index()
    description_id_map = description_id_map.set_index('Description')


    filtered_ordered_description_id_map = description_id_map[description_id_map.index.isin(combined_selected_terms)].reindex(combined_selected_terms)

    simple_df = simple_df[simple_df['Description'].isin (combined_selected_terms)].reindex(list(filtered_ordered_description_id_map['index']))

    print('greedy simplified terms: ', simple_df.shape)

    out_file_simplified = out_file_base+'_simplified.csv'
    simple_df.to_csv(out_file_simplified)

    # justification of simplifying method
    # print('parse_df', parsed_df['RL_geneID'])
    all_pairs_jaccard_coeffs_df_dict_before_simplification, all_pairs_jaccard_coeffs_dict_before_simplification= compare_selected_terms(parsed_df.copy(),list(parsed_df.index))
    all_pairs_jaccard_coeffs_df_dict_after_simplification, all_pairs_jaccard_coeffs_dict_after_simplification = compare_selected_terms(simple_df.copy(),list(filtered_ordered_description_id_map['index']))
    # comparing_jaccard_before_after_df_dict = overlap_stat(all_pairs_jaccard_coeffs_dict_before_simplification, all_pairs_jaccard_coeffs_dict_after_simplification )
    #
    # covered_prot_by_terms_before = number_of_prots_in_prediction_file/len(parsed_df)
    # covered_prot_by_terms_after = number_of_covered_prot/len(simple_df)
    #
    # covered_prot_by_terms_before_after_df = pd.DataFrame([covered_prot_by_terms_before,covered_prot_by_terms_after],columns = ['covered_prot_by_number_of_terms'],index = ['Before','After'])
    # print(out_file_base, covered_prot_by_terms_before_after_df)
    # #
    # compare_df = pd.concat([comparing_jaccard_before_after_df,covered_prot_by_terms_before_after_df],axis =1)
    #
    # out_file_overlap_stat = out_file_base + '_overlap_stat.csv'
    # compare_df.to_csv(out_file_overlap_stat)
    for prot_set in all_pairs_jaccard_coeffs_df_dict_before_simplification:
        all_pairs_jaccard_coeffs_df_dict_before_simplification[prot_set].to_csv(out_file_base + '_'+prot_set+'_before_jaccard.csv')

    for prot_set in all_pairs_jaccard_coeffs_df_dict_after_simplification:
        all_pairs_jaccard_coeffs_df_dict_after_simplification[prot_set].to_csv(out_file_base + '_'+prot_set+'_after_jaccard.csv')


    jaccard_histogram(all_pairs_jaccard_coeffs_df_dict_before_simplification.copy(),all_pairs_jaccard_coeffs_df_dict_after_simplification.copy(),out_file_base)
    prot_coverage_histogram(parsed_df.copy(), simple_df.copy(),out_file_base)
    write_uncovered_prot(parsed_df.copy(), simple_df.copy(),out_file_base)
    #overla
    # mean_composite_p_adjust_before = compute_composite_p_adjust_mean(parsed_df.copy())
    # mean_composite_p_adjust_after = compute_composite_p_adjust_mean(simple_df.copy())
    #
    # mean_composite_p_adjust_before_after_df = pd.DataFrame([mean_composite_p_adjust_before,mean_composite_p_adjust_after],columns = ['mean_composite_p.adjust'],index = ['Before','After'])
    #
    # compare_df = pd.concat([comparing_jaccard_before_after_df,mean_composite_p_adjust_before_after_df],axis =1)
    # compare_df.to_csv(out_file_jaccard.replace('simplified_Jaccard','')+'compare.csv')

    # all_pairs_jaccard_coeffs_df_before_simplification.to_csv(out_file_jaccard+'_before_simplification.csv')
    # all_pairs_jaccard_coeffs_df_after_simplification.to_csv(out_file_jaccard+'_after_simplification.csv')




    # return all_pairs_jaccard_coeffs_df_after_simplification, simple_df



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
    uniprot_to_gene = enrichment.load_gene_names(kwargs.get('id_mapping_file'))
    kwargs['uniprot_to_gene'] = uniprot_to_gene

    gene_to_uniprot = enrichment.load_uniprot(kwargs.get('id_mapping_file'))
    kwargs['gene_to_uniprot'] = gene_to_uniprot

    annotation_list = ['HIV_INTERACTION_PUBMED_ID', 'HIV_INTERACTION', 'HIV_INTERACTION_CATEGORY', 'UCSC_TFBS', 'UP_TISSUE', 'GAD_DISEASE']

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
        # print("\t%d prots in universe" % (len(prot_universe)))
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
            # print("\t%d original positive examples" % (len(orig_pos)))
            prot_universe = set(prots) | set(orig_pos)
            # print("\t%d prots in universe after adding them to the universe" % (len(prot_universe)))

        # now load the predictions, test at the various k values, and TODO plot
        k_to_test = get_k_to_test(dataset, **kwargs)
        # print("\ttesting %d k value(s): %s" % (len(k_to_test), ", ".join([str(k) for k in k_to_test])))

        # now load the prediction scores
        dataset_name = config_utils.get_dataset_name(dataset)
        alg_pred_files = config_utils.get_dataset_alg_prediction_files(
            output_dir, dataset, alg_settings, algs, **kwargs)
        for alg, pred_file in alg_pred_files.items():
            if not os.path.isfile(pred_file):
                print("Warning: %s not found. skipping" % (pred_file))
                continue
            num_algs_with_results += 1
            # print("reading: %s" % (pred_file))
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
                # print("writing %s" % (pred_filtered_file))
                df.to_csv(pred_filtered_file, sep='\t', index=None)

            for k in k_to_test:
                topk_predictions = list(df.iloc[:k]['prot'])

                # now run clusterProfiler from R
                out_dir = pred_filtered_file.split('.')[0] + '/'+ str(k)
                os.makedirs(os.path.dirname(out_dir), exist_ok=True)

                bp_df, mf_df, cc_df = enrichment.run_clusterProfiler_GO(
                    topk_predictions, out_dir, prot_universe=prot_universe, forced=kwargs.get('force_run'), **kwargs)
                for ont, df in [('BP', bp_df), ('MF', mf_df), ('CC', cc_df)]:
                    # make it into a multi-column-level dataframe
                    # print('fss_pval: ' , kwargs.get('fss_pval'))
                    # print('fss_pval type : ', type(kwargs.get('fss_pval')))
                    terms_to_keep_GO[ont] = terms_to_keep_GO[ont] + list(df[df['p.adjust']<=kwargs.get('fss_pval')]['ID'])
                    if kwargs.get('compare_krogan_terms'):
                        df = add_qval_ratio(df,ont, krogan_dir,alg)
                    tuples = [(dataset_name, alg, col) for col in df.columns]
                    index = pd.MultiIndex.from_tuples(tuples)
                    df.columns = index
                    all_dfs[ont] = pd.concat([all_dfs[ont], df], axis=1)


                KEGG_df = enrichment.run_clusterProfiler_KEGG(topk_predictions, out_dir, prot_universe=prot_universe, forced=kwargs.get('force_run'), **kwargs)
                if kwargs.get('compare_krogan_terms'):
                    KEGG_df = add_qval_ratio(KEGG_df,'KEGG',krogan_dir,alg)
                pathways_to_keep_KEGG = pathways_to_keep_KEGG + list(KEGG_df[KEGG_df['p.adjust']<=kwargs.get('fss_pval')]['ID'])
                tuples = [(dataset_name, alg, col) for col in KEGG_df.columns]
                index = pd.MultiIndex.from_tuples(tuples)
                KEGG_df.columns = index
                all_dfs_KEGG = pd.concat([all_dfs_KEGG, KEGG_df], axis=1)



                reactome_df = enrichment.run_ReactomePA_Reactome(topk_predictions, out_dir, prot_universe=prot_universe, forced=kwargs.get('force_run'),**kwargs)
                if kwargs.get('compare_krogan_terms'):
                    reactome_df = add_qval_ratio(reactome_df,'Reactome',krogan_dir,alg)
                pathways_to_keep_Reactome = pathways_to_keep_Reactome + list(reactome_df[reactome_df['p.adjust']<=kwargs.get('fss_pval')]['ID'])
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
        out_pref_dir = "%s/enrichment/combined%s-%s/" % (
            output_dir, "-krogan" if kwargs.get('compare_krogan_terms') else "",
            pval_str)

        network = os.path.basename(kwargs['config']).split('.')[0]

        out_pref = "%s/enrichment/combined%s-%s/%s-" % (
            output_dir, "-krogan" if kwargs.get('compare_krogan_terms') else "",
            pval_str, os.path.basename(kwargs['config']).split('.')[0])

    super_combined_file = "%s-k%s.xlsx" % (out_pref,k_to_test[0])
    super_combined_df = pd.DataFrame()

    #write combined KEGG Enrichment

    greedy_simplified_files_dir = out_pref_dir+'greedy_simplified/'

    os.makedirs(os.path.dirname(greedy_simplified_files_dir), exist_ok=True)

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

        processed_df = write_combined_table(all_dfs_KEGG, out_file, dataset_level=0)

        # write all the results from KEGG, GO, Reactome in one xlsx file.
        with pd.ExcelWriter(super_combined_file) as writer:
            # print('processed_df: ', processed_df.shape)
            processed_df.to_excel(writer, sheet_name = 'KEGG')

        # out_file_simplified = "%s_k%s_KEGG_simplified.csv" % (greedy_simplified_files_dir+network,k_to_test[0])
        # out_file_jaccard = "%s_k%s_KEGG_simplified_Jaccard" % (greedy_simplified_files_dir+network,k_to_test[0])

        out_file_base = "%s_k%s_KEGG" % (greedy_simplified_files_dir+network,k_to_test[0])
        simplify_enrichment_greedy_algo(processed_df.copy(),1,out_file_base)
        # greedy_simplified_df.to_csv(out_file_1)
        # all_pairs_jaccard_coeffs_df.to_csv(out_file_2)


    #write combined Reactome Enrichment
    if kwargs.get('file_per_alg'):
        all_dfs_reactome = all_dfs_reactome.swaplevel(0,1,axis=1)
        for alg, df_alg in all_dfs_reactome.groupby(level=0, axis=1):
            df_alg.dropna(how='all', inplace=True)
            # print('REACTOME FILE PER ALGO')
            out_file = "%s%s-k%s-Reactome.csv" % (out_pref, alg, k_to_test[0])
            write_combined_table(df_alg, out_file,dataset_level=1)

    else:
        print('REACTOME')
        out_file = "%sk%s-Reactome.csv" % (out_pref, k_to_test[0])
        # print('REACTOME ALL')
        # super_combined_df = pd.concat([super_combined_df, all_dfs_reactome],axis=0)
        processed_df = write_combined_table(all_dfs_reactome, out_file, dataset_level=0)
        with pd.ExcelWriter(super_combined_file, mode ='a') as writer:
            # print('processed_df: ', processed_df.shape)
            processed_df.to_excel(writer, sheet_name = 'Reactome')


        # out_file_simplified = "%s_k%s_Reactome_simplified.csv" % (greedy_simplified_files_dir+network,k_to_test[0])
        # out_file_jaccard = "%s_k%s_Reactome_simplified_Jaccard" % (greedy_simplified_files_dir+network,k_to_test[0])

        out_file_base = "%s_k%s_Reactome" % (greedy_simplified_files_dir+network,k_to_test[0])
        simplify_enrichment_greedy_algo(processed_df.copy(),1,out_file_base)
        # greedy_simplified_df.to_csv(out_file_1)
        # all_pairs_jaccard_coeffs_df.to_csv(out_file_2)

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
            print(geneset,'\n\n\n\n')
            out_file = "%sk%s-%s.csv" % (out_pref, k_to_test[0], geneset)
            # super_combined_df = pd.concat([super_combined_df, df],axis=0)
            processed_df= write_combined_table(df, out_file,dataset_level=0)
            with pd.ExcelWriter(super_combined_file, mode ='a') as writer:
                # print('processed_df: ', processed_df.shape)
                processed_df.to_excel(writer, sheet_name = 'GO-'+ geneset)
            if geneset == 'MF':
                continue


            # out_file_simplified = "%s_k%s_GO-%s_simplified.csv" % (greedy_simplified_files_dir+network,k_to_test[0],geneset)
            # out_file_jaccard = "%s_k%s_GO-%s_simplified_Jaccard" % (greedy_simplified_files_dir+network,k_to_test[0],geneset)

            out_file_base = "%s_k%s_GO-%s" % (greedy_simplified_files_dir+network,k_to_test[0],geneset)
            simplify_enrichment_greedy_algo(processed_df.copy(),1,out_file_base)
            # greedy_simplified_df.to_csv(out_file_1)
            # all_pairs_jaccard_coeffs_df.to_csv(out_file_2)




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
    df.drop(['ID','Description','Count','pvalue'], axis=1, level=2, inplace=True)
    return df

def write_combined_table(df, out_file, dataset_level=0):
    """
    """
    # for each term ID, store its name
    # id_to_name = {}
    # id_counts = defaultdict(int)
    # keep_indices = []

    df = process_df(df,dataset_level)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    print("writing %s" % (out_file))
    df.to_csv(out_file, sep=',')
    return df

def get_k_to_test(dataset, **kwargs):
    k_to_test = dataset['k_to_test'] if 'k_to_test' in dataset else kwargs.get('k_to_test', [])
    range_k_to_test = dataset['range_k_to_test'] if 'range_k_to_test' in dataset \
                        else kwargs.get('range_k_to_test')
    if range_k_to_test is not None:
        k_to_test += list(range(
            range_k_to_test[0], range_k_to_test[1], range_k_to_test[2]))
    # if nothing was set, use the default value
    if k_to_test is None or len(k_to_test) == 0:
        k_to_test = [100]
    return k_to_test

if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)
