"""
Murali's attempt to simplify enrichment results. This script is a one-off hack not integrated into the overall pipeline.
"""

import argparse
import yaml
from collections import defaultdict
import itertools
import os
import sys
#from tqdm import tqdm
import copy
import time
#import numpy as np
#from scipy import sparse
import pandas as pd
import numpy as np
# for median
import statistics

sys.path.insert(1, '/home/tasnina/SARS-CoV-2-network-analysis')
import enrichment


def reformat_enrichment_data(df):
    """
    Reformat the data frame containing enrichment results: for each term, get the gene set and the four numbers that are input to Fisher's exact test.
    """
    # convert gene ids into set of genes.
    df['GeneSet'] = df['geneID'].astype(str).apply(lambda x: (set(x.split('/'))))
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
#    print(df['BgRatioDenominator'])
    return(df)

def simplify_enriched_terms(df, min_odds_ratio = 1):
    """
    df: DataFrame containing all the enrichment information.
    min_odds_ratio: Stop selecting additional terms once the maximum odds ratio falls below this threshold.

    Use a greedy-set-cover-like algorithm to compute a small set of terms that cover all the proteins annotated by the terms in df. Here, we are considering only the proteins that also belong to the set for which we ran enrichment, e.g., top-k GeneManiaPlus predictions.

    In each iteration, the idea is to pick the term that maximises the odds ratio, which is the (GeneRatioNumerator/GeneRatioDenominator)/(BgRatioNumerator/BgRatioDenominator). After picking the term, we update the GeneSet for each other term by deleting all the genes annotated to the selected term. We also update GeneRatioNumerator and OddsRatio.
    """

    selected_terms = []
    #while (len(df['Description'])):
    while (True):
        # pick the term with the largest odds ratio.
        max_index = df['OddsRatio'].idxmax()
        # return if OddsRatio is < min_odds_ratio.
        if (df['OddsRatio'][max_index] < min_odds_ratio):
            return(selected_terms)
#        selected_terms.append(df['Description'][max_index])
        selected_terms.append(max_index)
        #print("Selected %s with odds ratio %f " % (df['Description'][max_index], df['OddsRatio'][max_index]))
        gene_set_covered = df['GeneSet'][max_index]
        # Update GeneSet for every term by removing all the genes covered by the selected term.
        df['GeneSet'] = df['GeneSet'].apply(lambda x: x.difference(gene_set_covered))
        # Update GeneRatioNumerator. It is just the number of genes in GeneSet (since they have not been covered by any already selected term)
        df['GeneRatioNumerator'] = df['GeneSet'].apply(lambda x: len(x))
        # Update OddsRatio

        df['OddsRatio'] = (df['GeneRatioNumerator']/df['GeneRatioDenominator'])/(df['BgRatioNumerator']/df['BgRatioDenominator'])

        # remove this term. I have commented out this line because I want to store the max_index above rather than the value in the Description column. If I store the Description column, then I have to map back to the index later, which I am not sure exactly how to do. Now if I store the max_index and also drop this row, then the rest of the indices are out of whack with the indices of df in the calling context, so I will access the wrong terms. I am safe in not dropping the row since this term's OddsRatio should be zero.
#        df.drop(max_index, axis = 0, inplace=True)

    return(selected_terms)

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
    # print(combined_df.shape)
    # print(combined_df.columns)
    # print(combined_df.head())

    while (True):
        # pick the term with the largest odds ratio.
        max_index = combined_df['OddsRatio'].idxmax()
        # return if OddsRatio is < min_odds_ratio.
        if (combined_df['OddsRatio'][max_index] < min_odds_ratio):
            return(selected_terms)
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
            # print(gene_set_covered)
            # Update GeneSet for every term by removing all the genes covered by the selected term.
            df['GeneSet'] = df['GeneSet'].apply(lambda x: x.difference(gene_set_covered))
            # Update GeneRatioNumerator. It is just the number of genes in GeneSet (since they have not been covered by any already selected term)
            df['GeneRatioNumerator'] = df['GeneSet'].apply(lambda x: len(x))
            # Update OddsRatio

            df['OddsRatio'] = (df['GeneRatioNumerator']/df['GeneRatioDenominator'])/(df['BgRatioNumerator']/df['BgRatioDenominator'])


        # now update OddsRatio in combined_df. since the indices do not match in dfs and combined_df, the safest thing to do is to repeat the concat and groupby.
        combined_df = pd.concat(dfs)
        combined_df = combined_df[['Description', 'OddsRatio']].groupby('Description', as_index=False).prod()

    return(selected_terms)

def compare_selected_terms(df, terms):
    """
    For all pairs of terms, compare the overlap of their gene sets. The goal of this Python function is to verify that the terms are not too similar.
    """
    all_pairs = itertools.combinations(terms, 2)
    all_pairs_jaccard_coeffs = []
    for (fn1, fn2) in all_pairs:
        # compare gene_set with
        # gs1 = df['GeneSet'][[df['Description'] == fn1].index]
        # gs2 = df['GeneSet'][[df['Description'] == fn2].index]
        gs1 = df['GeneSet'][fn1]
        gs2 = df['GeneSet'][fn2]
        # print("Function %s has size %d" % (fn1, len(gs1)) )
        # print("Function %s has size %d" % (fn2, len(gs2)) )
        jc = len(gs1.intersection(gs2))*1.0/len(gs1.union(gs2))
        all_pairs_jaccard_coeffs.append(jc)

    print("Maximum Jaccard coefficient is %f" % (max(all_pairs_jaccard_coeffs)))
    print("Median Jaccard coefficient is %f" % (statistics.median(all_pairs_jaccard_coeffs)))
    print("Mean Jaccard coefficient is %f" % (statistics.mean(all_pairs_jaccard_coeffs)))
#    print("Quantiles of Jaccard coefficient is %f" % (statistics.quantiles(all_pairs_jaccard_coeffs)))



def main():

    # should be a command-line option.
    # define files.
    out_dir = 'outputs/enrichment/murali'
    go_bp_files = {
        'krogan': 'outputs/enrichment/krogan/p1_0/enrich-BP-1_0.csv',

        'genemaniaplus': 'outputs/enrichment/networks/stringv11/400/2020-03-sarscov2-human-ppi-ace2/GM+/pred-scores-a0_01-tol1e-05-filtered-p0_05/enrich-BP-1_0.csv',
        'svm': 'outputs/enrichment/networks/stringv11/400/2020-03-sarscov2-human-ppi-ace2/SVM-rep100-nf5/pred-scores-rep100-nf5-svm-maxi1000-filtered-p0_05/enrich-BP-1_0.csv'
    }
    # should be a command-line option.
    #min_odds_ratio = 1
    min_odds_ratio = 5


    go_bp_enrichment = {}
    selected_terms = {}
    # read files.
    for alg in go_bp_files:
        # the first two lines seem useless so header = 2
        # the first two columns do not have a header. the third column specifies the term/pathway id, so using it as the column for indexing.
        go_bp_enrichment[alg] = pd.read_csv(go_bp_files[alg])
        print(go_bp_enrichment[alg].columns.values)


    # reformat data frames.
    for alg in go_bp_enrichment:
        reformat_enrichment_data(go_bp_enrichment[alg])

    # simplify.

    for alg in go_bp_enrichment:
        # trial run. simplify for each algorithm separately.
        print("Algorithm %s has %d terms before selection." %(alg, len(go_bp_enrichment[alg]['Description'])))
        selected_terms[alg] = simplify_enriched_terms(go_bp_enrichment[alg].copy(), min_odds_ratio)
        print("Algorithm %s has %d selected terms." %(alg, len(selected_terms[alg])))

    # now try all algorithms together
    combined_selected_terms = simplify_enriched_terms_multiple(copy.deepcopy(list(go_bp_enrichment.values())), min_odds_ratio)
    # combined_selected_terms.to_csv(out_dir+'/combined_simplified_file.csv')

    # compare similarity of selected terms as a gut check. I need to combine the dfs into one with a GeneSet that is the union of the individual gene sets.
    combined_df = pd.concat(go_bp_enrichment.values())
    print("Combined data frame includes %d frames" % (len(go_bp_enrichment.values())))
    # Got this trick for taking set unions with agg from https://stackoverflow.com/questions/32967201/how-to-concat-sets-when-using-groupby-in-pandas-dataframe
    combined_df = combined_df[['Description', 'GeneSet']].groupby('Description').agg({'GeneSet': lambda x: set.union(*x)})#.reset_index('Description')
    compare_selected_terms(combined_df, combined_selected_terms)


if __name__ == "__main__":
    # config_map, kwargs = parse_args()
    # main(config_map, **kwargs)
    main()
