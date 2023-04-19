import sys
import copy
#import numpy as np
import pandas as pd
import numpy as np
import itertools
import statistics
import scipy.stats as stats
from collections import defaultdict
#import subprocess
sys.path.insert(1, "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/")

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

def reformat_enrichment_data_v3(df):
    """
    Reformat the data frame containing enrichment results: for each term, get the gene set and the four numbers
    that are input to Fisher's exact test.
    """

    # convert gene ids into set of genes.
    # print(df["geneID"])

    df['GeneSet'] = df['geneID'].astype(str).apply(lambda x: (set(filter(None, x.split('/')))))
    # df['GeneSet'] = df['GeneSet'].apply(lambda x: x.discard('nan'))
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

    df['hypergeom_pval'] = pd.Series()


    for idx in df.index.values:
        # input to Fisher's exact test
        a = df.at[idx, 'GeneRatioNumerator']
        b = df.at[idx, 'GeneRatioDenominator'] - a
        c = df.at[idx, 'BgRatioNumerator'] - a
        d=  df.at[idx, 'BgRatioDenominator']-(a+b+c)

        if(a == 0):
             df.at[idx, 'hypergeom_pval'] = 1
        else:
            odd_ratio, df.at[idx, 'hypergeom_pval'] = oddsratio, pvalue = stats.fisher_exact([[a, b], [c, d]],'greater')
    return df


def simplify_enriched_terms_multiple_v3(dfs, prot_universe):
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
    total_covered_prots = set()
    hypergeom_pval_list = []
    # compute product of OddsRatio values in dfs. Use concat followed by groupby.
    combined_df = pd.concat(dfs)
    combined_df = combined_df[['Description', 'hypergeom_pval']].groupby('Description', as_index=False).prod()
    print('prot_universe', len(prot_universe))
    count=0
    while (True):
        # pick the term with the largest odds ratio.
        min_index = combined_df['hypergeom_pval'].idxmin()
        # hypergeom_pval_list.append(combined_df['hypergeom_pval'][min_index])
        # set another condition here
        print('covered_prot', len(total_covered_prots))
        n_uncovered = len(prot_universe.difference(total_covered_prots))
        if n_uncovered ==0:

            # print(dict(zip(selected_terms,hypergeom_pval_list)))
            return(selected_terms)

        selected_term = combined_df['Description'][min_index]
        selected_terms.append(selected_term)

        # print("Selected %s with pvalue%f " % (selected_term, combined_df['hypergeom_pval'][min_index]))

        for df in dfs:
            # now update each term in each df. We have to find where seleced_term is located.
            term_row = df[df['Description'] == selected_term]
            if (term_row.shape[0] == 0):
                # a df may not contain a row for selected_term
                continue
            # print(term_row)
            gene_set_covered = term_row['GeneSet'].values[0]
            # print('covered gene set: ', gene_set_covered)

            total_covered_prots = total_covered_prots.union(gene_set_covered)
            # Update GeneSet for every term by removing all the genes covered by the selected term.
            df['GeneSet'] = df['GeneSet'].apply(lambda x: x.difference(gene_set_covered))
            # Update GeneRatioNumerator. It is just the number of genes in GeneSet (since they have not been covered by any already selected term)
            df['GeneRatioNumerator'] = df['GeneSet'].apply(lambda x: len(x))
            # Update OddsRatio


            for idx in df.index.values:
                # input to Fisher's exact test
                a = df.at[idx, 'GeneRatioNumerator']
                b = df.at[idx, 'GeneRatioDenominator'] - a
                c = df.at[idx, 'BgRatioNumerator'] - a
                d=  df.at[idx, 'BgRatioDenominator']-(a+b+c)

                if(a == 0):
                     df.at[idx, 'hypergeom_pval'] = 1
                else:
                    odd_ratio, df.at[idx, 'hypergeom_pval'] = stats.fisher_exact([[a, b], [c, d]],'greater')

        # now update OddsRatio in combined_df. since the indices do not match in dfs and combined_df, the safest thing to do is to repeat the concat and groupby.
        combined_df = pd.concat(dfs)
        combined_df = combined_df[['Description', 'hypergeom_pval']].groupby('Description', as_index=False).prod()

        # check if no new protein is being covered for atleast 100 iteration.
        # If so then return the already selected terms
        new_n_uncovered = len(prot_universe.difference(total_covered_prots))
        if (n_uncovered - new_n_uncovered)==0:
            count+=1
        if count>100:
            return selected_terms

def simplify_enrichment_greedy_algo_v3(df):
    prot_universe = find_prot_universe(df.copy())

    description = df['Description']
    df.drop('Description', level = 0, axis = 1, inplace = True )

    parsed_df = pd.DataFrame({'Description':description})

    df_a_dict={}

    for dataset, df_d in df.groupby(level = 0, axis = 1):

        for prot_type, df_a in df_d.groupby(level=1, axis = 1):

            df_a.columns = df_a.columns.droplevel([0,1])

            # add description column
            df_a['Description'] = description

            # fill nan values
            df_a['pvalue']=df_a['pvalue'].fillna(1)
            df_a['geneID'] = df_a['geneID'].fillna('/')

            # fill the nan values in GeneRatio and BgRatio columns
            df_temp = df_a[df_a['GeneRatio']!=np.nan]
            generatio_denominator = df_temp.iloc[0].at['GeneRatio'].split('/')[1]
            filler_generatio = '0'+'/'+ generatio_denominator

            df_temp = df_a[df_a['BgRatio']!=np.nan]
            bgratio_denominator = df_temp.iloc[0].at['BgRatio'].split('/')[1]
            filler_bgratio = '0'+'/'+bgratio_denominator

            df_a['GeneRatio'] = df_a['GeneRatio'].fillna(filler_generatio)
            df_a['BgRatio'] = df_a['BgRatio'].fillna(filler_bgratio)

            if 'geneID' not in parsed_df.columns:
                parsed_df['geneID'] = df_a['geneID']
                parsed_df['geneName'] = df_a['geneName']
            else:
                parsed_df['geneID'] =parsed_df['geneID'].astype(str) +'/'+ df_a['geneID']
                parsed_df['geneName'] =parsed_df['geneName'].astype(str) +'/'+ df_a['geneName']


            if(prot_type !='-'):
                pval_col = prot_type+'_'+'pvalue'
                BgRatio_col = prot_type+'_'+'BgRatio'
                GeneRatio_col = prot_type+'_'+'GeneRatio'

            else:
                pval_col = dataset+'_'+'pvalue'
                BgRatio_col = dataset+'_'+'BgRatio'
                GeneRatio_col = dataset+'_'+'GeneRatio'

            parsed_df[pval_col] = df_a['pvalue']
            parsed_df[BgRatio_col] = df_a['BgRatio']
            parsed_df[GeneRatio_col] = df_a['GeneRatio']

            if(prot_type != '-'):
                df_a_dict[prot_type] = reformat_enrichment_data_v3(df_a)
            else:
                df_a_dict[dataset] = reformat_enrichment_data_v3(df_a)

    parsed_df['geneID'] = parsed_df['geneID'].astype(str).apply(lambda x: (set(filter(None,x.split('/')))))
    simple_df = parsed_df.copy()

    combined_selected_terms = simplify_enriched_terms_multiple_v3(copy.deepcopy(list(df_a_dict.values())), prot_universe)
    # print('Length of combined_selected_terms', len(combined_selected_terms))

    description_id_map = simple_df[['Description']].reset_index()
    description_id_map = description_id_map.set_index('Description')

    filtered_orederd_description_id_map = description_id_map[description_id_map.index.
        isin(combined_selected_terms)].reindex(combined_selected_terms)
    simple_df = simple_df[simple_df['Description'].isin(combined_selected_terms)].set_index('Description')
    simple_df = simple_df.reindex(list(filtered_orederd_description_id_map.index))

    print('greedy simplified terms: ', simple_df.shape)

    print('simple_df:' , simple_df)

    return  simple_df


def compare_selected_terms(df, terms):
    """
    For all pairs of terms, compare the overlap of their gene sets. The goal of this Python
    function is to verify that the terms are not too similar.
    """

    all_pairs = itertools.combinations(terms, 2)
    all_pairs_jaccard_coeffs = []
    all_pairs_jaccard_coeffs_dict = pd.DataFrame()
    gs1_list = []
    gs2_list = []
    description1_list = []
    description2_list = []
    order_diff_list = []

    for (fn1, fn2) in all_pairs:
        # compare gene_set with
        # gs1 = df['GeneSet'][[df['Description'] == fn1].index]
        # gs2 = df['GeneSet'][[df['Description'] == fn2].index]
        gs1 = df.at[fn1, 'geneID']
        gs2 = df.at[fn2, 'geneID']
        # print("Function %s has size %d" % (fn1, len(gs1)) )
        # print("Function %s has size %d" % (fn2, len(gs2)) )

        jc = len(gs1.intersection(gs2)) * 1.0 / len(gs1.union(gs2))

        # print('Jaccard: ' , fn1,fn2,jc)
        all_pairs_jaccard_coeffs.append(jc)

        gs1_list.append(len(gs1))
        gs2_list.append(len(gs2))
        description1_list.append(df.at[fn1, 'Description'])
        description2_list.append(df.at[fn2, 'Description'])
        order_diff_list.append(terms.index(fn2) - terms.index(fn1))

    all_pairs_jaccard_coeffs_dict['order_diff_2-1'] = pd.Series(order_diff_list)
    all_pairs_jaccard_coeffs_dict['Jaccard'] = pd.Series(all_pairs_jaccard_coeffs)

    all_pairs_jaccard_coeffs_dict['description_1'] = pd.Series(description1_list)
    all_pairs_jaccard_coeffs_dict['gs1'] = pd.Series(gs1_list)

    all_pairs_jaccard_coeffs_dict['description_2'] = pd.Series(description2_list)
    all_pairs_jaccard_coeffs_dict['gs2'] = pd.Series(gs2_list)

    # all_pairs_jaccard_coeffs_dict = all_pairs_jaccard_coeffs_dict.loc[all_pairs_jaccard_coeffs_dict[all_pairs_jaccard_coeffs_dict['order_diff_2-1']==1]]
    print("Maximum Jaccard coefficient is %f" % (max(all_pairs_jaccard_coeffs)))
    print("Median Jaccard coefficient is %f" % (statistics.median(all_pairs_jaccard_coeffs)))
    print("Mean Jaccard coefficient is %f" % (statistics.mean(all_pairs_jaccard_coeffs)))

    return all_pairs_jaccard_coeffs_dict


########### ********* CODE TO SIMPLIFY GO TERMS ON ENRICHED ON A SINGLE SET OF PROTEINS ******************
def single_set_find_prot_universe(df):

    df['geneID'] = df['geneID'].astype(str).apply(lambda x: (set(filter(None,x.split('/')))))

    # create protein universe
    prot_universe = set()
    for term, geneID_set in df['geneID'].items():
        prot_universe = prot_universe.union(geneID_set)

    return  prot_universe
def single_set_simplify_enriched_terms_v3(df, prot_universe):
    """
    dfs: a list of DataFrames containing the enrichment information for single sets of proteins.
    min_odds_ratio: Stop selecting additional terms once the maximum odds ratio falls below this threshold.

    Use a greedy-set-cover-like algorithm to compute a small set of terms that cover all the proteins annotated by the terms across the dfs.
    Here, we are considering only the proteins that also belong to the sets for which we ran enrichment, e.g., top-k GeneManiaPlus predictions and top-k SVM predictions.

    In each iteration, the idea is to pick the term that maximises a composite odds ratio. After picking the term,
    we update the GeneSet for each term in each df by deleting all the genes annotated to the selected term. We also update GeneRatioNumerator, OddsRatio, and CompositeOddsRatio.

    The odds ratio for a term in one df is (GeneRatioNumerator/GeneRatioDenominator)/(BgRatioNumerator/BgRatioDenominator).
    We define the composite odds ratio of a term to be the product across dfs of the odds ratios for the term.
    """

    selected_terms = []
    total_covered_prots = set()
    hypergeom_pval_list = []
    # compute product of OddsRatio values in dfs. Use concat followed by groupby.
    # df = df[['Description', 'hypergeom_pval']].groupby('Description', as_index=False).prod()
    print('prot_universe', len(prot_universe))

    while (True):
        # pick the term with the largest odds ratio.
        min_index = df['hypergeom_pval'].idxmin()
        # hypergeom_pval_list.append(combined_df['hypergeom_pval'][min_index])
        # set another condition here
        print(' covered_prot',len(total_covered_prots))
        if len(prot_universe.difference(total_covered_prots))==0:

            # print(dict(zip(selected_terms,hypergeom_pval_list)))
            return(selected_terms)

        selected_term = df['Description'][min_index]
        selected_terms.append(selected_term)

        #update remaining terms
        term_row = df[df['Description'] == selected_term]
        if (term_row.shape[0] == 0):
            # a df may not contain a row for selected_term
            continue
        # print(term_row)
        gene_set_covered = term_row['GeneSet'].values[0]
        # print('covered gene set: ', gene_set_covered)

        total_covered_prots = total_covered_prots.union(gene_set_covered)
        # Update GeneSet for every term by removing all the genes covered by the selected term.
        df['GeneSet'] = df['GeneSet'].apply(lambda x: x.difference(gene_set_covered))
        # Update GeneRatioNumerator. It is just the number of genes in GeneSet (since they have not been covered by any already selected term)
        df['GeneRatioNumerator'] = df['GeneSet'].apply(lambda x: len(x))
        # Update OddsRatio

        for idx in df.index.values:
            # input to Fisher's exact test
            a = df.at[idx, 'GeneRatioNumerator']
            b = df.at[idx, 'GeneRatioDenominator'] - a
            c = df.at[idx, 'BgRatioNumerator'] - a
            d = df.at[idx, 'BgRatioDenominator'] - (a + b + c)

            if (a == 0):
                df.at[idx, 'hypergeom_pval'] = 1
            else:
                odd_ratio, df.at[idx, 'hypergeom_pval'] = stats.fisher_exact([[a, b], [c, d]], 'greater')

def single_set_simplify_enrichment_greedy_algo_v3(df):
    prot_universe = single_set_find_prot_universe(df.copy())

    # fill nan values
    df['pvalue']=df['pvalue'].fillna(1)
    df['geneID'] = df['geneID'].fillna('/')

    # fill the nan values in GeneRatio and BgRatio columns
    df_temp = df[df['GeneRatio']!=np.nan]
    generatio_denominator = df_temp.iloc[0].at['GeneRatio'].split('/')[1]
    filler_generatio = '0'+'/'+ generatio_denominator

    df_temp = df[df['BgRatio']!=np.nan]
    bgratio_denominator = df_temp.iloc[0].at['BgRatio'].split('/')[1]
    filler_bgratio = '0'+'/'+bgratio_denominator

    df['GeneRatio'] = df['GeneRatio'].fillna(filler_generatio)
    df['BgRatio'] = df['BgRatio'].fillna(filler_bgratio)

    df = reformat_enrichment_data_v3(df)

    df['geneID'] = df['geneID'].astype(str).apply(lambda x: (set(filter(None,x.split('/')))))

    selected_terms = single_set_simplify_enriched_terms_v3(copy.deepcopy(df), prot_universe)


    return selected_terms



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
    df.drop(['Description','p.adjust', 'Count'], axis=1, level=2, inplace=True)
    return df