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
import enrichment_analysis as enrichment
# from src.utils import parse_utils as utils
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

    group.add_argument('--pval-cutoff', type=float, default=0.01,
                       help="Cutoff on the corrected p-value for enrichment.")
    group.add_argument('--qval-cutoff', type=float, default=0.05,
                       help="Cutoff on the Benjamini-Hochberg q-value for enrichment.")
    group.add_argument('--fss-pval', type=float, default=0.01,
                       help="Cutoff on the Benjamini-Hochberg q-value for enrichment.")

    group = parser.add_argument_group('FastSinkSource Pipeline Options')

    group.add_argument('--force-run', action='store_true', default=True,
                       help="Force re-running the enrichment tests, and re-writing the output files")

    return parser

def simplify_enrichment_result(df):
    description = df['Description']
    df.drop('Description', level = 0, axis = 1, inplace = True)
    parsed_df = pd.DataFrame({'Description':description})
    # filtered_simplified_df = pd.DataFrame({'Description':description})

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

            # filtered_simplified_df[pval_col] = df_a['pvalue']



    parsed_df['geneID'] = parsed_df['geneID'].astype(str).apply(lambda x: (set(filter(None,x.split('/')))))

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

    simple_df = simple_df[simple_df.index.isin(terms_to_keep)]


    # order accrding to selection
    simple_df = simple_df.reindex(terms_to_keep)

    return  simple_df


def compare_selected_terms(df, terms):
    """
    For all pairs of terms, compare the overlap of their gene sets. The goal of this Python
    function is to verify that the terms are not too similar.
    """
    #
    # print('df: ', df)
    # print('df index: ', df.index)

    all_pairs = itertools.combinations(terms, 2)
    all_pairs_jaccard_coeffs = []
    all_pairs_jaccard_coeffs_dict=pd.DataFrame()
    gs1_list=[]
    gs2_list = []
    description1_list = []
    description2_list = []
    order_diff_list=[]

    for (fn1, fn2) in all_pairs:
        # compare gene_set with
        # gs1 = df['GeneSet'][[df['Description'] == fn1].index]
        # gs2 = df['GeneSet'][[df['Description'] == fn2].index]
        gs1 = df.at[fn1, 'geneID']
        gs2 = df.at[fn2 , 'geneID']
        # print("Function %s has size %d" % (fn1, len(gs1)) )
        # print("Function %s has size %d" % (fn2, len(gs2)) )

        jc = len(gs1.intersection(gs2))*1.0/len(gs1.union(gs2))

        # print('Jaccard: ' , fn1,fn2,jc)
        all_pairs_jaccard_coeffs.append(jc)

        gs1_list.append(len(gs1))
        gs2_list.append(len(gs2))
        description1_list.append(df.at[fn1,'Description'])
        description2_list.append(df.at[fn2,'Description'])
        order_diff_list.append(terms.index(fn2)- terms.index(fn1))


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

    while (True):
        # pick the term with the largest odds ratio.
        min_index = combined_df['hypergeom_pval'].idxmin()
        # hypergeom_pval_list.append(combined_df['hypergeom_pval'][min_index])
        # set another condition here
        # print('prot_universe, covered_prot', len(prot_universe),len(total_covered_prots))
        if len(prot_universe.difference(total_covered_prots))==0:

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


def simplify_enrichment_greedy_algo_v3(df):

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

            if(alg != '-'):
                df_a_dict[alg] = reformat_enrichment_data_v3(df_a)
            else:
                df_a_dict[dataset] = reformat_enrichment_data_v3(df_a)

    parsed_df['geneID'] = parsed_df['geneID'].astype(str).apply(lambda x: (set(filter(None,x.split('/')))))
    simple_df = parsed_df.copy()

    combined_selected_terms = simplify_enriched_terms_multiple_v3(copy.deepcopy(list(df_a_dict.values())), prot_universe)
    # print('Length of combined_selected_terms', len(combined_selected_terms))

    description_id_map = simple_df[['Description']].reset_index()
    description_id_map = description_id_map.set_index('Description')

    filtered_orederd_description_id_map = description_id_map[description_id_map.index.isin(combined_selected_terms)].reindex(combined_selected_terms)
    simple_df = simple_df[simple_df['Description'].isin (combined_selected_terms)].reindex(list(filtered_orederd_description_id_map['index']))

    print('greedy simplified terms: ', simple_df.shape)

    print('simple_df:' , simple_df)
    all_pairs_jaccard_coeffs_df = compare_selected_terms(simple_df.copy(),list(filtered_orederd_description_id_map['index']))

    return all_pairs_jaccard_coeffs_df, simple_df


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
                k_to_test = n_pos
                print('k: ', k_to_test)
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
                        if kwargs.get('enrichment_on') =='top_preds':
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

                            # for k in k_to_test:
                            query_prots = list(df.iloc[:k_to_test]['prot'])
                            enrich_str = enrichment_on+'_'+str(k_to_test)

                        elif kwargs.get('enrichment_on') =='top_paths':
                            nsp_processed_paths_file = config_map['output_settings'][
                            'output_dir'] + "/viz/%s/%s/diffusion-path-analysis/%s/shortest_path_2ss/processed_shortest-paths-2ss-nsp%s-a%s%s.tsv" % (
                            dataset['net_version'], term, alg_name, nsp,alpha, sig_str)
                            #reading 'path_prots' column value as list
                            df = pd.read_csv(nsp_processed_paths_file, sep='\t', index_col=None, converters={'path_prots':pd.eval})
                            #get prots on top nsp paths
                            path_prots = list(df['path_prots']) #this is one nested list. flatten it.
                            query_prots = set([element for innerList in path_prots for element in innerList])

                            #remove source nodes
                            query_prots = query_prots.difference(set(orig_pos))
                            enrich_str = enrichment_on + '_' + str(nsp)
                        # now run clusterProfiler from R
                        out_dir = "%s/%s/%s/a-%s/" % (
                            base_out_dir, alg, term,str(alpha) )

                        direct_prot_goids_by_c,_,_,_ = \
                            go_prep_utils.parse_gaf_file(kwargs.get('gaf_file'))
                        go_category = {'BP':'P', 'MF':'F', 'CC':'C'}

                        for ont in ['BP', 'MF', 'CC']:
                            ##keep the prots that have atleast one go annotation
                            filtered_topk_predictions = go_prep_utils.keep_prots_having_atleast_one_go_ann(query_prots,
                                                        go_category[ont],direct_prot_goids_by_c)
                            filtered_prot_universe= go_prep_utils.keep_prots_having_atleast_one_go_ann(prot_universe,
                                                        go_category[ont],direct_prot_goids_by_c)
                            print('#prots in query: ', len(filtered_topk_predictions),
                                  '\n #prots in universe: ', len(filtered_prot_universe))
                            enrich_df = enrichment.run_clusterProfiler_GO(
                                filtered_prot_universe, filtered_topk_predictions, ont,
                                enrich_str, out_dir, forced=kwargs.get('force_run'), **kwargs)

                            print('number of enriched terms for ', ont, ': ', len(enrich_df))
                        print('Running enrichgo done: ', term, ' ', alg)


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
    df = process_df(df,dataset_level)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    print("writing %s" % (out_file))
    df.to_csv(out_file, sep=',')
    return df


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map,**kwargs)