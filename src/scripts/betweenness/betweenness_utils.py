import os, sys
import pickle

import argparse
import numpy as np
import pandas as pd
import matplotlib
import networkx as nx
import copy
import time
import scipy.sparse as sp
from pandas import DataFrame
from scipy.sparse import eye, diags
from scipy.linalg import inv
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu, describe
from scipy.stats import fisher_exact
from scipy.stats import kstest


sys.path.insert(1, "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/")
# sys.path.insert(0,"../../")
from src.FastSinkSource.src.main import setup_dataset
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils
from src.FastSinkSource.src.algorithms import rl_genemania as gm
from src.FastSinkSource.src.algorithms import PageRank_Matrix as rwr
import src.scripts.utils as script_utils
from src.setup_datasets import setup_mappings, map_nodes
from src.scripts.betweenness.betweenness_utils import *
from statsmodels.stats.multitest import multipletests

species_alias = {'yeast': 'Saccharomyces cerevisiae', 'human':'Homo sapiens'}


# ********************************************** INTERESTING PROTEIN PARSING ***********************************
def parse_viral_prot_file(viral_prot_file):
    '''This will return a dict where
    keys = ['sars2--','all'] '''
    viral_df = pd.read_csv(viral_prot_file, sep=',')
    #stat for viral interactors
    print('number of viruses: ', len(viral_df['Mnemonic_Tax'].unique()))

    viral_dict = {}

    #commented out the code which remove SARS2 prots even if it is from other virus
    # all_interactors = set(viral_df['Uniprot_Human'])
    # sars2_interactors = set(viral_df[viral_df['Mnemonic_Tax']=='SARS2']['Uniprot_Human'])
    # viral_dict['sars2--'] = all_interactors.difference(sars2_interactors)

    # the following will make sure that if a protein is common btn SARS2 and other protein then it will
    # appear in the dict
    viral_dict['sars2--'] = set(viral_df[viral_df['Mnemonic_Tax'] != 'SARS2']['Uniprot_Human'])
    print('Done parsing viral interactors file')
    return viral_dict


def parse_essential_prot_file(essential_prot_file, species):
    '''This function will take in a file containing essential proteins downloaded from
    DEG database and parse it to return the essential proteins from given species'''
    essential_prot_df = pd.read_csv(essential_prot_file, sep=';', header=None,
                                    usecols=[0, 2, 6, 7], names=['DEGID','prot', 'description', 'species',])
    essential_prot_df = essential_prot_df[essential_prot_df['species'] == species_alias[species]]
    print(essential_prot_df.columns)
    return essential_prot_df


def parse_pleiotropic_prot_file(pleiotropic_prot_file, species):
    '''This function will take in a file containing essential proteins downloaded from
    DEG database and parse it to return the essential proteins from given species'''
    pleiotropic_prot_df = pd.read_csv(pleiotropic_prot_file, sep=';', usecols=['Gene'])
    return list(set(pleiotropic_prot_df['Gene']))


def get_uniprot_set(datasets_dir, mapping_settings, prots,**kwargs):
    '''
    This function will return a set containing the uniprots of given prots. But there is
    no mapping per say from.
    '''
    # now map essential proteins to uniprot IDs
    namespace_mappings = setup_mappings(datasets_dir, mapping_settings, **kwargs)
    prot_to_uniprot, mapping_stats = \
        map_nodes(prots, namespace_mappings, prefer_reviewed=True)

    uniprots = list(prot_to_uniprot.values())
    uniprots = [item for sublist in uniprots for item in sublist]

    return set(uniprots)


def handle_essential_uniprot_mapping(master_config_map, **kwargs):
    '''
    Return a dict of essential uniprot IDs where
    key = type of essential prot : ['cell','org','all']
    value = set of uniprot of essential prots
    '''

    # mapping settings
    dataset_settings = master_config_map['dataset_settings']
    datasets_dir = dataset_settings['datasets_dir']
    mapping_settings = dataset_settings.get('mappings')
    species = mapping_settings[0]['species']

    # parse essential protein file
    essential_prot_file = kwargs.get('essential_prot_file')
    species_wise_essential_uniprot_file = essential_prot_file.replace('.csv', '_' + species + '.pkl')

    if not os.path.exists(species_wise_essential_uniprot_file):
        essential_uniprots_dict={}
        essential_df = parse_essential_prot_file(essential_prot_file, species)
        # separate organism-specific and cell-specific essential genes
        if species == 'human': #according to DEG database DEG2006, DEG2010 and DEG2011 are organism specific essential protein finding experiment
            essential_org = list(essential_df[essential_df['DEGID'].isin(['DEG2006','DEG2010','DEG2011'])]['prot'])
            essential_cell = list(essential_df[~(essential_df['DEGID'].isin(['DEG2006','DEG2010','DEG2011']))]['prot'])
        elif species=='yeast':
            essential_org = list(essential_df['prot'])
            essential_cell=[]

        # get essential uniprots
        essential_uniprots_dict['cell'] = get_uniprot_set(datasets_dir, mapping_settings, essential_cell, **kwargs)
        essential_uniprots_dict['org'] = get_uniprot_set(datasets_dir, mapping_settings, essential_org, **kwargs)
        essential_uniprots_dict['all'] =  essential_uniprots_dict['cell'].union(essential_uniprots_dict['org'])

        os.makedirs(os.path.dirname(species_wise_essential_uniprot_file), exist_ok=True)
        out_file = open(species_wise_essential_uniprot_file, 'wb')
        pickle.dump(essential_uniprots_dict, out_file)
        out_file.close()
        print('done saving essential prot mapping')

    else:
        out_file = open(species_wise_essential_uniprot_file, 'rb')
        essential_uniprots_dict = pickle.load(out_file)
        out_file.close()
        print('done reading essential prot mapping')

    return essential_uniprots_dict


# ********************************************** BETWEENNESS COMPUTATION ******************************************

def compute_1st_2nd_3rd_intermediate_node_contr(M1, M2, M3, X1, X2, X3):
    # paths of len 2
    first_intermediate_nodes_contr_pathlen2_mat = np.multiply(M1, X1)
    intermediate_nodes_contr_pathlen2_mat = (first_intermediate_nodes_contr_pathlen2_mat).T
    
    # paths of len  3
    first_intermediate_nodes_contr_pathlen3_mat = np.multiply(M2, X1)
    second_intermediate_nodes_contr_pathlen3_mat = np.multiply(M1, X2)
    intermediate_nodes_contr_pathlen3_mat = (first_intermediate_nodes_contr_pathlen3_mat+\
                                            second_intermediate_nodes_contr_pathlen3_mat).T
    # paths of len  4
    first_intermediate_nodes_contr_pathlen4_mat = np.multiply(M3, X1)
    second_intermediate_nodes_contr_pathlen4_mat = np.multiply(M2, X2)
    third_intermediate_nodes_contr_pathlen4_mat = np.multiply(M1, X3)
    # same node as 1st and 3rd intermediate node in paths of len 4
    M2_diag = np.diag(np.diag(M2))
    first_third_intermediate_nodes_contr_pathlen4_mat = np.multiply(M1, np.matmul(M2_diag, X1))
    intermediate_nodes_contr_pathlen4_mat = (first_intermediate_nodes_contr_pathlen4_mat+\
                                            second_intermediate_nodes_contr_pathlen4_mat + \
                                            third_intermediate_nodes_contr_pathlen4_mat-\
                                            first_third_intermediate_nodes_contr_pathlen4_mat).T
    intermediate_nodes_contr_mat = intermediate_nodes_contr_pathlen2_mat + intermediate_nodes_contr_pathlen3_mat+\
                                    intermediate_nodes_contr_pathlen4_mat
    # first_intermediate_nodes_contr = first_intermediate_nodes_contr_pathlen2_mat + \
    #                                  first_intermediate_nodes_contr_pathlen3_mat + \
    #                                  first_intermediate_nodes_contr_pathlen4_mat
    # second_intermediate_nodes_contr = second_intermediate_nodes_contr_pathlen3_mat + \
    #                                   second_intermediate_nodes_contr_pathlen4_mat
    # # total contribution from each node being a first, second, third intermediate node. Also take a transpose
    # # so we have sources along the columns and target along the rows again.
    # intermediate_nodes_contr_mat = (first_intermediate_nodes_contr + second_intermediate_nodes_contr + \
    #                                 third_intermediate_nodes_contr_pathlen4_mat - first_third_intermediate_nodes_contr_pathlen4_mat).T
    # del first_intermediate_nodes_contr, second_intermediate_nodes_contr

    del second_intermediate_nodes_contr_pathlen3_mat, second_intermediate_nodes_contr_pathlen4_mat
    del first_intermediate_nodes_contr_pathlen2_mat, first_intermediate_nodes_contr_pathlen3_mat, \
        first_intermediate_nodes_contr_pathlen4_mat
    del third_intermediate_nodes_contr_pathlen4_mat, first_third_intermediate_nodes_contr_pathlen4_mat

    del M1, X1, M2, X2, M3, X3
    return intermediate_nodes_contr_mat, intermediate_nodes_contr_pathlen2_mat, \
           intermediate_nodes_contr_pathlen3_mat,intermediate_nodes_contr_pathlen4_mat

# def compute_1st_2nd_3rd_intermediate_node_contr(M1, M2, M3, X1, X2, X3):
#     # 1st intermediate node in paths of len 2, 3, 4
#     first_intermediate_nodes_contr_pathlen2_mat = np.multiply(M1, X1)
#     first_intermediate_nodes_contr_pathlen3_mat = np.multiply(M2, X1)
#     first_intermediate_nodes_contr_pathlen4_mat = np.multiply(M3, X1)
#     first_intermediate_nodes_contr = first_intermediate_nodes_contr_pathlen2_mat + \
#                                      first_intermediate_nodes_contr_pathlen3_mat + \
#                                      first_intermediate_nodes_contr_pathlen4_mat
#     del first_intermediate_nodes_contr_pathlen2_mat, first_intermediate_nodes_contr_pathlen3_mat, \
#         first_intermediate_nodes_contr_pathlen4_mat
# 
#     # 2nd intermediate node in paths of len  3, 4
#     second_intermediate_nodes_contr_pathlen3_mat = np.multiply(M1, X2)
#     second_intermediate_nodes_contr_pathlen4_mat = np.multiply(M2, X2)
#     second_intermediate_nodes_contr = second_intermediate_nodes_contr_pathlen3_mat + \
#                                       second_intermediate_nodes_contr_pathlen4_mat
# 
#     del second_intermediate_nodes_contr_pathlen3_mat, second_intermediate_nodes_contr_pathlen4_mat
# 
#     # 3rd intermediate node in paths of len  4
#     third_intermediate_nodes_contr = np.multiply(M1, X3)
# 
#     # same node as 1st and 3rd intermediate node in paths of len 4
#     M2_diag = np.diag(np.diag(M2))
#     first_third_intermediate_nodes_contr = np.multiply(M1, np.matmul(M2_diag, X1))
# 
#     # total contribution from each node being a first, second, third intermediate node. Also take a transpose
#     # so we have sources along the columns and target along the rows again.
#     intermediate_nodes_contr_mat = (first_intermediate_nodes_contr + second_intermediate_nodes_contr + \
#                                     third_intermediate_nodes_contr - first_third_intermediate_nodes_contr).T
#     del first_intermediate_nodes_contr, second_intermediate_nodes_contr, \
#         third_intermediate_nodes_contr, first_third_intermediate_nodes_contr
# 
#     del M1, X1, M2, X2, M3, X3
#     return intermediate_nodes_contr_mat


def compute_btns(M, idx_to_prot):
    '''
    Given the network M, this function will compute on average how much contribution(via
    path length of atmost 4) is going through
    each node in the net considering all source-target pairs.
    Output: a betweenness-valuewise-sorted pandas Series.
    The fields are: 1.node_idx, 2. betweenness_score, 3. prot_name
    '''

    # Also take transpose of M to make the computation clearer i.e. after transpose
    # for every (u,v) edge we have u along the rows  and v along columns
    M1 = M.T
    X1 = np.sum(M1, axis=0).reshape(-1, 1)  # column matrix, cost of reaching 'a' via all paths of length 1 => X1[a]

    M2 = np.matmul(M1, M1) #M2[a][b] => cost of reaching b from a via all paths of length 2
    X2 = np.sum(M2, axis=0).reshape(-1, 1)  # column matrix, cost of reaching 'a' via all paths of length 2 => X2[a]

    M3 = np.matmul(M2, M1) #M3[a][b] => cost of reaching b from a via all paths of length 3
    X3 = np.sum(M3, axis=0).reshape(-1, 1)  # column matrix, cost of reaching 'a' via all paths of length 3 => X3[a]

    intermediate_nodes_contr_mat, intermediate_nodes_contr_pathlen2_mat, \
    intermediate_nodes_contr_pathlen3_mat,intermediate_nodes_contr_pathlen4_mat\
        = compute_1st_2nd_3rd_intermediate_node_contr(M1, M2, M3, X1, X2, X3)

    # compute the average contribution of an intermediate node to a target i.e. betweeness score
    intermediate_nodes_mean_contr = np.mean(intermediate_nodes_contr_mat, axis=0)
    del intermediate_nodes_contr_mat

    # sort nodes according to their betweenness score
    sorted_betweenness_score = pd.Series(intermediate_nodes_mean_contr).sort_values(ascending=False)
    sorted_betweenness_score_df = pd.DataFrame({'node_idx': list(sorted_betweenness_score.index),
                                                'betweenness': sorted_betweenness_score.values,
                                                'prot': [idx_to_prot[x] for x in list(sorted_betweenness_score.index)]})
    return sorted_betweenness_score_df


def handle_btns(M_pathmtx_loginv, prots, betweenness_score_file, force_run=False):
    if (not os.path.exists(betweenness_score_file) or force_run == True):

        # *******************************************
        # Compute betweenness scores.
        # The following analysis is focused on each node on the network and how important
        # they are as intermediate node considering all source target pairs.
        sorted_btns_scores_df = \
            compute_btns(M_pathmtx_loginv, prots)
        sorted_btns_scores_df['percent_rank'] = sorted_btns_scores_df['betweenness'].rank(pct=True)

        # save betweenness score in file
        os.makedirs(os.path.dirname(betweenness_score_file), exist_ok=True)
        sorted_btns_scores_df.to_csv(betweenness_score_file, sep='\t', index=False)
        print('done computing betweenness scores')

    else:
        sorted_btns_scores_df = pd.read_csv(betweenness_score_file,
                                            sep='\t', index_col=None)
        print('done betweenness file reading')
    return sorted_btns_scores_df


def compute_src_spec_btns(M, alg_name, a_d_norm_inv, idx_to_prot, pos_nodes_idx):
    '''
    Input: 1) Matrix, M. In M graph, cost of a path
    (multiplication of edge costs) from s to t means, source s's contribution to t's score.
    2) list, pos_nodes_index. Contains the index of positive nodes in the network

    Function: 1) This will compute the contribution going via each node in the network from
    all known source nodes to a target node. So, we will get each node's contribution
    as an intermediate node to each target node's score.
    2) Then we will take average contribution going via a node to a target node.

    Note: So far, I can consider paths of maximum length of 4.
    '''

    # Also take transpose of M to make the computation clearer i.e. after transpose
    # along the row I have u and along column I have v for every (u,v) edge.
    M1 = M.T
    X1 = np.sum(M1[pos_nodes_idx, :], axis=0).reshape(-1, 1)  # column vector

    M2 = np.matmul(M1, M1)
    X2 = np.sum(M2[pos_nodes_idx, :], axis=0).reshape(-1, 1)  # column matrix

    M3 = np.matmul(M2, M1)
    X3 = np.sum(M3[pos_nodes_idx, :], axis=0).reshape(-1, 1)  # column matrix

    #in the returned matrix again sources along columns and targets along rows
    intermediate_nodes_contr_mat, intermediate_nodes_contr_pathlen2_mat, \
    intermediate_nodes_contr_pathlen3_mat,intermediate_nodes_contr_pathlen4_mat=\
        compute_1st_2nd_3rd_intermediate_node_contr(M1, M2, M3, X1, X2, X3)

    #multiply the contributions with a_d_norm_inv=(I+a*D_norm^)-1 for RL
    if alg_name=='genemaniaplus':
        intermediate_nodes_contr_mat = np.matmul(a_d_norm_inv, intermediate_nodes_contr_mat)
        intermediate_nodes_contr_pathlen2_mat = np.matmul(a_d_norm_inv, intermediate_nodes_contr_pathlen2_mat)
        intermediate_nodes_contr_pathlen3_mat = np.matmul(a_d_norm_inv, intermediate_nodes_contr_pathlen3_mat)
        intermediate_nodes_contr_pathlen4_mat = np.matmul(a_d_norm_inv, intermediate_nodes_contr_pathlen4_mat)

    # compute the average contribution of an intermediate node to a target i.e. betweeness score
    src_spec_betweenness = np.mean(intermediate_nodes_contr_mat, axis=0)

    #now compute nodes contribution only via paths of length 2
    src_spec_mean_contr_via_pathlen_2 = np.mean(intermediate_nodes_contr_pathlen2_mat, axis=0)
    #now compute nodes contribution only via paths of length 3
    src_spec_mean_contr_via_pathlen_3 = np.mean(intermediate_nodes_contr_pathlen3_mat, axis=0)
    #now compute nodes contribution only via paths of length 4
    src_spec_mean_contr_via_pathlen_4 = np.mean(intermediate_nodes_contr_pathlen4_mat, axis=0)
    del intermediate_nodes_contr_mat,intermediate_nodes_contr_pathlen2_mat,\
        intermediate_nodes_contr_pathlen3_mat,intermediate_nodes_contr_pathlen4_mat

    # sort nodes according to their betweenness score
    sorted_src_spec_betweenness_score = pd.Series(src_spec_betweenness).sort_values(ascending=False)
    sorted_src_spec_betweenness_score_df = pd.DataFrame({'node_idx': list(sorted_src_spec_betweenness_score.index),
                    'betweenness': sorted_src_spec_betweenness_score.values,
                    'prot': [idx_to_prot[x] for x in
                    list(sorted_src_spec_betweenness_score.index)]})

    sorted_src_spec_betweenness_score_df['contr_pathlen_2'] = pd.Series(src_spec_mean_contr_via_pathlen_2)
    sorted_src_spec_betweenness_score_df['contr_pathlen_3'] = pd.Series(src_spec_mean_contr_via_pathlen_3)
    sorted_src_spec_betweenness_score_df['contr_pathlen_4'] = pd.Series(src_spec_mean_contr_via_pathlen_4)

    return sorted_src_spec_betweenness_score_df


def compute_src_spec_btns_simplified(M, alg_name, a_d_norm_inv, idx_to_prot, pos_nodes_idx):
    '''
    Input: 1) Matrix, M. In M graph, cost of a path
    (multiplication of edge costs) from s to t means, source s's contribution to t's score.
    2) list, pos_nodes_index. Contains the index of positive nodes in the network

    Function: 1) This will compute the contribution going via each node in the network from
    all known source nodes to a target node. So, we will get each node's contribution
    as an intermediate node to each target node's score.
    2) Then we will take average contribution going via a node to a target node.

    Note: So far, I can consider paths of maximum length of 4.
    '''

    # Also take transpose of M to make the computation clearer i.e. after transpose
    # along the row I have u and along column I have v for every (u,v) edge.
    N = M.shape[0]
    I = eye(N)
    X = inv(I - M)
    X1 = np.diag(np.sum(X[ :, pos_nodes_idx ], axis=1))  # column vector

    if alg_name=='genemaniaplus':
        X2=(np.matmul(a_d_norm_inv, X)).T
    else:
        X2=X.T

    X3 = (np.matmul(X1, X2))
    src_spec_betweenness = np.sum(X3[:,:], axis=1)

    del X, X1, X2, I

    # compute the average contribution of an intermediate node to a target i.e. betweeness score
    # sort nodes according to their betweenness score
    sorted_src_spec_betweenness_score = pd.Series(src_spec_betweenness).sort_values(ascending=False)
    sorted_src_spec_betweenness_score_df: DataFrame = pd.DataFrame({'node_idx': list(sorted_src_spec_betweenness_score.index),
                    'betweenness': sorted_src_spec_betweenness_score.values,
                    'prot': [idx_to_prot[x] for x in
                    list(sorted_src_spec_betweenness_score.index)]})

    return sorted_src_spec_betweenness_score_df

def handle_src_spec_btns(M_pathmtx_loginv, prots, betweenness_score_file,
                         pos_nodes_idx, alg_name, a_d_norm_inv, force_run=False):
    if (not os.path.exists(betweenness_score_file) or force_run==True):
        # *******************************************
        # Compute betweenness scores.
        # The following analysis is focused on each node on the network and how important
        # they are as intermediate node considering all source target pairs.
        sorted_src_spec_btns_scores_df = \
            compute_src_spec_btns(M_pathmtx_loginv,alg_name, a_d_norm_inv,  prots, pos_nodes_idx)
        # sorted_src_spec_btns_scores_df = \
        #     compute_src_spec_btns_simplified(M_pathmtx_loginv, alg_name, a_d_norm_inv, prots, pos_nodes_idx)
        sorted_src_spec_btns_scores_df['percent_rank'] = sorted_src_spec_btns_scores_df['betweenness'].rank(pct=True)

        # save betweenness score in file
        os.makedirs(os.path.dirname(betweenness_score_file), exist_ok=True)
        sorted_src_spec_btns_scores_df.to_csv(betweenness_score_file, sep='\t', index=False)
        print('done computing betweenness scores')

    else:
        sorted_src_spec_btns_scores_df = pd.read_csv(betweenness_score_file,
                                            sep='\t', index_col=None)
    print('done source specific betweenness score computation')
    return sorted_src_spec_btns_scores_df



def compute_network_centrality_scores(W, idx_to_prot, outfile, force_run=False):
    if ((not os.path.exists(outfile)) or force_run == True):
        # TODO: Currently considering symmetric W only.
        G = nx.Graph(W)
        betweenness_centrality_dict = nx.betweenness_centrality(G)

        # sort nodes according to their betweenness score
        sorted_btns_cntr = pd.Series(betweenness_centrality_dict).sort_values(ascending=False)
        sorted_btns_cntr_df = pd.DataFrame({'node_idx': list(sorted_btns_cntr.index),
                                            'betweenness': sorted_btns_cntr.values,
                                            'prot': [idx_to_prot[x] for x in
                                                     list(sorted_btns_cntr.index)]})
        # sorted_btns_cntr_df['percent_rank'] = sorted_btns_cntr_df['betweenness'].rank(pct=True)

        # save generic betweenness score in file
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        sorted_btns_cntr_df.to_csv(outfile, sep='\t', index=False)
    else:
        sorted_btns_cntr_df = pd.read_csv(outfile, sep='\t', index_col=None)
    print('done centrality score computation')
    return sorted_btns_cntr_df


#********************************* RELATION: BETWEENNESS AND INTERESTING PROTEINS *****************************
def handle_percentile_percent_corr(sorted_scores_df, interesting_uniprots):
    '''  This function will compute the pearson's correlation between percentage of essential
    proteins and percentile of betweenness score'''
    percentiles = []
    percentages_interesting = []
    bin_size = 100
    e = len(interesting_uniprots)  # number of essential prots
    for i in list(range(0, len(sorted_scores_df), bin_size)):
        bin_df = sorted_scores_df.iloc[i:i + bin_size, :]
        lowest_percentile_of_curr_bin = list(bin_df['percent_rank'])[-1]
        percentiles.append(lowest_percentile_of_curr_bin)

        # the following computes what percent of interesting prots is in a bin
        # percentage_of_essential_prot_in_curr_bin= len(set(bin_df['prot']).intersection(essential_uniprots))/float(e)*100

        # the following computes what percent of protein in a bin is interesting prots.
        percentage_of_interesting_prot_in_curr_bin = len(
            set(bin_df['prot']).intersection(interesting_uniprots)) / bin_size * 100

        percentages_interesting.append(percentage_of_interesting_prot_in_curr_bin)

    pearsons_corr, pvalue = script_utils.pearson_correlations(percentiles, percentages_interesting)

    # also compute MannWhiteney Pval for seeing how significant the difference in percentages_interesting
    # prots in top half ranks vs bottom half ranks
    half = int(len(percentages_interesting) / 2)
    _, mw_pval = mannwhitneyu(percentages_interesting[0:half],
                              percentages_interesting[half:], \
                              alternative='greater')

    print('Pearsons corr: ', pearsons_corr, 'pval: ', pvalue,
          '\nmw_pval: ', mw_pval)
    return pearsons_corr, pvalue, mw_pval, percentiles, percentages_interesting


def find_interesting_prot_in_src_top(sorted_df_pos,sorted_df_top_k,interesting_uniprots,percentiles,percentages_interesting ):
    # If we want to plot percentage_of_interesting_prots in the positive and top predictions separately

    pos_bin_percentile = list(sorted_df_pos['percent_rank'])[-1]
    pos_bin_percentage = len(set(sorted_df_pos['prot']).intersection(interesting_uniprots)) / float(
        len(sorted_df_pos)) * 100
    percentiles.append(pos_bin_percentile)
    percentages_interesting.append(pos_bin_percentage)

    top_bin_percentile = list(sorted_df_top_k['percent_rank'])[-1]
    top_bin_percentage = len(set(sorted_df_top_k['prot']).intersection(interesting_uniprots)) / float(
        len(sorted_df_top_k)) * 100
    percentiles.append(top_bin_percentile)
    percentages_interesting.append(top_bin_percentage)

    return percentiles, percentages_interesting

def interesting_prot_in_topks(sorted_btns_scores_df, essential_uniprots, ks):
    '''
    Return a dict: key= k, value= how many  proteins of interest (might be essential prots or prots that interact with virus) are present
    in corresponding top k prots
    '''
    sorted_prots = list(sorted_btns_scores_df['prot'])
    n_interesting_prots_per_k = {}
    for k in ks:
        top_k_prots = set(sorted_prots[0:k])
        n_interesting_prots_per_k[k] = len(top_k_prots.intersection(essential_uniprots))
    return n_interesting_prots_per_k


def find_new_prots_appearing_at_each_pathlens(sorted_scores_df, criteria = ['contr_pathlen_2','contr_pathlen_3','contr_pathlen_4']):
    '''
    Return the fraction of total prots appearing at each new path length
    '''
    total_prots = float(len(sorted_scores_df))
    new_prots_appearing_at_each_pathlens={}
    seen_prots=set()
    for criteron in criteria:
        #remove nodes having corresponding criteron value==0('contr_pathlen_2','contr_pathlen_3', or 'contr_pathlen_4' )
        criterion_prots= set(sorted_scores_df[~(sorted_scores_df[criteron]==0)]['prot'])
        new_prots = criterion_prots.difference(seen_prots)
        #Do not want 'contr_' to appear in the keys
        new_prots_appearing_at_each_pathlens[criteron.replace('contr_','')] = len(new_prots)/total_prots
        seen_prots = seen_prots.union(new_prots)
    return new_prots_appearing_at_each_pathlens

def handle_Fishers_exact_test_in_topks(sorted_btns_scores_df, interesting_uniprots, topks,
            ranking_criteria = ['betweenness']):
    '''
    Input: A sorted_btns_scores_df is a dataframe which is sorted according to the column 'betweenness'
    But it is not necessary to send a sorted sorted_btns_scores_df anymore.

    Output: A dict of dict.
    Inner dict: k=each k in topks
    and the value is a tuple(x,y)=> x is fraction of overlapping prots in topk and interesting prot,
    y is the pvalue of that overlap.
    Outer dict: keys are each criterion for ranking proteins i.e. betweenness,
    contr_as_intermediate_node_only_via_pathlen2 and so on.

    ranking_criteria=['contr_pathlen_2','contr_pathlen_3','contr_pathlen_4']
    '''
    #in sorted_btns_scores_df we have all the porteins for which we are considering the betweenness scores.
    #protein_universe should be only the proteins in  sorted_btns_scores_df
    scores_df = copy.deepcopy(sorted_btns_scores_df)
    prot_universe = set(scores_df['prot'])

    #filter out the proteins from essential_uniprots that are not present in prot universe
    filtered_interesting_uniprots = set(interesting_uniprots).intersection(prot_universe)
    print('number of interesting prot in network: ', len(filtered_interesting_uniprots))

    #computed fraction of overlap between top k ranked prots and prots of interest where topk can be defined
    #by betweenness_score, contr_as_intermediate_node_only_via_pathlen2, contr_as_intermediate_node_only_via_pathlen3,
    #contr_as_intermediate_node_only_via_pathlen4 and so on.

    all_criteria_frac_overlap_pval_topks={x:{} for x in ranking_criteria }

    for ranking_criteron in ranking_criteria:
        frac_overlap_pval_topks = {} #the value will be tuple(x,y)=> x is fraction of overlapping prots in topk and interesting prot.
                                                #y is the pvalue of that overlap
        # remove nodes having corresponding ranking_criteron value==0(i.e. 'betweenness','contr_pathlen_2','contr_pathlen_3', or 'contr_pathlen_4' )
        sorted_scores_df = scores_df[~(scores_df[ranking_criteron] == 0)]
        #sort prots in btns_scores_df according to the ranking_criterion
        sorted_scores_df = sorted_scores_df.sort_values(by=ranking_criteron, ascending=False )

        m = len(filtered_interesting_uniprots)
        N =  len(prot_universe)
        pvals=[]
        for k in topks:
            if len(sorted_scores_df['prot'])>=k:
                topk_prots = list(sorted_scores_df['prot'])[0:k]
                overlap = set(topk_prots).intersection(filtered_interesting_uniprots)
                a = len(overlap)
                b = k-a
                c = m - a
                d = N- k-m+a

                table = [[a,b ],[c,d]]
                pval = fisher_exact(table, alternative='greater')[1]
                pvals.append(pval)
                frac_overlap = a/float(k)
                frac_overlap_pval_topks[k] = (frac_overlap, pval)

        #correct for multiple hypothesis test
        _, corrected_pvals, _, _ = multipletests(pvals, alpha=0.01, method='fdr_bh')
        #now replace pval with corrected pval in frac_overlap_pval_topks
        count=0
        final_ks = list(frac_overlap_pval_topks.keys())
        for k in final_ks:
            frac_overlap_pval_topks[k] = (frac_overlap_pval_topks[k][0],corrected_pvals[count])
            count+=1
        all_criteria_frac_overlap_pval_topks[ranking_criteron] = frac_overlap_pval_topks
    return all_criteria_frac_overlap_pval_topks


def prepare_plotdata_for_Kolmogorov_Smirnov(filtered_src_spec_btns_df, ess_uniprots_dict_init,
                                            viral_uniprots_dict_init, marker ='range', ks=[], score='betweenness'):
    prot_universe =  set(filtered_src_spec_btns_df['prot'])
    btns_scores = list(filtered_src_spec_btns_df[score])
    n_ppi_prots = len(btns_scores)

    n_prots_ge_marker = {}  # score_marker value to index in descendingly_sorted_btns
    frac_prots_ge_btns_marker = {type_of_prot_sets: {} for type_of_prot_sets in
                                ['ppi_prots', 'ess_cell', 'ess_org', 'viral_sars2--']}
    if marker== 'range':
        # use fixed set of ranges
        btns_ranges = [1e-4, 1e-5, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]

        #for each milestone/marked btns score find out how many proteins have btns score>=that marker
        count = 0
        for i in range(0,n_ppi_prots):
            if btns_scores[i] < btns_ranges[count]:
                n_prots_ge_marker[btns_ranges[count]] = i
                # what fraction of total prot has btns score>=the given range
                frac_prots_ge_btns_marker['ppi_prots'][btns_ranges[count]] = i/float(n_ppi_prots)
                count += 1
                if count==len(btns_ranges):
                    break
    if marker== 'rank':
        for k in ks:
            #if betweenness score is zero than its rank is not valid anymore.
            #if for some rank k betwenness is zero then do not consider k or any later ranks
            #also if #number_of_total_prots in network is less than k then do not consider that k.
            if (len(btns_scores)<k) or (btns_scores[k]==0) :
                break
            n_prots_ge_marker[k]=k
            frac_prots_ge_btns_marker['ppi_prots'][k] = k / float(n_ppi_prots)


    #make sure than we consider only the essential/viral protein that are in the initial network
    ess_uniprots_dict = copy.deepcopy(ess_uniprots_dict_init)
    viral_uniprots_dict = copy.deepcopy(viral_uniprots_dict_init)

    for ess_type in ess_uniprots_dict:
        ess_uniprots_dict[ess_type] = prot_universe.intersection(ess_uniprots_dict[ess_type])
    for viral_type in viral_uniprots_dict:
        viral_uniprots_dict[viral_type] = prot_universe.intersection(viral_uniprots_dict[viral_type])

    for mark in n_prots_ge_marker: #n_prots_ge_marker[mark]=k, when based on ranks
        prots_ge_marker = set(list(filtered_src_spec_btns_df['prot'])[0:n_prots_ge_marker[mark]])

        # Compute what fraction of cell_spec ess prot has btns score>=a given range
        frac_prots_ge_btns_marker['ess_cell'][mark]=len(ess_uniprots_dict['cell'].
                        intersection(prots_ge_marker))/float(len(ess_uniprots_dict['cell']))

        # Compute what fraction of org_spec ess prot has btns score>=a given range
        frac_prots_ge_btns_marker['ess_org'][mark] = len(ess_uniprots_dict['org'].
                        intersection(prots_ge_marker))/ float(len(ess_uniprots_dict['org']))

        # Compute what fraction of viral_prot(excluding sars2--) has btns score>=a given range
        frac_prots_ge_btns_marker['viral_sars2--'][mark] = len(viral_uniprots_dict['sars2--'].
                        intersection(prots_ge_marker))/ float(len(viral_uniprots_dict['sars2--']))

    return frac_prots_ge_btns_marker

def handle_Kolmogorov_Smirnov_test(filtered_sorted_btns_df, ess_uniprots_dict,
                                   viral_uniprots_dict, score='betweenness' ):
    KS_dict = {}
    all_prots_btns = list(filtered_sorted_btns_df[score])
    for ess_type in ess_uniprots_dict:
        ess_df = filtered_sorted_btns_df[filtered_sorted_btns_df['prot'].
                isin(ess_uniprots_dict[ess_type])]
        ess_prots_btns = list(ess_df[score])
        ks_statistic, pvalue = kstest(ess_prots_btns, all_prots_btns)
        KS_dict['ess_'+ess_type] = pvalue

    for viral_type in viral_uniprots_dict:
        viral_df = filtered_sorted_btns_df[filtered_sorted_btns_df['prot'].
                isin(viral_uniprots_dict[viral_type])]
        viral_prots_btns = list(viral_df[score])
        ks_statistic, pvalue = kstest(viral_prots_btns, all_prots_btns)
        KS_dict['viral_' + viral_type] = pvalue
    return KS_dict

def filter_sorted_src_spec_btns(sorted_df, top_k_predictions_df, orig_pos):
    '''
    This function split all prots into positive/sources, top_k_preds and the rest.
    '''

    sorted_df_pos = sorted_df[sorted_df['prot'].isin(orig_pos)]
    sorted_df_wo_orig = sorted_df[~(sorted_df['prot'].isin(list(sorted_df_pos['prot'])))]
    sorted_df_top_k = sorted_df[sorted_df['prot'].isin(list(top_k_predictions_df['prot']))]
    sorted_df_filtered=sorted_df_wo_orig[~(sorted_df_wo_orig['prot'].isin(list(top_k_predictions_df['prot'])))]

    return sorted_df_filtered, sorted_df_pos,sorted_df_top_k

#***************************************** FILE SAVE ******************************************************
def save_btns_corr(alg_term_btns_corr_file, count, beta, pc, pval, mw):
    if count == 1:  # that means the result is for the first alpha value, so need a new file
        corr_term_fout = open(alg_term_btns_corr_file, 'w')
        corr_term_fout.write('alpha' + '\t' + 'pc' + '\t' + 'pval' + '\t' + 'mw'+'\n')
    else:
        corr_term_fout = open(alg_term_btns_corr_file, 'a')

        corr_term_fout.write(str(beta) + '\t' + str(pc) + '\t' + str(pval) +
                            '\t'+str(mw)+'\n')
    corr_term_fout.close()


#*************************************** MISCELLANEOUS******************************************************
def create_random_scores_df(sorted_scores_df, score='betweenness'):
    rand_df = copy.deepcopy(sorted_scores_df)
    rand_df[score] = pd.Series(np.random.rand(len(rand_df)))
    rand_df.sort_values(by=[score], ascending=False, inplace=True)
    return rand_df

def compute_frac_interesting_prot(prot_list, interesting_prot_list):
    prot_set = set(prot_list)
    interesting_prot_set = set(interesting_prot_list)

    return float(len(prot_set.intersection(interesting_prot_set))) / len(prot_set)


