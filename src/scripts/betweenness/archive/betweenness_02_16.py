# here's an example call to this script:
# python src/scripts/effective_diffusion_node_path.py --config fss_inputs/config_files/provenance/provenance_biogrid_y2h_go.yaml
# --run-algs genemaniaplus --k 500 --m 20 --n-sp 500

import os, sys
import pickle

import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib
import networkx as nx
import copy
import time
import scipy.sparse as sp
from scipy.sparse import eye, diags
from scipy.linalg import inv
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu, describe

import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Use this to save files remotely.
    matplotlib.use('Agg')
import pandas as pd

sys.path.insert(1, "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/")
# sys.path.insert(0,"../../")
from src.FastSinkSource.src.main import setup_dataset
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils
from src.FastSinkSource.src.algorithms import rl_genemania as gm
from src.FastSinkSource.src.algorithms import PageRank_Matrix as rwr
import src.scripts.utils as script_utils
from src.setup_datasets import setup_mappings, map_nodes

alg_alias = {'rwr': rwr, 'genemaniaplus': gm}
alg_plot_name = {'rwr': 'RWR', 'genemaniaplus': 'RL'}
species_alias = {'yeast': 'Saccharomyces cerevisiae', 'human':'Homo sapiens'}

def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        config_map = yaml.load(conf, Loader=yaml.FullLoader)
        # config_map = yaml.load(conf)
    with open(args.master_config, 'r') as conf:
        master_config_map = yaml.load(conf, Loader=yaml.FullLoader)
    # TODO check to make sure the inputs are correct in config_map
    return config_map, master_config_map, kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="Compute betweenness score for each node in the network")
    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                        "fss_inputs/config_files/provenance/biogrid_y2h_s12.yaml"
                       , help="Configuration file used when running FSS. ")
    group.add_argument('--master-config', type=str, default="/data/tasnina/Provenance-Tracing/"
                        "SARS-CoV-2-network-analysis/config-files/master-config.yaml"
                       , help="Configuration file used to do mappings")

    # group.add_argument('--config', type=str, default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
    #                                                  "fss_inputs/config_files/provenance/yeast/provenance_string900v11.5_s1.yaml"
    #                    , help="Configuration file used when running FSS. ")
    # group.add_argument('--master-config', type=str, default="/data/tasnina/Provenance-Tracing/"
    #                     "SARS-CoV-2-network-analysis/config-files/master-config-yeast.yaml"
    #                    , help="Configuration file used to do mappings")
    group.add_argument('--essential-prot-file', type=str,
                       default ="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                        "fss_inputs/essential-prots/deg_annotation_e.csv",
                       help="This file should contain the essential genes for corresponding species")
    group.add_argument('--viral-prot-file', type=str,
                       default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                               "fss_inputs/viral-prots/HVIDB.csv",
                       help="This file should contain the essential genes for corresponding species")
    # group.add_argument('--pleiotropic-prot-file', type=str,
    #                    default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
    #                     "fss_inputs/pleiotropic-prots/pleiotropic_gene_human_ra.xls",
    #                    help="This file should contain the pleiotropic genes for corresponding species")
    # group.add_argument('--pleiotropic-corr', action='store_true', default=False,
    #                    help='If true we will find correlation between pleoitrophy and betweenness.')
    group.add_argument('--run-algs', type=str, action='append', default=[])
    group.add_argument('--sep-pos-top', action='store_true', default=True,
                       help="If true, then the percentile-percent correlation will be computed only for the proteins"
                            "that are neither an original positive protein nor a top predicted protein"
                            "In such case, the scatter plot for percentile-percent will show two different colored points"
                            "indicating what the percent of essential/viral interactors are in positive_prot set and "
                            "top_predictions")
    group.add_argument('--force-download', action='store_true', default=False,
                       help="Force re-downloading and parsing of the input files")
    return parser

def log_inv_mat(M_pathmtx):
    '''
    This will convert each value in the input matrix to 10^-(value) which is equvalent to taking inverse
    of 10 base log.
    Also, it will convert any value=1 in the output matrix to be zero.
    '''
    M = np.power(np.full_like(M_pathmtx, 10), (-1) * M_pathmtx)
    # in M_pathmtx an index with value 0 means no edge. The above statement turns all 0 values to 1.
    # Now to preserve the same meaning for such indices, we need to convert all 1 to 0 in M.
    where_1 = np.where(M == 1)  # TODO: replace 1's with 0's in time efficient way
    M[where_1] = 0
    return M

def get_M_pathmtx_loginv(net_obj, alg_name, alpha):
    fluid_flow_mat_file_M = "%s/fluid-flow-mat-%s-a%s.npy" % (net_obj.out_pref, alg_name,
                                                              str(alpha).replace('.', '_'))
    ##########CREAE or LOAD THE DIFFUSION MAYTRICES
    force_matrix = False
    if (os.path.exists(fluid_flow_mat_file_M) and force_matrix == False):
        M_pathmtx = np.load(fluid_flow_mat_file_M)
        M_pathmtx_loginv = log_inv_mat(M_pathmtx)
    else:
        M_pathmtx_loginv = alg_alias[alg_name].get_M(net_obj.W, alpha)
    return M_pathmtx_loginv

def compute_1st_2nd_3rd_intermediate_node_contr(M1, M2, M3, X1, X2, X3):
    #1st intermediate node in paths of len 2, 3, 4
    first_intermediate_nodes_contr_pathlen2_mat = np.multiply(M1, X1)
    first_intermediate_nodes_contr_pathlen3_mat = np.multiply(M2, X1)
    first_intermediate_nodes_contr_pathlen4_mat = np.multiply(M3, X1)
    first_intermediate_nodes_contr = first_intermediate_nodes_contr_pathlen2_mat + \
                                     first_intermediate_nodes_contr_pathlen3_mat+ \
                                     first_intermediate_nodes_contr_pathlen4_mat
    del first_intermediate_nodes_contr_pathlen2_mat, first_intermediate_nodes_contr_pathlen3_mat,\
        first_intermediate_nodes_contr_pathlen4_mat


    #2nd intermediate node in paths of len  3, 4
    second_intermediate_nodes_contr_pathlen3_mat = np.multiply(M1, X2)
    second_intermediate_nodes_contr_pathlen4_mat = np.multiply(M2, X2)
    second_intermediate_nodes_contr = second_intermediate_nodes_contr_pathlen3_mat + \
                                     second_intermediate_nodes_contr_pathlen4_mat

    del second_intermediate_nodes_contr_pathlen3_mat, second_intermediate_nodes_contr_pathlen4_mat

    #3rd intermediate node in paths of len  4
    third_intermediate_nodes_contr = np.multiply(M1, X3)

    #same node as 1st and 3rd intermediate node in paths of len 4
    M2_diag = np.diag(np.diag(M2))
    first_third_intermediate_nodes_contr = np.multiply(M1, np.matmul(M2_diag, X1) )

    #total contribution from each node being a first, second, third intermediate node. Also take a transpose
    # so we have sources along the columns and target along the rows again.
    intermediate_nodes_contr_mat = (first_intermediate_nodes_contr + second_intermediate_nodes_contr + \
        third_intermediate_nodes_contr - first_third_intermediate_nodes_contr).T
    del first_intermediate_nodes_contr, second_intermediate_nodes_contr , \
        third_intermediate_nodes_contr, first_third_intermediate_nodes_contr

    del M1, X1, M2, X2, M3, X3
    return intermediate_nodes_contr_mat

def compute_betweenness_score(M, idx_to_prot):
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
    X1 = np.sum(M1, axis=0).reshape(-1, 1)  # column matrix

    M2 = np.matmul(M1, M1)
    X2 = np.sum(M2, axis=0).reshape(-1, 1)  # column matrix

    M3 = np.matmul(M2, M1)
    X3 = np.sum(M3, axis=0).reshape(-1, 1)  # column matrix

    intermediate_nodes_contr_mat = compute_1st_2nd_3rd_intermediate_node_contr(M1, M2, M3, X1, X2, X3)

    #compute the average contribution of an intermediate node to a target i.e. betweeness score
    intermediate_nodes_mean_contr = np.mean(intermediate_nodes_contr_mat, axis= 0)
    del intermediate_nodes_contr_mat

    #sort nodes according to their betweenness score
    sorted_betweenness_score = pd.Series(intermediate_nodes_mean_contr).sort_values(ascending=False)
    sorted_betweenness_score_df = pd.DataFrame({'node_idx': list(sorted_betweenness_score.index),
                                'betweenness':sorted_betweenness_score.values,
                                'prot': [idx_to_prot[x] for x in list(sorted_betweenness_score.index) ]})
    return sorted_betweenness_score_df


def handle_betweenness_score(M_pathmtx_loginv, prots, betweenness_score_file, force_run=False):
    if (not os.path.exists(betweenness_score_file) or force_run==True):
        
        # *******************************************
        # Compute betweenness scores.
        # The following analysis is focused on each node on the network and how important
        # they are as intermediate node considering all source target pairs.
        sorted_btns_scores_df = \
            compute_betweenness_score(M_pathmtx_loginv, prots)
        sorted_btns_scores_df['percent_rank'] = sorted_btns_scores_df['betweenness'].rank(pct=True)

        #save betweenness score in file
        os.makedirs(os.path.dirname(betweenness_score_file), exist_ok=True)
        sorted_btns_scores_df.to_csv(betweenness_score_file,sep='\t', index=False)
        print('done computing betweenness scores')

    else:
        sorted_btns_scores_df = pd.read_csv(betweenness_score_file,
                                     sep='\t',index_col=None)
        print('done betweenness file reading')
    return sorted_btns_scores_df


def compute_source_spec_betweenness_score(M, idx_to_prot, pos_nodes_idx):
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

    intermediate_nodes_contr_mat = compute_1st_2nd_3rd_intermediate_node_contr(M1, M2, M3, X1, X2, X3)

    # compute the average contribution of an intermediate node to a target i.e. betweeness score
    intermediate_nodes_mean_contr = np.mean(intermediate_nodes_contr_mat, axis=0)
    del intermediate_nodes_contr_mat

    # sort nodes according to their betweenness score
    sorted_src_spec_betweenness_score = pd.Series(intermediate_nodes_mean_contr).sort_values(ascending=False)
    sorted_src_spec_betweenness_score_df = pd.DataFrame({'node_idx': list(sorted_src_spec_betweenness_score.index),
                                                         'betweenness': sorted_src_spec_betweenness_score.values,
                                                         'prot': [idx_to_prot[x] for x in
                                                                  list(sorted_src_spec_betweenness_score.index)]})
    return sorted_src_spec_betweenness_score_df


def handle_src_spec_betweenness_score(M_pathmtx_loginv, prots, betweenness_score_file,
                                      pos_nodes_idx, force_run=False):
    if (not os.path.exists(betweenness_score_file) or force_run==True):
        # *******************************************
        # Compute betweenness scores.
        # The following analysis is focused on each node on the network and how important
        # they are as intermediate node considering all source target pairs.
        sorted_src_spec_btns_scores_df = \
            compute_source_spec_betweenness_score(M_pathmtx_loginv, prots, pos_nodes_idx)
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
    if ((not os.path.exists(outfile)) or force_run==True):
        #TODO: Currently considering symmetric W only.
        G = nx.Graph(W)
        betweenness_centrality_dict = nx.betweenness_centrality(G)

        # sort nodes according to their betweenness score
        sorted_btns_cntr = pd.Series(betweenness_centrality_dict).sort_values(ascending=False)
        sorted_btns_cntr_df = pd.DataFrame({'node_idx': list(sorted_btns_cntr.index),
                                    'betweenness': sorted_btns_cntr.values,
                                    'prot': [idx_to_prot[x] for x in
                                             list(sorted_btns_cntr.index)]})
        # sorted_btns_cntr_df['percent_rank'] = sorted_btns_cntr_df['betweenness'].rank(pct=True)

        #save generic betweenness score in file
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        sorted_btns_cntr_df.to_csv(outfile,sep='\t', index=False)
    else:
        sorted_btns_cntr_df = pd.read_csv(outfile, sep='\t',index_col=None)
    print('done centrality score computation')
    return sorted_btns_cntr_df

def parse_essential_prot_file(essential_prot_file, species):
    '''This function will take in a file containing essential proteins downloaded from
    DEG database and parse it to return the essential proteins from given species'''
    essential_prot_df = pd.read_csv(essential_prot_file, sep=';', header=None,
                        usecols =[2,6,7],names=['prot','description','species'])
    essential_prot_df = essential_prot_df[essential_prot_df['species']==species_alias[species]]
    print(essential_prot_df.columns)
    return list(essential_prot_df['prot'])

def parse_pleiotropic_prot_file(pleiotropic_prot_file, species):
    '''This function will take in a file containing essential proteins downloaded from
    DEG database and parse it to return the essential proteins from given species'''
    pleiotropic_prot_df = pd.read_csv(pleiotropic_prot_file, sep=';',usecols =['Gene'])
    return list(set(pleiotropic_prot_df['Gene']))

def get_uniprot_set(datasets_dir, mapping_settings, prots):
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
    Return a set of essential uniprot IDs
    '''

    #mapping settings
    dataset_settings = master_config_map['dataset_settings']
    datasets_dir = dataset_settings['datasets_dir']
    mapping_settings = dataset_settings.get('mappings')
    species = mapping_settings[0]['species']


    # parse essential protein file
    essential_prot_file = kwargs.get('essential_prot_file')
    species_wise_essential_uniprot_file = essential_prot_file.replace('.csv','_'+species+'.pkl')

    if not os.path.exists(species_wise_essential_uniprot_file):
        essential_prots = parse_essential_prot_file(essential_prot_file, species)
        #get essential uniprots
        essential_uniprots = get_uniprot_set(datasets_dir, mapping_settings, essential_prots)


        os.makedirs(os.path.dirname(species_wise_essential_uniprot_file), exist_ok=True)
        out_file = open(species_wise_essential_uniprot_file, 'wb')
        pickle.dump(essential_uniprots, out_file)
        out_file.close()
        print('done saving essential prot mapping')

    else:
        out_file = open(species_wise_essential_uniprot_file, 'rb')
        essential_uniprots = pickle.load(out_file)
        out_file.close()
        print('done reading essential prot mapping')

    # if kwargs.get('pleiotropic_corr'):
    #     pleiotropic_prot_file = kwargs.get('pleiotropic_prot_file')
    #     pleiotrophic_prots = parse_pleiotropic_prot_file(pleiotropic_prot_file, species)

    return essential_uniprots


def parse_viral_prot_file(viral_prot_file):
   
    viral_df = pd.read_csv(viral_prot_file, sep=',')
    return set(viral_df['Uniprot_Human'])

def handle_percentile_percent_corr(sorted_scores_df, interesting_uniprots, sorted_df_pos=pd.DataFrame(), sorted_df_top_k=pd.DataFrame(),
                                   score_name='betweenness', interesting_prot = 'essential prots'):
    '''  This function will compute the pearson's correlation between percentage of essential
    proteins and percentile of betweenness score'''
    percentiles = []
    percentages_interesting = []
    bin_size=100
    # sorted_scores_df['percent_rank'] = sorted_scores_df[score_name].rank(pct=True)
    e = len(interesting_uniprots) #number of essential prots
    for i in list(range(0, len(sorted_scores_df), bin_size)):
        bin_df = sorted_scores_df.iloc[i:i + bin_size, :]
        lowest_percentile_of_curr_bin = list(bin_df['percent_rank'])[-1]
        percentiles.append(lowest_percentile_of_curr_bin)

        #the following computes what percent of interesting prots is in a bin
        # percentage_of_essential_prot_in_curr_bin= len(set(bin_df['prot']).intersection(essential_uniprots))/float(e)*100

        #the following computes what percent of protein in a bin is interesting prots.
        percentage_of_interesting_prot_in_curr_bin= len(set(bin_df['prot']).intersection(interesting_uniprots)) / bin_size * 100

        percentages_interesting.append(percentage_of_interesting_prot_in_curr_bin)

    pearsons_corr, pvalue = script_utils.pearson_correlations(percentiles,percentages_interesting)

    #also compute MannWhiteney Pval for seeing how significant the difference in percentages_interesting
    #prots in top half ranks vs bottom half ranks
    half= int(len(percentages_interesting)/2)
    _, mw_pval = mannwhitneyu(percentages_interesting[0:half],
                              percentages_interesting[half:], \
                              alternative='greater')



    print('pearsons corr btn percentile of ' + score_name + ' score and percentage of '+ interesting_prot,
          '\ncorr: ', pearsons_corr, 'pval: ', pvalue,
          '\nmw_pval: ', mw_pval)

    # If we want to plot percentage_of_interesting_prots in the positive and top predictions separately
    if (not (sorted_df_pos.empty) and (not sorted_df_top_k.empty)):
        pos_bin_percentile =  list(sorted_df_pos['percent_rank'])[-1]
        pos_bin_percentage = len(set(sorted_df_pos['prot']).intersection(interesting_uniprots)) / float(len(sorted_df_pos)) * 100
        percentiles.append(pos_bin_percentile)
        percentages_interesting.append(pos_bin_percentage)

        top_bin_percentile =  list(sorted_df_top_k['percent_rank'])[-1]
        top_bin_percentage = len(set(sorted_df_top_k['prot']).intersection(interesting_uniprots)) / float(len(sorted_df_top_k)) * 100
        percentiles.append(top_bin_percentile)
        percentages_interesting.append(top_bin_percentage)

    return pearsons_corr, pvalue, mw_pval, percentiles, percentages_interesting


def create_random_scores_df(sorted_scores_df, score_name ='betweenness'):
    rand_df = copy.deepcopy(sorted_scores_df)
    rand_df[score_name] = pd.Series(np.random.rand(len(rand_df)))
    rand_df.sort_values(by=[score_name], ascending=False, inplace=True)
    return rand_df

def interesting_prot_in_topks(sorted_btns_scores_df, essential_uniprots, ks):
    '''
    Return a dict: key= k, value= how many  proteins of interest (might be essential prots or prots that interact with virus) are present
    in corresponding top k prots
    '''
    sorted_prots = list(sorted_btns_scores_df['prot'])
    n_interesting_prots_per_k= {}
    for k in ks:
        top_k_prots = set(sorted_prots[0:k])
        n_interesting_prots_per_k[k] = len(top_k_prots.intersection(essential_uniprots))
    return n_interesting_prots_per_k

def get_top_k_predictions(pred_file, alpha, k, orig_pos):
    '''
    This will return a dataframe with proteins sorted ascendingly according to their predicted score.
    It removes the original positive proteins from the output dataframe.
    '''


    if not os.path.isfile(pred_file):
        print("Warning: %s not found. skipping" % (pred_file))
    else:
        print("reading %s for alpha=%s" % (pred_file, alpha))
        df = pd.read_csv(pred_file, sep='\t')

        # remove the original positives for downstream analysis
        df = df[~(df['prot'].isin(orig_pos))]
        df.reset_index(inplace=True, drop=True)
        df = df[:k]
        return df
    return None

def filter_sorted_src_spec_btns(sorted_df, top_k_predictions_df, orig_pos):
    sorted_df_pos = sorted_df[sorted_df['prot'].isin(orig_pos)]
    sorted_df_wo_orig = sorted_df[~(sorted_df['node_idx'].isin(list(sorted_df_pos['node_idx'])))]
    sorted_df_top_k = sorted_df[sorted_df['prot'].isin(list(top_k_predictions_df['prot']))]
    sorted_df_filtered=sorted_df_wo_orig[~(sorted_df_wo_orig['prot'].isin(list(top_k_predictions_df['prot'])))]

    return sorted_df_filtered, sorted_df_pos,sorted_df_top_k

def compute_percent_interesting_prot(prot_list, interesting_prot_list):
    prot_set = set(prot_list)
    interesting_prot_set = set(interesting_prot_list)

    return float(len(prot_set.intersection(interesting_prot_set)))/len(prot_set) * 100

def main(config_map, master_config_map, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """
    # extract the general variables from the config map
    input_settings, input_dir, output_dir, alg_settings, kwargs \
        = config_utils.setup_config_variables(config_map, **kwargs)

    for dataset in input_settings['datasets']:
        print("Loading data for %s" % (dataset['net_version']))
        # load the network and the positive examples for each term
        net_obj, ann_obj, _ = setup_dataset(dataset, input_dir, **kwargs)
        prots, node2idx = net_obj.nodes, net_obj.node2idx
        ks = [100, 200, 300, 400, 500, 1000, 2000] #the top k betweenness proteins to consider

        # get essential protein list.
        # map essential prots to uniprot ids
        essential_uniprots = handle_essential_uniprot_mapping(master_config_map, **kwargs)
        #get human proteins that interact with viral prots
        viral_prot_file = kwargs.get('viral_prot_file')
        viral_uniprots =  parse_viral_prot_file(viral_prot_file)

        ###Directory for saving any betweenness related analysis result
        btns_out_dir = config_map['output_settings']['output_dir'] + '/betweenness/' + \
                       dataset['net_version'] + '/'

        # #*************************************************************************************************
        # ********************** NETWORK SPECIFIC NODE PROPERTIES AND ESSENTIAL PROTS *********************
        # all types of centrality (betweenness, degree) for nodes which is a graph node property
        # will be saved in the centrality.tsv file
        centr_file = "%s/net_centrality.tsv" % (btns_out_dir)
        sorted_cntr_df = compute_network_centrality_scores(net_obj.W, prots, centr_file)
        sorted_cntr_df['percent_rank'] = sorted_cntr_df['betweenness'].rank(pct=True)


        # compute correlation between network centrality and essential prots and save
        centr_corr_file = "%s/net_centrality_corr.txt" % (btns_out_dir)
        os.makedirs(os.path.dirname(centr_corr_file), exist_ok=True)
        cntr_corr_out = open(centr_corr_file, 'w')
        cntr_corr_out.write('pc_essential' + '\t' + 'pval_essential' + '\t'+'mw_essential' +
                            '\t'+'pc_viral' + '\t' + 'pval_viral'+'\t'+'mw_viral'+'\n')

        #compute pearson's correlation with centrality and essetiality
        #also output the lowest_percentile_of_centrality_score in each bin and corresponding percentage_of_essential_proteins
        #(or viral_interactors) in that bin.
        pc_cntr_ess, pval_cntr_ess, mw_pval_ess, prcntl_ess, prcnt_ess = handle_percentile_percent_corr(sorted_cntr_df, essential_uniprots,
                                                            score_name='betweenness', interesting_prot='essential prots')
        pc_cntr_viral, pval_cntr_viral, mw_pval_viral, prcntl_viral, prcnt_viral  = handle_percentile_percent_corr(sorted_cntr_df, viral_uniprots,
                                                            score_name='betweenness', interesting_prot='viral interactors')
        cntr_corr_out.write(str(pc_cntr_ess) + '\t' + str(pval_cntr_ess) + '\t'+str(mw_pval_ess)+
                            '\t'+str(pc_cntr_viral) + '\t' + str(pval_cntr_viral)+'\t'+str(mw_pval_viral))
        cntr_corr_out.close()

        ### PLOT essential prots in top ks central prots #################
        plot_ess_in_topk_cntr_file = btns_out_dir + 'topk_central_essential_a' + '.pdf'
        n_ess_prots_per_topk_cntr = interesting_prot_in_topks(sorted_cntr_df, essential_uniprots, ks)
        barplot_from_dict(n_ess_prots_per_topk_cntr, x_label='top k network centrality',
                          y_label='number of essential prots', ymax=9000, title= dataset['plot_exp_name'],
                          filename=plot_ess_in_topk_cntr_file)
            
        ### PLOT viral prots in top ks central prots #################
        plot_viral_in_topk_cntr_file = btns_out_dir + 'topk_central_viral_a' + '.pdf'
        n_viral_prots_per_topk_cntr = interesting_prot_in_topks(sorted_cntr_df, viral_uniprots, ks)
        barplot_from_dict(n_viral_prots_per_topk_cntr, x_label='top k network centrality',
                          y_label='number of viral interactors',ymax=8000, title=dataset['plot_exp_name'],
                          filename=plot_viral_in_topk_cntr_file)
        # ********************************************************************************************

        for alg_name in alg_settings:
            if (alg_settings[alg_name]['should_run'][0] == True) or (alg_name in kwargs.get('run_algs')):
                # load the top predictions
                print(alg_name)
                #alg specific betweenness scores and comparison of these btns
                # scores with other centrality measure of network
                # will be save in the following directory
                alg_spec_btns_out_dir = btns_out_dir + alg_name + '/'

                #across each alpha the correlation btn rwr(rl)_betweenness and essentiality is saved here
                alg_btns_corr_file = alg_spec_btns_out_dir + 'btns_corr.tsv'
                os.makedirs(os.path.dirname(alg_btns_corr_file), exist_ok=True)
                corr_fout = open(alg_btns_corr_file, 'w')
                corr_fout.write('alpha' + '\t' + 'pc_essential'+ '\t' + 'pc_pval_essential' +'\t'+'mw_essential'
                                +'\t'+ 'pc_viral'+ '\t' + 'pc_pval_viral'+'\t'+'mw_viral'
                                +'\t' + 'ktau_centrality' + '\t' + 'ktau_pval_centrality'  +'\n')

                plot_essential_in_topk_file = alg_spec_btns_out_dir + 'topk_btns_essential'+ '.pdf'
                plot_viral_in_topk_file = alg_spec_btns_out_dir + 'topk_btns_viral'+ '.pdf'


                n_interesting_prots_per_topk_df = pd.DataFrame()
                # n_interesting_prots_per_term_spec_topk_df = dict(pd.DataFrame())

                #Get prediction files
                alg_pred_files = config_utils.get_dataset_alg_prediction_files(
                    output_dir, dataset, alg_settings, [alg_name], **kwargs)
                # get the alpha values to use
                alphas = alg_settings[alg_name]['alpha']

                count=0
                for alpha, alg in zip(alphas, alg_pred_files):
                    count+=1
                    beta = alpha
                    if alg_name == 'genemaniaplus':
                        beta = round(1.0 / (1 + alpha), 2)
                    M_pathmtx_loginv = get_M_pathmtx_loginv(net_obj, alg_name, alpha)

                    # betweenness_score_file contains rwr(rl)betweenness score of
                    # all proteins in a network given certain alpha
                    btns_file = alg_spec_btns_out_dir + 'btns_a' + str(alpha) + '.tsv'
                    sorted_btns_scores_df = handle_betweenness_score(M_pathmtx_loginv, prots, btns_file)
                    
                    #compute Pearsons correlation btn betweenness score and essentiality
                    pc_ess, pval_ess,mw_pval_ess, prcntl_ess,prcnt_ess = handle_percentile_percent_corr(sorted_btns_scores_df, essential_uniprots)
                    # compute Pearsons correlation btn betweenness score and viral prots
                    pc_viral, pval_viral, mw_pval_viral, prcntl_viral, prcnt_viral = handle_percentile_percent_corr(sorted_btns_scores_df, viral_uniprots,
                                                                        interesting_prot='viral interactors')

                    # also compute kendal_tau score between protein list sorted by betweenness score and network centrality score respectively
                    ktau_corr, ktau_pvalue  = \
                        script_utils.kendal_tau(list(sorted_cntr_df['prot']), list(sorted_btns_scores_df['prot']))
                    #SAVE correlation scores
                    corr_fout.write(str(beta) + '\t'+str(pc_ess)+'\t'+ str(pval_ess)+'\t'+ str(mw_pval_ess)+
                                    '\t'+str(pc_viral)+'\t'+ str(pval_viral)+'\t'
                                    +str(ktau_corr)+'\t'+str(ktau_pvalue)+'\t'+ str(mw_pval_viral)+'\n')

                    #Also plot a scatter plot of percentile-betweenness-score and percentage of essential prot per bin
                    percentile_percentage_btns_ess_plot_file = alg_spec_btns_out_dir + 'percentile_percentage_btns_essential_a' + str(alpha) + '.pdf'
                    scatter_plot(prcntl_ess, prcnt_ess, x_label='percentile rank',
                                 y_label='percentage of essential prot', ymin=0, ymax=100,
                                 title=alg_plot_name[alg_name]+'_a_'+str(alpha)+'_'+dataset['plot_exp_name'],
                                 filename= percentile_percentage_btns_ess_plot_file )

                    # Also plot a scatter plot of percentile-betweenness-score and percentage of essential prot per bin
                    percentile_percentage_btns_viral_plot_file = alg_spec_btns_out_dir + 'percentile_percentage_btns_viral_a' + str( alpha) + '.pdf'
                    scatter_plot(prcntl_viral, prcnt_viral, x_label='percentile rank',
                                 y_label='percentage of viral interactors', ymin=0, ymax=100,
                                 title=alg_plot_name[alg_name]+'_a_'+str(alpha)+'_'+dataset['plot_exp_name'],
                                 filename=percentile_percentage_btns_viral_plot_file)

                    #compute how many essential proteins present in different top scoring k prots
                    n_essential_prots_per_topk = interesting_prot_in_topks(sorted_btns_scores_df, essential_uniprots, ks)
                    n_viral_prots_per_topk = interesting_prot_in_topks(sorted_btns_scores_df, viral_uniprots, ks)


                    temp_df = pd.DataFrame({'alpha': [beta]*len(ks), 'top_k':ks,
                                        'n_essential_prot':list(n_essential_prots_per_topk.values()),
                                        'n_viral_interactor': list(n_viral_prots_per_topk.values())})

                    n_interesting_prots_per_topk_df = pd.concat([n_interesting_prots_per_topk_df, temp_df], axis=0)

                    for term in ann_obj.terms:
                        term_idx = ann_obj.term2idx[term]
                        orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, term_idx)
                        orig_pos = [prots[p] for p in orig_pos_idx]
                        pos_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]

                        alg_term_spec_btns_out_dir = alg_spec_btns_out_dir+'/'+dataset['exp_name']+'/'
                        src_spec_btns_file = alg_term_spec_btns_out_dir +'btns_a' + str(alpha) + '.tsv'

                        #Compute BETWEENNESS score for each protein  and get a dataframe with proteins sorted in ascending order of btns score
                        sorted_src_spec_btns_df = handle_src_spec_betweenness_score(M_pathmtx_loginv,prots,src_spec_btns_file, pos_nodes_idx)

                        #Now get the top predicted proteins
                        #here k= len(orig_pos)
                        pred_file = alg_pred_files[alg]
                        pred_file = script_utils.term_based_pred_file(pred_file, term)
                        top_k_predictions_df = get_top_k_predictions(pred_file, alpha, len(orig_pos) , orig_pos)

                        ##If 'sep_pos_top' is true then filter the sorted_src_spec_btns_df to contain only
                        # the non-source, non-top predictions
                        if kwargs.get('sep_pos_top'):
                            sorted_src_spec_btns_df, sorted_df_pos, sorted_df_top_k = \
                                filter_sorted_src_spec_btns(sorted_src_spec_btns_df, top_k_predictions_df, orig_pos)
                        else:
                            sorted_df_top_k = pd.DataFrame()
                            sorted_df_pos = pd.DataFrame()
                        ###Do analysis to find relationship between betweenness score and  ESSENTAIL PROT
                        src_spec_pc_ess, src_spec_pval_ess, src_spec_mw_pval_ess, prcntl_ess, prcnt_ess =\
                            handle_percentile_percent_corr(sorted_src_spec_btns_df,essential_uniprots, sorted_df_pos, sorted_df_top_k)
                        # Also plot a scatter plot of percentile-score and percentage of essential prot per bin
                        percentile_percentage_btns_ess_plot_file = alg_term_spec_btns_out_dir + 'percentile_percentage_btns_essential_a' +\
                                                                   str(alpha) + '.pdf'

                        #If 'sep_pos_top' is true then compute percentage of essential(or viral interactors) in
                        # source proteins and  top predictions. then add them to prcnt

                        scatter_plot(prcntl_ess, prcnt_ess,  x_label='percentile rank',
                                     y_label='percentage of essential prot', sep_pos_top = kwargs.get('sep_pos_top'),
                                     ymin=0, ymax=100,
                                     title=alg_plot_name[alg_name] + '_a_'+str(alpha)+'_' +term+'_'+ dataset['plot_exp_name'],
                                     filename=percentile_percentage_btns_ess_plot_file)
                        ##Do analysis to find relationship between betweenness score and  VIRAL PROT
                        src_spec_pc_viral, src_spec_pval_viral, src_spec_mw_pval_viral, prcntl_viral, prcnt_viral = \
                            handle_percentile_percent_corr(sorted_src_spec_btns_df,viral_uniprots, sorted_df_pos, sorted_df_top_k)
                        # Also plot a scatter plot of percentile-score and percentage of essential prot per bin
                        percentile_percentage_btns_viral_plot_file = alg_term_spec_btns_out_dir + 'percentile_percentage_btns_viral_a' + str(alpha) + '.pdf'
                        scatter_plot(prcntl_viral, prcnt_viral, x_label='percentile rank',
                                     y_label='percentage of viral interactors', sep_pos_top = kwargs.get('sep_pos_top'),
                                     ymin=0, ymax=100,
                                     title=alg_plot_name[alg_name] + '_a_'+str(alpha)+'_' +term+'_'+ dataset['plot_exp_name'],
                                     filename=percentile_percentage_btns_viral_plot_file)

                        #plot percentile rank for source proteins and top_k_predictions
                        box_plot(data = pd.DataFrame({ 'percent_rank':list(sorted_df_pos['percent_rank'])+(list(sorted_df_top_k['percent_rank'])),
                                'node_spec': (['pos']*len(sorted_df_pos))+(['top_pred']*len(sorted_df_top_k)) }),
                                 x = 'node_spec', y='percent_rank', ymin=0, ymax=1,
                                 title=alg_plot_name[alg_name] + '_a_'+str(alpha)+'_' +term+'_'+ dataset['plot_exp_name'],
                                 filename=alg_term_spec_btns_out_dir+'percentile_rank_pos_top_a' + str(alpha) + '.pdf')



                        # also compute kendal_tau score between protein list sorted by betweenness score and
                        # network centrality score respectively
                        # ktau_corr, ktau_pvalue = script_utils.kendal_tau(list(sorted_cntr_df['prot']),
                        #                                                   list(sorted_src_spec_btns_df['prot']))
                        # SAVE
                        alg_term_btns_essentaility_corr_file = alg_term_spec_btns_out_dir + 'btns_corr.tsv'
                        os.makedirs(os.path.dirname(alg_term_btns_essentaility_corr_file), exist_ok=True)

                        if count==1: #that means the result is for the first alpha value, so need a new file
                            corr_term_fout = open(alg_term_btns_essentaility_corr_file, 'w')
                            # corr_term_fout.write('alpha' + '\t' + 'pc_essential' + '\t' + 'pc_pval_essential' + '\t' +
                            #                      'pc_viral' + '\t' + 'pc_pval_viral' + '\t' + 'ktau_centrality' + '\t' + 'ktau_pval_centrality' + '\n')

                            corr_term_fout.write('alpha' + '\t' + 'pc_essential' + '\t' + 'pc_pval_essential'+'\t' +'mw_essential'
                                                 + '\t' +'pc_viral' + '\t' + 'pc_pval_viral' +'\t' +'mw_viral'+'\n')

                        else:
                            corr_term_fout = open(alg_term_btns_essentaility_corr_file, 'a')

                        # corr_term_fout.write(str(beta) + '\t' + str(src_spec_pc_ess) +'\t' + str(src_spec_pval_ess) +
                        #                      '\t' + str(src_spec_pc_viral) + '\t' + str(src_spec_pval_viral)+
                        #                      '\t' + str(ktau_corr) + '\t' + str(ktau_pvalue) + '\n')

                        corr_term_fout.write(str(beta) + '\t' + str(src_spec_pc_ess) + '\t' + str(src_spec_pval_ess) +
                                            '\t'+str(src_spec_mw_pval_ess)+
                                             '\t' + str(src_spec_pc_viral) + '\t' + str(src_spec_pval_viral) +
                                             '\t'+ str(src_spec_mw_pval_viral)+'\n')
                        corr_term_fout.close()


                        # ##Amount of intersting prots in top k
                        # n_essential_prots_term_topk = interesting_prot_in_topks(sorted_src_spec_btns_df,
                        #                                                        essential_uniprots, ks)
                        # n_viral_prots_term_topk = interesting_prot_in_topks(sorted_src_spec_btns_df, viral_uniprots, ks)
                        # 
                        # temp_df = pd.DataFrame({'alpha': [beta] * len(ks), 'top_k': ks,
                        #                         'n_essential_prot': list(n_essential_prots_per_topk.values()),
                        #                         'n_viral_interactor': list(n_viral_prots_per_topk.values())})
                        # 
                        # n_interesting_prots_per_term_spec_topk_df[term] = \
                        #     pd.concat([n_interesting_prots_per_term_spec_topk_df[term]
                        #                if term in n_interesting_prots_per_term_spec_topk_df else pd.DataFrame(), temp_df], axis=0)

                #*************************** PLOT ****************************
                # plot how many essential proteins present in different top scoring k prots
                barplot_from_df(n_interesting_prots_per_topk_df, filename=plot_essential_in_topk_file, x='top_k',y='n_essential_prot',
                                ymax=9000, title= alg_plot_name[alg_name]+'_'+dataset['plot_exp_name'])
                #plot how many viral proteins present in different top scoring k prots
                barplot_from_df(n_interesting_prots_per_topk_df, filename=plot_viral_in_topk_file,  x='top_k', y='n_viral_interactor',
                                ymax= 8000, title= alg_plot_name[alg_name]+'_'+dataset['plot_exp_name'])
                corr_fout.close()

                # ##PLot term specific how many essential proteins present in different top scoring k prots
                # for term in n_interesting_prots_per_term_spec_topk_df:
                #     plot_filename = alg_term_spec_btns_out_dir + 'topk_btns_essential'+ '.pdf'
                #     barplot_from_df(n_interesting_prots_per_term_spec_topk_df[term], filename=plot_filename, x='top_k',
                #         y='n_essential_prot', ymax=9000, title=alg_plot_name[alg_name] + '_' +term+'_'+ dataset['plot_exp_name'])
                # 
                #     plot_filename = alg_term_spec_btns_out_dir + 'topk_btns_viral'+ '.pdf'
                #     barplot_from_df(n_interesting_prots_per_term_spec_topk_df[term], filename=plot_filename, x='top_k',
                #         y='n_viral_interactor', ymax=8000, title=alg_plot_name[alg_name] + '_' +term+'_'+ dataset['plot_exp_name'])
                # 
                # print('done')

def barplot_from_dict(n_essential_prots_per_topk, x_label, y_label,ymax, filename,title=''):
    '''
    n_essential_prots_per_topk: dict where key = k, value: number of essential prot in top k
    '''

    X = list(n_essential_prots_per_topk.keys())
    Y =  list(n_essential_prots_per_topk.values())
    sns.barplot(x=X, y=Y)

    plt.ylim([0,ymax])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.savefig(filename.replace('.pdf', '.png'))  # save plot in .png format as well
    # plt.show()
    plt.close()
    print('Save fig to ', filename)

def barplot_from_df(n_interesting_prots_per_topk_df, filename, x, y,ymax, title=''):
    '''
    n_essential_prots_per_topk: dict where key = k, value: number of essential prot in top k
    '''
    sns.barplot(n_interesting_prots_per_topk_df, x = x, y = y, hue='alpha')

    plt.ylim([0, ymax])
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.savefig(filename.replace('.pdf', '.png'))  # save plot in .png format as well
    # plt.show()
    plt.close()
    print('Save fig to ', filename)

def scatter_plot(X, Y,  x_label, y_label, sep_pos_top=False, ymin=None, ymax=None, title='', filename=''):

    if sep_pos_top:
        #in this case the last to points are for positive/src_bin and to_predicted_proteins_bin. And we want to show them in
        #different color
        color = ['blue']*(len(X)-2) + ['red','green']
    else:
        color = ['blue']*len(X)
    plt.scatter(X,Y,color = color)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim([ymin, ymax])
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.savefig(filename.replace('.pdf', '.png'))  # save plot in .png format as well
    # plt.show()
    plt.close()
    print('Save fig to ', filename)

def box_plot(data, x, y, ymin, ymax, title, filename):
    sns.boxplot(data, x=x, y=y )

    plt.ylim([ymin, ymax])
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.savefig(filename.replace('.pdf', '.png'))  # save plot in .png format as well
    # plt.show()
    plt.close()
    print('Save fig to ', filename)
    
def stat_on_random_score(sorted_btns_scores_df, essential_uniprots):
    # see what happens in terms of correlation if the betweenness score is given randomly
    rand_pearsons_corr_list=[]
    rand_pvalue_list = []
    for i in range(0, 1000):
        rand_sorted_btns_scores_df = create_random_scores_df(sorted_btns_scores_df[['prot']])
        rand_corr,rand_pval,percentiles,percentages_essential  = \
            handle_percentile_percent_corr(rand_sorted_btns_scores_df,
                                           essential_uniprots)
        rand_pearsons_corr_list.append(rand_corr)
        rand_pvalue_list.append(rand_pval)
    mean_rand_corr = np.mean(rand_pearsons_corr_list)
    max_rand_corr = np.max(rand_pearsons_corr_list)
    min_rand_corr = np.min(rand_pearsons_corr_list)
    print('rand corr:')
    print('mean: ', mean_rand_corr )
    print('max: ', max_rand_corr)
    print('min: ', min_rand_corr)

if __name__ == "__main__":
    config_map, master_config_map, kwargs = parse_args()
    main(config_map,master_config_map, **kwargs)
