import numpy as np
import scipy.stats as stats
import os
import pandas as pd

from src.FastSinkSource.src.algorithms import rl_genemania as gm
from src.FastSinkSource.src.algorithms import PageRank_Matrix as rwr

alg_alias = {'rwr': rwr, 'genemaniaplus': gm}

#************************************* NETWORK PROCESSING *********************************************
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

#***************************************** ALGO PREDICTIONS SCORE *********************************************
def get_top_k_predictions(pred_file, alpha, k, orig_pos):
    '''
    This will return a dataframe with proteins sorted descendingly according to their predicted score.
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

def term_based_pred_file(pred_file, term):
    pred_file = \
        pred_file.split('.')[0] + '-' +term + '.' + pred_file.split('.')[-1]
    #also in GO term we might see ':' which we replaced by '-' while writing the pediction to file. so consider that here.
    pred_file  = pred_file.replace(':','-')
    return pred_file


def get_balancing_alpha(config_map, dataset, alg_name,  term):
    alpha_summary_filename = config_map['output_settings']['output_dir'] + \
                             "/viz/%s/%s/param_select/" % (dataset['net_version'], dataset[
        'exp_name']) + '/' + alg_name + '/alpha_summary.tsv'
    alpha_summary_df = pd.read_csv(alpha_summary_filename, sep='\t', index_col=None)[
        ['term', 'balancing_alpha']]
    term_2_balancing_alpha_dict = dict(
        zip(alpha_summary_df['term'], alpha_summary_df['balancing_alpha']))

    balancing_alpha = term_2_balancing_alpha_dict[term]
    return balancing_alpha
#************************* NETWORK ANALYSIS*************************
def is_neighbor(W, source, target):
    ''' This will return True if there is an edge from the source to target'''
    ''' Consider the Weight matrix in a format that along rows we have targets, along columns we have sources'''
    if W[target][source]!=0:
        return True
    return False

def find_in_neighbors(W, target):
    ''' Find the nodes from which there are incoming edges to the target'''
    neighbors = np.where(W[target]>0)[0]
    return list(neighbors)

def is_diag_zero(W):
    if W.diagonal().sum()==0:
        return True
    else:
        return False


#********************** STATISTICAL TEST ******************************
def pearson_correlations(list1, list2):
    #write code for finding pearson correlations coefficient here
    return stats.pearsonr(list1, list2)

def kendal_tau(list1, list2):
    kt = stats.kendalltau(list1, list2)
    return kt.correlation, kt.pvalue


#*************************** FILE WRITE *********************************
def save_dict(dict1, filename):
    #write key and values of a dict in tab separated way
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    f = open(filename, 'w')
    for key in dict1:
        val = "{:.2e}".format(dict1[key])
        f.write(str(key) + '\t' + val +'\n')
    f.close()


