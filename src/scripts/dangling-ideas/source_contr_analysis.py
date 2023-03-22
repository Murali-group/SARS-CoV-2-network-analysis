# here's an example call to this script:
# python src/scripts/effective_diffusion_node_path.py --config fss_inputs/config_files/provenance/provenance_biogrid_y2h_go.yaml
# --run-algs genemaniaplus --k 500 --m 20 --n-sp 500

import os, sys
import yaml
import argparse
import numpy as np
import matplotlib
import networkx as nx
import copy
import time
import scipy.sparse as sp
from scipy.sparse import eye, diags
from scipy.linalg import inv


HIGH_WEIGHT = 1e6

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

alg_alias = {'rwr': rwr, 'genemaniaplus': gm}

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
    parser = argparse.ArgumentParser(description="Script to analyze the top contributors of each prediction's "
                                                 "diffusion score, as well as the effective diffusion (i.e.,"
                                                 " fraction of diffusion received from non-neighbors)")
    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                        "fss_inputs/config_files/provenance/provenance_string700v11.5_s12.yaml"
                       , help="Configuration file used when running FSS. ")
    group.add_argument('--run-algs', type=str, action='append', default=[])
    group.add_argument('--k-to-test', '-k', type=int, action='append', default=[332],
                       help="k-value(s) for which to get the top-k predictions to test. " +
                            "If not specified, will check the config file.")
    group.add_argument('--pos-k', action='store_true', default=False,
                       help="if true get the top-k predictions to test is equal to the number of positive annotations")
    group.add_argument('--m', type=int, default=350,
                       help="How many top predictions to consider")
    group.add_argument('--stat-sig-cutoff', type=float,
                       help="Cutoff on the node p-value for a node to be considered in the topk. " + \
                            "The p-values should already have been computed with run_eval_algs.py")
    group.add_argument('--balancing-alpha-only', action='store_true', default=True,
                       help="Ignore alpha from config file rather take the alpha value\
                            that balanced the two loss terms in quad loss function for the corresponding\
                            network-term-alg")
    return parser


def find_top_m_contributing_sources_per_pred(m, top_pred_idx,pos_nodes_idx, M_inv, cutoff=0):
    '''
    Returns two dictionaries:
    1. dict with top predicted proteins as keys and list of( atleast size m) top contributing sources of
    each prediction as values. If the contribution from a source<cutoff, it will not be in the list.
    2. dict with key = source and value = #of_top_preds for which this source was a top contributor.
    '''
    # any M_inv where sources-> columns and targets -> rows, this function will work
    top_sources_per_pred = {}
    src_as_top_contr = {x:0 for x in pos_nodes_idx} #key=source, value = #of top preds for which this source was a top contributor
    # find some top m contributing sources for each of top k(same as k_to_test) predictions.
    mask = np.zeros(M_inv.shape[1], dtype=bool)
    mask[pos_nodes_idx] = True
    M_inv_new = M_inv[:, mask]  # keep the columns with source/pos_nodes only
    # print(M_inv.shape, M_inv_new.shape)
    # for prot_idx in list(range(M_inv.shape[0])):
    for prot_idx in top_pred_idx:

        # get contribution from source nodes for each top predictions and sort. Also keep track of which source node
        # contributed how much
        pos_nodes_idx.sort(reverse=False)  # sort ascending
        contr_vals = list(M_inv_new[prot_idx])  # in this 'contr' list, values are
                                  # sorted according to the source nodes' index values
        per_src_contr_dict = dict(zip(pos_nodes_idx, contr_vals))
        #sort sources according to their contribution
        per_src_contr_dict = dict(sorted(per_src_contr_dict.items(),
                            key=lambda item: item[1], reverse=True))

        # contribtution of the source has to be atleast the cutoff times the total score.
        # cutoff=0.01 means 1% of total score
        final_cutoff = sum(M_inv_new[prot_idx])*(cutoff)
        per_src_contr_dict = {key: val for key, val in per_src_contr_dict.items() if val > final_cutoff}
        # take only top m contributing sources whose contribution is atleast>final_cutoff
        top_sources_per_pred[prot_idx] = list(per_src_contr_dict.keys())[0:min(m, len(per_src_contr_dict))]
        for src in top_sources_per_pred[prot_idx]:
            src_as_top_contr[src]+=1
    src_as_top_contr = dict(sorted(src_as_top_contr.items(), key=lambda item: item[1], reverse=True))
    return top_sources_per_pred, src_as_top_contr
#
# def compute_frac_intermediate_nodes_contr(M, M_inv, pred_scores, top_k_preds, pos_nodes_idx, alpha,
#                                           mat_inverse_file, force_run = False):
#     '''
#     This function will compute the contribution from intermediate nodes to final score of other nodes whereas the intermediate
#     nodes can be the first, second or third node in a path via which the contribution to score in coming along.
#     '''
#     pred_score_inv = np.divide(1., pred_scores).reshape(-1, 1)  # a column vector
#     pred_score_inv[np.isinf(pred_score_inv)] = 0
#
#     e = np.zeros(M.shape[1])
#     e[pos_nodes_idx] = 1./len(pos_nodes_idx)
#     e = sp.csc_matrix(e.reshape(-1,1))
#
#     # there might be some source from which no edge is going out i.e. source in a dangling node. For those
#     # what we did in RWR is broadcast to every other node with equal probability. To account for that, here we
#     # can look for sources for which we do not have any outgoing edges and create edges from those sources
#     # to all other nodes with equal edge_weight=1/M.shape[0]
#     #TODO verify the above logic with Murali
#     N=M.shape[1]
#     norm_deg = np.array(np.sum(M, axis=0))[0]
#
#     # for i in range(N):
#     #     if norm_deg[i]==0: #the i'th node(column) has no outgoing edges
#     #         M[:,i] = 1.0/N
#
#     M1 = (1-alpha)*M
#     M2 = M1.dot(M1)
#     I = eye(M.shape[0])
#
#     mat2_inverse_file = mat_inverse_file.replace('diffusion-mat', 'diffusion-mat2')
#     if (not os.path.exists(mat2_inverse_file)|(force_run==True)):
#         M2_inv = inv((I- M2).A)
#          # save M2_inv for later use
#         os.makedirs(os.path.dirname(mat2_inverse_file), exist_ok=True)
#         np.save(mat2_inverse_file, M2_inv)
#     else:
#         M2_inv = np.load(mat2_inverse_file)
#
#     P2 = ((alpha*M1 + alpha*I).dot(e)).A.reshape(1,-1) #conver P2 into a row matrix
#     # elementwise multiplication. The one row of P2 multiplies with each row of M2_inv elementwise
#     first_intermediate_nodes_contr = np.multiply(M2_inv, P2)
#
#     # elementwise multiplication. pred_score_inv is a column matrix.
#     # The one column of <pred_score_inv> multiplies with each column of <first_intermediate_nodes_contr> elementwise
#     frac_first_intermediate_nodes_contr = np.multiply(first_intermediate_nodes_contr, pred_score_inv)
#
#     #testing
#     print('pred score of 4109: ', pred_scores[4109])
#     print('contr intermediate: 14042, target: 4109 ', np.asarray(first_intermediate_nodes_contr)[4109][14042])
#     print('frac contr: ', np.asarray(frac_first_intermediate_nodes_contr)[4109][14042])
#
#     M3 = M2.dot(M1)
#     mat3_inverse_file = mat_inverse_file.replace('diffusion-mat', 'diffusion-mat3')
#     if (not os.path.exists(mat3_inverse_file)|force_run==True):
#         M3_inv = inv((I-M3).A)
#         # save M2_inv for later use
#         os.makedirs(os.path.dirname(mat3_inverse_file), exist_ok=True)
#         np.save(mat3_inverse_file, M3_inv)
#     else:
#         M3_inv = np.load(mat3_inverse_file)
#
#     P3 = ((alpha*M2 + alpha*M1 + alpha*I).dot(e)).A.reshape(1,-1)
#     second_intermediate_nodes_contr = np.multiply(M3_inv, P3)  # do elementwise multiplication
#     frac_second_intermediate_nodes_contr = np.multiply(second_intermediate_nodes_contr, pred_score_inv)
#
#     #total contribution
#     intermediate_nodes_contr = first_intermediate_nodes_contr
#     # frac_intermediate_nodes_contr = np.multiply(intermediate_nodes_contr_1_2, pred_score_inv)
#
#
#
#     ##sanity check
#     scores_0 = alpha*np.matmul(M_inv, e.A.reshape(-1,1))
#     scores_1 = np.matmul(M2_inv, P2.reshape(-1,1))
#     scores_2 = np.matmul(M3_inv, P3.reshape(-1,1))
#
#
#     for idx in top_k_preds:
#         assert ((abs(pred_scores[idx]-scores_0[idx])/pred_scores[idx])<10e-4), print('non-matching score')
#         assert ((abs(pred_scores[idx]-scores_1[idx])/pred_scores[idx])<10e-4), print('non-matching score')
#         assert ((abs(pred_scores[idx]-scores_2[idx])/pred_scores[idx])<10e-4), print('non-matching score')
#
#
#     assert ((frac_first_intermediate_nodes_contr < 1).all()), \
#         print('greater than 1 values in frac_first_intermediate_contr')
#     #
#     # assert (frac_intermediate_nodes_contr < 1).all(), \
#     #     print('greater than 1 values in frac_intermediate_contr')
#
#     del M1, M2, I, M2_inv, M3_inv, P2, P3, frac_first_intermediate_nodes_contr, \
#         frac_second_intermediate_nodes_contr, first_intermediate_nodes_contr, second_intermediate_nodes_contr
#     return np.array(intermediate_nodes_contr)

def compute_frac_intermediate_nodes_contr(M_pathmtx, R, pos_nodes_idx, top_k_pred_idx):
    R_sum = np.sum(R[:, pos_nodes_idx], axis=1)
    R_sum_inv = np.divide(1., R_sum).reshape([-1, 1])  # column vector
    R_sum_inv[np.isinf(R_sum_inv)] = 0

    M = np.power(np.full_like(M_pathmtx, 10), (-1) * M_pathmtx)
    # in M_pathmtx an index with value 0 means no edge. The above statement turns all 0 values to 1.
    # Now to preserve the same meaning for such indices, we need to convert all 1 to 0 in M.
    where_1 = np.where(M == 1)  # TODO: replace 1's with 0's in time efficient way
    M[where_1] = 0

    # Also take transpose of M to make the computation clearer i.e. after transpose
    # along the row I have u and along column I have v for every (u,v) edge.
    M1 = M.T
    X1 = np.sum(M1[pos_nodes_idx, :], axis=0).reshape(-1,1) #column vector

    M2 = np.matmul(M1,M1)
    X2 = np.sum(M2[pos_nodes_idx, :], axis=0).reshape(-1,1) #column matrix

    M3 = np.matmul(M2,M1)
    X3 = np.sum(M3[pos_nodes_idx, :], axis=0).reshape(-1,1) #column matrix

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

    frac_intermediate_nodes_contr_mat = np.multiply(intermediate_nodes_contr_mat, R_sum_inv)

    del M1, X1, M2, X2,M3, X3, R_sum, R_sum_inv
    return intermediate_nodes_contr_mat, frac_intermediate_nodes_contr_mat


def analyse_intermediate_nodes_for_top_k(frac_intermediate_nodes_contr, top_k_pred_idx, node2prots, c=5):

    top_c_intermediate_nodes_for_top_k_dict = {}
    top_c_intermediate_contr_for_top_k_dict = {}
    top_c_intermediate_prots_for_top_k_dict = {}
    for pred_idx in top_k_pred_idx:
        # this sort is in ascending order
        intermediate_nodes_sorted_by_contr = np.argsort(frac_intermediate_nodes_contr[pred_idx])

        top_c_intermediate_nodes_for_top_k_dict[pred_idx] = intermediate_nodes_sorted_by_contr[-c:]
        top_c_intermediate_contr_for_top_k_dict[pred_idx] = frac_intermediate_nodes_contr[pred_idx]\
                            [top_c_intermediate_nodes_for_top_k_dict[pred_idx]]

        top_c_intermediate_prots_for_top_k_dict[node2prots[pred_idx]] = [node2prots[x]
                                for x in top_c_intermediate_nodes_for_top_k_dict[pred_idx]]
    return top_c_intermediate_nodes_for_top_k_dict, top_c_intermediate_contr_for_top_k_dict,\
           top_c_intermediate_prots_for_top_k_dict


def main(config_map, k, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """

    # extract the general variables from the config map
    input_settings, input_dir, output_dir, alg_settings, kwargs \
        = config_utils.setup_config_variables(config_map, **kwargs)

    sig_cutoff = kwargs.get('stat_sig_cutoff')
    sig_str = "-sig%s" % (str(sig_cutoff).replace('.', '_')) if sig_cutoff else ""
    # m = kwargs.get('m')

    # for each dataset, extract the path(s) to the prediction files,
    # read in the predictions, and test for the statistical significance of overlap

    for dataset in input_settings['datasets']:
        print("Loading data for %s" % (dataset['net_version']))
        # load the network and the positive examples for each term
        net_obj, ann_obj, _ = setup_dataset(
            dataset, input_dir, **kwargs)
        prots, node2idx = net_obj.nodes, net_obj.node2idx

        for term in ann_obj.terms:
            term_idx = ann_obj.term2idx[term]
            orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, term_idx)
            orig_pos = [prots[p] for p in orig_pos_idx]
            pos_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]
            n_pos = len(pos_nodes_idx)

            # If 'pos_k'=True, then the number of top predictions is equal to the number of positively annotated nodes
            # for this certain term.
            if kwargs.get('pos_k'):
                k = n_pos
                print('k: ', k)
            for alg_name in alg_settings:
                if (alg_settings[alg_name]['should_run'][0] == True) or (alg_name in kwargs.get('run_algs')):
                    # load the top predictions
                    print(alg_name)
                    if kwargs.get('balancing_alpha_only'):  # in alg_setting[alg_name]['alpha'] put the balancing alpha
                        # get the balancing alpha for this network - alg - term
                        alpha_summary_filename = config_map['output_settings']['output_dir'] + \
                                                 "/viz/%s/%s/param_select/" % (dataset['net_version'], dataset[
                            'exp_name']) + '/' + alg_name + '/alpha_summary.tsv'
                        alpha_summary_df = pd.read_csv(alpha_summary_filename, sep='\t', index_col=None)[
                            ['term', 'balancing_alpha']]
                        term_2_balancing_alpha_dict = dict(
                            zip(alpha_summary_df['term'], alpha_summary_df['balancing_alpha']))

                        balancing_alpha = term_2_balancing_alpha_dict[term]
                        alg_settings[alg_name]['alpha'] = [balancing_alpha]

                    alg_pred_files = config_utils.get_dataset_alg_prediction_files(
                        output_dir, dataset, alg_settings, [alg_name], **kwargs)
                    # get the alpha values to use
                    alphas = alg_settings[alg_name]['alpha']

                    for alpha, alg in zip(alphas, alg_pred_files):
                        pred_file = alg_pred_files[alg]
                        pred_file = script_utils.term_based_pred_file(pred_file, term)
                        if not os.path.isfile(pred_file):
                            print("Warning: %s not found. skipping" % (pred_file))
                            continue
                        print("reading %s for alpha=%s" % (pred_file, alpha))
                        df = pd.read_csv(pred_file, sep='\t')

                        # remove the original positives for downstream analysis
                        df = df[~df['prot'].isin(orig_pos)]
                        df.reset_index(inplace=True, drop=True)

                        if sig_cutoff:
                            df = config_utils.get_pvals_apply_cutoff(df, pred_file, **kwargs)

                        if k > len(df['prot']):
                            print("ERROR: k %s > num predictions %s. Quitting" % (k, len(df['prot'])))
                            sys.exit()
                        pred_scores = np.zeros(len(net_obj.nodes))
                        df = df[:k]
                        top_k_pred = df['prot']
                        top_k_pred_idx = [net_obj.node2idx[n] for n in top_k_pred]
                        pred_scores[top_k_pred_idx] = df['score'].values

                        # No need for including dataset['exp_name'] as the following matrix are seed node independent.
                        diff_mat_file = "%s/diffusion-mat-%s-a%s.npy" % (net_obj.out_pref, alg_name,
                                                                         str(alpha).replace('.', '_'))
                        fluid_flow_mat_file_M = "%s/fluid-flow-mat-%s-a%s.npy" % (net_obj.out_pref, alg_name,
                                                                                  str(alpha).replace('.', '_'))
                        fluid_flow_mat_file_R = "%s/fluid-flow-mat-R-%s-a%s.npy" % (net_obj.out_pref, alg_name,
                                                                                    str(alpha).replace('.', '_'))

                        ##########CREAE or LOAD THE DIFFUSION MAYTRICES
                        force_matrix = False

                        # M_inv = alg_alias[alg_name].get_diffusion_matrix(net_obj.W, alpha=alpha,
                        #                                                  diff_mat_file=diff_mat_file,
                        #                                                  force_run=force_matrix)
                        M_pathmtx, R = alg_alias[alg_name].get_fluid_flow_matrix(net_obj.W, alpha=alpha, \
                                                                                 fluid_flow_mat_file_M=fluid_flow_mat_file_M, \
                                                                                 fluid_flow_mat_file_R=fluid_flow_mat_file_R,
                                                                                 force_run=force_matrix)
                        # if alg_name == 'rwr':
                        #     # M_inv = M_inv * (alpha / float(len(pos_nodes_idx)))
                        #     M = alg_utils._net_normalize(net_obj.W, norm='full')
                        #
                        #     # #test
                        #     # deg = np.asarray(net_obj.W.sum(axis=0)).flatten()
                        #     # deg = np.divide(1., deg)
                        #     # deg[np.isinf(deg)] = 0
                        #     # D = sp.diags(deg)
                        #     # P = net_obj.W.A.dot(D.A)
                        #     # print(np.array_equal(M.A, P))
                        #     # del D, P
                        #
                        # top_sources_per_pred, src_as_top_contr = \
                        #     find_top_m_contributing_sources_per_pred(kwargs.get('m'),\
                        #             top_k_pred_idx, pos_nodes_idx, M_inv)

                        #the following analysis is only for RWR for now.

                        intermediate_nodes_contr, frac_intermediate_nodes_contr  = compute_frac_intermediate_nodes_contr\
                                                        (M_pathmtx, R, pos_nodes_idx, top_k_pred_idx)

                        top_c_intermediate_nodes_for_top_k_dict,top_c_intermediate_contr_for_top_k_dict,\
                        top_c_intermediate_prots_for_top_k_dict = \
                        analyse_intermediate_nodes_for_top_k(frac_intermediate_nodes_contr, top_k_pred_idx,prots, c=10)
                        print('done')


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    for k in kwargs.get('k_to_test'):
        main(config_map, k=k, **kwargs)
