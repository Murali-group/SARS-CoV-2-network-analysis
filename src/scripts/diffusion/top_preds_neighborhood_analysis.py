import os, sys
import yaml
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy

import logging

logging.basicConfig(filename='diffisuion_eppsteins.log', filemode='a', level=logging.INFO, \
                    format='%(name)s - %(levelname)s - %(message)s')

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
import src.scripts.utils as dfsn_utils


def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        # config_map = yaml.load(conf, Loader=yaml.FullLoader)
        config_map = yaml.load(conf)
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
                                                     "fss_inputs/config_files/provenance/provenance_string700_s12.yaml"
                       , help="Configuration file used when running FSS. ")

    group.add_argument('--run-algs', type=str, action='append', default=[])
    group.add_argument('--k-to-test', '-k', type=int, action='append', default=[332],
                       help="k-value(s) for which to get the top-k predictions to test. " +
                            "If not specified, will check the config file.")

    group.add_argument('--stat-sig-cutoff', type=float,
                       help="Cutoff on the node p-value for a node to be considered in the topk. " + \
                            "The p-values should already have been computed with run_eval_algs.py")

    group.add_argument('--force-run', action='store_true', default=True,
                       help="Force re-running the path diffusion analysis")


    group.add_argument('--plot-only', action='store_true', default=False,
                       help="Force re-running the path diffusion analysis")

    return parser


def get_sorted_source_contribution_per_pred(top_preds, pos_nodes_idx, M_inv, cutoff=0):
    '''
    Returns a dictionary with top predicted proteins as keys and list of( atleast size m) top contributing sources of
    each prediction as values. If the contribution from a source<cutoff, it will not be in the list.
    '''
    # any M_inv where sources-> columns and targets -> rows, this function will work
    top_sources_per_pred = {}
    # find some top m contributing sources for each of top k(same as k_to_test) predictions.
    mask = np.zeros(M_inv.shape[1], dtype=bool)
    mask[pos_nodes_idx] = True
    M_inv_new = M_inv[:, mask]  # keep the columns with source/pos_nodes only
    # print(M_inv.shape, M_inv_new.shape)
    for prot_idx in top_preds:
        # get contribution from source nodes for each top predictions and sort. Also keep track of which source node
        # contributed how much
        pos_nodes_idx.sort(reverse=False)  # sort ascending
        contr_vals = list(
            M_inv_new[prot_idx])  # in this 'contr' list, values are sorted according to the source nodes' index values

        per_src_contr_dict = dict(zip(pos_nodes_idx, contr_vals))
        per_src_contr_dict = dict(sorted(per_src_contr_dict.items(), key=lambda item: item[1], reverse=True))

        top_sources_per_pred[prot_idx] = copy.deepcopy(per_src_contr_dict)

    return top_sources_per_pred



def main(config_map, k, **kwargs):
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
        net_obj, ann_obj, _ = setup_dataset(
            dataset, input_dir, **kwargs)
        prots, node2idx = net_obj.nodes, net_obj.node2idx

        for term in ann_obj.terms:
            term_idx = ann_obj.term2idx[term]
            orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, term_idx)
            orig_pos = [prots[p] for p in orig_pos_idx]
            pos_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]

            for alg_name in alg_settings:
                if (alg_settings[alg_name]['should_run'][0] == True) or (alg_name in kwargs.get('run_algs')):
                    # load the top predictions
                    print(alg_name)
                    alg_pred_files = config_utils.get_dataset_alg_prediction_files(
                        output_dir, dataset, alg_settings, [alg_name], **kwargs)
                    # get the alpha values to use
                    alphas = alg_settings[alg_name]['alpha']
                    for alpha, alg in zip(alphas, alg_pred_files):
                        path_length_wise_contr = {}
                        pred_file = alg_pred_files[alg]
                        pred_file = dfsn_utils.term_based_pred_file(pred_file, term)
                        if not os.path.isfile(pred_file):
                            print("Warning: %s not found. skipping" % (pred_file))
                            continue
                        print("reading %s for alpha=%s" % (pred_file, alpha))
                        df = pd.read_csv(pred_file, sep='\t')


                        # remove the original positives for downstream analysis
                        df = df[~df['prot'].isin(orig_pos)]
                        df.reset_index(inplace=True, drop=True)


                        pred_scores = np.zeros(len(net_obj.nodes))
                        df = df[:k]
                        top_k_pred = df['prot']
                        top_k_pred_idx = [net_obj.node2idx[n] for n in top_k_pred]
                        pred_scores[top_k_pred_idx] = df['score'].values

                        # No need for including dataset['exp_name'] as the following matrix are seed node independent.
                        diff_mat_file = "%s/diffusion-mat-%s-a%s.npy" % (net_obj.out_pref, alg_name,
                                                                         str(alpha).replace('.', '_'))

                        top_pred_neighborhood_file = config_map['output_settings'][
                                         'output_dir'] + "/viz/%s/%s/diffusion-source-analysis/%s/top_pred_neighborhood-k%s-m%s-a%s.tsv" % (
                                         dataset['net_version'], term, alg_name, k, kwargs.get('m'),
                                         alpha)
                        os.makedirs(os.path.dirname(top_pred_neighborhood_file), exist_ok=True)
                        if (not os.path.isfile(top_pred_neighborhood_file)) or kwargs.get('force_run') == True:

                            if alg_name == 'genemaniaplus':
                                M_inv = gm.get_diffusion_matrix(net_obj.W, alpha=alpha, diff_mat_file=diff_mat_file,
                                                                force_run=False)

                            if alg_name == 'rwr':
                                M_inv = rwr.get_diffusion_matrix(net_obj.W, alpha=alpha,
                                                                 diff_mat_file=diff_mat_file, force_run=False)
                                M_inv = M_inv * (alpha / float(len(pos_nodes_idx)))

                            source_contrs_per_pred = \
                                get_sorted_source_contribution_per_pred(top_k_pred_idx,
                                                                         copy.deepcopy(pos_nodes_idx), M_inv)

                            adj_matrix = (net_obj.W).toarray(order='C')

                            target_list= []
                            score_list = [] #score of target
                            source_list = []
                            neighborhood_list = [] # if source and target are neighbors or not (1 or 0)
                            source_contr_list = []

                            # for a target if n seeds are direct neighbors, find out how many of them(neighboring seeds) made
                            # it to the top n contributing sources.
                            target_neighborhood = {} #key = target, value = [a,b] here a=#neighboring seeds,
                            # b=#neighbors_in_the_top_a_contributing sources

                            for target in top_k_pred_idx:
                                neighbors_in_top_x_contrs= []
                                target_score = pred_scores[target]
                                source_contrs = source_contrs_per_pred[target]
                                target_neighborhood [target] = [0,0]
                                for source in source_contrs:
                                    frac_score_contr_from_s_t = M_inv[target][source] / target_score
                                    if adj_matrix[target][source] != 0:
                                        neighbor = 1
                                        target_neighborhood[target][0]+=1
                                    else:
                                        neighbor = 0

                                    target_list.append(target)
                                    score_list.append(target_score)
                                    source_list.append(source)
                                    neighborhood_list.append(neighbor)
                                    source_contr_list.append(frac_score_contr_from_s_t)
                                    neighbors_in_top_x_contrs.append(target_neighborhood[target][0])
                                #now extract numbers of neighbors in top n contributors,
                                # where n=total number of neighboring seeds
                                target_neighborhood[target][1] = neighbors_in_top_x_contrs[target_neighborhood[target][0]]

                                #newlyd computed target_neighborhood : b/a
                                # target_neighborhood[target] = target_neighborhood[target][1]/target_neighborhood[target][0]

                            plt.plot(list(target_neighborhood.values()))
                            plt.show()
                            plt.close()

                            del M_inv
                            neighborhood_df = pd.DataFrame({'source':source_list,'target': target_list,
                                            'neighbor':neighborhood_list,\
                                          'score':score_list, 'source_to_target_contr': source_contr_list})
                            neighborhood_df.to_csv(top_pred_neighborhood_file,sep='\t',index=False)


                        neighborhood_df = pd.read_csv(top_pred_neighborhood_file, sep='\t', index_col=None)


                    # #plot neighborhood analysis plots
                    # targets = list(neighborhood_df['target'].unique())
                    # for target in targets:
                    #
                    #     target_spec_df = neighborhood_df[neighborhood_df['target']==target]
                    #     n_neighbors = target_spec_df['neighbor'].sum()
                    #



if __name__ == "__main__":
    config_map, kwargs = parse_args()
    for k in kwargs.get('k_to_test'):
        main(config_map, k=k, **kwargs)
