# here's an example call to this script:
# python src/scripts/path_based_effective_diffusion_eppstein.py --config fss_inputs/config_files/provenance/provenance_biogrid_y2h_go.yaml
# --run-algs genemaniaplus --k 500 --m 20 --n-sp 500


import os, sys
import yaml
import argparse
import numpy as np
import matplotlib
import networkx as nx
import copy
import time

import subprocess
import logging

logging.basicConfig(filename='diffisuion_eppsteins.log', filemode='a', level=logging.INFO, \
                    format='%(message)s')

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
    group.add_argument('--k-to-test', '-k', type=int, action='append', default=[],
                       help="k-value(s) for which to get the top-k predictions to test. " +
                            "If not specified, will check the config file.")

    group.add_argument('--n-sp', '-n', type=int, default=200,
                       help="n-sp is the number of shortest paths to be considered" +
                            "If not specified, will check the config file. Default=20")
    group.add_argument('--m', type=int, default=20,
                       help="for each top prediction, for how many top contributing sources we wanna analyse the path" +
                            "Default=20")
    group.add_argument('--max-len', type=int, default=100,
                       help="for each top prediction, for how many top contributing sources we wanna analyse the path" +
                            "Default=20")
    group.add_argument('--stat-sig-cutoff', type=float,
                       help="Cutoff on the node p-value for a node to be considered in the topk. " + \
                            "The p-values should already have been computed with run_eval_algs.py")

    group.add_argument('--force-ksp', action='store_true', default=False,
                       help="Force re-running the path diffusion analysis")
    group.add_argument('--force-contr', action='store_true', default=False,
                       help="Force re-running the path diffusion analysis")

    group.add_argument('--plot-only', action='store_true', default=False,
                       help="Force re-running the path diffusion analysis")

    return parser


def find_top_m_contributing_sources_per_pred(m, top_preds, pos_nodes_idx, M_inv, cutoff=0):
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

        per_src_contr_dict = {key: val for key, val in per_src_contr_dict.items() if val > cutoff}
        # take only top m contributing sources whose contribution is atleast>cutoff
        top_sources_per_pred[prot_idx] = list(per_src_contr_dict.keys())[0:min(m, len(per_src_contr_dict))]

    return top_sources_per_pred


def process_java_computed_eppstein_output(path_df):
    # process tha output from java code of Eppsitein's
    path_df['cost'] = path_df['cost'].astype(str).apply(lambda x: x.split(':')[0])
    path_df['cost'] = path_df['cost'].apply(lambda x: float(x))
    # find the actual cost i.e. cost without log implied.
    path_df['actual_cost'] = path_df['cost'].apply(lambda x: 10 ** (-x))

    # path has been saved as source-intermediate1-intermediate2-target in this form. So counting '-' will
    # give the path length
    path_df['length'] = path_df['path'].astype(str).apply(lambda x: x.count('-'))
    return path_df


def compute_path_len_wise_contr(R, source, target, path_df, path_length_wise_rate_contr):
    total_rate_contr_from_s_2_t = R[target][source]  # in R, columns are the sources and rows are the targets
    # paths is a tuple as (path_length, actual_cost)
    paths = tuple(zip(path_df['length'], path_df['actual_cost']))

    for path in paths:
        path_length = path[0]
        path_cost = path[1]

        if path_length in path_length_wise_rate_contr[(source, target)]:
            path_length_wise_rate_contr[(source, target)][path_length] += path_cost
        else:
            logging.info(str(source) + '\t' + str(target) + '\t'+str(path_length) +\
                         '\t'+str(path_cost/total_rate_contr_from_s_2_t))

    # compute fraction of total contribution coming via each path length
    for path_length in path_length_wise_rate_contr[(source, target)]:
        path_length_wise_rate_contr[(source, target)][path_length] /= total_rate_contr_from_s_2_t


def write_path_len_wise_contr(source, target, neighbor, score, frac_score_contr_from_s_t, path_length_wise_rate_contr,
                              filename):
    out_f = open(filename, 'a')
    rounded_score = round(score, 6)
    rounded_contr_s_t = round(frac_score_contr_from_s_t, 6)
    out_str = str(source) + '\t' + str(target) + '\t' + str(neighbor) + '\t' + str(rounded_score) + '\t' + str(
        rounded_contr_s_t)
    for path_length in path_length_wise_rate_contr[(source, target)]:
        rounded_val = round(path_length_wise_rate_contr[(source, target)][path_length], 6)
        out_str = out_str + '\t' + str(rounded_val)
    out_f.write(out_str + '\n')

    # print(out_str+'\n')
    out_f.close()


def save_paths(path_df, shortest_path_file):
    path_df['cost'] = path_df['cost'].astype(str).apply(lambda x: x.split(':')[0])

    # read already existing items in the file if not empty
    if os.path.getsize(shortest_path_file) > 0:
        df = pd.read_csv(shortest_path_file, sep='\t', index_col=None)
        df = pd.concat([df, path_df], axis=0)
    else:
        df = path_df
    df.to_csv(shortest_path_file, sep='\t', index=False)


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
    m = kwargs.get('m')
    max_pathlen = kwargs.get('max_len')

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
                        pred_file = script_utils.term_based_pred_file(pred_file, term)
                        if not os.path.isfile(pred_file):
                            print("Warning: %s not found. skipping" % (pred_file))
                            continue
                        print("reading %s for alpha=%s" % (pred_file, alpha))
                        df = pd.read_csv(pred_file, sep='\t')

                        # analyse pos
                        df_1 = df[df['prot'].isin(node2idx)]
                        df_1['prot_idx'] = df_1['prot'].apply(lambda x: node2idx[x])
                        all_pred_scores = dict(zip(df_1['prot_idx'], df_1['score']))
                        all_pred_scores = dict(
                            sorted(all_pred_scores.items(), key=lambda item: item[0], reverse=False))
                        all_pred_scores = list(all_pred_scores.values())

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

                        shortest_path_input_graph_file = "%s/shortest-path-input-graph-%s-a%s.txt" % \
                                                         (net_obj.out_pref, alg_name,
                                                          str(alpha).replace('.', '_'))
                        all_same_weight_input_graph_file = "%s/all_same_weight_input-graph-%s-a%s.txt" % \
                                                         (net_obj.out_pref, alg_name,
                                                          str(alpha).replace('.', '_'))

                        # shortest_path_file will be created only when all shortest paths for
                        # the targets for this certain setup (i.e. nsp, m, k values) have been computed.
                        shortest_path_file = config_map['output_settings'][
                                                 'output_dir'] + "/viz/%s/%s/diffusion-path-analysis/%s/shortest_path/" \
                                                                 "shortest-paths-k%s-nsp%s-m%s-a%s%s.txt" % (
                                                 dataset['net_version'], term, alg_name, k,
                                                 kwargs.get('n_sp'), kwargs.get('m'), alpha, sig_str)

                        contr_file = config_map['output_settings'][
                                         'output_dir'] + "/viz/%s/%s/diffusion-path-analysis/%s/length_wise_contr-k%s-nsp%s-m%s-a%s%s-max%s.tsv" % (
                                         dataset['net_version'], term, alg_name, k, kwargs.get('n_sp'), kwargs.get('m'),
                                         alpha, sig_str, max_pathlen)

                        os.makedirs(os.path.dirname(contr_file), exist_ok=True)

                        if (not os.path.isfile(contr_file)) or (kwargs.get('force_contr') == True):

                            if alg_name == 'genemaniaplus':
                                M_inv = gm.get_diffusion_matrix(net_obj.W, alpha=alpha, diff_mat_file=diff_mat_file,
                                                                force_run=False)
                                M_pathmtx, R = gm.get_fluid_flow_matrix(net_obj.W, alpha=alpha, \
                                                                        fluid_flow_mat_file_M=fluid_flow_mat_file_M, \
                                                                        fluid_flow_mat_file_R=fluid_flow_mat_file_R,
                                                                        force_run=False)
                            if alg_name == 'rwr':
                                M_inv = rwr.get_diffusion_matrix(net_obj.W, alpha=alpha,
                                                                 diff_mat_file=diff_mat_file, force_run=False)
                                M_inv = M_inv * (alpha / float(len(pos_nodes_idx)))
                                M_pathmtx, R = rwr.get_fluid_flow_matrix(net_obj.W, alpha,
                                                                         fluid_flow_mat_file_M=fluid_flow_mat_file_M,
                                                                         fluid_flow_mat_file_R=fluid_flow_mat_file_R,
                                                                         force_run=False)
                            top_m_contrs_per_pred = \
                                find_top_m_contributing_sources_per_pred(m, top_k_pred_idx,
                                                                         copy.deepcopy(pos_nodes_idx), M_inv)

                            #taking transpose as newtorkx considers rows to be source and cols to be target but
                            # in our case we normlaized M_pathmtx such a way thay cols are sources and rows are target.
                            G = nx.from_numpy_matrix(M_pathmtx.transpose(), create_using=nx.DiGraph())
                            print('networkx graph creation done')

                            # save this graph as list of edges with weights.
                            nx.write_weighted_edgelist(G, shortest_path_input_graph_file)

                            #now to find out actual number of shortest paths of less than a length l, we need the graph
                            # G where all edges are of weight 1.
                            nx.set_edge_attributes(G, values=1, name='weight')
                            nx.write_weighted_edgelist(G, all_same_weight_input_graph_file)

                            del G

                            del M_pathmtx
                            adj_matrix = (net_obj.W).toarray(order='C')

                            n_shortest_path = kwargs.get('n_sp')

                            f = open(contr_file, 'w')
                            out_str = 'source\ttarget\tneighbour\tscore\tfrac_source_contr'
                            for i in range(1, max_pathlen + 1, 1):
                                out_str += '\t' + 'frac_contr_via_pathlen_' + str(i)
                            f.write(out_str + '\n')
                            f.close()

                            count = 0
                            # **************RUN EPPSTEIN*********************
                            wd = os.getcwd()
                            # change directory to the java base Eppstein's algo
                            eppstein_code_dir = '/data/tasnina/k-shortest-paths/'

                            t0 = time.time()
                            os.chdir(eppstein_code_dir)
                            p1 = subprocess.Popen(['javac', './edu/ufl/cise/bsmock/graph/ksp/test/TestEppstein.java'])
                            p1.wait()
                            os.chdir(wd)
                            # print('t0:', time.time()-t0)

                            for target in top_k_pred_idx:
                                # target_spec_shortest_path_file
                                target_spec_sp_file = shortest_path_file.replace('.txt', '_' + str(target) + '.tsv')
                                if (not os.path.isfile(target_spec_sp_file)) or (kwargs.get('force_ksp') == True):
                                    os.makedirs(os.path.dirname(target_spec_sp_file), exist_ok=True)
                                    f = open(target_spec_sp_file, 'w')
                                    f.close()

                                    print('target: ', target)
                                    sources = str(top_m_contrs_per_pred[target])
                                    source_contrs = []
                                    for source in top_m_contrs_per_pred[target]:
                                        source_contrs.append(M_inv[target][source])
                                    source_contrs = str(source_contrs)

                                    # if source and target are not in same connected component then do not pass this
                                    # source-target pair to Eppstein's
                                    t1 = time.time()
                                    # ****************** RUN EPPSTEIN's KSP***********************
                                    # write code for running Eppstein's ksp algo here.
                                    # get current directory

                                    # Two values govern the increase of K in k-shortest path alg.
                                    # first is the tolerance. The ksp alg will keep increasing
                                    # k until the sum of frac-contribution from shortest paths is >= (1-tolerance)
                                    # second is the hard-coded condition in the java function which checks if with increasing
                                    # k, the sum of the contribution changes. If the change is as small as (10^-10) then the
                                    # alg stops.
                                    tolerance = 0.001
                                    eppstein_inputs = [shortest_path_input_graph_file, \
                                                       target_spec_sp_file, str(sources), str(target), \
                                                       str(n_shortest_path), source_contrs, str(tolerance)]
                                    os.chdir(eppstein_code_dir)
                                    p = subprocess.Popen(['java', 'edu.ufl.cise.bsmock.graph.ksp.test.TestEppstein'] + \
                                                         eppstein_inputs)
                                    p.wait()
                                    os.chdir(wd)

                                # read parse output of Eppstein's code to compute diffusion
                                path_df = pd.read_csv(target_spec_sp_file, sep=' ', header=None,
                                                      index_col=None, names=['source', 'target', 'cost', 'path'])
                                path_df = process_java_computed_eppstein_output(path_df)

                                total_score = pred_scores[target]
                                for source in top_m_contrs_per_pred[target]:
                                    frac_score_contr_from_s_t = M_inv[target][source] / total_score
                                    path_length_wise_contr[(source, target)] = {x: 0 for x in
                                                                                range(1, max_pathlen + 1, 1)}
                                    if adj_matrix[target][source] != 0:
                                        neighbor = 1
                                    else:
                                        neighbor = 0

                                    source_target_spec_df = path_df[(path_df['source'] == source)]
                                    compute_path_len_wise_contr(R, source, target, \
                                                                source_target_spec_df, path_length_wise_contr)

                                    write_path_len_wise_contr(source, target, neighbor, total_score, \
                                                              frac_score_contr_from_s_t, path_length_wise_contr,
                                                              contr_file)
                                # print('t: ', time.time()-t1)
                            f = open(shortest_path_file, 'w')
                            f.write('Finished shortest path calculation\n')
                            f.close()

                            del M_inv, R

                        # new effective diffusion
                        path_length_based_effective_diffusion_file = config_map['output_settings']['output_dir'] + \
                                                                     "/viz/%s/%s/diffusion-path-analysis/%s/path-length-wise-effective-diff-k%s-nsp%s-m%s-a%s%s.tsv" \
                                                                     % (dataset['net_version'], term, alg_name,
                                                                         k, kwargs.get('n_sp'),
                                                                         kwargs.get('m'), alpha, sig_str)

                        source_target_contr_df = pd.read_csv(contr_file, sep='\t', index_col=None)

                        # 'frac_total_score_via_len_1' holds fraction of total target score that is coming from source via path length 1
                        source_target_contr_df['frac_total_score_via_len_1'] = \
                            source_target_contr_df['frac_source_contr'] * source_target_contr_df[
                                'frac_contr_via_pathlen_1']

                        source_target_contr_df['frac_total_score_via_len_2'] = \
                            source_target_contr_df['frac_source_contr'] * source_target_contr_df[
                                'frac_contr_via_pathlen_2']
                        source_target_contr_df['frac_total_score_via_len_3'] = \
                            source_target_contr_df['frac_source_contr'] * source_target_contr_df[
                                'frac_contr_via_pathlen_3']

                        path_length_based_effective_diffusion = \
                            source_target_contr_df.groupby('target')[
                                ['frac_total_score_via_len_1', 'frac_total_score_via_len_2',
                                 'frac_total_score_via_len_3']].sum()

                        path_length_based_effective_diffusion = \
                            path_length_based_effective_diffusion.reset_index()

                        path_length_based_effective_diffusion['effective_diffusion'] = 1 - \
                                                                                       path_length_based_effective_diffusion[
                                                                                           'frac_total_score_via_len_1']

                        # round up the decimal value upto 3 decimal point
                        path_length_based_effective_diffusion['effective_diffusion'] = \
                            path_length_based_effective_diffusion['effective_diffusion'].apply(lambda x: round(x, 6))

                        path_length_based_effective_diffusion. \
                            to_csv(path_length_based_effective_diffusion_file, sep='\t', index=False)

                        print('dataset: ', dataset, 'alg: ', alg_name, 'alpha: ', alpha)
                        print(path_length_based_effective_diffusion)


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    for k in kwargs.get('k_to_test'):
        main(config_map, k=k, **kwargs)
