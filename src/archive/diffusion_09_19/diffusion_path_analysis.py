# here's an example call to this script:
#python src/scripts/diffusion_path_analysis.py --config fss_inputs/config_files/provenance/provenance_biogrid_y2h_s12.yaml
# --k 332 --m 20 --n-sp 50



import os, sys
import yaml
import argparse
import numpy as np
import matplotlib
import networkx as nx
import copy

if __name__ == "__main__":
    # Use this to save files remotely. 
    matplotlib.use('Agg')
import pandas as pd
sys.path.insert(1,"/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/")
# sys.path.insert(0,"../../")
from src.FastSinkSource.src.main import setup_dataset
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils
from src.FastSinkSource.src.algorithms import rl_genemania as gm
from src.FastSinkSource.src.algorithms import PageRank_Matrix as rwr
import submodules.PathLinker.ksp_Astar as ksp
import src.scripts.utils as script_utils


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
    parser = argparse.ArgumentParser(description="Script to analyze the top contributors of each prediction's "
                                                 "diffusion score, as well as the effective diffusion (i.e.,"
                                                 " fraction of diffusion received from non-neighbors)")
    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str,default = "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                                                      "fss_inputs/config_files/provenance/provenance_biogrid_y2h_s12.yaml"
                    ,help="Configuration file used when running FSS. ")
    group.add_argument('--analysis-type', type=str, default="diffusion_path_analysis",
                       help="Type of network analysis to perform. Options: 'diffusion_analysis', "
                            "'shortest_paths', 'degrees'. Default: 'diffusion_analysis")
    #group.add_argument('--node', type=str, action="append",
    #                   help="Check the distance of the given node")
    group.add_argument('--cutoff', type=float, default=0.01,
                       help="Cutoff of fraction of diffusion recieved to use to choose the main contributors.")

    group.add_argument('--run-algs', type=str, action='append', default=[])
    group.add_argument('--k-to-test', '-k', type=int, action='append', default = [332],
                       help="k-value(s) for which to get the top-k predictions to test. " +
                       "If not specified, will check the config file. Default=332")

    group.add_argument('--n-sp', type=int, default=20,
                       help="n-sp is the number of shortest paths to be considered" +
                            "If not specified, will check the config file. Default=20")
    group.add_argument('--m', type=int, default=20,
                       help="for each top prediction, for how many top contributing sources we wanna analyse the path" +
                            "Default=20")
    group.add_argument('--stat-sig-cutoff', type=float,
                       help="Cutoff on the node p-value for a node to be considered in the topk. " + \
                       "The p-values should already have been computed with run_eval_algs.py")
    group.add_argument('--terms-file', type=str, 
                       help="Plot the effective diffusion values per term.")
    group.add_argument('--sample-method', type=str, default='kmeans',
                       help="Approach used to sample random sets of positive examples. " + \
                       "Options: 'kmeans' (bin nodes by kmeans clustring on the degree distribution), 'simple'"
                       " (uniformly at random)")
    group.add_argument('--num-random-sets', type=int, default=1000,
                       help="Number of random sets used when computing pvals. Default: 1000")
    group.add_argument('--force-run', action='store_true', default=True,
                       help="Force re-running the path diffusion analysis")
    group.add_argument('--id-mapping-file', type=str, default="datasets/mappings/human/uniprot-reviewed-status.tab.gz",
                       help="Table downloaded from UniProt to map to gene names. Expected columns: 'Entry'"
                            " and 'Gene names'")
    group.add_argument('--eval-only', action='store_true', default=False,
                       help="No computation. Just read the file where k-shortest path with their length has been saved")

    return parser


def find_top_m_contributing_sources_per_pred(m, top_preds, pos_nodes_idx, M_inv):
    '''
    Returns a dictionary with top predicted proteins as keys and list of top contributing sources of
    each prediction as values.
    '''
    #any M_inv where sources-> columns and targets -> rows, this function will work
    top_sources_per_pred = {}
    #find some top m contributing sources for each of top k(same as k_to_test) predictions.
    mask = np.zeros(M_inv.shape[1], dtype=bool)
    mask[pos_nodes_idx] = True
    M_inv_new = M_inv[:, mask] #keep the columns with source/pos_nodes only
    # print(M_inv.shape, M_inv_new.shape)
    for prot_idx in top_preds:
        # get contribution from source nodes for each top predictions and sort. Also keep track of which source node
        # contributed how much
        pos_nodes_idx.sort(reverse=False) #sort ascending
        contr_vals = list(M_inv_new[prot_idx]) #in this 'contr' list, values are sorted according to the source nodes' index values

        per_src_contr_dict = dict(zip(pos_nodes_idx, contr_vals))
        per_src_contr_dict = dict(sorted(per_src_contr_dict.items(), key=lambda item: item[1], reverse=True))

        #take only top m contributing sources
        top_sources_per_pred[prot_idx] = list(per_src_contr_dict.keys())[0:m]

    return top_sources_per_pred


def compute_path_len_wise_contr(R, source, target, paths, path_length_wise_rate_contr):
    total_rate_contr_from_s_2_t = R[target][source] #in R, columns are the sources and rows are the targets
    for path in paths:
        path_length = len(path)-1
        path_cost = path[-1][1]
        contr_via_path = 10 ** (-path_cost)
        path_length_wise_rate_contr[(source, target)][path_length] += contr_via_path

    # compute fraction of total contribution coming via each path length
    for path_length in path_length_wise_rate_contr[(source, target)]:
        path_length_wise_rate_contr[(source, target)][path_length] /= total_rate_contr_from_s_2_t


def write_path_len_wise_contr(source, target, neighbor, score, frac_score_contr_from_s_t, path_length_wise_rate_contr, filename):

    out_f = open(filename, 'a')
    out_str = str(source)+'\t'+str(target)+'\t'+str(neighbor)+'\t'+str(score)+'\t'+str(frac_score_contr_from_s_t)
    for path_length in path_length_wise_rate_contr[(source, target)]:
        out_str = out_str + '\t'+str(path_length_wise_rate_contr[(source, target)][path_length])
    out_f.write(out_str+'\n')

    # print(out_str+'\n')
    out_f.close()

def main(config_map, k, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """
    #
    # print(kwargs.get('force_run'))
    # return
    # extract the general variables from the config map
    input_settings, input_dir, output_dir, alg_settings, kwargs \
        = config_utils.setup_config_variables(config_map, **kwargs)

    sig_cutoff = kwargs.get('stat_sig_cutoff')
    sig_str = "-sig%s" % (str(sig_cutoff).replace('.', '_')) if sig_cutoff else ""
    # k = kwargs.get('k', 332)
    m = kwargs.get('m')
    # for each dataset, extract the path(s) to the prediction files,
    # read in the predictions, and test for the statistical significance of overlap

    for dataset in input_settings['datasets']:
        print("Loading data for %s" % (dataset['net_version']))

        dataset_name = config_utils.get_dataset_name(dataset)
        # load the network and the positive examples for each term
        net_obj, ann_obj, _ = setup_dataset(
            dataset, input_dir, **kwargs)
        prots, node2idx = net_obj.nodes, net_obj.node2idx
        orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, 0)
        orig_pos = [prots[p] for p in orig_pos_idx]
        pos_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]

        for alg_name in alg_settings:
            if (alg_settings[alg_name]['should_run'][0]==True) or (alg_name in kwargs.get('run_algs')):
                # load the top predictions
                print(alg_name)
                alg_pred_files = config_utils.get_dataset_alg_prediction_files(
                    output_dir, dataset, alg_settings, [alg_name], **kwargs)
                # get the alpha values to use
                alphas = alg_settings[alg_name]['alpha']
                for alpha, alg in zip(alphas, alg_pred_files):
                    path_length_wise_contr = {}
                    pred_file = alg_pred_files[alg]
                    pred_file = script_utils.term_based_pred_file(pred_file, dataset['exp_name'])
                    if not os.path.isfile(pred_file):
                        print("Warning: %s not found. skipping" % (pred_file))
                        continue
                    print("reading %s for alpha=%s" % (pred_file, alpha))
                    df = pd.read_csv(pred_file, sep='\t')

                    #analyse pos
                    df_1 = df[df['prot'].isin(node2idx)]
                    df_1['prot_idx'] = df_1['prot'].apply(lambda x: node2idx[x])
                    all_pred_scores = dict(zip(df_1['prot_idx'],df_1['score']))
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


                    # diff_mat_file = "%s/diffusion-mat-%s-%s-a%s.npy" % (net_obj.out_pref, alg_name, dataset['exp_name'],
                    #                                                    str(alpha).replace('.', '_'))
                    # fluid_flow_mat_file_M = "%s/fluid-flow-mat-%s-%s-a%s.npy" % (net_obj.out_pref,alg_name,
                    #                                     dataset['exp_name'], str(alpha).replace('.','_'))
                    # fluid_flow_mat_file_R = "%s/fluid-flow-mat-R%s-%s-a%s.npy" % (net_obj.out_pref, alg_name,
                    #                                     dataset['exp_name'],str(alpha).replace('.', '_'))

                    #No need for including dataset['exp_name'] as the following matrix are seed node independent.
                    diff_mat_file = "%s/diffusion-mat-%s-a%s.npy" % (net_obj.out_pref, alg_name,
                                                                        str(alpha).replace('.', '_'))
                    fluid_flow_mat_file_M = "%s/fluid-flow-mat-%s-a%s.npy" % (net_obj.out_pref, alg_name,
                                                                                 str(alpha).replace('.', '_'))
                    fluid_flow_mat_file_R = "%s/fluid-flow-mat-R-%s-a%s.npy" % (net_obj.out_pref, alg_name,
                                                                                  str(alpha).replace('.', '_'))

                    shortest_path_input_graph_file = "%s/shortest-path-input-graph-%s-a%s.txt" % \
                                                     (net_obj.out_pref, alg_name,
                                                      str(alpha).replace('.', '_'))

                    shortest_path_file =config_map['output_settings']['output_dir'] +"/viz/%s/%s/diffusion-path-analysis/%s/orig-shortest-paths-k%s-nsp%s-m%s-a%s%s.tsv" % (
                        dataset['net_version'], dataset['exp_name'], alg_name, k, kwargs.get('n_sp'),kwargs.get('m'), alpha, sig_str)
                    contr_file = config_map['output_settings']['output_dir'] +"/viz/%s/%s/diffusion-path-analysis/%s/orig-length_wise_contr-k%s-nsp%s-m%s-a%s%s.tsv" % (
                        dataset['net_version'], dataset['exp_name'], alg_name, k, kwargs.get('n_sp'), kwargs.get('m'), alpha, sig_str)

                    os.makedirs(os.path.dirname(shortest_path_file), exist_ok=True)

                    if (not os.path.isfile(shortest_path_file)) or (not os.path.isfile(contr_file))\
                            or kwargs.get('force_run')==True:

                        if alg_name == 'genemaniaplus':
                            M_inv = gm.get_diffusion_matrix(net_obj.W, alpha=alpha, diff_mat_file=diff_mat_file,
                                                            force_run=True)

                            M_pathmtx, R = gm.get_fluid_flow_matrix(net_obj.W, alpha = alpha, \
                                                                fluid_flow_mat_file_M=fluid_flow_mat_file_M, \
                                                                fluid_flow_mat_file_R=fluid_flow_mat_file_R,
                                                                force_run=True)


                        if alg_name =='rwr':
                            M_inv = rwr.get_diffusion_matrix(net_obj.W, alpha=alpha,
                                                             diff_mat_file=diff_mat_file, force_run=True)
                            M_inv = M_inv * (alpha / float(len(pos_nodes_idx)))
                            M_pathmtx, R = rwr.get_fluid_flow_matrix(net_obj.W, alpha,
                                                                    fluid_flow_mat_file_M=fluid_flow_mat_file_M,
                                                                    fluid_flow_mat_file_R=fluid_flow_mat_file_R,
                                                                    force_run=True)
                        top_m_contrs_per_pred = \
                            find_top_m_contributing_sources_per_pred(m, top_k_pred_idx, copy.deepcopy(pos_nodes_idx), M_inv)

                        G = nx.from_numpy_matrix(M_pathmtx.transpose(), create_using=nx.DiGraph())
                        print('networkx graph creation done')

                        #save this graph as list of edges with weights.

                        nx.write_weighted_edgelist(G, shortest_path_input_graph_file)

                        del M_pathmtx
                        adj_matrix = (net_obj.W).toarray(order='C')

                        n_shortest_path = kwargs.get('n_sp') #ksp_Astar runtime does not depend much on this n-sp, runtime same for n=10 and 20
                        max_pathlen = 20

                        f1 = open(shortest_path_file, 'w')
                        f1.close()

                        f = open(contr_file, 'w')
                        out_str = 'source\ttarget\tneighbour\tscore\tfrac_source_contr'
                        for i in range(1, max_pathlen+1, 1):
                            out_str += '\t'+'frac_contr_via_pathlen_'+str(i)
                        f.write(out_str+'\n')
                        f.close()

                        count=0
                        #pos mask
                        # mask = np.zeros(M_inv.shape[1], dtype=bool)
                        # mask[pos_nodes_idx] = True
                        # M_inv_new = M_inv[:, mask]  # keep the columns with source/pos_nodes only
                        for target in top_k_pred_idx:
                            total_score = pred_scores[target]  #this is from M_inv
                            # print('all source contr: ', M_inv_new[target])
                            for source in top_m_contrs_per_pred[target]:
                                # if alg_name == 'rwr':
                                #     assert abs(M_inv[target][source]-R[target][source]*all_pred_scores[source])<0.0000001,\
                                #     print('R, M_inv worng')

                                path_length_wise_contr[(source, target)]={x:0 for x in range(1,max_pathlen+1, 1)}
                                frac_score_contr_from_s_t = M_inv[target][source]/total_score
                                if adj_matrix[target][source] != 0:
                                    neighbor = 1
                                else:
                                    neighbor = 0

                                #just replace the following two lines and
                                paths = ksp.k_shortest_paths_yen(G, source, target, k = n_shortest_path, weight='weight')
                                # TODO save paths variable as pickle file
                                ksp.printKSPPaths(shortest_path_file, paths, source, target)

                                compute_path_len_wise_contr(R, source, target,\
                                                            paths, path_length_wise_contr)

                                write_path_len_wise_contr(source, target, neighbor, total_score,\
                                            frac_score_contr_from_s_t, path_length_wise_contr, contr_file)

                                count+=1

                        del G, M_inv, R

                    #new effective diffusion
                    path_length_based_effective_diffusion_file = config_map['output_settings']['output_dir'] + \
                                                                 "/viz/%s/%s/diffusion-path-analysis/%s/path-length-wise-effective-diff-k%s-nsp%s-m%s-a%s%s.tsv" \
                                                                 % (
                                                                 dataset['net_version'], dataset['exp_name'], alg_name,
                                                                 k, kwargs.get('n_sp'),
                                                                 kwargs.get('m'), alpha, sig_str)

                    source_target_contr_df = pd.read_csv(contr_file, sep='\t', index_col=None)

                    # 'frac_total_score_via_len_1' holds fraction of total target score that is coming from source via path length 1
                    source_target_contr_df['frac_total_score_via_len_1'] = \
                        source_target_contr_df['frac_source_contr'] * source_target_contr_df['frac_contr_via_pathlen_1']

                    path_length_based_effective_diffusion = \
                        source_target_contr_df.groupby('target')['frac_total_score_via_len_1'].sum()

                    path_length_based_effective_diffusion = \
                        path_length_based_effective_diffusion.to_frame().reset_index()
                    path_length_based_effective_diffusion['effective_diffusion'] = 1 - \
                                                                                   path_length_based_effective_diffusion[
                                                                                       'frac_total_score_via_len_1']

                    path_length_based_effective_diffusion.to_csv(path_length_based_effective_diffusion_file, sep='\t')

                    print('dataset: ', dataset, 'alg: ', alg_name, 'alpha: ', alpha)
                    print(path_length_based_effective_diffusion)

if __name__ == "__main__":
    config_map, kwargs = parse_args()
    for k in kwargs.get('k_to_test'):
        main(config_map, k=k, **kwargs)


