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
from scipy.sparse import eye, diags


import subprocess
import logging

logging.basicConfig(filename='diffisuion_eppsteins.log', filemode='a', level=logging.INFO, \
                    format='%(message)s')

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
                     "fss_inputs/config_files/provenance/string700_s12.yaml"
                       , help="Configuration file used when running FSS. ")
    group.add_argument('--run-algs', type=str, action='append', default=[])
    group.add_argument('--k-to-test', '-k', type=int, action='append', default=[332],
                       help="k-value(s) for which to get the top-k predictions to test. " +
                            "If not specified, will check the config file.")
    group.add_argument('--pos-k', action='store_true', default=False,
                       help="if true get the top-k predictions to test is equal to the number of positive annotations")

    group.add_argument('--n-sp', '-n', type=int, default=1000,
                       help="n-sp is the number of shortest paths to be considered" +
                            "If not specified, will check the config file. Default=1000")

    group.add_argument('--stat-sig-cutoff', type=float,
                       help="Cutoff on the node p-value for a node to be considered in the topk. " + \
                            "The p-values should already have been computed with run_eval_algs.py")

    group.add_argument('--force-ksp', action='store_true', default=False,
                       help="Force re-running the path diffusion analysis")
    group.add_argument('--force-contr', action='store_true', default=False,
                       help="Force re-running the path diffusion analysis")

    group.add_argument('--balancing-alpha-only', action='store_true', default=True,
                       help="Ignore alpha from config file rather take the alpha value\
                            that balanced the two loss terms in quad loss function for the corresponding\
                            network-term-alg")
    return parser

def add_supernode(G, ss_index, pos_idices):
    # Add a supernode with index=ss_index, where there will be edges between ss_index and pos_indices.
    # The edge weight will be very high
    edges_ss_to_s = list(zip([ss_index]*len(pos_idices), pos_idices))
    edges_s_to_ss = list(zip(pos_idices,[ss_index]*len(pos_idices)))
    G.add_edges_from(edges_ss_to_s, weight = HIGH_WEIGHT )
    G.add_edges_from(edges_s_to_ss, weight = HIGH_WEIGHT )

def save_weighted_graph_with_supersource_supersink(M_pathmtx, pos_nodes_idx, top_preds_idx, shortest_path_input_graph_file):

    # taking transpose as newtorkx considers rows to be source and cols to be target but
    # in our case we normlaized M_pathmtx such a way that cols are sources and rows are target.
    G = nx.from_numpy_matrix(M_pathmtx.transpose(), create_using=nx.DiGraph())

    # adding the supersource here
    ssource_index = G.number_of_nodes()  # TODO make sure that in existng G I had max nodeidx = G.number_of_nodes - 1
    ssink_index = ssource_index+1 # TODO make sure that in existng G I had max nodeidx = G.number_of_nodes - 1

    add_supernode(G, ssource_index, pos_nodes_idx)
    add_supernode(G, ssink_index, top_preds_idx)


    # Now save this graph G as a list of edges with weights to pass it to java based k-sp algorithm
    nx.write_weighted_edgelist(G, shortest_path_input_graph_file)
    print('networkx graph creation done')

    return  ssource_index, ssink_index


def process_java_computed_eppstein_output_for_supersource_supersink(path_df, idx_to_prot):
    # process tha output from java code of Eppsitein's
    path_df['cost'] = path_df['cost'].astype(str).apply(lambda x: x.split(':')[0])
    # remove cost of connecting to supersource and super sink
    path_df['cost'] = path_df['cost'].apply(lambda x: float(x)-2*HIGH_WEIGHT)
    # find the actual cost i.e. cost without log implied.
    # multiplying with (10^HIGH_WEIGHT) at the end to remove the edge cost incorporated by edges from supersource
    path_df['cost'] = path_df['cost'].apply(lambda x: 10 ** (-x))
    # path has been saved as source-intermediate1-intermediate2-target in this form. So counting '-' will
    # give the path length
    #Subtract 2 to remove the edge from supersource, and edge to supersink
    #TODO Make sure that in path supersource_idx comes up only once.
    path_df['length'] = path_df['path'].astype(str).apply(lambda x: x.count('-')-2)
    path_df['path'] = path_df['path'].apply(lambda x: x.split('-')[1:-1])
    path_df['path_prots'] = path_df['path'].apply(lambda x: [idx_to_prot[int(idx)] for idx in x] )
    path_df['unique_path_nodes'] = path_df['path'].apply(lambda x: len(set(x)) )
    return path_df



def find_loop_subtype(loop_info, path_length, path):
    #see if a path is of length 3 and if it is a loop
    is_looped_path_len_3 = loop_info and (path_length==3) #changed from & to and
    if is_looped_path_len_3:
        if path[1]==path[3]: #either stst, stut
            if  path[0]==path[2]: #then stst
                loop_subtype = 'stst'
            else:
                loop_subtype = 'stut'
        else:
            loop_subtype = 'sust'
    else:
        loop_subtype = '-'
    return loop_subtype


def find_paths_of_interest(path_df, R, pos_nodes_idx, to_k_pred_idx, network):
    '''
    path_df = this df contains 7 columns i.e. source, target, cost, path, path_prots, length,unique_path_nodes

    This function will extract interesting paths from this input df. Initially we are interested in paths that have
    more than 3 unique nodes in them.
    '''
    # def convert_idx_list_to_prots(indices, idx_to_prot):
    #     return [idx_to_prot[x] for x in indices]
    
    # interesting_paths_df = copy.deepcopy(path_df[path_df['unique_path_nodes']>=3])
    temp_paths_df = copy.deepcopy(path_df)

    #sum(R[int(x[-1]), pos_nodes_idx] = total incoming contribution to target node.
    # total_incoming_contr !=  predicted_score. To get final pred score we have to consider
    # outgoing flow as well.
    temp_paths_df['source'] = temp_paths_df['path']. \
        apply(lambda x: int(x[0]))
    temp_paths_df['target'] = temp_paths_df['path']. \
        apply(lambda x: int(x[-1]))
    temp_paths_df['target_rank'] = temp_paths_df['target']. \
        apply(lambda x: to_k_pred_idx.index(x))
    temp_paths_df['total_incoming_contr'] = temp_paths_df['target'].apply(lambda x: sum(R[x, pos_nodes_idx]))
    temp_paths_df['frac_contr_to_score'] = temp_paths_df['cost'] / \
                                           temp_paths_df['total_incoming_contr']
    frac_score_covered_per_target = temp_paths_df.groupby(['target'])['frac_contr_to_score'].sum().to_dict()
    #by taking some top nsp paths, how much of each target's total score is covered.
    temp_paths_df['frac_score_covered'] = temp_paths_df['target'].apply(lambda x: frac_score_covered_per_target[x])
    temp_paths_df['neighbor'] = temp_paths_df.apply(lambda x:
                                    int(bool(network[x['target'] , x['source']])), axis=1)
    #figure out if a certain path contain loops or not
    temp_paths_df['loop_info'] = temp_paths_df.apply(lambda x:
                                int(bool(x['unique_path_nodes']<=x['length'])), axis=1)
    #figure out if there is a loop of length 3, what are sub-type of loop it belongs to. Subtypes:
    # 1.stst 2.stut 3.sust,  where s = source, t = target, u = any other node
    temp_paths_df['len_3_loop_type'] = temp_paths_df.apply(lambda x: find_loop_subtype(x['loop_info'],\
                                                                x['length'],x['path_prots']),axis=1)

    temp_paths_df = temp_paths_df[['source','target','target_rank','path_prots','length','unique_path_nodes','cost',\
                    'total_incoming_contr','frac_contr_to_score','frac_score_covered', 'neighbor', 'loop_info', 'len_3_loop_type']]
    interesting_paths_df = temp_paths_df[temp_paths_df['unique_path_nodes']>=3]
    return interesting_paths_df, temp_paths_df


def write_summary(processed_nsp_paths_df, summary_file, **kwargs):
    #find  out how many targets are there and what is the range of ranking of the targets
    targets_covered = len(processed_nsp_paths_df['target'].unique())
    max_rank = processed_nsp_paths_df['target_rank'].max()
    min_rank = processed_nsp_paths_df['target_rank'].min()

    max_score_covered = processed_nsp_paths_df['frac_score_covered'].max()
    min_score_covered = processed_nsp_paths_df['frac_score_covered'].min()

    f = open(summary_file,'w')
    f.write('#considered shortest paths: '+ str(kwargs.get('n_sp'))+'\n')
    f.write('targets covered: ' + str(targets_covered)+'\n')
    f.write('range of target\'s rank: ' + str(min_rank) + ' - '+ str(max_rank)+'\n')
    f.write('range of target\'s covered score (frac): ' + str(min_score_covered) + ' - '+\
            str(max_score_covered)+'\n')
    f.close()

def find_top_preds_only_via_direct_edges(M, top_k_pred_idx, pos_nodes_idx):
    '''This function will sort(descending) the proteins(index) according to the contribution they get via direct edges from pos_nodes_idx.
     Then take the list from beginning till the last top_k_pred_idx in the list'''
    total_prots = M.shape[0]
    st_contr_all_targets = {}
    st_contr_only_top_targets = {}

    for target in range(total_prots):
        #TODO for RL contribution the following sum has to be divided by some degree i.e. (I+aD)^-1
        contr = sum(M[pos_nodes_idx,target])
        st_contr_all_targets[target]=contr
        if target in top_k_pred_idx:
            st_contr_only_top_targets[target] = contr

    #sort all proteins in descending order of contribution via direct edge
    st_contr_all_targets = {key: val for key, val in sorted(st_contr_all_targets.items(),
                                    key= lambda element: element[1], reverse=True)}

    n_nonzero_st_contr = list(st_contr_all_targets.values()).index(0) #this many target got nonzero contr from st paths
    # sort the top predicted proteins in descending order of contribution via direct edge
    st_contr_only_top_targets = {key: val for key, val in sorted(st_contr_only_top_targets.items(),
                                    key=lambda element: element[1], reverse=True)}

    #now take the first m proteins from st_contr_all_targets.keys() that ensures that all the top preds are in that.
    last_top_pred_protein = list(st_contr_only_top_targets.keys())[-1]
    m = list(st_contr_all_targets.keys()).index(last_top_pred_protein)

    return list(st_contr_all_targets.keys())[0:m+1]

def compute_and_save_contr_from_different_types_of_paths(M_pathmtx, R, pos_nodes_idx,
                top_k_pred_idx, paths_stat_file):
    '''This fucntion will compute the contribution coming via a specific type of looped path i.e.
    'stst' : source -> target -> source -> target'''
    print('start computing contribution from different path types')
    M = np.power(np.full_like(M_pathmtx,10), (-1)*M_pathmtx)
    #in M_pathmtx an index with value 0 means no edge. The above statement turns all 0 values to 1.
    # Now to preserve the same meaning for such indices, we need to convert all 1 to 0 in M.
    where_1 = np.where(M == 1) #TODO: replace 1's with 0's in time efficient way
    M[where_1]=0

    #Also take transpose of M to make the computation clearer i.e. after transpose
    #along the row I have u and along column I have v for every (u,v) edge.
    M = M.T
    top_direct_targets = find_top_preds_only_via_direct_edges(M,top_k_pred_idx,pos_nodes_idx)

    frac_contr_via_pathlen_1 = {}
    frac_contr_via_pathlen_2 = {}
    frac_contr_via_pathlen_3 = {}
    frac_contr_via_pathlen_4 = {}

    frac_contr_from_stst_path = {}
    frac_contr_from_stut_path = {}
    frac_contr_from_sust_path = {}
    frac_contr_from_suvt_path = {}
    frac_contr_from_path_len_beyond_4 = {}

    M2 = np.matmul(M,M) #M^2
    stut_contr_matrix = np.matmul(M, diags(M2.diagonal()).A)
    sust_contr_matrix = np.matmul(diags(M2.diagonal()).A, M)
    M3 = np.matmul(M,M2) #M^3
    M4 = np.matmul(M,M3)

    for target in top_direct_targets:
        pathlen_1_cost = 0
        patheln_2_cost = 0
        patheln_3_cost = 0
        patheln_4_cost = 0

        stst_path_cost = 0
        stut_path_cost = 0
        sust_path_cost = 0
        suvt_path_cost = 0
        total_incoming_contr = sum(R[target, pos_nodes_idx])
        for source in pos_nodes_idx:
            pathlen_1_cost += M[source][target]
            patheln_2_cost += M2[source][target]
            patheln_3_cost += M3[source][target]
            patheln_4_cost += M4[source][target]

            stst_path_cost_indv = M[source][target]*\
                   M[target][source]*M[source][target]
            stst_path_cost+= stst_path_cost_indv

            stut_path_cost += stut_contr_matrix[source][target] - stst_path_cost_indv
            sust_path_cost += sust_contr_matrix[source][target] - stst_path_cost_indv

            suvt_path_cost += M3[source][target] - sust_contr_matrix[source][target] \
                              - stut_contr_matrix[source][target] + stst_path_cost_indv

        frac_contr_via_pathlen_1[target] = pathlen_1_cost/total_incoming_contr
        frac_contr_via_pathlen_2[target] = patheln_2_cost/total_incoming_contr
        frac_contr_via_pathlen_3[target] = patheln_3_cost/total_incoming_contr
        frac_contr_via_pathlen_4[target] = patheln_4_cost/total_incoming_contr

        frac_contr_from_stst_path[target] = stst_path_cost/total_incoming_contr
        frac_contr_from_stut_path[target] = stut_path_cost/total_incoming_contr
        frac_contr_from_sust_path[target] = sust_path_cost/total_incoming_contr
        frac_contr_from_suvt_path[target] = suvt_path_cost/total_incoming_contr
        frac_contr_from_path_len_beyond_4[target] = 1-(frac_contr_via_pathlen_1[target]+\
            frac_contr_via_pathlen_2[target]+ frac_contr_via_pathlen_3[target] + frac_contr_via_pathlen_4[target])


    del M2, M3, M4, stut_contr_matrix,sust_contr_matrix,

    frac_contr_from_different_paths_df = pd.DataFrame({'target':list(frac_contr_from_stst_path.keys()),
                        'frac_pathlen_1':list(frac_contr_via_pathlen_1.values()),
                        'frac_pathlen_2':list(frac_contr_via_pathlen_2.values()),
                        'frac_stst': list(frac_contr_from_stst_path.values()),
                        'frac_stut':list(frac_contr_from_stut_path.values()),
                        'frac_sust':list(frac_contr_from_sust_path.values()),
                        'frac_suvt':list(frac_contr_from_suvt_path.values()),
                        'frac_pathlen_3': list(frac_contr_via_pathlen_3.values()),
                        'frac_pathlen_4': list(frac_contr_via_pathlen_4.values()),
                        'frac_beyond_pathlen4':list(frac_contr_from_path_len_beyond_4.values()),
                        })

    frac_contr_from_different_paths_df_only_top_preds = frac_contr_from_different_paths_df\
                                        [frac_contr_from_different_paths_df['target'].isin(top_k_pred_idx)]
    frac_contr_from_different_paths_df_non_top_preds = frac_contr_from_different_paths_df \
        [~(frac_contr_from_different_paths_df['target'].isin(top_k_pred_idx))]

    #save path_type wise contribution for all
    frac_contr_from_different_paths_df_only_top_preds.to_csv(paths_stat_file.replace('stat','top'), sep='\t',index=False)
    frac_contr_from_different_paths_df_non_top_preds.to_csv(paths_stat_file.replace('stat','bot'), sep='\t',index=False)

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

            #If 'pos_k'=True, then the number of top predictions is equal to the number of positively annotated nodes
            # for this certain term.
            if kwargs.get('pos_k'):
                k = n_pos
                print('k: ', k)
            for alg_name in alg_settings:
                if (alg_settings[alg_name]['should_run'][0] == True) or (alg_name in kwargs.get('run_algs')):
                    # load the top predictions
                    print(alg_name)
                    if kwargs.get('balancing_alpha_only'): #in alg_setting[alg_name]['alpha'] put the balancing alpha
                        # get the balancing alpha for this network - alg - term
                        alpha_summary_filename = config_map['output_settings']['output_dir'] + \
                            "/viz/%s/%s/param_select/" % (dataset['net_version'], dataset[
                            'exp_name']) + '/' + alg_name + '/alpha_summary.tsv'
                        alpha_summary_df = pd.read_csv(alpha_summary_filename, sep='\t', index_col=None)[['term','balancing_alpha']]
                        term_2_balancing_alpha_dict = dict(zip(alpha_summary_df['term'], alpha_summary_df['balancing_alpha']))

                        balancing_alpha = term_2_balancing_alpha_dict[term]
                        alg_settings[alg_name]['alpha'] = [balancing_alpha]


                    alg_pred_files = config_utils.get_dataset_alg_prediction_files(
                        output_dir, dataset, alg_settings, [alg_name], **kwargs)
                    # get the alpha values to use
                    alphas = alg_settings[alg_name]['alpha']

                    for alpha, alg in zip(alphas, alg_pred_files):
                        t1=time.time()
                        path_length_wise_contr = {}
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

                        shortest_path_input_graph_file = "%s/shortest-path-input-graph-2ss-%s-a%s-%s.txt" % \
                                                         (net_obj.out_pref, alg_name,
                                                          str(alpha).replace('.', '_'), term)

                        # shortest_path_file will be created only when all shortest paths for
                        # the targets for this certain setup (i.e. nsp, m, k values) have been computed.
                        ssource_2_ssink_paths_file = config_map['output_settings'][
                                        'output_dir'] + "/viz/%s/%s/diffusion-path-analysis/%s/shortest_path_2ss/" \
                                        "shortest-paths-2ss-nsp%s-a%s%s.tsv" % (
                                         dataset['net_version'], term, alg_name,
                                         kwargs.get('n_sp'), alpha, sig_str)

                        paths_of_different_types_stat_file = config_map['output_settings'][
                                        'output_dir'] + "/viz/%s/%s/diffusion-path-analysis/%s/type_based_path_stat-a%s%s.tsv" % (
                                        dataset['net_version'], term, alg_name,
                                        alpha, sig_str)
                        #the following file will contain all nsp paths as in ssource_2_ssink_paths_file, but this file will contain
                        # some extra information/stat about those paths.
                        nsp_processed_paths_file = config_map['output_settings'][
                                                       'output_dir'] + "/viz/%s/%s/diffusion-path-analysis/%s/shortest_path_2ss/processed_shortest-paths-2ss-nsp%s-a%s%s.tsv" % (
                                                       dataset['net_version'], term, alg_name, kwargs.get('n_sp'),
                                                       alpha, sig_str)

                        summary_file = config_map['output_settings'][
                                                         'output_dir'] + "/viz/%s/%s/diffusion-path-analysis/%s/shortest_path_2ss/" \
                                                                         "summary-2ss-nsp%s-a%s%s.tsv" % (
                                                         dataset['net_version'], term, alg_name,
                                                         kwargs.get('n_sp'), alpha, sig_str)

                        interesting_paths_file = config_map['output_settings'][
                                     'output_dir'] + "/viz/%s/%s/diffusion-path-analysis/%s/shortest_path_2ss/paths_of_interest-nsp%s-a%s%s.tsv" % (
                                     dataset['net_version'], term, alg_name, kwargs.get('n_sp'),
                                     alpha, sig_str)

                        os.makedirs(os.path.dirname(interesting_paths_file), exist_ok=True)

                        ##########CREAE or LOAD THE DIFFUSION MAYTRICES
                        force_matrix = False
                        # M_inv = alg_alias[alg_name].get_diffusion_matrix(net_obj.W, alpha=alpha,
                        #         diff_mat_file=diff_mat_file,force_run=force_matrix)
                        M_pathmtx, R = alg_alias[alg_name].get_fluid_flow_matrix(net_obj.W, alpha=alpha, \
                                        fluid_flow_mat_file_M=fluid_flow_mat_file_M, \
                                        fluid_flow_mat_file_R=fluid_flow_mat_file_R, force_run=force_matrix)


                        # if alg_name =='rwr':
                        #     M_inv = M_inv * (alpha / float(len(pos_nodes_idx)))


                        compute_and_save_contr_from_different_types_of_paths \
                            (M_pathmtx, R, pos_nodes_idx, top_k_pred_idx, paths_of_different_types_stat_file)

                        if (not os.path.exists(nsp_processed_paths_file)) or (kwargs.get('force_contr') == True):
                            ## The following function has to run for every term. as it adds edges from
                            # supersource to pos_idx which is different for every term and supersink to
                            # top preds.
                            n_shortest_path = kwargs.get('n_sp')
                            # **************RUN EPPSTEIN*********************
                            wd = os.getcwd()
                            # change directory to the java base Eppstein's algo
                            eppstein_code_dir = '/data/tasnina/k-shortest-paths/'

                            t0 = time.time()
                            os.chdir(eppstein_code_dir)
                            p1 = subprocess.Popen(['javac', './edu/ufl/cise/bsmock/graph/ksp/test/TestEppstein.java'])
                            p1.wait()
                            os.chdir(wd)

                            if (not os.path.isfile(ssource_2_ssink_paths_file)) or (kwargs.get('force_ksp') == True):
                                ssource_idx, ssink_idx = \
                                    save_weighted_graph_with_supersource_supersink(M_pathmtx, pos_nodes_idx,
                                                                                   top_k_pred_idx,
                                                                                   shortest_path_input_graph_file)
                                target = ssink_idx
                                os.makedirs(os.path.dirname(ssource_2_ssink_paths_file), exist_ok=True)
                                f = open(ssource_2_ssink_paths_file, 'w')
                                f.close()
                                sources = str([ssource_idx])
                                # if source and target are not in same connected component then do not pass this
                                # source-target pair to Eppstein's
                                # ****************** RUN EPPSTEIN's KSP***********************
                                # write code for running Eppstein's ksp algo here.
                                # get current directory
                                eppstein_inputs = ["pathnum", shortest_path_input_graph_file,
                                                     ssource_2_ssink_paths_file, str(sources), str(target),
                                                     str(n_shortest_path) ]
                                os.chdir(eppstein_code_dir)

                                p = subprocess.Popen(['java', 'edu.ufl.cise.bsmock.graph.ksp.test.TestEppstein'] + \
                                                     eppstein_inputs)
                                p.wait()
                                os.chdir(wd)
                                os.remove(shortest_path_input_graph_file)

                            ##process the output file given by Java
                            path_df = pd.read_csv(ssource_2_ssink_paths_file, sep=' ', header=None,
                                      index_col=None, names=['source', 'target', 'cost', 'path'])
                            path_df = process_java_computed_eppstein_output_for_supersource_supersink(path_df, prots)
                            interesting_paths_df, processed_nsp_paths_df = \
                                find_paths_of_interest(path_df, R, pos_nodes_idx, top_k_pred_idx, net_obj.W )
                            interesting_paths_df.to_csv(interesting_paths_file, sep='\t', index=False)
                            processed_nsp_paths_df.to_csv(nsp_processed_paths_file, sep='\t', index=False)

                            #write overall summary
                            write_summary(processed_nsp_paths_df, summary_file, **kwargs)

                            #REMOVE THE GRAPH FILES HERE
                        del R , M_pathmtx


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    for k in kwargs.get('k_to_test'):
        main(config_map, k=k, **kwargs)
