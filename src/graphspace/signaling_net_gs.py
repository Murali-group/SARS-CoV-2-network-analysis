import os, sys
import yaml
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


sys.path.insert(1, "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/")
from src.FastSinkSource.src.main import setup_dataset
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils
import src.scripts.utils as script_utils
from src.scripts.plot_utils import *

from src.graphspace import post_to_graphspace_base as gs
from src.FastSinkSource.src.utils import file_utils as utils


def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        config_map = yaml.load(conf, Loader=yaml.FullLoader)
        # config_map = yaml.load(conf)
    return config_map, kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="Script to visualize top contributing paths to k top scoring predictions")
    # general parameters
    group = parser.add_argument_group('Main Options')

    group.add_argument('--config', type=str, default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                        "fss_inputs/config_files/provenance/signor_s12.yaml"
                       , help="Configuration file used when running FSS. ")
    group.add_argument('--id-mapping-file', type=str,
                       help="Table downloaded from UniProt to map to gene names. Expected columns: 'Entry', 'Gene names', 'Protein names'")
    group.add_argument('--k-to-test', '-k', type=int, action='append', default=[332],
                       help="k-value(s) for which to get the top-k predictions to test. " +
                            "If not specified, will check the config file.")
    group.add_argument('--pos-k', action='store_true', default=False,
                       help="if true get the top-k predictions to test is equal to the number of positive annotations")
    group.add_argument('--n-sp', '-n', type=int, default=1000,
                       help="n-sp is the number of shortest paths to be considered" +
                            "If not specified, will check the config file. Default=20")
    group.add_argument('--n-sp-viz', type=int, default=20,
                       help="How many top paths to vizualize" +
                            "Default=20")
    group.add_argument('--stat-sig-cutoff', type=float,
                       help="Cutoff on the node p-value for a node to be considered in the topk. " + \
                            "The p-values should already have been computed with run_eval_algs.py")
    group.add_argument('--balancing-alpha-only', action='store_true', default=True,
                       help="Ignore alpha from config file rather take the alpha value\
                            that balanced the two loss terms in quad loss function for the corresponding\
                            network-term-alg")


    ########## GRAPH CONFIG
    # posting options
    group.add_argument('-U', '--username', type=str,  default='tasnina@vt.edu',
                      help='GraphSpace account username to post graph to. Required')
    group.add_argument('-P', '--password', type=str, default='1993Hello#GraphSpace',
                      help='Username\'s GraphSpace account password. Required')
    group.add_argument('--graph-name', type=str,  default='test',
                      help='Graph name for posting to GraphSpace. Default: "test".')
    # replace with an option to write the JSON file before/after posting
    # parser.add_option('', '--outprefix', type='string', metavar='STR', default='test',
    #                  help='Prefix of name to place output files. Required.')
    group.add_argument('--group', type=str,
                      help='Name of group to share the graph with.')
    group.add_argument('--make-public', action="store_true", default=False,
                      help='Option to make the uploaded graph public.')
    # TODO implement and test this option
    # parser.add_argument('', '--group-id', type='string', metavar='STR',
    #                  help='ID of the group. Could be useful to share a graph with a group that is not owned by the person posting')
    group.add_argument( '--tag', type=str,action="append",
                      help='Tag to put on the graph. Can list multiple tags (for example --tag tag1 --tag tag2)')
    group.add_argument( '--apply-layout', type=str,
                      help='Specify the name of a graph from which to apply a layout. Layout name specified by the --layout-name option. ' +
                           'If left blank and the graph is being updated, it will attempt to apply the --layout-name layout.')
    group.add_argument( '--layout-name', type=str, default='layout1',
                      help="Name of the layout (of the graph specified by the --apply-layout option). " +
                           "X and y coordinates of nodes from that layout will be applied to matching node IDs in this graph. Default: 'layout1'")
    return parser

def post_graph_gs(edges_file, **kwargs):
    '''
    edges_file: a file containing edges in a tab separated  format i.e. edge-head \t edge-tail.
    '''

    lines = utils.readColumns(edges_file, 1, 2)
    prededges = set(lines)

    node_labels = {}
    if kwargs.get('id_mapping_file') is not None:
        node_labels = utils.readDict(kwargs.get('id_mapping_file'), 1, 2)

    # get attributes of nodes and edges from the graph_attr file
    graph_attr = {}
    attr_desc = {}
    if kwargs.get('graph_attr'):
        graph_attr, attr_desc = gs.readGraphAttr( kwargs.get('graph_attr'))

    if  kwargs.get('net') is not None:
        # add the edge weight from the network to attr_desc which will be used for the popup
        edge_weights = {(u, v): float(w) for u, v, w in utils.readColumns(kwargs.get('net'), 1, 2, 3)}
        for e in prededges:
            if e not in attr_desc:
                attr_desc[e] = {}
            attr_desc[e]["edge weight"] = edge_weights[e]

    # set the width of the edges by the network weight
    if kwargs.get('net') is not None and kwargs.get('set_edge_width'):
        graph_attr = gs.set_edge_width(prededges, edge_weights, graph_attr, a=1, b=12)

    # TODO build the popups here. That way the popup building logic can be separated from the
    # GSGraph building logic
    popups = {}
    prednodes = set([n for edge in prededges for n in edge])
    for n in prednodes:
        popups[n] = gs.buildNodePopup(n, attr_val=attr_desc)
    for u, v in prededges:
        popups[(u, v)] = gs.buildEdgePopup(u, v, node_labels=node_labels, attr_val=attr_desc)

    # Now post to graphspace!
    G = gs.constructGraph(prededges, node_labels=node_labels, graph_attr=graph_attr, popups=popups)

    # TODO add an option to build the 'graph information' tab legend/info
    # build the 'Graph Information' metadata
    desc = gs.buildGraphDescription(edges_file, kwargs.get('net'))
    metadata = {'description': desc, 'tags': [], 'title': ''}

    G.set_data(metadata)
    G.set_name(kwargs.get('graph_name'))

    gs.post_graph_to_graphspace(G, kwargs.get('username'), kwargs.get('password'), kwargs.get('graph_name'),
                             apply_layout=kwargs.get('apply_layout'),
                             layout_name=kwargs.get('layout_name'),
                             group=kwargs.get('group'), make_public=kwargs.get('make_public'))


def main(config_map, k, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """

    # TEST GRAPH SPACE GRAPH POSTING
    sample_edges_file = '/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/fss_inputs/graphspace/test_graphs/graph_1.txt'
    post_graph_gs(sample_edges_file, **kwargs)

    # # extract the general variables from the config map
    # input_settings, input_dir, output_dir, alg_settings, kwargs \
    #     = config_utils.setup_config_variables(config_map, **kwargs)
    #
    # sig_cutoff = kwargs.get('stat_sig_cutoff')
    # sig_str = "-sig%s" % (str(sig_cutoff).replace('.', '_')) if sig_cutoff else ""
    # # m = kwargs.get('m')
    # max_pathlen = kwargs.get('max_len')
    #
    # # for each dataset, extract the path(s) to the prediction files,
    # # read in the predictions, and test for the statistical significance of overlap
    #
    # for dataset in input_settings['datasets']:
    #     print("Loading data for %s" % (dataset['net_version']))
    #     # load the network and the positive examples for each term
    #     net_obj, ann_obj, _ = setup_dataset(
    #         dataset, input_dir, **kwargs)
    #     prots, node2idx = net_obj.nodes, net_obj.node2idx
    #


        ##post Graph to GraphSpace
        # edges_file = input_dir + '/' + dataset['net_version'] + '/' + dataset['net_files'][0]
        # post_graph_gs(edges_file, **kwargs)


        # for term in ann_obj.terms:
        #     term_idx = ann_obj.term2idx[term]
        #     orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, term_idx)
        #     orig_pos = [prots[p] for p in orig_pos_idx]
        #     pos_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]
        #     n_pos = len(pos_nodes_idx)
        #
        #     #If 'pos_k'=True, then the number of top predictions is equal to the number of positively annotated nodes
        #     # for this certain term.
        #     if kwargs.get('pos_k'):
        #         k = n_pos
        #         print('k: ', k)
        #     for alg_name in alg_settings:
        #         if (alg_settings[alg_name]['should_run'][0] == True):
        #             # load the top predictions
        #             print(alg_name)
        #
        #             if kwargs.get('balancing_alpha_only'): #in alg_setting[alg_name]['alpha'] put the balancing alpha
        #                 # get the balancing alpha for this network - alg - term
        #                 alpha_summary_filename = config_map['output_settings']['output_dir'] + \
        #                     "/viz/%s/%s/param_select/" % (dataset['net_version'], dataset[
        #                     'exp_name']) + '/' + alg_name + '/alpha_summary.tsv'
        #                 alpha_summary_df = pd.read_csv(alpha_summary_filename, sep='\t', index_col=None)[['term','balancing_alpha']]
        #                 term_2_balancing_alpha_dict = dict(zip(alpha_summary_df['term'], alpha_summary_df['balancing_alpha']))
        #
        #                 balancing_alpha = term_2_balancing_alpha_dict[term]
        #                 alg_settings[alg_name]['alpha'] = [balancing_alpha]
        #
        #
        #             alg_pred_files = config_utils.get_dataset_alg_prediction_files(
        #                 output_dir, dataset, alg_settings, [alg_name], **kwargs)
        #             # get the alpha values to use
        #             alphas = alg_settings[alg_name]['alpha']
        #
        #             for alpha, alg in zip(alphas, alg_pred_files):
        #                 pred_file = alg_pred_files[alg]
        #                 pred_file = script_utils.term_based_pred_file(pred_file, term)
        #                 if not os.path.isfile(pred_file):
        #                     print("Warning: %s not found. skipping" % (pred_file))
        #                     continue
        #                 print("reading %s for alpha=%s" % (pred_file, alpha))
        #                 df = pd.read_csv(pred_file, sep='\t')
        #
        #
        #                 # remove the original positives for downstream analysis
        #                 df = df[~df['prot'].isin(orig_pos)]
        #                 df.reset_index(inplace=True, drop=True)
        #
        #                 if sig_cutoff:
        #                     df = config_utils.get_pvals_apply_cutoff(df, pred_file, **kwargs)
        #
        #                 if k > len(df['prot']):
        #                     print("ERROR: k %s > num predictions %s. Quitting" % (k, len(df['prot'])))
        #                     sys.exit()
        #
        #                 pred_scores = np.zeros(len(net_obj.nodes))
        #                 df = df[:k]
        #                 top_k_pred = df['prot']
        #                 top_k_pred_idx = [net_obj.node2idx[n] for n in top_k_pred]
        #
        #                 #the following file contains paths top nsp paths from considering supersource-to-supersink. However,
        #                 #here I removed the supersource and supersource from head and tail of the paths to get the actual paths
        #                 # 'path_prots' column cotains uniprots along a path.
        #                 nsp_processed_paths_file = config_map['output_settings'][
        #                                                'output_dir'] + "/viz/%s/%s/diffusion-path-analysis/%s/shortest_path_2ss/processed_shortest-paths-2ss-nsp%s-a%s%s.tsv" % (
        #                                                dataset['net_version'], term, alg_name, kwargs.get('n_sp'),
        #                                                alpha, sig_str)
        #
        #
        #                 # shortest_path_file = config_map['output_settings'][
        #                 #                          'output_dir'] + "/viz/%s/%s/diffusion-path-analysis/%s/shortest_path/" \
        #                 #                                          "shortest-paths-ss-k%s-nsp%s-a%s.txt" % (
        #                 #                          dataset['net_version'], term, alg_name, k,
        #                 #                          kwargs.get('n_sp'), alpha)
        #                 # for target in top_k_pred_idx:
        #                 #     target_spec_sp_file = shortest_path_file.replace('.txt', '_' + str(target) + '.tsv')
        #                 #     target_spec_sp_file = target_spec_sp_file.replace('-k' + str(k), '-pl' + str(
        #                 #         max_pathlen))


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    for k in kwargs.get('k_to_test'):
        main(config_map, k=k, **kwargs)


