import os, sys
import yaml
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

sys.path.insert(1, "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/")
from src.FastSinkSource.src.main import setup_dataset
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils
import src.scripts.utils as script_utils
from src.scripts.plot_utils import *
import json

from src.graphspace import post_to_graphspace_base as gs
from src.graphspace.signor_gs_utils import *
from src.graphspace import backend_utils as back_utils
from src.graphspace import post_to_graphspace_wrapper as wrapper
from src.utils import network_processing_utils as net_utils


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

    #ALGO specific arguments
    group.add_argument('--config', type=str, default="/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                        "fss_inputs/config_files/provenance/signor_s12.yaml"
                       , help="Configuration file used when running FSS. ")
    group.add_argument('--id-mapping-file', type=str, default = "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                        "datasets/mappings/human/uniprot-reviewed-status.tab.gz",
                       help="Table downloaded from UniProt to map to gene names. Expected columns: 'Entry', 'Gene names', 'Protein names'")
    group.add_argument('--evidence-file', default = "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                        "datasets/networks/signor-cc/all_data_22_12_22.tsv", help='File containing the evidence for each edge in the interactome.')
    group.add_argument('--k-to-test', '-k', type=int, action='append', default=[332],
                       help="k-value(s) for which to get the top-k predictions to test. " +
                            "If not specified, will check the config file.")
    group.add_argument('--pos-k', action='store_true', default=True,
                       help="if true get the top-k predictions to test is equal to the number of positive annotations")
    group.add_argument('--n-sp', '-n', type=int, default=1000,
                       help="n-sp is the number of shortest paths to be considered" +
                            "If not specified, will check the config file. Default=20")
    group.add_argument('--n-sp-viz', type=int, default=100,
                       help="How many top paths to vizualize" +
                            "Default=20")
    group.add_argument('--stat-sig-cutoff', type=float,
                       help="Cutoff on the node p-value for a node to be considered in the topk. " + \
                            "The p-values should already have been computed with run_eval_algs.py")
    group.add_argument('--balancing-alpha-only', action='store_true', default=True,
                       help="Ignore alpha from config file rather take the alpha value\
                            that balanced the two loss terms in quad loss function for the corresponding\
                            network-term-alg")

    #GO term related arguments
    parser.add_argument('--gaf-file', type=str, default = "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                        "datasets/go/goa_human.gaf", help="File containing GO annotations in GAF format. Required")
    parser.add_argument('--obo-file', type=str, default = "/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/"
                        "datasets/go/go-basic.obo", help="GO OBO file which contains the GO DAG. Required")
    group.add_argument('--pval-cutoff', type=float, default=0.01,
                       help="Cutoff on the corrected p-value for enrichment.")



    ############### GRAPH ATTR

    group.add_argument('--force-attr', action='store_true', default=True,
                       help="If true create new graph_attr_file.")


    ########## GRAPHSPACE GRAPH Prep
    # posting options
    group.add_argument('--viz-type', type=str, default='top_paths_enrich_GO',
                       help="Options: 'top_paths_enrich_GO'/ 'top_paths'. "
                            "top_paths_enrich_GO: show the proteins in cluster under enriched GO terms on top paths."
                            "top_paths: show the most contributing paths (nodes and edges)  " )
    group.add_argument('--n-go', type=int, default=20,
                       help="How many enriched GO terms to visualize")
    group.add_argument('--freq-cutoff', type=float, default=0.5,
                       help="Apply this frequency cutoff on REVIGO simplified GO terms. More frequency"
                       "means more general terms. If the frequency>0.75,do not consider that term. ")

    #GRAPHSPACE POSTING

    group.add_argument('-U', '--username', type=str,  default='tasnina@vt.edu',
                      help='GraphSpace account username to post graph to. Required')
    group.add_argument('-P', '--password', type=str, default='1993Hello#GraphSpace',
                      help='Username\'s GraphSpace account password. Required')
    group.add_argument('--graph-name', type=str,  default='signor',
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
    group.add_argument( '--tags', type=str,action="append",
                      help='Tag to put on the graph. Can list multiple tags (for example --tag tag1 --tag tag2)')
    group.add_argument( '--apply-layout', type=str,
                      help='Specify the name of a graph from which to apply a layout. Layout name specified by the --layout-name option. ' +
                           'If left blank and the graph is being updated, it will attempt to apply the --layout-name layout.')
    group.add_argument( '--layout-name', type=str, default='layout1',
                      help="Name of the layout (of the graph specified by the --apply-layout option). " +
                           "X and y coordinates of nodes from that layout will be applied to matching node IDs in this graph. Default: 'layout1'")
    group.add_argument('--parent-nodes', action="store_true", default=True,
                      help='Use parent/group/compound nodes for the different node types')

    group.add_argument('--out-pref', type=str, metavar='STR',default='/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/outputs/graphspace/',
                      help='Prefix of name to place output files. ')
    return parser


def prepare_for_graphspace(prededges, orig_pos, all_prots, node_info_dict, go_id_2_name, graph_attr_file,uniprot_to_protein_names,
                           uniprot_to_gene, force_attr=True):
    '''
    This function will prepapre graphspace data such as node_labels, graph_attr, popups
    Output:
    node_labels:
    graph_attr:
    popups:
    '''
    prednodes = set([n for edge in prededges for n in edge])
    #******************** GRAPH ATTR *******************************
    #Write/read graph attributes specific for nodes and edges. Basically save the attributes that are not dynamic here.
    if (not os.path.isfile(graph_attr_file)) or (force_attr == True):
        createGraphAttr(orig_pos, all_prots, graph_attr_file)
    graph_attr, _ = gs.readGraphAttr(graph_attr_file)
    #We can add more graph_attributes even after reading from the file.
    # set the width of the edges by the network weight
    # if kwargs.get('net') is not None and kwargs.get('set_edge_width'):
    #     graph_attr = gs.set_edge_width(edges, edge_weights, graph_attr, a=1, b=12)

    #******************** Node Labels *******************************

    #**************************** POPUPS **********************************

    # create popups for nodes and edges
    attr_desc = {n:{} for n in prednodes}
    # convert into dict of dict. Key=prot, value={'score':a, 'rank':b, 'GO_C_ID':c, 'GO_C_term':d, 'GO_F_ID':e, 'GO_F_term':f,
    # 'GO_P_ID':g, 'GO_P_term':h}
    attrs_of_interest = ['score','rank','GO_BP_ID']
    for node in prednodes:
        # TODO: Find out why some uniprots do not have protein names
        # add protein names for the uniprot
        if node in uniprot_to_protein_names:
            attr_desc[node]['Protein names'] = uniprot_to_protein_names[node]
        for attr_name in attrs_of_interest:
            attr_desc[node][attr_name] = node_info_dict[node][attr_name]

    popups = {}
    for n in prednodes:
        popups[n] = gs.buildNodePopup(n, go_id_2_name, attr_val=attr_desc)
    # for u, v in prededges:
    #     popups[(u, v)] = gs.buildEdgePopup(u, v, node_labels=node_labels, attr_val=attr_desc)
    return graph_attr, uniprot_to_gene, popups


def post_graph_gs(edges,node_labels, graph_attr, popups, **kwargs):
    ''' Inputs: edges: a list tuples where each tuple is an edge'''

    # Now post to graphspace!
    G = gs.constructGraph(edges, node_labels=node_labels, graph_attr=graph_attr, popups=popups)

    # TODO add an option to build the 'graph information' tab legend/info
    # build the 'Graph Information' metadata
    # desc = gs.buildGraphDescription(edges_file, kwargs.get('net'))
    # metadata = {'description': desc, 'tags': [], 'title': ''}
    #
    # G.set_data(metadata)
    graph_name = kwargs.get('graph_name')+'-'+str(datetime.now())
    G.set_name(graph_name)

    # before posting, see if we want to write the Graph's JSON to a file
    if kwargs.get('out_pref') is not None:
        print("Writing graph and style JSON files to:\n\t%s-graph.json \n\t%s-style.json" % (
        kwargs['out_pref'], kwargs['out_pref']))
        graph_json_f = kwargs['out_pref'] + graph_name + "-graph.json"
        style_json_f = kwargs['out_pref'] + graph_name + "-style.json"

        with open(graph_json_f, 'w') as out:
            json.dump(G.get_graph_json(), out, indent=2)
        with open(style_json_f, 'w') as out:
            json.dump(G.get_style_json(), out, indent=2)

    gs.post_graph_to_graphspace(G, kwargs.get('username'), kwargs.get('password'),graph_name,
                             apply_layout=kwargs.get('apply_layout'),
                             layout_name=kwargs.get('layout_name'),
                             group=kwargs.get('group'), make_public=kwargs.get('make_public'))


def main(config_map, k, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """

    # TEST GRAPH SPACE GRAPH POSTING
    # sample_edges_file = '/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/fss_inputs/graphspace/test_graphs/graph_1.txt'
    # post_graph_gs(sample_edges_file, **kwargs)


    nsp = kwargs.get('n_sp')
    sig_cutoff = kwargs.get('stat_sig_cutoff')
    sig_str = "-sig%s" % (str(sig_cutoff).replace('.', '_')) if sig_cutoff else ""
    force_attr = kwargs.get('force_attr')

    gaf_file = kwargs.get('gaf_file')
    obo_file = kwargs.get('obo_file')

    input_settings, input_dir, output_dir, alg_settings, kwargs \
        = config_utils.setup_config_variables(config_map, **kwargs)
    # gene name for corresponding uniprot is the node label
    if kwargs.get('id_mapping_file') is not None:
        uniprot_to_gene, uniprot_to_protein_names = load_gene_names(kwargs.get('id_mapping_file'))
        # Nure: There is a node 'P01019-PRO_0000420660' for which we do not have gene_name but
        # for P01019 we have. So add 'PRO_0000420660' to node_labels.
        uniprot_to_gene['P01019-PRO_0000420660'] = uniprot_to_gene['P01019']
        uniprot_to_protein_names['P01019-PRO_0000420660'] = uniprot_to_protein_names['P01019']

    for dataset in input_settings['datasets']:
        print("Loading data for %s" % (dataset['net_version']))
        base_out_dir = "%s/enrichment/%s/%s" % (output_dir, dataset['net_version'], dataset['exp_name'])
        # load the network and the positive examples for each term
        net_obj, ann_obj, _ = setup_dataset(dataset, input_dir, **kwargs)
        prots, node2idx = net_obj.nodes, net_obj.node2idx
        prot_universe = set(prots) #TODO discuss with Murali about what prot set to use as universe
        dataset_name = config_utils.get_dataset_name(dataset)

        print("\t%d prots in universe" % (len(prot_universe)))
        for term in ann_obj.terms:
            term_idx = ann_obj.term2idx[term]
            orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, term_idx)
            orig_pos = [prots[p] for p in orig_pos_idx]
            targets = list(set(prots).difference(set(orig_pos)))

            pos_nodes_idx = [node2idx[n] for n in orig_pos if n in node2idx]
            n_pos = len(pos_nodes_idx)
            # If 'pos_k'=True, then the number of top predictions is equal to the number of positively annotated nodes
            # for this certain term.
            if kwargs.get('pos_k'):
                k = n_pos
                print('k: ', k)

            for alg_name in alg_settings:
                if (alg_settings[alg_name]['should_run'][0] == True):
                    # load the top predictions
                    print(alg_name)
                    if kwargs.get('balancing_alpha_only'):
                        balancing_alpha = script_utils.get_balancing_alpha(config_map, dataset, alg_name, term)
                        alg_settings[alg_name]['alpha'] = [balancing_alpha]
                    alphas = alg_settings[alg_name]['alpha']
                    alg_pred_files = config_utils.get_dataset_alg_prediction_files(
                        output_dir, dataset, alg_settings, [alg_name], **kwargs)

                    for alpha, alg in zip(alphas, alg_pred_files):
                        nsp_processed_paths_file = config_map['output_settings'][
                                                    'output_dir'] + "/viz/%s/%s/diffusion-path-analysis/%s/shortest_path_2ss/" \
                                                    "processed_shortest-paths-2ss-k%s-nsp%s-a%s%s.tsv" % (
                                                    dataset['net_version'], term, alg_name,k, nsp, alpha, sig_str)
                        # reading 'path_prots' column value as list
                        paths_df = pd.read_csv(nsp_processed_paths_file, sep='\t', index_col=None,
                                               converters={'path_prots': pd.eval})
                        paths = list(paths_df['path_prots'])  # this is one nested list. flatten it.

                        # pred score file
                        pred_file = alg_pred_files[alg]
                        pred_file = script_utils.term_based_pred_file(pred_file, term)
                        df = pd.read_csv(pred_file, sep='\t')
                        df_nonpos = df[~df['prot'].isin(orig_pos)].reset_index(drop=True)
                        top_targets = list(df_nonpos['prot'])[0:k]

                        if kwargs.get('viz_type')=='top_paths':
                            node_info_dict = back_utils.handle_get_node_info(pred_file, orig_pos, gaf_file)
                            # get go_id to go_term(name) mapping
                            go_id_2_name = back_utils.get_go_id_2_name_mapping(obo_file)

                            #now take top paths to visualize
                            paths = paths[0: kwargs.get('n_sp_viz')]
                            top_edges = net_utils.get_edges_from_list_of_paths(paths)
                            #start preparing for posting graph to GraphSpace
                            graph_attr_file = config_map['input_settings']['input_dir'] + "/graphspace/%s-%s/%s-a%s%s/nsp%s.tsv" % (
                                                  dataset['net_version'], term, alg_name, alpha, sig_str, nsp)
                            graph_attr, node_labels, popups = prepare_for_graphspace(list(top_edges.keys()), orig_pos, prots,
                                                            node_info_dict, go_id_2_name, graph_attr_file,
                                                            uniprot_to_protein_names, uniprot_to_gene )
                            post_graph_gs(top_edges, node_labels, graph_attr, popups,**kwargs)

                        elif kwargs.get('viz_type')=='top_paths_enrich_GO':
                            #how many GO term to visualize
                            n_go=kwargs.get('n_go')
                            # enrichment file
                            enrich_out_dir = "%s/%s/%s/a-%s/" % (base_out_dir, alg, term, str(alpha))
                            for ont in ['BP']:
                            # for ont in ['BP','MF','CC']:
                                #take the edges in the top paths on which the enrichment had been computed.
                                paths = paths[0: kwargs.get('n_sp')]
                                top_edges = net_utils.get_edges_from_list_of_paths(paths)
                                #the nodes in top_edges after removing the original nodes
                                # top_nodes_non_pos = (set([t for t, h in top_edges]).union(set([h for t, h in top_edges]))).\
                                #     difference(set(orig_pos))
                                # m = len(top_nodes_non_pos)
                                # #now take the top m predicted nodes
                                # top_targets = list(df_nonpos['prot'])[0:m]

                                path_enrich_file = "%s/enrich-%s-%s-%s_revigo_simplified.csv" % (enrich_out_dir, ont,
                                                    'top_paths'+'_k'+str(k)+'_nsp'+str(nsp),
                                                    str(kwargs.get('pval_cutoff')).replace('.', '_'))
                                path_enrich_df = pd.read_csv(path_enrich_file).set_index('ID')
                                # path_enrich_df.sort_values(by=['p.adjust'], ascending=True, inplace=True)

                                #take n_go GO terms at each turn
                                path_enrich_df['Frequency']=path_enrich_df['Frequency'].apply(lambda x: float(x.replace('%','')))
                                freq_cutoff = kwargs.get('freq_cutoff')
                                path_enrich_df = path_enrich_df[path_enrich_df['Frequency']<freq_cutoff][['Description', 'geneID','p.adjust']]
                                #convert into dict of dict. outer key = ID, inner keys = ['Description', 'geneID','p.adjust']
                                path_enrich_dict = path_enrich_df.to_dict(orient='index')

                                sorted_go_terms = list(path_enrich_df.index)
                                count=0

                                while (((count+n_go)<len(list(path_enrich_dict.keys()))) and count<100) :
                                    selected_goids = sorted_go_terms[count:count+n_go]
                                    term_names = {goid:path_enrich_dict[goid]['Description'] for goid in selected_goids}
                                    term_prots = {goid:path_enrich_dict[goid]['geneID'] for goid in selected_goids}
                                    term_pvals = {goid:path_enrich_dict[goid]['p.adjust'] for goid in selected_goids}
                                    kwargs['graph_name'] = dataset_name+ '-' + ont + '-top-paths-'+str(nsp)+'-'+str(n_go)+'_'+\
                                                           str(count)+'-freq-'+str(freq_cutoff)+'-'+str(datetime.now())
                                    wrapper.call_post_to_graphspace(top_edges, orig_pos, targets , top_targets, uniprot_to_gene,
                                                            uniprot_to_protein_names, selected_goids, term_names, term_prots, term_pvals,
                                                            out_file=output_dir+'/graphspace/color.txt', **kwargs)
                                    count+=n_go

def load_gene_names(id_mapping_file):
    df = pd.read_csv(id_mapping_file, sep='\t', header=0)
    #keep the 'reviewed' uniprot to Gene Names mapping only.
    df = df[df['Reviewed']=='reviewed']
    ## keep only the first gene for each UniProt ID
    uniprot_to_gene = {p: genes.split(' ')[0] for p, genes in zip(df['Entry'], df['Gene Names'].astype(str))}
    if 'Protein names' in df.columns:
        uniprot_to_prot_names = dict(zip(df['Entry'], df['Protein names'].astype(str)))
        #node_desc = {n: {'Protein names': uniprot_to_prot_names[n]} for n in uniprot_to_prot_names}
    return uniprot_to_gene, uniprot_to_prot_names


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    for k in kwargs.get('k_to_test'):
        main(config_map, k=k, **kwargs)

