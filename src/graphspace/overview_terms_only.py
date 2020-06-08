
import os, sys
import yaml
import argparse
import itertools
from tqdm import tqdm
from collections import defaultdict
# TODO give the path to this repo
import pandas as pd
import numpy as np
from scipy import sparse as sp
# GSGraph already implements networkx
import networkx as nx
import json

# local imports
sys.path.insert(0,"")
from src import setup_datasets
from src.FastSinkSource.src import main as run_eval_algs
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils
from src.graphspace import post_to_graphspace_base as gs
from src.graphspace import gs_utils
from src.graphspace.overview_gs import *


def setup_opts2():
    # use the same arguments as the other script
    parser = setup_opts()

#    # additional parameters
#    group = parser.add_argument_group('Additional options')
#    group.add_argument('--forcealg', action="store_true", default=False,
#            help="Force re-running algorithms if the output files already exist")
#    group.add_argument('--forcenet', action="store_true", default=False,
#            help="Force re-building network matrix from scratch")
#    group.add_argument('--verbose', action="store_true", default=False,
#            help="Print additional info about running times and such")

    return parser


def main(config_map, **kwargs):
    """
    *config_map*: everything in the config file
    *kwargs*: all of the options passed into the script
    """
    # extract the general variables from the config map
    input_settings, input_dir, output_dir, alg_settings, kwargs \
        = config_utils.setup_config_variables(config_map, **kwargs)
    #algs = config_utils.get_algs_to_run(alg_settings, **kwargs)
    #del kwargs['algs']
    for dataset in input_settings['datasets']:
        print("Loading data for %s" % (dataset['net_version']))
        # load the network and the positive examples for each term
        net_obj, ann_obj, eval_ann_obj = run_eval_algs.setup_dataset(
            dataset, input_dir, **kwargs) 
        # find the shortest path(s) from each drug node to any virus node
        if kwargs.get('edge_weight_cutoff'):
            print("\tapplying edge weight cutoff %s" % (kwargs['edge_weight_cutoff']))
            W = net_obj.W
            W = W.multiply((W > kwargs['edge_weight_cutoff']).astype(int))
            net_obj.W = W
            num_nodes = np.count_nonzero(W.sum(axis=0))  # count the nodes with at least one edge
            num_edges = (len(W.data) / 2)  # since the network is symmetric, the number of undirected edges is the number of entries divided by 2
            print("\t%d nodes and %d edges" % (num_nodes, num_edges))

    graph_attr = defaultdict(dict)
    attr_desc = defaultdict(dict)
    if kwargs.get('graph_attr_file'):
        graph_attr, attr_desc = gs.readGraphAttr(kwargs['graph_attr_file'])
    # load the namespace mappings
    uniprot_to_gene = None
    # also add the protein name
    uniprot_to_prot_names = None
    # these 
    node_desc = defaultdict(dict)
    if kwargs.get('id_mapping_file'):
        df = pd.read_csv(kwargs['id_mapping_file'], sep='\t', header=0) 
        ## keep only the first gene for each UniProt ID
        uniprot_to_gene = {p: genes.split(' ')[0] for p, genes in zip(df['Entry'], df['Gene names'].astype(str))}
        if 'Protein names' in df.columns:
            uniprot_to_prot_names = dict(zip(df['Entry'], df['Protein names'].astype(str)))
            node_desc = {n: {'Protein names': uniprot_to_prot_names[n]} for n in uniprot_to_prot_names}

    # load human-virus ppis
    print("reading %s" % (kwargs['sarscov2_human_ppis']))
    df = pd.read_csv(kwargs['sarscov2_human_ppis'], sep='\t')
    edges = zip(df[df.columns[0]], df[df.columns[1]])
    #edges = [(v.replace("SARS-CoV2 ",""), uniprot_to_gene[h]) for v,h in edges]
    edges = [(v.replace("SARS-CoV2 ",""), h) for v,h in edges]
    virus_nodes = [v for v,h in edges]
    krogan_nodes = [h for v,h in edges]
    virhost_edges = edges 

    print("reading %s" % (kwargs['simplified_terms_file']))
    df = pd.read_csv(kwargs['simplified_terms_file'], sep='\t')
    kwargs['terms_to_highlight'] = df[df.columns[0]]
    term_names = dict(zip(df[df.columns[0]], df[df.columns[1]]))
    term_names = {t: name.replace('\\n','\n') for t, name in term_names.items()}
    # if there's a term_colors column, use that
    term_colors = dict(zip(df[df.columns[0]], df['term_colors']))
    print(df.head())

    # read the enrichment results
    df, term_ann = load_enrichment_file(**kwargs)
    # make a parent node per term in the simplified file
    # also get the term name, and the enrichment p-value(?)
    #term_names = dict(zip(df.index, df[('Description', 'Unnamed: 1_level_1', 'Unnamed: 1_level_2')]))
    pred_nodes = set()
    # setup their graph_attributes for posting
    covered_prots = set()
    prot_per_parent_term = defaultdict(set)
    # TODO need to order these correctly so that each term will have prots
    for i, t in enumerate(kwargs['terms_to_highlight']):
        name = term_names[t]
        color = term_colors[t]
        # UPDATE, make the terms be regular nodes
        graph_attr[name].update(group_node_styles)
        graph_attr[name]["background-opacity"] = 0.8
        graph_attr[name]['color'] = color
        graph_attr[name]['shape'] = 'ellipse'
        graph_attr[name]['text-valign'] = 'bottom'
        # add this node as the parent to the other nodes
        if kwargs.get('virus_nodes'):
            pass
        else:
            #for n in set(term_ann[t]) - covered_prots:
            for n in set(term_ann[t]) & set(krogan_nodes):
                graph_attr[n]['parent'] = name
                graph_attr[n]['color'] = color
                pred_nodes.add(n)
                covered_prots.add(n) 
                prot_per_parent_term[name].add(n) 
        # TODO add the link
        # this is used to make the popup
        #attr_desc[('parent', name)] = t
    for parent, prots in prot_per_parent_term.items():
        print("%d prots for %s" % (len(prots), parent))
    # if the node list file is not specified, then set the annotated nodes as the node list
    #if not kwargs.get('node_list_file'):
    #    node_list = nodes

    # make the group node be a name, and the individual nodes be the terms(?)
    # for now, just leave the terms as regular nodes
    # make the size of the node be the number of proteins annotated
    #setup_term_nodes(term_ann, term_names)
    term_num_ann = {term_names[t]: len(ann) for t, ann in term_ann.items() if t in term_names}
    graph_attr = gs.set_node_size(term_names.values(), term_num_ann, graph_attr,)  # a=20, b=80, min_weight=None, max_weight=None)
    # add the number of ann to the popup, and the GO ID
    attr_desc.update({('# ann', t): num_ann for t, num_ann in term_num_ann.items()})
    attr_desc.update({('GO ID', term_names[t]): t for t in term_names})

    krogan_attr = setup_krogan_group_nodes(virhost_edges, graph_attr=graph_attr, **kwargs)
    for n in krogan_attr:
        graph_attr[n].update(krogan_attr[n])
    # add an edge from each virus group node to the GO term nodes, and weight by... 
    curr_pred_nodes, pred_edges = setup_virus_to_term_edges(net_obj, virhost_edges, term_ann, term_names)
    if not kwargs.get('virus_nodes'):
        pred_nodes.update(set([n for n in curr_pred_nodes if n in set(krogan_nodes)]))
    pred_nodes.update(set([t for t,h in pred_edges]).union(set([h for t,h in pred_edges])))
    # make sure all the terms are added as nodes
    pred_nodes.update(set(term_names.values()))

    # also add styles to the edges
    for e in pred_edges:
        graph_attr[e]['color'] = dark_gray
        graph_attr[e]['width'] = 5
        graph_attr[e]['opacity'] = 0.5
    graph_attr = gs.set_edge_width(
        pred_edges, pred_edges, graph_attr,
        a=kwargs.get('min_edge_width',2), b=kwargs.get('max_edge_width',12))

    #print(graph_attr['Q92769'])
    popups = {}
    for n in pred_nodes:
        if n in attr_desc:
            popups[n] = gs.buildNodePopup(n, attr_val=attr_desc)
            #popups[n] = gs.buildNodePopup(n, node_type=node_type, attr_val=attr_desc)
    node_labels = {}
    G = gs.constructGraph(
        pred_edges, prednodes=pred_nodes, node_labels=uniprot_to_gene,
        graph_attr=graph_attr, popups=popups)

    # set of group nodes to add to the graph
    #if kwargs.get('parent_nodes'):
        #group_nodes_to_add = [virus_group, krogan_group, drug_group, human_group]
        #group_nodes_to_add = set(attr['parent'] for n, attr in graph_attr.items() if 'parent' in attr) 
        #add_group_nodes(G, group_nodes_to_add, graph_attr, attr_desc)
    desc = ''
    metadata = {'description':desc,'tags':kwargs.get('tags',[]), 'title':''}
    G.set_data(metadata)
    if 'graph_exp_name' in dataset:
        graph_exp_name = dataset['graph_exp_name']
    else:
        #graph_exp_name = config_utils.get_dataset_name(dataset) 
        graph_exp_name = "%s-%s" % (dataset['exp_name'].split('/')[-1], dataset['net_version'].split('/')[-1])
    #graph_name = "%s-k%s%s" % (
    #    graph_exp_name, k_to_test[0], kwargs.get('name_postfix',''))
    graph_name = "%s%s" % (
        graph_exp_name, kwargs.get('name_postfix',''))
        #"test","", "")

    if kwargs.get('term_to_highlight'):
        graph_name += "-%sterms" % (len(kwargs['term_to_highlight']))
    G.set_name(graph_name)
    # also set the legend
    #G = set_legend(G)
    # write the posted network to a file if specified
    if kwargs.get('out_pref'):
        out_file = "%s%s.txt" % (kwargs['out_pref'], graph_name)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        print("writing network to %s" % (out_file))
        net_data = nx.node_link_data(G)
        net_data.update(G.get_style_json())
        #print(list(net_data.items())[:10])
        print("writing %s" % (out_file.replace('.txt','.json')))
        with open(out_file.replace('.txt','.json'), 'w') as outfile:
            json.dump(net_data, outfile, indent=4, sort_keys=True)
        # remove any newlines from the node name if they're there
        node_labels = {n: n.replace('\n','-') for n in G.nodes(data=False)}
        # TODO write the node data as well
        G2 = nx.relabel_nodes(G, node_labels, copy=True)
        nx.write_edgelist(G2, out_file)
        sys.exit()

    # put the parent nodes and the nodes in the parent nodes in a grid layout automatically
    print("Setting the x and y coordinates of each node in a grid layout")
    # relabel the nodes to their names
    graph_attr = {uniprot_to_gene.get(n,n): attr for n, attr in graph_attr.items()}
    layout = gs_utils.grid_layout(G, graph_attr)
    for node, (x,y) in layout.items():
        G.set_node_position(node_name=node, x=x, y=y)

    print("%d nodes and %d edges to post" % (G.number_of_nodes(), G.number_of_edges()))
    gs.post_graph_to_graphspace(
            G, kwargs['username'], kwargs['password'], graph_name, 
            apply_layout=kwargs['apply_layout'], layout_name=kwargs['layout_name'],
            group=kwargs['group'], make_public=kwargs['make_public'])


def setup_krogan_group_nodes(virhost_edges, graph_attr=None, **kwargs):
    if graph_attr is None:
        graph_attr = defaultdict(dict)

    for v, h in virhost_edges:
        if kwargs.get('virus_nodes'):
            graph_attr[v].update(virus_node_styles)
            graph_attr[v]['font-weight'] = "bolder"
            graph_attr[v]['font-size'] = "32px"
            graph_attr[v]['text-outline-width'] = 3
        else:
            graph_attr[v].update(group_node_styles)
            # update to have no background color, but have a red border
            #graph_attr[v]['color'] = virus_node_styles['color']
            graph_attr[v]['color'] = "#ffffff"
            graph_attr[v]['background-opacity'] = 0.6
            graph_attr[v]['border-color'] = virus_node_styles['color']
            graph_attr[v]['border-width'] = 5
            graph_attr[h]['parent'] = v
            color = graph_attr[h].get('color')
            # apply the krogan node styles
            graph_attr[h].update(krogan_node_styles)
            # change the default krogan node color to gray to indiciate it's not annotated with a term.
            graph_attr[h]['color'] = gray
            if color is not None:
                graph_attr[h]['color'] = color
    #print(graph_attr['Q92769'])
    # this krogan node doesn't have a virus parent, likely because it has not direct edges to a term
    #print(graph_attr['Q9UJZ1'])
    return graph_attr


def load_enrichment_file(enriched_terms_file, **kwargs):
    print("Reading %s" % (enriched_terms_file))
    df = pd.read_csv(enriched_terms_file, sep='\t', header=[0,1,2], index_col=0, comment='#') 
    df2 = df.copy()
    #print(df.head())
    # get the prots per term
    #df2 = df2[("STRING", "GM+", "Rank")]
    df2.columns = df2.columns.droplevel([0,1])
    df2 = df2['geneID']
    df2.columns = list(range(len(df2.columns)))
    for i in range(1,len(df2.columns)):
        df2[0] = df2[0] + '/' + df2[i]
    #print(df2.head())
    #print(df2['geneID'].head())
    term_ann = dict(zip(df2.index, df2[0].values))
    # UPDATE: get only the term annotations for a single method
    #df2 = df2[df2.columns[0]]
    #term_ann = dict(zip(df2.index, df2[0]))
    term_ann = {t: str(ann).split('/') for t,ann in term_ann.items()}
    print("\t%d terms, %s" % (
        len(term_ann),
        ", ".join("%s: %d ann" % (t, len(term_ann[t])) for t in kwargs['terms_to_highlight']))) 
    
    return df, term_ann


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)
