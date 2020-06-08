
import os, sys
import yaml
import argparse
import itertools
from tqdm import tqdm
from collections import defaultdict
# TODO give the path to this repo
from graphspace_python.api.client import GraphSpace
from graphspace_python.graphs.classes.gsgraph import GSGraph
from graphspace_python.graphs.classes.gslegend import GSLegend
import pandas as pd
import numpy as np
from scipy import sparse as sp
# GSGraph already implements networkx
import networkx as nx

# local imports
sys.path.insert(0,"")
from src import setup_datasets
from src.FastSinkSource.src import main as run_eval_algs
from src.FastSinkSource.src.utils import config_utils
from src.FastSinkSource.src.algorithms import alg_utils
from src.graphspace import post_to_graphspace_base as gs
from src.graphspace import gs_utils


red = "#e32727"
orange = "#eb9007"
#orange = "#d88c00"
    #'color': "#f2630a",  # orange
    #'color': "#f27d29",  # orange
green = "#15ab33"
green2 = "#21c442"
light_blue = "#9ecbf7"
light_blue2 = "#479aed"  # a bit darker blue than the node
gray = "#D8D8D8"
dark_gray = "#6b6b6b"

# I manually put these colors together
GO_term_colors = [
    #"#FFDC00",  # yellow
    "#ffca37",  # yellow2
    "#009933",  # green
    "#00cc99",  # turquoise
    "#0099cc",  # blue-green
    "#7FDBFF",  # bright blue
    "#9966ff",  # bright purple
    "#ff5050",  # light red
    "#e67300",  # burnt orange
    "#86b300",  # lime green
    "#0074D9",  # blue
    "#6699ff",  # light blue
    "#85144b",  # maroon
    ]

# names of group nodes
virus_group = 'SARS-CoV-2 Proteins'
krogan_group = 'Krogan Identified Proteins'
drug_group = 'DrugBank Drugs'
human_group = 'Human Proteins'
default_node_styles = {
    #'font-weight': "bolder",
    'color': green, 
    'shape': 'ellipse',
    'width': 35,
    'height': 35,
    #'group': human_group,
    }
default_edge_styles = {
    #'color': gray,
    'color': green2,
    'opacity': 0.6,
    }
drug_node_styles = {
    #'font-weight': "bolder",
    'color': orange,  
    'shape': 'star',
    'text-valign':'bottom',
    'width': 40,
    'height': 40,
    'group': drug_group
    }
drug_edge_styles = {
    'color': orange,
    'width': 3,
    'line-style': 'dashed',
    }
virus_node_styles = {
    'color': red,
    'shape': 'diamond',
    'width': 80,
    'height': 80,
    'background-opacity': 0.9,
    'group': virus_group,
    }
krogan_node_styles = {
    #'font-weight': "bolder",
    'color': light_blue,
    'shape': 'rectangle',
    'width': 30,
    'height': 30,
    'background-opacity': 0.9,
    #'group': krogan_group,
    }
virhost_edge_styles = {
    'color': light_blue2,
    'width': 3,
    'opacity': 0.6,
    }
node_styles = {
    'drug': drug_node_styles,
    'virus': virus_node_styles,
    'krogan': krogan_node_styles,
    'default': default_node_styles,
    }
edge_styles = {
    'drug': drug_edge_styles,
    'virhost': virhost_edge_styles,
    'default': default_edge_styles,
    }
node_style_names = {
    "drug": "DrugBank drug",
    "virus": "SARS-CoV-2 protein",
    "krogan": "Krogan-identified protein",
    "default": "Human protein"
    }
edge_style_names = {
    "drug": "DrugBank drug target",
    "virhost": "SARS-CoV-2 - Human PPI",
    "default": "Human PPI"
    }
group_node_styles = {
    "background-opacity": 0.3,
    'font-weight': "bolder",
    'font-size': "32px",
    'text-outline-width': 3,
    # remove the border around the group nodes 
    'border-width': 0,
    'text-valign': 'bottom',
    }


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
    parser = argparse.ArgumentParser(description="Script to test for enrichment of the top predictions among given genesets")

    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, required=True,
                       help="Configuration file used when running FSS. ")
    group.add_argument('--k-to-test', '-k', type=int, action="append",
                       help="k-value(s) for which to get the top-k predictions to test. " +
                       "If not specified, will check the config file. Default=100")
    group.add_argument('--mini-krogan-nodes', action="store_true",
                       help="Don't show the names of the nodes, just show the small versions of the nodes")
    group.add_argument('--virus-nodes', action="store_true",
                       help="Show only the virus nodes rather than the krogan nodes inside the virus nodes")
    #group.add_argument('--range-k-to-test', '-K', type=int, nargs=3,
    #                   help="Specify 3 integers: starting k, ending k, and step size. " +
    #                   "If not specified, will check the config file.")

    group = parser.add_argument_group('Data sources options')
    group.add_argument('--sarscov2-human-ppis', default='datasets/protein-networks/2020-03-biorxiv-krogan-sars-cov-2-human-ppi-ace2.tsv',
                       help="Table of virus and human ppis. Default: datasets/protein-networks/2020-03-biorxiv-krogan-sars-cov-2-human-ppi-ace2.tsv")
    group.add_argument('--id-mapping-file', type=str, default="datasets/mappings/human/uniprot-reviewed-status.tab.gz",
                       help="Table downloaded from UniProt to map to gene names. Expected columns: 'Entry', 'Gene names', 'Protein names'")
    group.add_argument('--enriched-terms-file', '-E', type=str, 
                       help="File containing the enriched terms, and the genes that are annotated to each. " + \
                       "Should be output by src/Enrichment/fss_enrichment.py Can also include the Krogan comparison")
    group.add_argument('--simplified-terms-file', type=str,
                       help="File containing list of terms to show")
    group.add_argument('--edge-weight-cutoff', type=float, 
                       help="Cutoff to apply to the edges to view (e.g., 900 for STRING)")
    group.add_argument('--edge-evidence-file', type=str,
                       help="File containing evidence for each edge. See XXX for the file format")

    # posting options
    group = parser.add_argument_group('GraphSpace Options')
    group.add_argument('--username', '-U', type=str, 
                      help='GraphSpace account username to post graph to. Required')
    group.add_argument('--password', '-P', type=str,
                      help='Username\'s GraphSpace account password. Required')
    #group.add_argument('', '--graph-name', type=str, metavar='STR', default='test',
    #                  help='Graph name for posting to GraphSpace. Default = "test".')
    group.add_argument('--out-pref', type=str, metavar='STR',
                      help='Prefix of name to place output files. ')
    group.add_argument('--name-postfix', type=str, default='',
                      help='Postfix of graph name to post to graphspace.')
    group.add_argument('--group', type=str,
                      help='Name of group to share the graph with.')
    group.add_argument('--make-public', action="store_true", default=False,
                      help='Option to make the uploaded graph public')
    # TODO implement and test this option
    #group.add_argument('--group-id', type=str, metavar='STR',
    #                  help='ID of the group. Could be useful to share a graph with a group that is not owned by the person posting')
    group.add_argument('--tag', type=str, action="append",
                      help='Tag to put on the graph. Can list multiple tags (for example --tag tag1 --tag tag2)')
    group.add_argument('--apply-layout', type=str,
                      help='Specify the name of a graph from which to apply a layout. Layout name specified by the --layout-name option. ' + 
                      'If left blank and the graph is being updated, it will attempt to apply the --layout-name layout.')
    group.add_argument('--layout-name', type=str, default='layout1',
                      help="Name of the layout of the graph specified by the --apply-layout option to apply. Default: 'layout1'")
    group.add_argument('--parent-nodes', action="store_true", default=False,
                      help='Use parent/group/compound nodes for the different node types')
    group.add_argument('--graph-attr-file',
                       help='File used to specify graph attributes. Tab-delimited with columns: 1: style, 2: style attribute, ' + \
                       '3: nodes/edges to which styles will be applied separated by \'|\' (edges \'-\' separated), 4th: Description of style to add to node popup.')
    #group.add_argument('--force-post', action='store_true', default=False,
    #                   help="Update graph if it already exists.")

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

    df = pd.read_csv(kwargs['simplified_terms_file'], sep='\t')
    kwargs['terms_to_highlight'] = df[df.columns[0]]
    term_names = dict(zip(df[df.columns[0]], df[df.columns[1]]))
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
        graph_attr[name].update(group_node_styles)
        graph_attr[name]['color'] = color
        # add this node as the parent to the other nodes
        for n in set(term_ann[t]) - covered_prots:
            graph_attr[n]['parent'] = name
            graph_attr[n]['color'] = color
            pred_nodes.add(n)
            covered_prots.add(n) 
            prot_per_parent_term[name].add(n) 
        # TODO add the link
        # this is used to make the popup
        attr_desc[('parent', name)] = t
    for parent, prots in prot_per_parent_term.items():
        print("%d prots for %s" % (len(prots), parent))
    # if the node list file is not specified, then set the annotated nodes as the node list
    #if not kwargs.get('node_list_file'):
    #    node_list = nodes

    krogan_attr = setup_krogan_group_nodes(virhost_edges, graph_attr=graph_attr)
    for n in krogan_attr:
        graph_attr[n].update(krogan_attr[n])
    # add an edge from each virus group node to the GO term nodes, and weight by... 
    curr_pred_nodes, pred_edges = setup_virus_to_term_edges(net_obj, virhost_edges, term_ann, term_names)
    pred_nodes.update(curr_pred_nodes)
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
        a=kwargs.get('min_edge_width',1), b=kwargs.get('max_edge_width',12))

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
        # remove any newlines from the node name if they're there
        node_labels = {n: n.replace('\n','-') for n in G.nodes(data=False)}
        # TODO write the node data as well
        G2 = nx.relabel_nodes(G, node_labels, copy=True)
        nx.write_edgelist(G2, out_file)

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


def setup_virus_to_term_edges(net_obj, virhost_edges, term_ann, term_names, **kwargs):

    W = net_obj.W
    print("converting network to networkx graph")
    G = nx.from_scipy_sparse_matrix(W)
    # now convert the edges back to prots
    G = nx.relabel_nodes(G, {i: p for i, p in enumerate(net_obj.nodes)})
    G.add_edges_from(virhost_edges)
    print("\t%d nodes and %d edges after adding %d virus edges" % (G.number_of_nodes(), G.number_of_edges(), len(virhost_edges)))
    krogan_nodes = set(h for v,h in virhost_edges)

    virus_term_num_edges = {v: defaultdict(int) for v,h in virhost_edges}
    pred_nodes = set()
    for v,h in virhost_edges:
        for t, ann in term_ann.items():
            if t not in term_names:
                continue
            t = term_names[t]
            for p in ann:
                if G.has_edge(h,p):
                    #pred_edges.add((v,t))
                    virus_term_num_edges[v][t] += 1
                    pred_nodes.add(h)
                    pred_nodes.add(p)

    # instead of adding an edge to every term, just add an edge to the terms that have at least 10% of the virus_krogan - term_prot edges
    # UPDATE: also weight the edges by the number
    perc_edges_cutoff = kwargs.get('perc_edges_cutoff', 0.05)
    pred_edges = {}
    for v, term_num_edges in virus_term_num_edges.items():
        total_num = 0
        for t, num_edges in term_num_edges.items():
            total_num += num_edges
        for t, num_edges in term_num_edges.items():
            perc_edges = num_edges / float(total_num)
            # TODO I'm adding this edge since zinc ion transport doesn't have any other edges.
            # this should be automated
            if t == "GO:0006829" or t == "zinc ion transport":
                print(f"manualling adding edge between {v}, zinc ion transport ({t}), {perc_edges}")
                pred_edges[(v,t)] = perc_edges
            if perc_edges > perc_edges_cutoff:
                pred_edges[(v,t)] = perc_edges

    print("%d edges, %d vir, %d terms" % (len(pred_edges), len(set(v for v,t in pred_edges)), len(set(t for v,t in pred_edges))))
    # TODO add styles
    return pred_nodes, pred_edges


def setup_krogan_group_nodes(virhost_edges, graph_attr=None):
    if graph_attr is None:
        graph_attr = defaultdict(dict)

    for v, h in virhost_edges:
        graph_attr[v].update(group_node_styles)
        # update to have no background color, but have a red border
        #graph_attr[v]['color'] = virus_node_styles['color']
        graph_attr[v]['color'] = "#ffffff"
        graph_attr[v]['background-opacity'] = 0.4
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
