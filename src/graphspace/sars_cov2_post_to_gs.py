
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
gray = "#6b6b6b"

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
    'width': 60,
    'height': 60,
    'background-opacity': 0.9,
    'group': virus_group,
    }
krogan_node_styles = {
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
    #group.add_argument('--range-k-to-test', '-K', type=int, nargs=3,
    #                   help="Specify 3 integers: starting k, ending k, and step size. " +
    #                   "If not specified, will check the config file.")
    group.add_argument('--node-list-file', type=str, 
                      help="File containing a list of nodes for which to post a subgraph")
    group.add_argument('--node-to-post', type=str, action="append",
                      help="UniProt ID of a taxon node for which to get neighbors. Can specify multiple")

    group = parser.add_argument_group('Data sources options')
    group.add_argument('--sarscov2-human-ppis', default='datasets/protein-networks/2020-03-biorxiv-krogan-sars-cov-2-human-ppi-ace2.tsv',
                       help="Table of virus and human ppis. Default: datasets/protein-networks/2020-03-biorxiv-krogan-sars-cov-2-human-ppi-ace2.tsv")
    group.add_argument('--id-mapping-file', type=str, default="datasets/mappings/human/uniprot-reviewed-status.tab.gz",
                       help="Table downloaded from UniProt to map to gene names. Expected columns: 'Entry', 'Gene names', 'Protein names'")
    group.add_argument('--drug-id-mapping-file', type=str, 
                       help="Table parsed from DrugBank xml with drug names and other info. " + \
                       "Will post the subnetwork of the shortest path from the top k drugs to the virus nodes")
    group.add_argument('--drug-targets-file', type=str, 
                       help="This option can be specified to add the drug-target edges to the predictions")
    group.add_argument('--drug-target-info-file', type=str, 
                       help="Table of drug target info to add to popups")
    group.add_argument('--drug-list-file', type=str, 
                      help="File containing a list of drugs for which to filter the drug targets")
    group.add_argument('--drug-targets-only', action="store_true",
                      help="Only include nodes that are targets of a drug")
    group.add_argument('--term-to-highlight', '-T', type=str, action="append",
                       help="One or more terms for which to highlight (i.e., make a parent node) in the graph")
    group.add_argument('--enriched-terms-file', '-E', type=str, 
                       help="File containing the enriched terms, and the genes that are annotated to each. " + \
                       "Should be output by src/Enrichment/fss_enrichment.py Can also include the Krogan comparison")
    group.add_argument('--edge-weight-cutoff', type=float, 
                       help="Cutoff to apply to the edges to view (e.g., 900 for STRING)")
    group.add_argument('--edge-evidence-file', type=str,
                       help="File containing evidence for each edge. See XXX for the file format")
    #/home/jeffl/src/python/graphspace/trunk/graphspace-human/post_to_new_graphspace_evidence.py
    #evidence, edge_types, edge_dir = getev.getEvidence(prededges.keys(), evidence_file=evidence_file, add_ev_to_family_edges=add_ev_to_family_edges)
    # edge_popup = getEdgeAnnotation(u,v, evidence, k=k_value

    group = parser.add_argument_group('FastSinkSource Pipeline Options')
    group.add_argument('--alg', '-A', dest='algs', type=str, action="append",
                       help="Algorithms for which to get results. Must be in the config file. " +
                       "If not specified, will get the list of algs with 'should_run' set to True in the config file")
    group.add_argument('--num-reps', type=int, 
                       help="Number of times negative sampling was repeated to compute the average scores. Default=1")
    group.add_argument('--sample-neg-examples-factor', type=float, 
                       help="Factor/ratio of negatives to positives used when making predictions. " +
                       "Not used for methods which use only positive examples.")

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
    algs = config_utils.get_algs_to_run(alg_settings, **kwargs)
    del kwargs['algs']

    node_list = [] 
    if kwargs.get('node_list_file'):
        print("Reading %s" % (kwargs['node_list_file']))
        node_list = set(pd.read_csv(kwargs['node_list_file'], sep='\t', comment='#', header=None, squeeze=True).tolist())
    if kwargs.get('drug_list_file'):
        print("Reading %s" % (kwargs['drug_list_file']))
        drug_list = set(pd.read_csv(kwargs['drug_list_file'], sep='\t', comment='#', header=None, usecols=[0], squeeze=True).tolist())
        #node_list |= drug_list
    if kwargs.get('node_to_post'):
        node_list |= set(kwargs['node_to_post'])

    # this dictionary will hold all the styles 
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
    drug_nodes = None
    if kwargs.get('drug_id_mapping_file'):
        print("Reading %s" % (kwargs['drug_id_mapping_file']))
        df = pd.read_csv(kwargs['drug_id_mapping_file'], sep='\t', header=0) 
        # just add the drug name mapping to the mapping dictionary already in place
        uniprot_to_gene.update({d: name for d, name in zip(df['drugbank_id'], df['name'].astype(str))})
        for d, name in uniprot_to_gene.items():
            new_name = fix_drug_name(name)
            uniprot_to_gene[d] = new_name
        # now get extra drug info
        #uniprot_to_prot_names.update({d: group_nodes for d, group_nodes in zip(df['drugbank_id'], df['group_nodes'].astype(str))})
        drug_nodes = set(list(df['drugbank_id'].values))
    drugG = None 
    if kwargs.get('drug_targets_file'):
        print("Reading %s" % (kwargs['drug_targets_file']))
        df = pd.read_csv(kwargs['drug_targets_file'], sep='\t', header=None) 
        drugG = nx.from_pandas_edgelist(df, source=0, target=1)
    if kwargs.get('drug_target_info_file'):
        print("Reading %s" % (kwargs['drug_target_info_file']))
        df = pd.read_csv(kwargs['drug_target_info_file'], sep='\t') 
        # also get the pmid and references from the table to add as popup info
        drug_target_pmids = {}
        drug_target_action = {}
        pmid_citations = {}
        for d, p, action, pmids, citations in df[['drugbank_id', 'uniprot_id', 'actions', 'pubmed_ids', 'citations']].values:
            drug_target_action[(d,p)] = action
            if pd.isnull(pmids):
                continue
            drug_target_pmids[(d,p)] = str(pmids).split('|')
            pmid_citations.update(dict(zip(str(pmids).split('|'), str(citations).split('|'))))
    if kwargs.get('enriched_terms_file'):
        print("Reading %s" % (kwargs['enriched_terms_file']))
        df = pd.read_csv(kwargs['enriched_terms_file'], header=[0,1,2], index_col=0) 
        df2 = df.copy()
        print(df.head())
        # get the prots per term
        df2.columns = df2.columns.droplevel([0,1])
        df2 = df2['geneID']
        df2.columns = list(range(len(df2.columns)))
        for i in range(1,len(df2.columns)):
            df2[0] = df2[0] + '/' + df2[i]
        print(df2.head())
        #print(df2['geneID'].head())
        term_ann = dict(zip(df2.index, df2[0].values))
        term_ann = {t: str(ann).split('/') for t,ann in term_ann.items()}
        print("\t%d terms, %s" % (
            len(term_ann),
            ", ".join("%s: %d ann" % (t, len(term_ann[t])) for t in kwargs['term_to_highlight']))) 
        # also get the term name, and the enrichment p-value(?)
        term_names = dict(zip(df.index, df[('Description', 'Unnamed: 1_level_1', 'Unnamed: 1_level_2')]))
        nodes = set()
        # setup their graph_attributes for posting
        # reverse so that if a gene is annotated to multiple terms, then the first term gets priority of parent
        for i, t in enumerate(kwargs['term_to_highlight'][::-1]):
            name = term_names[t]
            color = GO_term_colors[i]
            graph_attr[name]['color'] = color
            # add this node as the parent to the other nodes
            for n in term_ann[t]:
                graph_attr[n]['parent'] = name
                graph_attr[n]['color'] = color
                nodes.add(n)
            # TODO add the link
            # this is used to make the popup
            attr_desc[('parent', name)] = t
        # if the node list file is not specified, then set the annotated nodes as the node list
        if not kwargs.get('node_list_file'):
            node_list = nodes
    if kwargs.get('drug_targets_only'):
        new_node_list = set([n for n in node_list if drugG.has_node(n)])
        if len(new_node_list) > 2:
            node_list = new_node_list

    # load human-virus ppis
    df = pd.read_csv(kwargs['sarscov2_human_ppis'], sep='\t')
    edges = zip(df[df.columns[0]], df[df.columns[1]])
    edges = [(v.replace("SARS-CoV2 ",""), h) for v,h in edges]
    virus_nodes = [v for v,h in edges]
    krogan_nodes = [h for v,h in edges]
    virhost_edges = edges 

#    genesets_to_test = config_map.get('genesets_to_test')
#    if genesets_to_test is None or len(genesets_to_test) == 0:
#        print("ERROR: no genesets specified to test for overlap. " +
#              "Please add them under 'genesets_to_test'. \nQuitting") 
#        sys.exit()
#
#    # first load the gene sets
#    # TODO use these
#    geneset_group_nodes = {}
#    for geneset_to_test in genesets_to_test:
#        name = geneset_to_test['name'] 
#        gmt_file = "%s/genesets/%s/%s" % (
#            input_dir, name, geneset_to_test['gmt_file'])
#        if not os.path.isfile(gmt_file):
#            print("WARNING: %s not found. skipping" % (gmt_file))
#            sys.exit()
#
#        geneset_group_nodes[name] = setup_datasets.parse_gmt_file(gmt_file)  

    # for each dataset, extract the path(s) to the prediction files,
    # read in the predictions, and test for the statistical significance of overlap 
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
        prots = net_obj.nodes
        print("\t%d total prots" % (len(prots)))
        # TODO using this for the SARS-CoV-2 project,
        # but this should really be a general purpose script
        # and to work on any number of terms 
        orig_pos_idx, _ = alg_utils.get_term_pos_neg(ann_obj.ann_matrix, 0)
        orig_pos = [ann_obj.prots[p] for p in orig_pos_idx]
        print("pos & krogan: %d; pos - krogan: %d; krogan - pos: %d" % (
            len(set(orig_pos) & set(krogan_nodes)),
            len(set(orig_pos) - set(krogan_nodes)),
            len(set(krogan_nodes) - set(orig_pos))))
        print("\t%d original positive examples" % (len(orig_pos)))
        #pos_neg_file = "%s/%s" % (input_dir, dataset['pos_neg_file'])
        #df = pd.read_csv(pos_neg_file, sep='\t')
        #orig_pos = df[df['2020-03-sarscov2-human-ppi'] == 1]['prots']
        #print("\t%d original positive examples" % (len(orig_pos)))

        # now load the predictions, test at the various k values, and TODO plot
        k_to_test = get_k_to_test(dataset, **kwargs)
        print("\tposting %d k values: %s" % (len(k_to_test), ", ".join([str(k) for k in k_to_test])))

        # now load the prediction scores
        alg_pred_files = config_utils.get_dataset_alg_prediction_files(
            output_dir, dataset, alg_settings, algs, use_alg_name=False, **kwargs)
        for alg, pred_file in alg_pred_files.items():
            if not os.path.isfile(pred_file):
                print("Warning: %s not found. skipping" % (pred_file))
                continue
            print("reading: %s" % (pred_file))
            df = pd.read_csv(pred_file, sep='\t')
            scores = dict(zip(df['prot'], df['score']))
            # remove the original positives
            df = df[~df['prot'].isin(orig_pos)]
            df.reset_index(inplace=True, drop=True)
            df = df[['prot', 'score']]
            curr_scores = dict(zip(df['prot'], df['score']))
            df.sort_values(by='score', ascending=False, inplace=True)
            #print(df.head())
            if len(node_list) == 0:
                node_list = set(list(df[:k_to_test[0]]['prot']))
            pred_nodes = node_list
            if kwargs.get('paths_to_virus'):
                pred_nodes = get_paths_to_virus_nodes(node_list, net_obj, virhost_edges, **kwargs)
            node_types = {} 
            # if the drug nodes were part of the network, then get the top predicted drugs
            # from the prediction scores
            if drug_nodes is not None and drugG is None:
                if len(node_list) > 0:
                    top_k_drug_nodes = node_list
                elif kwargs.get('paths_to_virus'):
                    top_k_drug_nodes = list(df[df['prot'].isin(drug_nodes)][:k_to_test[0]]['prot'])
                pred_nodes = get_paths_to_virus_nodes(top_k_drug_nodes, net_obj, virhost_edges, **kwargs)
                node_types = {d: 'drug' for d in drug_nodes}

            pred_nodes -= (set(virus_nodes) | set(krogan_nodes))
            all_nodes = set(pred_nodes) | set(krogan_nodes)

            # build the network to post
            pred_edges, graph_attr, attr_desc2, node_type_rank = build_subgraph(
                alg, pred_nodes, curr_scores, all_nodes,
                net_obj, graph_attr, node_types,
                min_edge_width=2 if not kwargs.get('edge_weight_cutoff') else 3,
                max_edge_width=8 if not kwargs.get('edge_weight_cutoff') else 6,
                **kwargs)
            attr_desc.update(attr_desc2)

            # add the node rank to the name of the node
            for n, rank in node_type_rank.items():
                uniprot_to_gene[n] += "\n%s"%rank 

            # now also add the virus edges for the human prots that interact with predicted prots
            net_nodes = set([n for e in pred_edges for n in e])
            # add the drugs that target the predicted nodes, if specified
            if drugG is not None:
                drugs_skipped = set()
                before = len(pred_edges)
                if len(node_list) > 0:
                    drugs_with_target = [n for n in node_list if drugG.has_node(n)]
                else:
                    drugs_with_target = [n for n in net_nodes if drugG.has_node(n)]
                for n in drugs_with_target:
                    for d in drugG.neighbors(n):
                        if drugG.degree[d] >= kwargs.get('degree_cutoff',1000):
                            drugs_skipped.add(d)
                        elif kwargs.get('drug_list_file') and d not in drug_list:
                            continue
                        else:
                            pred_edges.add((d,n))
                            graph_attr[d].update(drug_node_styles)
                            graph_attr[(d,n)].update(drug_edge_styles) 
                            # also add a popup with the # targets
                            attr_desc[d]['# targets'] = drugG.degree[d]
                print("\tadded %d drug-target edges" % (len(pred_edges) - before))
                if len(drugs_skipped) > 0:
                    print("\t%d drug-target edges skipped from drug with > %s targets: %s" % (len(drugs_skipped), kwargs.get('degree_cutoff',100), ', '.join(drugs_skipped)))
            pred_edges.update(set([(v,h) for v,h in virhost_edges if h in net_nodes]))
            net_nodes = set([n for e in pred_edges for n in e])
            print("\t%d edges, %d nodes" % (len(pred_edges), len(net_nodes)))

            # add the styles for the virus and krogan nodes
            for n in net_nodes:
                if n in virus_nodes:
                    # styles for the virus nodes
                    graph_attr[n] = virus_node_styles 
                    #graph_attr[n]['group'] = virus_group
                elif n in krogan_nodes:
                    # styles for the human nodes
                    graph_attr[n].update(krogan_node_styles)
                    #graph_attr[n]['group'] = 
                #elif drug_nodes is not None and n in drug_nodes:
                #    graph_attr[n]['group'] = "DrugBank Drugs"
                #else: 
                #    graph_attr[n]['group'] = "Human Proteins"
            for e in virhost_edges:
                graph_attr[e] = virhost_edge_styles

            evidence=None
            if kwargs.get('edge_evidence_file'):
                evidence, _,_ = gs_utils.getEvidence(pred_edges, evidence_file=kwargs['edge_evidence_file'])
            if kwargs.get('drug_target_info_file'):
                evidence = defaultdict(dict) if evidence is None else evidence
                for (drug, target), pmids in drug_target_pmids.items():
                    references = [{'pmid': pmid, 'text': pmid_citations[pmid]} for pmid in pmids]
                    evidence[(drug, target)]['DrugBank'] = references

            # Now post to graphspace!
            print("Building GraphSpace graph")
            popups = {}
            for i, n in enumerate(net_nodes):
                if n in virus_nodes:
                    continue
                if node_desc is not None and n in node_desc:
                    attr_desc[n].update(node_desc[n])
                node_type = 'drugbank' if drug_nodes and n in drug_nodes else 'uniprot'
                popups[n] = gs.buildNodePopup(n, node_type=node_type, attr_val=attr_desc)
            for u,v in pred_edges:
                popups[(u,v)] = gs.buildEdgePopup(u,v, node_labels=uniprot_to_gene, attr_val=attr_desc, evidence=evidence)
            G = gs.constructGraph(pred_edges, node_labels=uniprot_to_gene, graph_attr=graph_attr, popups=popups)
            
            # set of group nodes to add to the graph
            if kwargs.get('parent_nodes'):
                #group_nodes_to_add = [virus_group, krogan_group, drug_group, human_group]
                group_nodes_to_add = set(attr['parent'] for n, attr in graph_attr.items() if 'parent' in attr) 
                add_group_nodes(G, group_nodes_to_add, graph_attr, attr_desc)

            # TODO add an option to build the 'graph information' tab legend/info
            # build the 'Graph Information' metadata
            #desc = gs.buildGraphDescription(opts.edges, opts.net)
            desc = ''
            metadata = {'description':desc,'tags':kwargs.get('tags',[]), 'title':''}
            G.set_data(metadata)
            if 'graph_exp_name' in dataset:
                graph_exp_name = dataset['graph_exp_name']
            else:
                #graph_exp_name = config_utils.get_dataset_name(dataset) 
                graph_exp_name = "%s-%s" % (dataset['exp_name'].split('/')[-1], dataset['net_version'].split('/')[-1])
            graph_name = "%s-%s-k%s%s" % (
                alg, graph_exp_name, k_to_test[0], kwargs.get('name_postfix',''))
                #"test","", "")

            if kwargs.get('term_to_highlight'):
                graph_name += "-%s-%s" % (kwargs['term_to_highlight'][0], term_names[kwargs['term_to_highlight'][0]].replace(' ','-')[:25])
            if kwargs.get('node_to_post') is not None:
                graph_name += '-'.join(kwargs['node_to_post'])
            G.set_name(graph_name)
            # also set the legend
            G = set_legend(G)
            # write the posted network to a file if specified
            if kwargs.get('out_pref'):
                out_file = "%s%s.txt" % (kwargs['out_pref'], graph_name)
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                print("writing network to %s" % (out_file))
                # remove any newlines from the node name if they're there
                node_labels = {n: n.replace('\n','-') for n in G.nodes(data=False)}
                # TODO write the node data as well
                G2 = nx.relabel_nodes(G, node_labels, copy=True)
                nx.write_edgelist(G2, out_file)

            gs.post_graph_to_graphspace(
                    G, kwargs['username'], kwargs['password'], graph_name, 
                    apply_layout=kwargs['apply_layout'], layout_name=kwargs['layout_name'],
                    group=kwargs['group'], make_public=kwargs['make_public'])


def get_paths_to_virus_nodes(top_k_drug_nodes, net_obj, virhost_edges, **kwargs):
    W = net_obj.W
    print("converting network to networkx graph")
    G = nx.from_scipy_sparse_matrix(W)
    # now convert the edges back to prots
    G = nx.relabel_nodes(G, {i: p for i, p in enumerate(net_obj.nodes)})
    G.add_edges_from(virhost_edges)
    print("\t%d nodes and %d edges after adding %d virus edges" % (G.number_of_nodes(), G.number_of_edges(), len(virhost_edges)))
    krogan_nodes = set(h for v,h in virhost_edges)
    # remove the edges from drugs to krogan nodes if they are there
    # since those weren't used during network propagation
    # UPDATE: I can control that with the drug network passed in
    #for d in top_k_drug_nodes:
    #    for n in krogan_nodes:
    #        if G.has_edge(d,n):
    #            G.remove_edge(d,n)
    print("\tfinding the shortest paths from %d nodes to the virus nodes" % (len(top_k_drug_nodes)))
    #print(top_k_drug_nodes)
    no_path = set()
    nodes_on_paths = set()
    for d in tqdm(top_k_drug_nodes):
        shortest_paths = defaultdict(set)
        for v in set(v for v,h in virhost_edges):
            try:
                shortest_path = nx.shortest_path(G, source=d, target=v)
            except nx.exception.NetworkXNoPath:
                shortest_path = []
                no_path.add(d)
            sp_len = len(shortest_path)-1
            shortest_paths[sp_len].update(set(shortest_path))
        min_len = min(shortest_paths.keys())
        nodes_on_paths.update(shortest_paths[min_len])
    print("\t%d total nodes on the shortest paths" % (len(nodes_on_paths)))
    if len(no_path) > 0:
        print("\t%d nodes with no path to a virus node" % (len(no_path)))
    return nodes_on_paths


def build_subgraph(
        alg, predicted_nodes, scores, all_nodes,
        net_obj, graph_attr, node_types, **kwargs):

    W = net_obj.W
    prots, node2idx = net_obj.nodes, net_obj.node2idx
    ranks = {p: i+1 for i,p in enumerate(sorted(scores, key=scores.get, reverse=True))}
#            # now add the virus edges
#            prededges = list(subG.edges()) + virhost_edges
    # UPDATE: get a different rank for each node
    node_type_rank = ranks.copy()
    nodes_per_type = defaultdict(set)
    for n,t in node_types.items():
        if n in scores:
            nodes_per_type[t].add(n) 
    for t, nodes in nodes_per_type.items():
        for i,n in enumerate(sorted(nodes, key=scores.get, reverse=True)):
            node_type_rank[n] = i+1

    prededges = set()
    # get the edges in the induced subgraph 
    #get_subnet()
    for u in predicted_nodes:
        if u not in node2idx:
            continue
        neighbors = set([prots[v] for v in get_mat_neighbors(W, node2idx[u])])
        for v in neighbors:
            if v in all_nodes and v != u:
                prededges.add(tuple(sorted((u,v))))

    # set the default styles
    for n in predicted_nodes:
        graph_attr[n].update(default_node_styles.copy())
        if node_types is not None and n in node_types:
            graph_attr[n].update(node_styles[node_types[n]])
    for u,v in prededges:
        graph_attr[(u,v)] = default_edge_styles.copy()
        if node_types is not None:
            if u in node_types:
                graph_attr[(u,v)].update(edge_styles[node_types[u]]) 
            if v in node_types:
                graph_attr[(u,v)].update(edge_styles[node_types[v]]) 

    # this is used for the popups
    attr_desc = defaultdict(dict)

    # and set their styles
    for n in predicted_nodes:
        # maybe put the labels below the nodes?
        # helps with visualizing the background opacity
        graph_attr[n]['text-valign'] = 'bottom'
        # add the strain name to the popup
        #attr_desc[n]['Strain'] = species_names[curr_taxon]
        if n in predicted_nodes:
            # UPDATE: use the node rank instead of the node score
            #graph_attr[n]['background-opacity'] = pred_local_conf[n]
            if n not in ranks:
                graph_attr[n]['background-opacity'] = scores.get(n,1)
            else:
                #graph_attr[n]['background-opacity'] = scores[n]
                #graph_attr[n]['background-opacity'] = max([0.9 - (ranks[n] / 1000.0), float(scores[n])])
                graph_attr[n]['background-opacity'] = 0.9
                if n in node_types:
                    attr_desc[n]["%s %s rank"%(node_types[n], alg)] = node_type_rank[n]
                    attr_desc[n]["%s overall rank"%(alg)] = ranks[n]
                else:
                    attr_desc[n]["%s rank"%(alg)] = ranks[n]
            if n not in scores:
                print("WARNING: no score for node %s" % (n))
                continue
            attr_desc[n]["%s prediction score"%(alg)] = "%0.4f" % (scores[n])

    prednodes = set([n for e in prededges for n in e])
    print("%d nodes, %d nodes, %d edges so far" % (len(all_nodes), len(prednodes), len(prededges)))

    # set the width of the edges by the network weight
    edge_weights = defaultdict(float)
    for u,v in tqdm(prededges):
        e = (u,v)
        w = W[node2idx[u]][:,node2idx[v]].A.flatten()[0]
        edge_weights[e] = w
        attr_desc[e]["Edge weight"] = "%0.1f" % (w)
        # make the edges somewhat opaque for a better visual style
        #graph_attr[e]['opacity'] = 0.6

    # set the width of the edges by the network weight
    #edge_weights = {(u,v): float(W[node2idx[u]][:,node2idx[v]].A.flatten()[0]) for u,v in prededges}
    #for e,w in edge_weights.items():
    #    attr_desc[e]["Final edge weight"] = "%0.1f" % (w)
    # TODO set the min and max as parameters or something
    #max_weight = 180 
    #if net_obj.multi_net:
    #    # the swsn weights give a scalar for each network
    #    max_weight = net_obj.swsn_weights[0]*180
    #    min_weight = net_obj.swsn_weights[0]
    #    print(max_weight)
    #else:
    max_weight = max(W.data) 
    min_weight = min(W.data)
    for e in edge_weights:
        if edge_weights[e] > max_weight:
            edge_weights[e] = max_weight 
    if min_weight != max_weight:
        graph_attr = gs.set_edge_width(
            prededges, edge_weights, graph_attr,
            a=kwargs.get('min_edge_width',2), b=kwargs.get('max_edge_width',12),
            min_weight=min_weight, max_weight=max_weight)

    return prededges, graph_attr, attr_desc, node_type_rank


def get_k_to_test(dataset, **kwargs):
    k_to_test = dataset['k_to_test'] if 'k_to_test' in dataset else kwargs.get('k_to_test', [])
    range_k_to_test = dataset['range_k_to_test'] if 'range_k_to_test' in dataset \
                        else kwargs.get('range_k_to_test')
    if range_k_to_test is not None:
        k_to_test += list(range(
            range_k_to_test[0], range_k_to_test[1], range_k_to_test[2]))
    # if nothing was set, use the default value
    if k_to_test is None or len(k_to_test) == 0:
        k_to_test = [100]
    return k_to_test


def get_mat_neighbors(A, n):
    neighbors = A[n].nonzero()[1]
    return neighbors


#def get_subnet():
#            # get all edges between these nodes
#            print("converting network to networkx graph")
#            netG = nx.from_scipy_sparse_matrix(net_obj.W)
#            # get the subgraph of these nodes as the network 
#            # convert the nodes to ids
#            all_nodes_idx = [node2idx[n] for n in all_nodes if n in node2idx]
#            subG = nx.subgraph(netG, all_nodes_idx)
#            # now convert the edges back to prots
#            subG = nx.relabel_nodes(subG, {i: p for i, p in enumerate(prots)})


def set_legend(G):
    Ld = GSLegend()
    curr_node_styles = {}
    for node_type, styles in node_styles.items():
        styles["background-color"] = styles['color']
        #node_type = node_type[0].upper()+node_type[1:]
        node_type = node_style_names.get(node_type, node_type)
        curr_node_styles[node_type] = styles 
    curr_edge_styles = {}
    for edge_type, styles in edge_styles.items():
        styles["line-color"] = styles['color']
        edge_type = edge_style_names.get(edge_type, edge_type)
        curr_edge_styles[edge_type] = styles 
    legend_json = {
        "legend":{
            "nodes": curr_node_styles,
            "edges": curr_edge_styles,
            }
        }
    # Adding JSON representation of legend
    Ld.set_legend_json(legend_json)
    # Adding the legend 'Ld' to the graph 'G'.
    G.set_legend(Ld)
    return G


def fix_drug_name(name, split_over_lines=25):
    # remove the special characters from the name, since graphspace can't handle those apgrouply
    name = name.replace("'","")
    if len(name) > 50:
        name = name[:15] + '[...]'
        return name
    # split the name to two lines if its too long
    if split_over_lines is not None and len(name) > split_over_lines:
        if " " in name:
            sep = " "
        elif "-" in name:
            sep = "-"
        else:
            return name
        split_name = name.split(sep)
        combined_name = ""
        curr_combined = ""
        for i, sub_name in enumerate(split_name):
            curr_combined += sub_name
            if i < len(split_name)-1:
                curr_combined += sep
            if len(curr_combined) > split_over_lines and i < len(split_name)-1:
                combined_name += curr_combined 
                combined_name += "\n"
                curr_combined = ""
        combined_name += curr_combined
        #print(name)
        #print(combined_name)
        name = combined_name
    return name


def add_group_nodes(G, group_nodes_to_add, graph_attr, attr_desc):
    for group in group_nodes_to_add:
        # TODO add a popup for the group which would be a description of the function or pathway
        if ('parent', group) in attr_desc:
            popup = attr_desc[('parent', group)]
        else:
            popup = group

        if group in graph_attr and 'label' in graph_attr[group]:
            label = graph_attr[group]['label']
        else:
            label = group.replace("_", " ")

        #popup = group
        # leave off the label for now because it is written under the edges and is difficult to see in many cases
        # I requested this feature on the cytoscapejs github repo: https://github.com/cytoscape/cytoscape.js/issues/1900
        #G.add_node(group, popup=popup, k=k_value, label=label)
        G.add_node(group, popup=popup, label=label)

        # TODO streamline this better
        color = None
        for g2, styles in [
                (virus_group, virus_node_styles),
                (krogan_group, krogan_node_styles),
                (drug_group, drug_node_styles),
                (human_group, default_node_styles)]:
            if group == g2:
                color = styles['color']
        attr_dict = {} 
        color = color if color else graph_attr[group].get('color') 
        # set the background opacity so the background is not the same color as the nodes
        attr_dict["background-opacity"] = 0.3
        attr_dict['font-weight'] = "bolder"
        attr_dict['font-size'] = "32px"
        attr_dict['text-outline-width'] = 3
        # remove the border around the group nodes 
        border_width = 0
        valign="bottom"

        G.add_node_style(group, attr_dict=attr_dict, color=color, border_width=border_width, bubble=color, valign=valign)
    return G


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)
