
# code to post toxcast results to graphspace

import argparse
import os
import sys
import json
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from src.FastSinkSource.src.utils import file_utils as utils
#from graphspace_python.api.client import GraphSpace
from graphspace_python.graphs.classes.gsgraph import GSGraph

#import post_to_new_graphspace_evidence as post_to_gs
from src.graphspace import post_to_graphspace_base as gs_base
from graphspace_python.graphs.classes.gslegend import GSLegend
from src.graphspace import gs_utils
from datetime import datetime

# Dictionaries of node and edge properties
NODE_COLORS = {
        'target' : '#FFFF60',  # yellow
        'source' : '#056e05', #green
        'top_target': '#3349ff',  # blue
        'default' : '#D8D8D8',  # gray
        'intermediate_rec' : '#D8D8D8',  # gray
        'intermediate_tf' : '#D8D8D8',  # gray
        # for the parent/compound nodes
        'Targets' : '#FFFF60',  # yellow
        'Sources' : '#3349ff',  # blue
        'Top_targets': '#FF3933', #red
        'Intermediate Proteins' : '#D8D8D8',  # gray
        'No Enriched Terms': '#D8D8D8',  # gray
        'Intermediate Receptors' : '#D8D8D8',  # gray
        'Intermediate TFs' : '#D8D8D8',  # gray

}

NODE_SHAPES = {
        'source'           : 'diamond',
        'top_target'       : 'triangle',
        'target'           : 'rectangle',
        # 'default'          : 'ellipse',
        # 'intermediate_rec' : 'triangle',
        # 'intermediate_tf'  : 'rectangle',
}

EDGE_DIR_MAP = {
        'physical':False,
        'phosphorylation':True,
        'enzymatic':True,
        'activation':True,
        'inhibition':True,
        'spike_regulation':True,
        'dir_predicted': True,
        'signaling' : True,
}

EDGE_COLORS = {
        'physical': '#27AF47',  # green
        'phosphorylation': '#F07406',  # orange
        'enzymatic': 'brown',
        #'enzymatic': '#DD4B4B',  # red
        'activation': 'grey',
        'inhibition': 'grey',
        'spike_regulation': 'red',
        'dir_predicted': '#2A69DC',
        'signaling' : '#34bdeb',
}


# order to use when deciding which color an edge will get
edge_type_order = ['phosphorylation', 'enzymatic', 'spike_regulation', 'activation', 'inhibition', 'dir_predicted', 'physical']

# define a set of colors to pull from
# TODO what if there are more than 16 pathways?
# here's a great set of colors: http://clrs.cc/
COLORS = [
    # organge,  pink, deep-maroon,  light-purple,  light-green,,  lime-green?, turquoise, aqua,
   "#eb7f21",   "#d92bc5",   "#8c1f5b",     "#e3b8f5",  "#4bfa4b",       "#00cc99", "#0099cc", "#7FDBFF",
    # orange-red, orange, light-green, yellow, blue,
    "#ff5050", "#86b300", "#FFDC00", "#0074D9",
    # blue2,    orange2, light-green2, light_purple, dark green
    "#6699ff", "#f79760",  "#4efc77", "#eb46e2", "#34a352",
    # LIGHT purple, DARK purple, dark brown, dark blue, salmon, orange
    "#cc99ff", "#6600cc", "#996633", "#006699", "#ffcccc", "#e68a00",
]


BORDER_COLORS={
    "#eb7f21":"#592d07"   ,
    "#d92bc5": "#3b0d35",
    "#8c1f5b": "#211c1f" ,
    "#e3b8f5": "#421157"  ,
    "#4bfa4b":"#0c360c",
    "#00cc99":"#034d3a",
}
# for some of the terms that are common for multiple chemicals, give them the same color
FIXED_TERM_COLORS = {
    "GO:0042493": "#85144b",  # response to drug
    "GO:0043066": "#f79760",  # neg. reg. apoptotic process
    "GO:0016032": "#f7608b",  # viral process
    "GO:0038095": "#6699ff",  # Fc-epsilon receptor\nsignaling pathway
    "GO:0030168": "#0099cc",  # platelt activation
    # make this term red because it makes sense
    "GO:1900034": "#ff5050",  # regulation of cellular response to heat
    }

node_legend_dict={}

def init_node_legend_dict():
    global node_legend_dict
    node_legend_dict = {'source': {'background-color': NODE_COLORS['source'], 'shape': NODE_SHAPES['source']},
                        'target': {'background-color': NODE_COLORS['target'], 'shape': NODE_SHAPES['target']},
                        'top_target': {'background-color': NODE_COLORS['top_target'], 'shape': NODE_SHAPES['top_target']}}

def write_revigo_color_files(selected_terms, term_names, term_prots, term_pvals,out_file):
    """
    selected_terms: List of GO terms
    term_names: dict: (key->term_ID, value-> term_name)
    term_prots: dict: (key-> term_ID, value->prots annotated to that term in a string format where prots are separated by '/')
    term_pvals: dict: (key->term_ID, value-> pval)

    out_file = term_color_file : the term colors will be saved in out_file
    new_func_colors: dict of dict. outer key=term_ID, inner keys=['prots','color','link','name'] for each term.
    """

    # assign a color to each selected term

    term_popups = {}
    link_template = "<a style=\"color:blue\" href=\"https://www.ebi.ac.uk/QuickGO/GTerm?id=%s\" target=\"DB\">%s</a>"
    for term in selected_terms:
        term_link = link_template % (term, term)
        popup = "<b>QuickGO</b>: %s" % (term_link)
        popup += "<br><b>p-value</b>: %0.2e" % (float(term_pvals[term]))
        term_popups[term] = popup

    function_colors = write_colors_file(out_file, selected_terms, term_names, term_prots, term_popups)
    term_color_file = out_file

    new_func_colors = defaultdict(dict)
    for term in function_colors:
        new_func_colors[term]['prots'] = term_prots[term]
        new_func_colors[term]['color'] = function_colors[term]
        new_func_colors[term]['link'] = "https://www.ebi.ac.uk/QuickGO/GTerm?id=%s" % (term)
        new_func_colors[term]['name'] = term_names[term]
        # if uid in pathway_colors[pathway]['prots']:
        #     pathway_link = '<a style="color:%s" href="%s">%s</a>' % (pathway_colors[pathway]['color'], pathway_colors[pathway]['link'], pathway)
    return term_color_file, new_func_colors


def write_colors_file(out_file, functions, function_names, function_prots, function_popups=None, colors=None):
    # first, shorten the funciton names and make them wrap over two lines
    function_names = shorten_names(function_names)
    if colors is None:
        colors = COLORS
    if len(functions) > len(colors):
        print("\tWarning: # functions %d exceeds # colors %d. Limiting functions to the %d colors" % (len(functions), len(colors), len(colors)))
        prots_used = set()
        new_funcs = []
        #for f in sorted(function_prots, key=lambda f: len(function_prots[f]), reverse=True):
        for f in functions:
            f_prots = set(function_prots[f].split('/'))
            # only keep functions if they have some unique annotations
            if len(f_prots - prots_used) < 2:
                continue
            else:
                new_funcs.append(f)
                prots_used.update(f_prots)
        functions = new_funcs
        if len(new_funcs) > len(colors):
            # limit the pathways to the number of colors
            functions = new_funcs[:len(colors)]

    function_colors = {}
    available_colors = set(colors)
    for function in functions:
        if function in FIXED_TERM_COLORS:
            function_colors[function] = FIXED_TERM_COLORS[function]
            available_colors.discard(function_colors[function])
    available_colors = [colors[i] for i, color in enumerate(colors) if color in available_colors]
    for i, function in enumerate(sorted(set(functions) - set(function_colors.keys()))):
        function_colors[function] = available_colors[i]

    # if no popup was provided, just show an empty popup
    if function_popups is None:
        function_popups = {function:"" for function in functions}

    print("\tWriting graphspace colors file: %s" % (out_file))
    ## UPDATE 2017-07-13: create a compound or "parent" node for each goterm where the name of the parent node is the goterm name
    ## TODO The 4th column is the description/popup of the parent node
    # write the prots along with a color for posting to graphspace
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as out:
        # first write the style of the individual nodes
        #out.write("#style\tstyle_val\tprots\tdescription\n")
        out.write('\n'.join(['\t'.join(['color', function_colors[function], function_prots[function], '-']) for function in functions])+'\n')
        # out.write('\n'.join(['\t'.join(['border_color', BORDER_COLORS[function_colors[function]], function_prots[function], '-']) for function in functions])+'\n')
        # out.write('\n'.join(['\t'.join(['border_style', 'solid', function_prots[function], '-']) for function in functions])+'\n')
        # out.write('\n'.join(['\t'.join(['border_width', '2px', function_prots[function], '-']) for function in functions])+'\n')

        # then write the parent nodes
        #out.write("#parent\tfunction\tprots\tpopup\n")
        # for function in functions:
        #     out.write('\t'.join(['parent', function_names[function], function_prots[function], function_popups[function]])+'\n')
        out.write('\n'.join(['\t'.join(['parent', function_names[function], function_prots[function], '-']) for function in functions])+'\n')

        # then write the style of the parent node
        #out.write("#color\tcolor_val\tparent_node\tdescription\n")
        out.write('\n'.join(['\t'.join(['color', function_colors[function], function_names[function], '-']) for function in functions])+'\n')
    #save function color legends
    global node_legend_dict
    for function in function_colors:
        node_legend_dict[function_names[function]] = {'background-color': function_colors[function], 'shape': 'ellipse'}

    return function_colors


def shorten_names(function_names):
    new_func_names = {}
    for term, name in function_names.items():
        if len(name) > 20:
            split_name = name.split(' ')
            curr_len = 0
            curr_name = ''
            for i, word in enumerate(split_name):
                curr_name += word
                curr_len += len(word) + 1
                if curr_len > 20 and i < len(split_name)-1:
                    curr_name += '\\n'
                    curr_len = 0
                else:
                    curr_name += ' '
                if len(curr_name) > 60:
                    break
            name = curr_name
        new_func_names[term] = name
    return new_func_names


def call_post_to_graphspace(prededges, sources, targets, top_targets, uniprot_to_gene,uniprot_to_protein_names,
                            selected_terms, term_names, term_prots, term_pvals,out_file, **kwargs):

    init_node_legend_dict() #fresh node_legend_dict to remove any lingering legend

    term_color_file, function_colors = write_revigo_color_files(selected_terms, term_names,
                                        term_prots, term_pvals, out_file)
    kwargs['function_colors'] = function_colors

    #TODO plot only the nodes that are in some GO terms
    build_graph_and_post(
        prededges, sources, targets, top_targets, uniprot_to_gene, uniprot_to_protein_names,
        graph_attr_file=term_color_file,
        ev_file=kwargs['evidence_file'],
        **kwargs)


def build_graph_and_post(prededges, sources, targets,top_targets, uniprot_to_gene,uniprot_to_protein_names,
        graph_attr_file=None, ev_file=None, **kwargs):

    graph_name=  kwargs.get('graph_name')

    # get attributes of nodes and edges from the graph_attr file
    graph_attr = {}
    # description of a style, style_attr tuple
    attr_desc = {}
    if graph_attr_file is not None:
        graph_attr, attr_desc = gs_base.readGraphAttr(graph_attr_file)

    # get the evidence supporting each edge
    evidence, edge_types, edge_dir = gs_utils.getEvidence(list(prededges.keys()),evidence_file=ev_file)

    # Now post to graphspace!
    #G = gs.constructGraph(pred_edges, node_labels=uniprot_to_gene, graph_attr=graph_attr, popups=popups)
    #G = gs_base.constructGraph(prededges, node_labels=uniprot_to_gene, graph_attr=graph_attr, attr_desc=attr_desc)

    #********************** CONSTRUCT GRAPH *******************
    G = constructGraph(
        prededges, sources, targets, top_targets, uniprot_to_gene,uniprot_to_protein_names,
        node_labels=uniprot_to_gene,
        evidence=evidence, edge_types=edge_types, edge_dir=edge_dir,
        graph_attr=graph_attr, attr_desc=attr_desc, **kwargs)
    print("Graph has %d nodes and %d edges" % (G.number_of_nodes(), G.number_of_edges()))

    #************ ADD GRAPH LEGEND *******************
    Ld = create_graph_legend()
    G.set_legend(Ld)

    # put the parent nodes and the nodes in the parent nodes in a grid layout automatically
    print("Setting the x and y coordinates of each node in a grid layout")
    # relabel the nodes to their names
    #NURE: TODO uncomment later
    graph_attr = {uniprot_to_gene.get(n,n): attr for n, attr in graph_attr.items()}
    layout = gs_utils.grid_layout(G, graph_attr)
    for node, (x,y) in layout.items():
        G.set_node_position(node_name=node, x=x, y=y)

    # before posting, see if we want to write the Graph's JSON to a file
    if kwargs.get('out_pref') is not None:
        print("Writing graph and style JSON files to:\n\t%s-graph.json \n\t%s-style.json" % (kwargs['out_pref']+graph_name, kwargs['out_pref']+graph_name))
        graph_json_f= kwargs['out_pref'] + graph_name+"-graph.json"
        style_json_f= kwargs['out_pref'] + graph_name+"-style.json"
        legend_json_f = kwargs['out_pref'] + graph_name + "-legend.json"

        with open(graph_json_f, 'w') as out:
            json.dump(G.get_graph_json(), out, indent=2)
        with open(style_json_f, 'w') as out:
            json.dump(G.get_style_json(), out, indent=2)
        with open(legend_json_f, 'w') as out:
            json.dump(Ld.get_legend_json(), out, indent=2)

    # G.set_tags(kwargs.get('tags',[]))
    G.set_name(graph_name)
    gs_base.post_graph_to_graphspace(
            G, kwargs['username'], kwargs['password'], graph_name,
            apply_layout=kwargs['apply_layout'], layout_name=kwargs['layout_name'],
            group=kwargs['group'], make_public=kwargs['make_public'])




# def readNetwork(paths=None, ranked_edges=None, k_limit=200, no_k=False):
#     """ Read the PathLinker paths or ranked_edges output.
#         Get all of the edges that have a k less than the k_limit.
#     """
#     if no_k is False:
#         if paths is not None:
#             # Predicted paths from pathlinker
#             lines = utils.readColumns(paths,1,2,3)
#             prededges = {}
#             edges = set()
#             for k, path_score, path in lines:
#                 # get all of the edges in the paths that have a k less than the k_limit
#                 if int(k) > k_limit:
#                     break
#
#                 path = path.split('|')
#
#                 for i in range(len(path)-1):
#                     edge = (path[i], path[i+1])
#                     if edge not in edges:
#                         edges.add(edge)
#                         prededges[edge] = int(k)
#
#         if ranked_edges is not None:
#             # Predicted edges from pathlinker
#             lines = utils.readColumns(ranked_edges,1,2,3)
#             # get all of the edges that have a k less than the k_limit
#             prededges = {(u,v):int(k) for u,v,k in lines if int(k) <= k_limit}
#     else:
#         if ranked_edges:
#             # Set of edges from another source such as a pathway
#             lines = utils.readColumns(ranked_edges,1,2)
#             # keep the edges in a dictionary to work with the rest of the code
#             prededges = {(u,v):None for u,v in lines}
#     return prededges


def create_graph_legend():
    global node_legend_dict
    Ld = GSLegend()
    for node_type in node_legend_dict:
        #TODO convert all 'target' -> 'intermediate', 'top_target' -> 'target' in all over the code and
        # remove the following hack

        if node_type=='target':
            node_type_l = 'intermediate'
        elif node_type=='top_target':
            node_type_l = 'target'
        else:
            node_type_l = node_type
        Ld.add_legend_entries('nodes', node_type_l, node_legend_dict[node_type])

    return Ld



def constructGraph(
        prededges, sources, targets,top_targets, uniprot_to_gene, uniprot_to_protein_names, node_labels={},
        evidence={}, edge_types={}, edge_dir={},
        graph_attr={}, attr_desc={}, **kwargs):
    '''
    Posts the toxcast pathlinker result to graphspace

    :param source: list of source nodes
    :param targets: list of target nodes
    :param graphid: name of graph to post
    :param outfile: output JSON file that will be written
    '''
    # NetworkX object
    G = GSGraph()

    prednodes = set([t for t,h in prededges]).union(set([h for t,h in prededges]))

    # GSGraph does not allow adding multiple nodes with the same name.
    # Use this set to make sure each node has a different gene name.
    # if the gene name is the same, then use the gene + uniprot ID instead
    genes_added = set()
    # set of parent nodes to add to the graph
    parents_to_add = {}

    ## add GraphSpace/Cytoscape.js attributes to all nodes.
    for n in prednodes:
        #default is gray circle
        node_type = 'default'
        # default parent
        #parent = 'Intermediate Proteins'
        parent = 'No Enriched Terms'
        if n in sources:
            node_type = 'source'
            # set the parent node for this source
            parent = 'Sources'
        elif n in top_targets:
            node_type = 'top_target'
            # set the parent node for this source
            parent = 'Top_targets'
        elif n in targets:
            node_type = 'target'
            parent = 'Targets'
        # #when node is enriched in some go term then add that gor-term-function-parent with
        # #node_type to get the final parent
        # if n in graph_attr:
        #
        if n not in graph_attr:
            graph_attr[n] = {}
            graph_attr[n]['parent'] = parent
        # only overwrite the parent if it is a source
        elif node_type == 'source':
            graph_attr[n]['parent'] = parent

        # set the name of the node to be the gene name and add the k to the label
        gene_name = node_labels[n]

        # if this gene name node was already added, then add another node with the name: gene-uniprot
        # TODO some uniprot IDs map to the same Gene Name (such as GNAS)
        if gene_name in genes_added:
            gene_name = "%s-%s" % (gene_name, n)
            node_labels[n] = gene_name
            #continue
        genes_added.add(gene_name)

        short_name = gene_name
        # # TODO if the the family name is too long, then just choose one of the genes and add -family to it

        edgeswithnode = set([(t,h) for t,h in prededges if t==n or h==n])
        #Previously:
        # pathswithnode = set([int(prededges[e]) for e in edgeswithnode])
        pathswithnode = [list(prededges[e]) for e in edgeswithnode]
        #flatten pathswithnode
        pathswithnode =  set([item for sublist in pathswithnode for item in sublist])
        k_value = min(pathswithnode)

        attr_dict = {}
        if kwargs.get('parent_nodes') is True:
            # set the parent if specified
            if n in graph_attr and 'parent' in graph_attr[n]:
                # set the parent of this node
                parent = graph_attr[n]['parent']
                attr_dict['parent'] = parent
            # also add this parent to the set of parent/compound nodes to add to the graph
            # keep track of the lowest k value so it will work correctly with the sliding bar
            # if parent not in parents_to_add or k_value < parents_to_add[parent]:
            #     parents_to_add[parent] = k_value

        node_popup = buildNodePopup(n,uniprot_to_protein_names[n], pathswithnode,
                    pathway_colors=kwargs.get('function_colors'))

        label = short_name

        # TODO set the node label smaller than the gene name for large family node labels
        G.add_node(gene_name, attr_dict=attr_dict, popup=node_popup,
                   label=label,k=k_value)

        attr_dict = {}
        shape = NODE_SHAPES[node_type]
        color = NODE_COLORS[node_type]
        style = 'solid'
        width = 45
        height = 45
        border_width = 2
        border_color = None
        if kwargs.get('case_study') is True:
            border_color = "#7f8184"
        bubble = None
        if n in graph_attr:
            # TODO allow for any attribute to be set. Need to update add_node_style first so that attributes aren't overwritten
            #for style in graph_attr[n]:
            #    attr_dict[style] = graph_attr[n][style]
            if 'color' in graph_attr[n]:
                color = graph_attr[n]['color']
            if 'shape' in graph_attr[n]:
                shape = graph_attr[n]['shape']
            if 'style' in graph_attr[n]:
                style = graph_attr[n]['style']
            if 'border_color' in graph_attr[n]:
                border_color= graph_attr[n]['border_color']
            if 'border_width' in graph_attr[n]:
                border_width= graph_attr[n]['border_width']
        border_color = color if border_color is None else border_color
        # I updated the bubble function in graphspace_python gsgraph.py so it wouldn't overwrite the border color.
        bubble = color if bubble is None else bubble
        luminance = get_color_luminance(color)
        # this sets the text color to white for dark colors
        if luminance < 0.45:
            attr_dict['color'] = 'white'

        G.add_node_style(gene_name, shape=shape, attr_dict=attr_dict, color=color, width=width, height=height,
                         style=style, border_color=border_color, border_width=border_width, bubble=bubble)


    # now add the parent nodes
    for parent, k_value in parents_to_add.items():
        # TODO add a popup for the parent which would be a description of the function or pathway
        if ('parent', parent) in attr_desc:
            popup = attr_desc[('parent', parent)]
        else:
            popup = parent

        if parent in graph_attr and 'label' in graph_attr[parent]:
            label = graph_attr[parent]['label']
        else:
            label = parent.replace("_", " ")
        label = label.replace("\\n","\n")

        #popup = parent
        # leave off the label for now because it is written under the edges and is difficult to see in many cases
        # I requested this feature on the cytoscapejs github repo: https://github.com/cytoscape/cytoscape.js/issues/1900
        G.add_node(parent, popup=popup, k=k_value, label=label)

        parent_label = parent.replace('\n','\\n')
        # the only style we need to set for the parent/compound nodes is the color
        # default color for sources and targets, and intermediate nodes
        if parent in NODE_COLORS:
            color = NODE_COLORS[parent]
        if parent_label in graph_attr and 'color' in graph_attr[parent_label]:
            color = graph_attr[parent_label]['color']
        attr_dict = {}
        # set the background opacity so the background is not the same color as the nodes
        attr_dict["background-opacity"] = 0.3
        attr_dict['font-weight'] = "bolder"
        attr_dict['font-size'] = "32px"
        attr_dict['text-outline-width'] = 3
        luminance = get_color_luminance(color)
        # this sets the text color to white for dark colors
        if luminance < 0.45:
            attr_dict['color'] = 'white'
        # remove the border around the parent nodes
        border_width = 0
        valign="bottom"

        G.add_node_style(parent, attr_dict=attr_dict, color=color, border_width=border_width, bubble=color, valign=valign)


    #Add legend to graph
    # Add all of the edges and their Graphspace/Cytoscape.js attributes
    for (u,v) in prededges:
        # get the main edge type
        if edge_types: #if we have type information for edges
            main_edge_type = getMainEdgeType(edge_types[(u,v)])
        else:
            main_edge_type=None

        if main_edge_type is None:
            # sys.stderr.write("WARNING: %s has no edge type. edge_types[%s]: %s. " % (str((u,v)),str((u,v)), str(edge_types[(u,v)])))
            # sys.stderr.write("Evidence for edge: %s\n" % (str(evidence[(u,v)])))
            # sys.stderr.write("\tSetting to 'physical'\n")
            #CHECK
            #sys.exit(1)
            # TODO: how to set edge type
            main_edge_type = "signaling"

        #if main_edge_type == '':
        #    raise NoEdgeTypeError("ERROR: %s,%s has no edge type. edge_types[%s,%s]: %s" % (u,v,u,v, str(edge_types[(u,v)])))
        k_value = prededges[(u,v)]

        # if 'activation' not in edge_types[(u,v)] and 'inhibition' in edge_types[(u,v)]:
        #     # I don't think the graphspace interface has anything for this, so add it here
        #     arrow_shape = 'tee'
        # else:
        arrow_shape = 'triangle'

        gene_name_u = ','.join(sorted(uniprot_to_gene[u].split(',')))
        gene_name_v = ','.join(sorted(uniprot_to_gene[v].split(',')))

        # family_ppi_evidence will be None if we are not including famliy edges
        edge_popup = buildEdgePopup(u,v, evidence, uniprot_to_gene, k=k_value)
        # G.add_edge(gene_name_u,gene_name_v,directed=edge_dir[(u,v)],popup=edge_popup,k=k_value)
        #Nure: TODO for undirected edge give directed=False
        G.add_edge(gene_name_u,gene_name_v, directed=True,popup=edge_popup,k=k_value)

        attr_dict = {}

        color = EDGE_COLORS[main_edge_type]
        edge_style = 'solid'
        edge_str = "%s-%s" % (u,v)
        if edge_str in graph_attr:
            if 'color' in graph_attr[edge_str]:
                color = graph_attr[edge_str]['color']
            if 'arrow_shape' in graph_attr[edge_str]:
                arrow_shape= graph_attr[edge_str]['arrow_shape']
            if 'edge_style' in graph_attr[edge_str]:
                edge_style = graph_attr[edge_str]['edge_style']

        G.add_edge_style(gene_name_u, gene_name_v, attr_dict=attr_dict,
                         directed=EDGE_DIR_MAP[main_edge_type], color=color, width=1.5, arrow_shape=arrow_shape,edge_style=edge_style)
    return G


def get_color_luminance(color, hex_format=True):
    if hex_format:
        c = color.replace('#','')
        # this is from here: https://stackoverflow.com/a/29643643
        r,g,b = tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
    else:
        print("ERROR: only hex_format is implemented for get_color_luminance()")
        sys.exit()
    # got this from here: https://stackoverflow.com/a/1855903
    luminance = (0.299 * r + 0.507 * g + 0.194 * b)/255.
    return luminance


def getMainEdgeType(edge_types):
    """ a single edge can have multiple edge types according to the different sources or databases
    Choose a main edge type here
    *edge_types* the set of edge types for a given edge
    """
    main_edge_type = None
    for edge_type in edge_type_order:
        if edge_type in edge_types:
            main_edge_type = edge_type
            break

    return main_edge_type


def buildNodePopup(n, protein_names=None, pathswithnode=None, pathway_colors=None):
    '''
    Converts the node data html for the node popup.

    :param data: dictionary of data from the NetworkX node.
    :systematic_name: The systematic name of the yeast gene

    :returns: HTML string.
    '''

    htmlstring = ''
    uniproturl = 'http://www.uniprot.org/uniprot/%s'

    #List Uniprot accession number
    uid = n
    htmlstring += '<b>Uniprot ID</b>: <a style="color:blue" href="%s" target="UniProtKB">%s</a><br>' % (uniproturl%uid, uid)

    if pathswithnode is not None:
        htmlstring += '<hr />'
        htmlstring += '<b>Paths</b>: %s<br>' %(','.join(str(i) for i in sorted(pathswithnode)))
    if protein_names is not None:
        htmlstring += '<hr />'
        htmlstring += '<b>Protein names</b>: %s<br>' %(protein_names)

    # if the pathways are specified for this node, add them to the list
    if pathway_colors is not None:
        htmlstring += "<hr /><b>Functions</b>:<ul>"
        for pathway in pathway_colors:
            if uid in pathway_colors[pathway]['prots']:
                pathway_link = '<a style="color:%s" href="%s">%s (%s)</a>' % (
                    pathway_colors[pathway]['color'], pathway_colors[pathway]['link'], pathway_colors[pathway]['name'], pathway)
                htmlstring += '<li>%s</li>' % (pathway_link)
        htmlstring+='</ul>'

    return htmlstring


##########################################################
def buildEdgePopup(t, h, evidence, uniprot_to_gene, PPIWEIGHTS=None, k=None, family_ppi_evidence=None):
    #if t == "Q13315" and h == "Q01094":
    #    pdb.set_trace()
    annotation = ''
    annotation+='<b>%s - %s</b></br>'%(','.join(sorted(uniprot_to_gene[t].split(','))), ','.join(sorted(uniprot_to_gene[h].split(','))))
    annotation+='<b>%s - %s</b></br>'%(t,h)
    if PPIWEIGHTS is not None:
        annotation+='<b>Weight</b>: %.3f</br>' % (PPIWEIGHTS[(t,h)])
    if k is not None:
        annotation+='<b>Edge Ranking</b>: %s' % (k)

    family_edge = True if len(t.split(',')) > 1 or len(h.split(',')) > 1 else False

    if family_ppi_evidence is not None and family_edge is True:
        annotation += '<hr /><h><b>Direct Sources of Evidence</b></h>'
        annotation += gs_utils.evidenceToHTML(t,h,evidence[(t,h)])
        if (t,h) in family_ppi_evidence:
            annotation += '<hr /><h><b>Sources of Evidence</b></h>'
            annotation += gs_utils.evidenceToHTML(t,h,family_ppi_evidence[(t,h)])
    else:
        # annotation += '<hr /><h><b>Sources of Evidence</b></h>'
        annotation += gs_utils.evidenceToHTML(t,h,evidence[(t,h)])

    return annotation


# def setup_parser():
#     """
#     """
#     #parser = post_to_gs.setup_parser()
#     ## Parse command line args.
#     parser = argparse.ArgumentParser(description="Script to test for enrichment of the top predictions among given genesets")
#
#     # general parameters
#     group = parser.add_argument_group('Main Options')
#     #
#     # group.add_argument('--mapping-file', default="inputs/2018_01-toxcast-net/2018_01_uniprot_mapping.tsv",
#     #         help='File to map to a different namespace. Network/edge IDs (uniprot ids) should be in the first column with the other namespace (gene name) in the second')
#     # group.add_argument('--evidence-file', default="inputs/2018_01-toxcast-net/2018_01interactome-evidence.tsv",
#     #         help='File containing the evidence for each edge in the interactome.')
#     # group.add_argument('--revigo-file',
#     #         help="File containing the outputs of REVIGO for coloring the nodes in the graph")
#     # group.add_argument('--term-counts-file',
#     #         help="File containing the frequency of terms among the chemicals. Terms with a frequency < .75 will be selected.")
#     # group.add_argument('--ctd-support-file',
#     #         help="File containing the CTD phosphorylations interactions per chemical. Will be used to add a double border to nodes with support.")
#     # group.add_argument('--single-run','-S', type=str, action='append',
#     #         help='Run only a single chemical. Can specify multiple chemicals by listing this option multiple times.')
#     # group.add_argument('--k-to-post', type=int, default=200,
#     #         help='Value of k to test for significance. Multiple k values can be given.')
#     # group.add_argument('--forcepost',action='store_true', default=False,
#     #         help='Force the network to be posted to graphspace even if json file already exists')
#
#
#     # posting options
#     # group = parser.add_argument_group('GraphSpace Options')
#     # group.add_argument('--username', '-U', type=str,
#     #                   help='GraphSpace account username to post graph to. Required')
#     # group.add_argument('--password', '-P', type=str,
#     #                   help='Username\'s GraphSpace account password. Required')
#     # #group.add_argument('', '--graph-name', type=str, metavar='STR', default='test',
#     # #                  help='Graph name for posting to GraphSpace. Default = "test".')
#     # # group.add_argument('--out-pref', type=str, metavar='STR',
#     # #                   help='Prefix of name to place output files. ')
#     # group.add_argument('--name-postfix', type=str, default='',
#     #                   help='Postfix of graph name to post to graphspace.')
#     # group.add_argument('--group', type=str,
#     #                   help='Name of group to share the graph with.')
#     # group.add_argument('--case-study', action="store_true", default=False,
#     #                   help='Use the Case study colors and labels (no k in labels, gray border around nodes)')
#     # group.add_argument('--make-public', action="store_true", default=False,
#     #                   help='Option to make the uploaded graph public')
#     # # TODO implement and test this option
#     # #group.add_argument('--group-id', type=str, metavar='STR',
#     # #                  help='ID of the group. Could be useful to share a graph with a group that is not owned by the person posting')
#     # group.add_argument('--tag', dest="tags", type=str, action="append",
#     #                   help='Tag to put on the graph. Can list multiple tags (for example --tag tag1 --tag tag2)')
#     # group.add_argument('--apply-layout', type=str,
#     #                   help='Specify the name of a graph from which to apply a layout. Layout name specified by the --layout-name option. ' +
#     #                   'If left blank and the graph is being updated, it will attempt to apply the --layout-name layout.')
#     # group.add_argument('--layout-name', type=str, default='layout1',
#     #                   help="Name of the layout of the graph specified by the --apply-layout option to apply. Default: 'layout1'")
#     # group.add_argument('--parent-nodes', action="store_true", default=False,
#     #                   help='Use parent/group/compound nodes for the different node types')
#     # group.add_argument('--graph-attr-file',
#     #                    help='File used to specify graph attributes. Tab-delimited with columns: 1: style, 2: style attribute, ' + \
#     #                    '3: nodes/edges to which styles will be applied separated by \'|\' (edges \'-\' separated), 4th: Description of style to add to node popup.')
#     return parser
#
#
