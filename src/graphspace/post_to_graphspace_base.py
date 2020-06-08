#!/usr/bin/python

# base code to post a network with attributes to graphspace

#print("Importing Libraries")

import sys, os
from collections import defaultdict
from optparse import OptionParser
from graphspace_python.api.client import GraphSpace
from graphspace_python.graphs.classes.gsgraph import GSGraph
# GSGraph already implements networkx
#import networkx as nx
#import utils.file_utils as utils
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from src.utils import file_utils as utils
import sys
import pandas as pd

from src.graphspace import gs_utils


def main(args):
    #global PARENTNODES

    opts, args = parseArgs(args)

    #PARENTNODES = opts.include_parent_nodes

    # Set of edges from another source such as a pathway
    lines = utils.readColumns(opts.edges,1,2)
    prededges = set(lines)

    node_labels = {} 
    if opts.mapping_file is not None:
        node_labels = utils.readDict(opts.mapping_file, 1, 2)

    # get attributes of nodes and edges from the graph_attr file
    graph_attr = {}
    attr_desc = {} 
    if opts.graph_attr:
        graph_attr, attr_desc = readGraphAttr(opts.graph_attr)

    if opts.net is not None:
        # add the edge weight from the network to attr_desc which will be used for the popup
        edge_weights = {(u,v):float(w) for u,v,w in utils.readColumns(opts.net,1,2,3)}
        for e in prededges:
            if e not in attr_desc:
                attr_desc[e] = {}
            attr_desc[e]["edge weight"] = edge_weights[e]

    # set the width of the edges by the network weight
    if opts.net is not None and opts.set_edge_width:
        graph_attr = set_edge_width(prededges, edge_weights, graph_attr, a=1, b=12)

    # TODO build the popups here. That way the popup building logic can be separated from the
    # GSGraph building logic
    popups = {}
    prednodes = set([n for edge in prededges for n in edge])
    for n in prednodes:
        popups[n] = buildNodePopup(n, attr_val=attr_desc)
    for u,v in prededges:
        popups[(u,v)] = buildEdgePopup(u,v, node_labels=node_labels, attr_val=attr_desc)

    # Now post to graphspace!
    G = constructGraph(prededges, node_labels=node_labels, graph_attr=graph_attr, popups=popups)

    # TODO add an option to build the 'graph information' tab legend/info
    # build the 'Graph Information' metadata
    desc = buildGraphDescription(opts.edges, opts.net)
    metadata = {'description':desc,'tags':[], 'title':''}
    if opts.tag:
        metadata['tags'] = opts.tag
    G.set_data(metadata)
    G.set_name(opts.graph_name)

    post_graph_to_graphspace(G, opts.username, opts.password, opts.graph_name, apply_layout=opts.apply_layout, layout_name=opts.layout_name,
                             group=opts.group, make_public=opts.make_public)


def readGraphAttr(graph_attr_file):
    """ 
    Read attributes of nodes and edges from the graph_attr file
    Must have 4 tab-delimited columns. 
    1: Style name
    2: Style value
    3: Nodes/Edges (joined by '|') to apply the style to
    4: This is intended to be either a popup or part of the Graph Description / Legend, but it isn't built yet

    # example node attribute:
    color blue    p1|p2|p3  - 
    # example edge attribute:
    edge_style dotted    p1-p2|p2-p3  - 
    # example compound node. Here p1, p2, and p3 will have the parent attribute set to 'parent1' (i.e. they will belong to the same compound node parent1)
    parent    parent1  p1|p2|p3  - 
    # then to set the attributes of 'parent1', specify it as the node
    color blue    parent1  -
    """
    graph_attr = defaultdict(dict)
    # description of a style, style_attr tuple 
    # can also contain edge-str: name: value
    # which can be used when building popups
    attr_desc = defaultdict(dict)

    # keep the order of the pathways by order of highest posterior probability
    #pathway_colors = collections.OrderedDict()
    print("Adding graph attributes from '%s' (must have 3 tab-delimited columns)" % (graph_attr_file))
    # TODO the last column (here always '-') can be given as a description
    #lines = utils.readColumns(graph_attr_file, 1,2,3,4)
    #lines = utils.readColumns(graph_attr_file, 1,2,3)
    df = pd.read_csv(graph_attr_file, sep='\t', header=None, comment='#')
    print("\tread %d lines" % (len(df)))
    # reverse the lines so the pathways at the top of the file will overwrite the pathways at the bottom
    #for style, style_attr, items, desc in lines[::-1]:
    for style, style_attr, items, desc in df.values[::-1]:
        for item in str(items).split('|'):
            # if this is an edge, then split it by the '-'
            if len(item.split('-')) == 2:
                item = tuple(item.split('-'))
            elif len(item.split('-')) > 2:
                print("Error: '-' found in node name for edge: %s. '-' is used to split an edge." % (item))
                sys.exit(1)

            if item not in graph_attr:
                graph_attr[item] = {}
            graph_attr[item][style] = style_attr
        attr_desc[(style, style_attr)] = desc
        #graph_attributes[group_number] = {"style": style, "style_attr": style_attr, "prots": prots.split(','), "desc":desc}

    return graph_attr, attr_desc


def set_node_size(nodes, node_weights, graph_attr, a=60, b=160, min_weight=None, max_weight=None):
    """
    Set the width of edges according to edge weights that will be normalized between *a* and *b*
    Width will be stored in graph_attr

    20 and 80 seemed like good minimum and maximum widths
    """
    # TODO make this into a function
    #def normalize_between_vals(values, a, b, max_val=None, min_val=None):
    if max_weight is None:
        max_weight = max(node_weights.values())
    if min_weight is None:
        min_weight = min(node_weights.values())
    print("node max_weight = %s, min_weight = %s" % (max_weight, min_weight))
    for n in nodes:
        normalized_weight = (b-a) * (float(node_weights[n] - min_weight) / float(max_weight - min_weight)) + a
        graph_attr[n]['width'] = normalized_weight
        graph_attr[n]['height'] = normalized_weight

    return graph_attr


def set_edge_width(edges, edge_weights, graph_attr, a=1, b=12, min_weight=None, max_weight=None):
    """
    Set the width of edges according to edge weights that will be normalized between *a* and *b*
    Width will be stored in graph_attr

    1 and 12 seemed like good minimum and maximum widths
    """
    # TODO make this into a function
    #def normalize_between_vals(values, a, b, max_val=None, min_val=None):
    if max_weight is None:
        max_weight = max(edge_weights.values())
    if min_weight is None:
        min_weight = min(edge_weights.values())
    print("edge max_weight = %s, min_weight = %s" % (max_weight, min_weight))
    for e in edges:
        normalized_weight = (b-a) * (float(edge_weights[e] - min_weight) / float(max_weight - min_weight)) + a
        graph_attr[e]['width'] = normalized_weight

    return graph_attr


def post_graph_to_graphspace(G, username, password, graph_name, apply_layout=None, layout_name='layout1',
                             group=None, group_id=None, make_public=None):
    """
    Post a graph to graphspace and perform other layout and sharing tasks
    *G*: Costructed GSGraph object
    *username*: GraphSpace username 
    *password*: GraphSpace password 
    *graph_name*: Name to give to graph when posting. If a graph with that name already exists, it will be updated
    *apply_layout*: Graph name to check for x and y positions of a layout (layout_name) and apply them to nodes of this graph 
    *layout_name*: Name of layout to check for in the apply_layout graph. Default: 'layout1' 
    *group*: Name of group to share graph with
    *group_id*: Not implemented yet. ID of group to share graph with. Could be useful if two groups have the same name
    *make_public*: Make the graph public
    """
    # post to graphspace
    gs = GraphSpace(username, password)
    #print("\nPosting graph '%s' to graphspace\n" % (graph_name))
    gs_graph = gs.get_graph(graph_name, owner_email=username)

    layout = None
    # I often use the layout 'layout1', so I set that as the default
    # first check if the x and y coordinates should be set from another graph
    if apply_layout is not None:
        # if a layout was created for a different graph name, try to copy that layout here
        print("checking if layout '%s' exists for graph %s" % (layout_name, apply_layout))
        layout = gs.get_graph_layout(graph_name=apply_layout,layout_name=layout_name)
    # if the graph already exists, see if the layout can be copied
    if gs_graph is not None:
        print("checking if layout '%s' exists for this graph (%s)" % (layout_name, graph_name))
        layout = gs.get_graph_layout(graph=gs_graph,layout_name=layout_name)
    # now apply the layout if applicable
    if layout is not None:
        # set the x and y position of each node in the updated graph to the x and y positions of the layout you created
        print("Setting the x and y coordinates of each node to the positions in %s" % (layout_name))
        for node, positions in layout.positions_json.items():
            G.set_node_position(node_name=node, x=positions['x'], y=positions['y'])
        # also check nodes that may have added a little more to the name
        for node in G.nodes():
            for n2, positions in layout.positions_json.items():
                # remove the newline from the node name if its there
                n2 = n2.split('\n')[0]
                if n2 in node:
                    G.set_node_position(node_name=node, x=positions['x'], y=positions['y'])

    if gs_graph is None:
        print("\nPosting graph '%s' to graphspace\n" % (graph_name))
        gsgraph = gs.post_graph(G)
    else:
        # "re-post" or update the graph 
        print("\nGraph '%s' already exists. Updating it\n" % (graph_name))
        gsgraph = gs.update_graph(G, graph_name=graph_name, owner_email=username)
    if make_public is True:
        print("Making graph '%s' public." % (graph_name))
        gsgraph = gs.publish_graph(graph=G)
    print(gsgraph.url)

    # TODO implement the group_id. This will allow a user to share a graph with a group that is not their own
    if group is not None:
        # create the group if it doesn't exist
        #group = gs.post_group(GSGroup(name='icsb2017', description='sample group'))
        # or get the group you already created it
        print("sharing graph with group '%s'" % (group))
        group = gs.get_group(group_name=group)
        #print(group.url)
        gs.share_graph(graph=gs_graph, graph_name=graph_name, group=group)


def constructGraph(edges, prednodes=None, node_labels={}, graph_attr={}, popups={}, edge_dirs={}):
    """
    Posts the set of edges to graphspace

    *edges*: set of edges to post 
    *graph_attr*: optional dictionary containing styles for nodes and edges. For example:
        n1: {color: red, border_width: 10}
        n1-n2: {line-color: blue}
    *node_labels*: optional dictionary containing the desired label for each node
    *popups*: optional dictionary containing html popups for nodes and edges 
    *edge_dirs*: optional dictionary specifying if an edge is directed (True) or not (False). Default is not directed (False)

    returns the constructed GSGraph
    """
    # NetworkX object
    #G = nx.DiGraph(directed=True)
    G = GSGraph()

    if prednodes is None:
        prednodes = set([t for t,h in edges]).union(set([h for t,h in edges]))

    # GSGraph does not allow adding multiple nodes with the same name.
    # Use this set to make sure each node has a different gene name.
    # if the gene name is the same, then use the gene + uniprot ID instead
    labels_added = set()
    # set of parent nodes to add to the graph
    #parents_to_add = {}

    ## add GraphSpace/Cytoscape.js attributes to all nodes.
    for n in prednodes:
        # this dictionary will pass along any extra parameters that are not usually handled by GraphSpace
        attr_dict = {}
        # A 'group' node is also known as a 'compound' or 'parent' node
        if n in graph_attr and 'parent' in graph_attr[n]:
            # set the group of this node
            group = graph_attr[n]['parent']
            attr_dict['parent'] = group

        # if there is no popup, then have the popup just be the node name
        node_popup = popups.pop(n, n)
        # leave the gene name as the node ID if there are no node_labels provided
        node_label = node_labels.pop(n,n)
        # if this gene name node was already added, then add another node with the name: gene-uniprot
        if node_label in labels_added:
            node_label = "%s-%s" % (node_label, n)
        node_labels[n] = node_label
        labels_added.add(node_label)

        G.add_node(node_label, attr_dict=attr_dict, popup=node_popup, label=node_label)

        # these are the default values I like. Any styles set in the graph_attr dictionary will overwrite these defaults
        attr_dict = {
            #"border-color": "#7f8184"
        }
        #border_color = "#7f8184"  # slightly darker grey
        shape = 'ellipse'
        color = '#D8D8D8'  # grey - background-color
        #border_style = 'solid'
        width = 45
        height = 45
        border_width = 2
        bubble = None  # default is the node color
        if n in graph_attr:
            # any attribute can be set in the graph_attr dict and the defaults will be overwritten
            for style in graph_attr[n]:
                # for some reason, the text color is updated with this color... 
                if style == 'color':
                    continue
                attr_dict[style] = graph_attr[n][style]
            # get the color so the bubble is updated correctly
            # color is actually used for the background color
            if 'color' in graph_attr[n]:
                color = graph_attr[n]['color']
            #if 'border_color' in graph_attr[n]:
            #    border_color= graph_attr[n]['border_color']
            #if 'bubble' in graph_attr[n]:
            #    border_color= graph_attr[n]['bubble']
        if 'border-color' not in attr_dict:
            attr_dict['border-color'] = color 
        #border_color = color if border_color is None else border_color
        # I updated the bubble function in graphspace_python gsgraph.py so it wouldn't overwrite the border color.
        bubble = color if bubble is None else bubble
        #if n == "P11021":
        #    print(attr_dict) 
        #    sys.exit()

        G.add_node_style(node_label, shape=shape, attr_dict=attr_dict, color=color, width=width, height=height,
                         #style=border_style,
                         #border_color=border_color,
                         #border_width=border_width,
                         bubble=bubble)

    # Add all of the edges and their Graphspace/Cytoscape.js attributes
    for (u,v) in edges:
        # if there is no popup, then have the popup just be the edge name
        edge_popup = popups.pop((u,v), "%s-%s" % (u,v))
        directed = edge_dirs.pop((u,v), False)
        # TODO directed vs undirected edges into an option
        G.add_edge(node_labels[u],node_labels[v],directed=directed,popup=edge_popup)

        # edge style defaults:
        attr_dict = {
            'width': 1.5,
            # this should be the default for directed edges
            #'target-arrow-shape': 'triangle',
            'line-style': 'solid'}
        color = "#D8D8D8"  # in the attr_dict, this is 'line-color'
        if (u,v) in graph_attr:
            # any attribute can be set in the graph_attr dict and the defaults will be overwritten
            for style in graph_attr[(u,v)]:
                attr_dict[style] = graph_attr[(u,v)][style]
            if 'color' in graph_attr[(u,v)]:
                color = graph_attr[(u,v)]['color']

        #print(width, color, arrow_shape, edge_style)
        G.add_edge_style(node_labels[u], node_labels[v], attr_dict=attr_dict,
                         directed=directed, color=color) 
    return G


def buildNodePopup(n, node_type='uniprot', attr_val=None):
    """
    Builds the node data html for the node popup.

    *n*: node
    *uniprot*: if the node is a uniprot ID, then add a link to uniprot
    *attr_val*: dict containing an attribute name and value for nodes

    *returns* HTML popup string.
    """

    htmlstring = ''
    uniproturl = 'http://www.uniprot.org/uniprot/%s' 
    #entrezurl = 'http://www.ncbi.nlm.nih.gov/gene/%s'
    drugbankurl = 'https://www.drugbank.ca/drugs/%s' 

    #List Uniprot accession number
    uid = n
    if node_type == "uniprot":
        htmlstring += '<b>Uniprot ID</b>: <a style="color:blue" href="%s" target="UniProtKB">%s</a><br>' % (uniproturl%uid, uid)
    elif node_type == "drugbank":
        htmlstring += '<b>DrugBank ID</b>: <a style="color:blue" href="%s" target="DrugBank">%s</a><br>' % (drugbankurl%uid, uid)
    else:
        htmlstring += "%s</br>" % (n)
    #    htmlstring += '<li><i>Recommended name</i>: %s</li>' % (db.get_description(uid))
#    # get the alternate names from the AltName-Full namespace
#    alt_names = db.map_id(uid, 'uniprotkb', 'AltName-Full')
#    if len(alt_names) > 0:
#        htmlstring += '<li><i>Alternate name(s)</i>: <ul><li>%s</li></ul></li>' %('</li><li>'.join(sorted(alt_names)))

    # TODO have a better way to build the popups
    if attr_val is not None and n in attr_val:
        htmlstring += "<hr />"
        # now add all of the specified node annotatinos
        for attr, val in attr_val[n].items():
            htmlstring += "<b>%s</b>: %s</br>" % (attr, val)

    return htmlstring


def buildEdgePopup(u, v, node_labels={}, attr_val=None, evidence=None, **kwargs):
    """
    Builds the edge html for the edge popup.

    *u*: tail of edge
    *v*: head of edge
    *attr_val*: dict containing an attribute name and value for nodes

    *returns* HTML popup string.
    """

    htmlstring = '' 
    htmlstring +='<b>%s - %s</b></br>'%(u,v)
    if u in node_labels and v in node_labels:
        htmlstring +='<b>%s - %s</b></br>'%(node_labels[u], node_labels[v])

    # TODO design a better method to build the popups
    if attr_val is not None and (u,v) in attr_val:
        htmlstring += "<hr />"
        # now add all of the specified edge annotations
        for attr, val in attr_val[(u,v)].items():
            htmlstring += "<b>%s</b>: %s</br>" % (attr, val)

    # if the evidence file was passed in, then also add the evidence for this edge
    if evidence is not None:
        htmlstring += '<hr /><h><b>Sources of Evidence</b></h>'
        htmlstring += gs_utils.evidenceToHTML(u,v,evidence[(u,v)])

    return htmlstring


def buildGraphDescription(edgesfile, net):
    # TODO build a graph description using the --graph-attr option 
    desc = ''
    # I used this website to get the shapes: http://shapecatcher.com/index.html
    unicode_shapes = {'triangle': '&#x25b2', 'square': '&#x25a0', 'circle': '&#x25cf',
            'arrow': '&#10230', 'T': '&#x22a3', 'dash': '&#x2014'}

    desc += '<hr /><b>Edges file</b>: %s<br>' % (edgesfile)
    if net is not None:
        desc += '<hr /><b>Background network file</b>: %s<br>' % (net)

    return desc


def parseArgs(args):
    ## Parse command line args.
    usage = 'post_to_graphspace_human.py [options]\n'
    parser = OptionParser(usage=usage)
    parser.add_option('', '--edges', type='string', metavar='STR',
                      help='File of edges to post. Tab-delimited file with columns TAIL, HEAD. Required')
    parser.add_option('', '--net', type='string', metavar='STR',
                      help='File of weighted directed edges. Can be used to get the weight of each edge for the popup. Tab-delimited file with columns TAIL,  HEAD,  WEIGHT.')
    parser.add_option('', '--mapping-file', type='string', metavar='STR',
                      help='File used to map to a different namespace. Network/edge IDs (uniprot ids) should be in the first column with the other namespace (gene name) in the second')

    # extra options
    parser.add_option('', '--graph-attr', type='string', metavar='STR',
            help='File used to specify graph attributes. Tab-delimited with columns: 1: style, 2: style attribute, ' + 
            '3: nodes/edges to which styles will be applied separated by \'|\' (edges \'-\' separated), 4th: Description of style to add to graph legend (not yet implemented).')
    parser.add_option('', '--set-edge-width', action="store_true", default=False,
                      help='Set edge widths according to the weight in the network file.')
    parser.add_optoin('', '--edge-evidence-file', type='string',
                       help="File containing evidence for each edge. See XXX for the file format")

    # posting options
    parser.add_option('-U', '--username', type='string', metavar='STR', 
                      help='GraphSpace account username to post graph to. Required')
    parser.add_option('-P', '--password', type='string', metavar='STR', 
                      help='Username\'s GraphSpace account password. Required')
    parser.add_option('', '--graph-name', type='string', metavar='STR', default='test',
                      help='Graph name for posting to GraphSpace. Default: "test".')
    # replace with an option to write the JSON file before/after posting
    #parser.add_option('', '--outprefix', type='string', metavar='STR', default='test',
    #                  help='Prefix of name to place output files. Required.')
    parser.add_option('', '--group', type='string', metavar='STR', 
                      help='Name of group to share the graph with.')
    parser.add_option('', '--make-public', action="store_true", default=False,
                      help='Option to make the uploaded graph public.')
    # TODO implement and test this option
    #parser.add_option('', '--group-id', type='string', metavar='STR',
    #                  help='ID of the group. Could be useful to share a graph with a group that is not owned by the person posting')
    parser.add_option('', '--tag', type='string', metavar='STR', action="append",
                      help='Tag to put on the graph. Can list multiple tags (for example --tag tag1 --tag tag2)')
    parser.add_option('', '--apply-layout', type='string', metavar='STR', 
                      help='Specify the name of a graph from which to apply a layout. Layout name specified by the --layout-name option. ' + 
                      'If left blank and the graph is being updated, it will attempt to apply the --layout-name layout.')
    parser.add_option('', '--layout-name', type='string', metavar='STR', default='layout1',
                      help="Name of the layout (of the graph specified by the --apply-layout option). " +
                      "X and y coordinates of nodes from that layout will be applied to matching node IDs in this graph. Default: 'layout1'")
    # TODO implement parent nodes
    #parser.add_option('', '--include-parent-nodes', action="store_true", default=False,
    #                  help="Include source, target, intermediate parent nodes, and whatever parent nodes are in the --graph-attr file")

    (opts, args) = parser.parse_args(args)

    if not opts.edges: 
        parser.print_help()
        sys.exit("\n\t--edges option required")
    #if not opts.ppifile:
    #    parser.print_help()
    #    sys.exit("\n--ppifile option required")

    if opts.username is None or opts.password is None:
        parser.print_help()
        sys.exit("\n\t--username and --password required")

    return opts, args


if __name__ == '__main__':
    main(sys.argv)
