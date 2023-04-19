import os
import pandas as pd

def createGraphAttr(pos_nodes, all_nodes, graph_attr_file):
    '''
    create and save graph attributes. save attr in graph_attr_file in 4 tab separated columns.
    4 cols are:
    1: Style name
    2: Style value
    3: Nodes/Edges (joined by '/') to apply the style to
    4: This is intended to be either a popup or part of the Graph Description / Legend, but it isn't built yet

    # example node attribute:
    color blue    p1/p2/p3  -

    Currently used attributes:
    1. give positive/source nodes different colors than other nodes.
    '''

    #********** COLOR **************
    # give positive/source nodes different colors than other nodes
    pos_nodes_str = '/'.join(pos_nodes)
    non_pos_nodes = set(all_nodes).difference(set(pos_nodes))
    non_pos_nodes_str = '/'.join(non_pos_nodes)

    os.makedirs(os.path.dirname(graph_attr_file), exist_ok=True)
    f = open(graph_attr_file, 'w')
    f.write('color\tgreen\t'+pos_nodes_str+'\t-\n')
    f.write('color\tgrey\t'+non_pos_nodes_str+'\t-\n')

    f.close()