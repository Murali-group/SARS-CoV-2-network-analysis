import os, sys
from collections import defaultdict
import networkx as nx

# See the tutorial of obonet here:
# https://github.com/dhimmel/obonet/blob/master/examples/go-obonet.ipynb
try:
    import obonet
except:
    print("\tFailed to import obonet")

# global variables
id_to_name = {}
name_to_id = {}
goid_to_category = {}  # mapping from a GO term ID and the category it belongs to ('C', 'F' or 'P')


def parse_obo_file_and_build_dags(obo_file, out_dir, forced=False):
    """
    Parse the GO OBO into a networkx MultiDiGraph using obonet.
    Then construct a DAG for each category using the 'is_a' relationships
    *forced*: this function will store the dags as an edgelist for faster parsing
        If forced is true, it will overwrite those

    *returns*: a dictionary containing a DAG for each of the 3 GO categories 'C', 'F', and 'P'
    """
    global id_to_name, name_to_id, goid_to_category

    dag_edgelist_file = out_dir + obo_file.split('/')[-1].replace(".obo", "-isa-edgelist.txt")
    goid_names_file = out_dir+ obo_file.split('/')[-1].replace(".obo", "-names.txt")

    os.makedirs(os.path.dirname(dag_edgelist_file), exist_ok=True)

    if not forced and os.path.isfile(dag_edgelist_file) and os.path.isfile(goid_names_file):
        print("Reading GO dags from %s" % (dag_edgelist_file))
        go_dags = {}
        for c in ['C', 'F', 'P']:
            go_dags[c] = nx.DiGraph()
        with open(dag_edgelist_file, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                g1, g2, c = line.rstrip().split('\t')[:3]
                go_dags[c].add_edge(g1, g2)

        for c, dag in go_dags.items():
            print("\tDAG for %s has %d nodes, %d edges" % (c, dag.number_of_nodes(), dag.number_of_edges()))
            # also set the category for each GO term
            for n in dag.nodes():
                goid_to_category[n] = c

        with open(goid_names_file, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                goid, name, c = line.rstrip().split('\t')[:3]
                name_to_id[name] = goid
                id_to_name[goid] = name
    else:
        print("Reading GO OBO file from %s" % (obo_file))
        # obonet returns a networkx MultiDiGraph object containing all of the relationships in the ontology
        graph = obonet.read_obo(obo_file)
        # build a mapping from the GO term IDs to the name of the GO term
        id_to_name = {id_: data['name'] for id_, data in graph.nodes(data=True)}
        name_to_id = {data['name']: id_ for id_, data in graph.nodes(data=True)}
        print("\t%d nodes, %d edges" % (graph.number_of_nodes(), graph.number_of_edges()))

        # make sure this really is a DAG
        if nx.is_directed_acyclic_graph(graph) is False:
            print("\tWarning: graph is not a dag")

        # copied this section from cell 19 of https://github.com/IGACAT/DataPreprocessing/blob/master/scripts/populate_go_terms.ipynb
        # Extract all edges with "is_a" relationship.
        # I did not include "part_of" relationships because the molecular_function and biological_process DAGs are not separate from each other if I do
        is_a_edge_list = []
        for child, parent, key in graph.out_edges(keys=True):
            if key == 'is_a':
                is_a_edge_list.append((child, parent))

        # get a is_a-type edge-induced subgraph
        is_a_subG = nx.MultiDiGraph(is_a_edge_list)
        full_to_category = {'cellular_component': 'C', 'biological_process': 'P', 'molecular_function': 'F'}
        go_dags = {}
        # there are 3 weakly_connected_components. One for each category
        for wcc in nx.weakly_connected_components(is_a_subG):
            G = is_a_subG.subgraph(wcc)

            # store this DAG in the dictionary of GO DAGs
            # find the root node
            root_node = None  # find root_node  (no out_edge)
            for node in G.nodes():
                if G.out_degree(node) == 0:
                    root_node = node
                    # print(root_node, id_to_name[node])
                    break
            c = full_to_category[id_to_name[root_node]]
            print("\tDAG for %s has %d nodes" % (id_to_name[root_node], len(wcc)))
            go_dags[c] = G

            # also set the category for each GO term
            for n in G.nodes():
                goid_to_category[n] = c
        print("\twriting dags to %s" % (dag_edgelist_file))
        with open(dag_edgelist_file, 'w') as out:
            out.write("#child\tparent\thierarchy\n")
            for c, dag in go_dags.items():
                out.write(''.join("%s\t%s\t%s\n" % (g1, g2, c) for g1, g2 in dag.edges()))

        # also write the names to a file
        print("\twriting goid names to %s" % (goid_names_file))
        with open(goid_names_file, 'w') as out:
            for goid in id_to_name:
                out.write("%s\t%s\t%s\n" % (goid, id_to_name[goid], goid_to_category[goid]))
    return go_dags


# TODO just use the goid_to_category object
def find_terms_category(terms, go_dags):
    # figure out which DAG these terms are part of
    # for now, just use a single term
    for h, dag in go_dags.items():
        if dag.has_node(list(terms)[0]):
            break
    print("\tusing %s category" % (h))
    return h, go_dags[h]


def get_most_specific_terms(terms, dag=None, ann_obj=None):
    """
    Given a set of terms, remove all ancestors those terms from the set
        to get only those which are the leaf or most specific terms

    *terms*: a collection of terms
    *dag*: a networkx directed graph of an ontology (e.g., GO BP)
    *returns*: the most specific terms
    """
    if dag is None and ann_obj is None:
        print("ERROR: must give either the DAG or ann_obj")
        return terms
    if dag is None:
        G = nx.DiGraph()
        dag = nx.from_scipy_sparse_matrix(ann_obj.dag_matrix, create_using=G)
        nx.relabel_nodes(dag, {i: t for i, t in enumerate(ann_obj.terms)}, copy=False)
    ancestor_terms = set()
    for t in terms:
        if t in ancestor_terms:
            continue
        # get the ancestors, which are technically descendants in the DAG
        ancestors = set(nx.descendants(dag, t))
        ancestor_terms |= ancestors
    # remove the ancestors to get the specific terms
    specific_terms = set(terms) - ancestor_terms
    return specific_terms


go_basic_obo_url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
project_dir = '/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/'
data_dir = 'datasets/go/'
out_dir = project_dir+data_dir
parse_obo_file_and_build_dags(go_basic_obo_url, project_dir+data_dir)