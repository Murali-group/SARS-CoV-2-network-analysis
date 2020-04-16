
from collections import defaultdict
import numpy as np
import scipy.sparse as sp


def print_net_stats(W, train_ann_mat, test_ann_mat=None, term_idx=None):
    """
    Get statistics about the # of connected components, and the number of left-out positive examples that are in a component without a positive example
    *term_idx*: indices of terms for which to limit the train and test annotations 
        e.g., current terms for a species
    """
    # count the nodes with at least one edge
    num_nodes = np.count_nonzero(W.sum(axis=0))
    # since the network is symmetric, the number of undirected edges is the number of entries divided by 2
    num_edges = (len(W.data) / 2)
    print("\t%s nodes, %s edges" % (num_nodes, num_edges))
    # get the number of connected components (ccs), as well as the nodes in each cc
    num_ccs, cc_labels = sp.csgraph.connected_components(W, directed=False, return_labels=True)
    # now split the nodes into their respective connected components
    ccs = defaultdict(set)
    for i, label in enumerate(cc_labels):
        ccs[label].add(i)
    cc_sizes = [len(nodes) for cc, nodes in ccs.items()]
    print("%s connected_components; sizes: max: %d, 75%%: %d, median: %d, 25%%: %d" % (
        num_ccs, np.max(cc_sizes), np.percentile(cc_sizes, 75),
        np.median(cc_sizes), np.percentile(cc_sizes, 25)))
    print("\t%0.2f%% nodes in the largest cc" % ((np.max(cc_sizes) / float(num_nodes))*100))
    print("top 20 connected_components sizes:")
    print(' '.join(str(cc_size) for cc_size in sorted(cc_sizes, reverse=True)[:20]))

    if test_ann_mat is None:
        print_cc_ann_stats(ccs, train_ann_mat, term_idx=term_idx) 
    else:
        print_cc_stats_for_train_test_pos(ccs, train_ann_mat, test_ann_mat, term_idx=term_idx) 

    return num_nodes, num_edges


def print_cc_ann_stats(ccs, ann_mat, term_idx=None):
    if term_idx is not None:
        # get just the rows of the terms specified
        ann_mat = ann_mat[term_idx]
    # sum over the columns to get the prots with at least 1 positive example
    pos_prots = np.ravel((ann_mat > 0).astype(int).sum(axis=0))
    pos_prot_idx = set(list(pos_prots.nonzero()[0]))
    #ccs_no_pos = set(cc for cc, nodes in ccs.items() \
    #                       if len(nodes & pos_prot_idx) == 0)

    #cc_sizes = {cc: len(nodes) for cc, nodes in ccs.items()}
    largest_cc = (None, 0)
    for cc, nodes in ccs.items():
        if len(nodes) > largest_cc[1]:
            largest_cc = (cc, len(nodes))
    largest_cc = largest_cc[0]
    pos_prots_in_largest_cc = 0
    for p in pos_prot_idx:
        if p in ccs[largest_cc]:
            pos_prots_in_largest_cc += 1
    print("%d/%d (%0.2f%%) pos prots are in the largest cc" % (
        pos_prots_in_largest_cc, len(pos_prot_idx),
        (float(pos_prots_in_largest_cc) / len(pos_prot_idx))*100))


def print_cc_stats_for_train_test_pos(ccs, train_ann_mat, test_ann_mat, term_idx=None):
    # to get stats about the train and test pos and neg examples,
    # first limit the annotations to the terms with annotations for this species
    curr_train_mat, curr_test_mat = train_ann_mat, test_ann_mat
    if term_idx is not None:
        # get just the rows of the terms specified
        curr_train_mat = train_ann_mat[term_idx]
        curr_test_mat = train_ann_mat[term_idx]
    # sum over the columns to get the prots with at least 1 positive example
    train_pos_prots = np.ravel((curr_train_mat > 0).astype(int).sum(axis=0))
    train_pos_prot_idx = set(list(train_pos_prots.nonzero()[0]))
    test_pos_prots = np.ravel((curr_test_mat > 0).astype(int).sum(axis=0))
    test_pos_prot_idx = set(list(test_pos_prots.nonzero()[0]))
    ccs_no_train_pos = set(cc for cc, nodes in ccs.items() \
                           if len(nodes & train_pos_prot_idx) == 0)
    ccs_test_pos     = set(cc for cc, nodes in ccs.items() \
                           if len(nodes & test_pos_prot_idx) != 0)
    # for the target species, get the prots which have at least one annotation 
    # and find the connected_components with target species annotations, yet no train positive annotations
    # in other words, get the percentage of nodes with a test ann that are in a cc with no train ann
    nodes_ccs_test_pos = set()
    for cc in ccs_no_train_pos & ccs_test_pos:
        nodes_ccs_test_pos.update(ccs[cc] & test_pos_prot_idx)
    print("%d/%d (%0.2f%%) test pos prots are in a cc with no train pos" % (
        len(nodes_ccs_test_pos), len(test_pos_prot_idx),
        (len(nodes_ccs_test_pos) / float(len(test_pos_prot_idx)))*100))

    # TODO move this somewhere else
#    # also check how many ccs have only positive or only negative examples
#    # and the proportion of positive to negative examples in ccs
#    train_neg_prots = np.ravel((curr_train_mat < 0).astype(int).sum(axis=0))
#    train_neg_prot_idx = set(list(train_neg_prots.nonzero()[0]))
#    cc_stats = {}
#    for cc, nodes in ccs.items():
#        num_pos_in_cc = len(nodes & train_pos_prot_idx)
#        num_neg_in_cc = len(nodes & train_neg_prot_idx)
#        if num_pos_in_cc > 1 or num_neg_in_cc > 1:
#            cc_stats[cc] = {
#                'num_pos': num_pos_in_cc,
#                'num_neg': num_neg_in_cc,
#                'pos/(pos+neg)': num_pos_in_cc / (num_pos_in_cc+num_neg_in_cc)}
#    df = pd.DataFrame(cc_stats).T
#    df[['num_pos', 'num_neg']] = df[['num_pos', 'num_neg']].astype(int)
#    df['pos/(pos+neg)'] = df['pos/(pos+neg)'].round(3)
#    df.sort_values('pos/(pos+neg)', inplace=True, ascending=False)
#    print("head and tail of ratio of pos to neg per cc:")
#    print(df.head())
#    print(df.tail())

    #print("%d ccs have no train ann, %d have test ann, %d have test ann and no train ann" % (
    #    len(ccs_no_train_ann), len(ccs_test_ann), len(ccs_no_train_ann & ccs_test_ann)))
    #return num_nodes, num_edges, df

