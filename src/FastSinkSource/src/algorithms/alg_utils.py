
import os, sys
from scipy import sparse as sp
from scipy.sparse import csr_matrix, csgraph, diags
import numpy as np
from collections import defaultdict
import time
from tqdm import tqdm, trange
import src.utils.file_utils as utils
import gzip


ALGORITHMS = [
    "sinksourceplus-bounds",
    "sinksource-bounds",
    "fastsinksourceplus",  
    "fastsinksource",  
    "localplus",  
    "local",  
    "birgrank",
    "aptrank",
    "genemania",
    ]


def str_(s):
    return str(s).replace('.','_')


def select_terms(only_functions_file=None, terms=None):
    selected_terms = set()
    if only_functions_file is not None:
        selected_terms = utils.readItemSet(only_functions_file, 1)
        print("%d functions from only_functions_file: %s" % (len(selected_terms), only_functions_file))
    terms = set() if terms is None else set(terms)
    selected_terms.update(terms)
    if len(selected_terms) == 0:
        selected_terms = None
    return selected_terms


def setup_sparse_network(network_file, node2idx_file=None, forced=False):
    """
    Takes a network file and converts it to a sparse matrix
    """
    sparse_net_file = network_file.replace('.'+network_file.split('.')[-1], '.npz')
    if node2idx_file is None:
        node2idx_file = sparse_net_file + "-node-ids.txt"
    if forced is False and (os.path.isfile(sparse_net_file) and os.path.isfile(node2idx_file)):
        print("Reading network from %s" % (sparse_net_file))
        W = sp.load_npz(sparse_net_file)
        print("\t%d nodes and %d edges" % (W.shape[0], len(W.data)/2))
        print("Reading node names from %s" % (node2idx_file))
        node2idx = {n: int(n2) for n, n2 in utils.readColumns(node2idx_file, 1, 2)}
        idx2node = {n2: n for n, n2 in node2idx.items()}
        prots = [idx2node[n] for n in sorted(idx2node)]
    elif os.path.isfile(network_file):
        print("Reading network from %s" % (network_file))
        u,v,w = [], [], []
        open_func = gzip.open if network_file.endswith('.gz') else open
        with open_func(network_file, 'r') as f:
            for line in f:
                line = line.decode() if network_file.endswith('.gz') else line
                if line[0] == '#':
                    continue
                line = line.rstrip().split('\t')
                u.append(line[0])
                v.append(line[1])
                if len(line > 2):
                    w.append(float(line[2]))
                else:
                    w.append(float(1))
        print("\tconverting uniprot ids to node indexes / ids")
        # first convert the uniprot ids to node indexes / ids
        prots = sorted(set(list(u)) | set(list(v)))
        node2idx = {prot: i for i, prot in enumerate(prots)}
        i = [node2idx[n] for n in u]
        j = [node2idx[n] for n in v]
        print("\tcreating sparse matrix")
        #print(i,j,w)
        W = sp.coo_matrix((w, (i, j)), shape=(len(prots), len(prots))).tocsr()
        # make sure it is symmetric
        if (W.T != W).nnz == 0:
            pass
        else:
            print("### Matrix not symmetric!")
            W = W + W.T
            print("### Matrix converted to symmetric.")
        #print("\t%d nodes, %d edges")
        #name = os.path.basename(net_file)
        print("\twriting sparse matrix to %s" % (sparse_net_file))
        sp.save_npz(sparse_net_file, W)
        print("\twriting node2idx labels to %s" % (node2idx_file))
        with open(node2idx_file, 'w') as out:
            out.write(''.join(["%s\t%d\n" % (prot,i) for i, prot in enumerate(prots)]))
    else:
        print("Network %s not found. Quitting" % (network_file))
        sys.exit(1)

    return W, prots


def align_mat(mat, new_shape, row_labels, row_label_to_new_index, 
        map_to=False, verbose=False):
    """
    This function is to algin a matrix with rows ordered differently, and a differeint shape
    *map_to*: Use the row_label_to_new_index mapping in the new matrix, and *row_labels* in the "old" matrix. 
        Otherwise, use *row_labels* in the new matrix, and row_label_to_new_index in the "old" matrix
    """
    new_mat = sp.lil_matrix(new_shape)
    # need to realign the pos_mat and the leaf_ann_mat, since the terms could be ordered differently
    for i in trange(len(row_labels), disable=False if verbose else True):
        old_index = i
        new_index = row_label_to_new_index[row_labels[i]]
        if map_to:
            old_index = new_index
            new_index = i
        new_mat[new_index] = mat[old_index]
    return new_mat.tocsr()


def normalizeGraphEdgeWeights(W, ss_lambda=None, axis=1):
    """
    *W*: weighted network as a scipy sparse matrix in csr format
    *ss_lambda*: SinkSourcePlus lambda parameter
    *axis*: The axis to normalize by. 0 is columns, 1 is rows
    """
    # normalize the matrix
    # by dividing every edge weight by the node's degree 
    deg = np.asarray(W.sum(axis=axis)).flatten()
    if ss_lambda is None:
        deg = np.divide(1., deg)
    else:
        deg = np.divide(1., ss_lambda + deg)
    deg[np.isinf(deg)] = 0
    # make sure we're dividing by the right axis
    if axis == 1:
        deg = csr_matrix(deg).T
    else:
        deg = csr_matrix(deg)
    P = W.multiply(deg)
    return P.asformat(W.getformat())


def _net_normalize(W, axis=0):
    """
    Normalize W by multiplying D^(-1/2) * W * D^(-1/2)
    This is used for GeneMANIA
    *W*: weighted network as a scipy sparse matrix in csr format
    """
    # normalizing the matrix
    # sum the weights in the columns to get the degree of each node
    deg = np.asarray(W.sum(axis=axis)).flatten()
    deg = np.divide(1., np.sqrt(deg))
    deg[np.isinf(deg)] = 0
    D = sp.diags(deg)
    # normalize W by multiplying D^(-1/2) * W * D^(-1/2)
    P = D.dot(W.dot(D))
    return P


def get_term_pos_neg(ann_matrix, i):
    """
    The matrix should be lil format as others don't have the getrowview option
    """
    # get the row corresponding to the current terms annotations 
    #term_ann = ann_matrix[i,:].toarray().flatten()
    #positives = np.where(term_ann > 0)[0]
    #negatives = np.where(term_ann < 0)[0]
    # may be faster with a lil matrix, but takes a lot more RAM
    #term_ann = ann_matrix.getrowview(i)
    term_ann = ann_matrix[i,:]
    positives = (term_ann > 0).nonzero()[1]
    negatives = (term_ann < 0).nonzero()[1]
    return positives, negatives


def setup_fixed_scores(P, positives, negatives=None, a=1, 
        remove_nonreachable=True, verbose=False):
    """
    Remove the positive and negative nodes from the matrix P 
    and compute the fixed vector f which contains the score contribution 
    of the positive nodes to the unknown nodes.
    """
    #print("Initializing scores and setting up network")
    pos_vec = np.zeros(P.shape[0])
    pos_vec[positives] = 1
    #if negatives is not None:
    #    pos_vec[negatives] = -1

    # f contains the fixed amount of score coming from positive nodes
    f = a*csr_matrix.dot(P, pos_vec)

    if remove_nonreachable is True:
        node2idx, idx2node = {}, {}
        # remove the negatives first and then remove the non-reachable nodes
        if negatives is not None:
            node2idx, idx2node = build_index_map(range(len(f)), negatives)
            P = delete_nodes(P, negatives)
            f = np.delete(f, negatives)
            #fixed_nodes = np.concatenate([positives, negatives])
            positives = set(node2idx[n] for n in positives)
        positives = set(list(positives))
        fixed_nodes = positives 

        start = time.time()
        # also remove nodes that aren't reachable from a positive 
        # find the connected_components. If a component doesn't have a positive, then remove the nodes of that component
        num_ccs, node_comp = csgraph.connected_components(P, directed=False)
        # build a dictionary of nodes in each component
        ccs = defaultdict(set)
        # check to see which components have a positive node in them
        pos_comp = set()
        for n in range(len(node_comp)):
            comp = node_comp[n]
            ccs[comp].add(n)
            if comp in pos_comp:
                continue
            if n in positives:
                pos_comp.add(comp)

        non_reachable_ccs = set(ccs.keys()) - pos_comp
        not_reachable_from_pos = set(n for cc in non_reachable_ccs for n in ccs[cc])
#        # use matrix multiplication instead
#        reachable_nodes = get_reachable_nodes(P, positives)
#        print(len(reachable_nodes), P.shape[0] - len(reachable_nodes))
        if verbose:
            print("%d nodes not reachable from a positive. Removing them from the graph" % (len(not_reachable_from_pos)))
            print("\ttook %0.4f sec" % (time.time() - start))
        # combine them to be removed
        fixed_nodes = positives | not_reachable_from_pos

        node2idx2, idx2node2 = build_index_map(range(len(f)), fixed_nodes)
        if negatives is not None:
            # change the mapping to be from the deleted nodes to the original node ids
            node2idx = {n: node2idx2[node2idx[n]] for n in node2idx if node2idx[n] in node2idx2}
            idx2node = {node2idx[n]: n for n in node2idx}
        else:
            node2idx, idx2node = node2idx2, idx2node2 
    else:
        fixed_nodes = positives 
        if negatives is not None:
            fixed_nodes = np.concatenate([positives, negatives])
        node2idx, idx2node = build_index_map(range(len(f)), set(list(fixed_nodes)))
    # removing the fixed nodes is slightly faster than selecting the unknown rows
    # remove the fixed nodes from the graph
    fixed_nodes = np.asarray(list(fixed_nodes)) if not isinstance(fixed_nodes, np.ndarray) else fixed_nodes
    if remove_nonreachable is True:
        newP = delete_nodes(P, fixed_nodes)
        # and from f
        f = np.delete(f, fixed_nodes)
    else:
        # UPDATE: Instead of deleting the nodes, which takes a long time for large matrices, 
        # just set them to 0
        newP = remove_node_edges(P, fixed_nodes)
        f[fixed_nodes] = 0
    assert newP.shape[0] == newP.shape[1], "Matrix is not square"
    assert newP.shape[1] == len(f), "f doesn't match size of P"

    if remove_nonreachable is True:
        return newP, f, node2idx, idx2node
    else:
        return newP, f


def remove_node_edges(W, nodes_idx):
    nodes = np.zeros(W.shape[0])
    nodes[nodes_idx] = 1
    # now set all of the non-annotated prot rows and columns to 0
    diag = diags(nodes)
    edges_to_remove = diag.dot(W) + W.dot(diag)
    newW = W - edges_to_remove
    # network should be in csr form. Make sure the 0s aren't left over
    newW.eliminate_zeros()
    return newW


def build_index_map(nodes, nodes_to_remove):
    """
    *returns*: a dictionary of the original node ids/indices to the current indices, as well as the reverse
    """
    # keep track of the original node integers 
    # to be able to map back to the original node names
    node2idx = {}
    idx2node = {}
    index_diff = 0
    for i in nodes:
        if i in nodes_to_remove:
            index_diff += 1
            continue
        idx2node[i - index_diff] = i
        node2idx[i] = i - index_diff

    return node2idx, idx2node 


def delete_nodes(mat, indices):
    """
    Remove the rows and columns denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask, :][:, mask]


# slightly faster than mat[indices, :][:, indices]
def select_nodes(mat, indices):
    """
    Select the rows and columns denoted by ``indices`` form the CSR sparse matrix ``mat``.
    Equivalent to getting a subnetwork of a graph
    """
    mask = np.zeros(mat.shape[0], dtype=bool)
    mask[indices] = True
    return mat[mask, :][:, mask]


# no longer needed
def parse_pos_neg_files(pos_neg_files, terms=None):
    # get the positives and negatives from the matrix
    all_term_prots = {}
    all_term_neg = {}
    for pos_neg_file in pos_neg_files:
        #term_prots, term_neg = self.parse_pos_neg_matrix(self.pos_neg_file)
        term_prots, term_neg = parse_pos_neg_file(pos_neg_file, terms=terms)
        all_term_prots.update(term_prots)
        all_term_neg.update(term_neg)

    return all_term_prots, all_term_neg


def parse_pos_neg_file(pos_neg_file, terms=None):
    print("Reading positive and negative annotations for each protein from %s" % (pos_neg_file))
    term_prots = {}
    term_neg = {}
    all_prots = set()
    # TODO possibly use pickle
    if not os.path.isfile(pos_neg_file):
        print("Warning: %s file not found" % (pos_neg_file))
        return term_prots, term_neg

        #for term, pos_neg_assignment, prots in utils.readColumns(pos_neg_file, 1,2,3):
    with open(pos_neg_file, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            term, pos_neg_assignment, prots = line.rstrip().split('\t')[:3]
            if terms and term not in terms:
                continue
            prots = set(prots.split(','))
            if int(pos_neg_assignment) == 1:
                term_prots[term] = prots
            elif int(pos_neg_assignment) == -1:
                term_neg[term] = prots

            all_prots.update(prots)

    print("\t%d GO terms, %d prots" % (len(term_prots), len(all_prots)))

    return term_prots, term_neg


def write_output(term_scores, terms, prots, out_file, 
        num_pred_to_write=10, term2idx=None):
    """
    *num_pred_to_write* can either be an integer, or a dictionary with a number of predictions to write for each term
    *term2idx*: if only a subset of terms will be written, term2idx gives the index (row) of those terms in the matrix
    """
    # make sure the output file exists
    if '/' in out_file:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
    # now write the scores to a file
    if isinstance(num_pred_to_write, dict):
        #print("\twriting top %d*num_pred scores to %s" % (kwargs['factor_pred_to_write'], out_file))
        print("\twriting top <factor>*num_pred scores to %s" % (out_file))
    else:
        print("\twriting %s scores to %s" % (
            "top %d"%num_pred_to_write if num_pred_to_write != -1 else "all", out_file))

    with open(out_file, 'w') as out:
        out.write("#term\tprot\tscore\n")
        for i, term in enumerate(terms):
            if len(terms) < term_scores.shape[0]:
                if term2idx is not None:
                    i = term2idx[term]
                else:
                    raise Exception("%d terms < %d term rows in scores matrix. Must pass term2idx dict" % (
                        len(terms), term_scores.shape[0]))
            num_to_write = num_pred_to_write
            scores = term_scores[i].toarray().flatten()
            #print("debug: %d nodes with a non-zero score" % (np.count_nonzero(scores)))
            #print(np.sort(scores)[::-1][:num_to_write])
            # convert the nodes back to their names, and make a dictionary out of the scores
            # UPDATE: only write the non-zero scores since those nodes don't have a score
            scores = {prots[j]:s for j, s in enumerate(scores) if s != 0}
            if isinstance(num_to_write, dict):
                num_to_write = num_pred_to_write[terms[i]]
            write_scores_to_file(scores, term=term, file_handle=out,
                    num_pred_to_write=int(num_to_write))


def write_scores_to_file(scores, term='', out_file=None, file_handle=None,
        num_pred_to_write=100, header="", append=True):
    """
    *scores*: dictionary of node_name: score
    *num_pred_to_write*: number of predictions (node scores) to write to the file 
        (sorted by score in decreasing order). If -1, all will be written
    """

    if num_pred_to_write == -1:
        num_pred_to_write = len(scores) 

    if out_file is not None:
        if append:
            print("Appending %d scores for term %s to %s" % (num_pred_to_write, term, out_file))
            out_type = 'a'
        else:
            print("Writing %d scores for term %s to %s" % (num_pred_to_write, term, out_file))
            out_type = 'w'

        file_handle = open(out_file, out_type)
    elif file_handle is None:
        print("Warning: out_file and file_handle are None. Not writing scores to a file")
        return 

    # write the scores to a file, up to the specified number of nodes (num_pred_to_write)
    file_handle.write(header)
    for n in sorted(scores, key=scores.get, reverse=True)[:num_pred_to_write]:
        file_handle.write("%s\t%s\t%0.4e\n" % (term, n, scores[n]))
    return

