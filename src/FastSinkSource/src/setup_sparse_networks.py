
# Script to setup network and annotations files as sparse matrices
# also used for weighting the networks with the
# Simultaneous Weight with Specific Negatives (SWSN) method

from collections import defaultdict
import os, sys, time
import networkx as nx
import numpy as np
from scipy.io import savemat, loadmat
from scipy import sparse as sp
from tqdm import tqdm
import gzip
# this warning prints out a lot when normalizing the networks due to nodes having no edges.
# RuntimeWarning: divide by zero encountered in true_divide
# Ignore it for now
import warnings
warnings.simplefilter('ignore', RuntimeWarning)
available_weight_methods = ['swsn', 'gmw', 'gmw2008', 'add']

# my local imports
from .utils import file_utils as utils
from .utils.string_utils import full_column_names, STRING_NETWORKS
from .algorithms import alg_utils as alg_utils
from .weight_networks.findKernelWeights import findKernelWeights
from .weight_networks.combineNetworksSWSN import combineNetworksSWSN


class Sparse_Networks:
    """
    An object to hold the sparse network (or sparse networks if they are to be joined later), 
        the list of nodes giving the index of each node, and a mapping from node name to index
    *sparse_networks*: either a list of scipy sparse matrices, or a single sparse matrix
        If a list is given, then the sparse matrices must be aligned (i.e., nodes and indexes match)
    *weight_method*: method to combine the networks if multiple sparse networks are given.
        Possible values: 'swsn', 'gmw', or 'add'
        'swsn': Simultaneous Weighting with Specific Negatives (all terms)
        'gmw': GeneMANIA Weighting (term-by-term). May also called gm2008
        'add': Simply add the networks together
    *unweighted*: set the edge weights to 1 for all given networks.
    *term_weights*: a dictionary of tuples containing the weights and indices to use for each term.
        Would be used instead of running 'gmw'
    """
    def __init__(self, sparse_networks, nodes, net_names=None,
                 weight_method='swsn', unweighted=False, term_weights=None, verbose=False):
        self.multi_net = False
        if isinstance(sparse_networks, list):
            if len(sparse_networks) > 1:
                self.sparse_networks = sparse_networks
                self.multi_net = True
            else:
                self.W = sparse_networks[0]
        else:
            self.W = sparse_networks
        self.nodes = nodes
        # used to map from node/prot to the index and vice versa
        self.node2idx = {n: i for i, n in enumerate(nodes)}
        self.net_names = net_names
        self.weight_method = weight_method
        self.unweighted = unweighted
        self.verbose = verbose
        # make sure the values are correct
        if self.multi_net is True:
            if weight_method.lower() not in available_weight_methods:
                print("ERROR: weight_method '%s' not recognized. Available options: '%s'.\nQuitting" % (
                    "', '".join(available_weight_methods)))
                sys.exit()
            self.weight_swsn = True if weight_method.lower() == 'swsn' else False
            self.weight_gmw = True if weight_method.lower() in ['gmw', 'gm2008'] else False
            self.term_weights = term_weights  # extra setting for gmw
            # if specified, just add them together
            if weight_method.lower() == "add":
                self.W = sparse_networks[0]
                for W in sparse_networks[1:]:
                    self.W += W
                self.multi_net = False
        else:
            self.weight_swsn = False
            self.weight_gmw = False

        # set a weight str for writing output files
        self.weight_str = '%s%s%s' % (
            '-unw' if self.unweighted else '', 
            '-gmw' if self.weight_gmw else '',
            '-swsn' if self.weight_swsn else '')

        if self.unweighted is True:
            print("\tsetting all edge weights to 1 (unweighted)")
            if self.multi_net is False:
                # convert all of the entries to 1s to "unweight" the network
                self.W = (self.W > 0).astype(int) 
            else:
                new_sparse_networks = []
                for i in range(len(self.sparse_networks)):
                    net = self.sparse_networks[i]
                    # convert all of the entries to 1s to "unweight" the network
                    net = (net > 0).astype(int) 
                    new_sparse_networks.append(net)
                self.sparse_networks = new_sparse_networks

        if self.multi_net is True:
            print("\tnormalizing the networks")
            self.normalized_nets = []
            for net in self.sparse_networks:
                self.normalized_nets.append(_net_normalize(net))

    def weight_SWSN(self, ann_matrix):
        self.W, self.swsn_time, self.swsn_weights = weight_SWSN(
            ann_matrix, normalized_nets=self.normalized_nets, 
            net_names=self.net_names, nodes=self.nodes, verbose=self.verbose)
        return self.W, self.swsn_time

    def combine_using_weights(self, weights):
        """ Combine the different networks using the specified weights
        *weights*: list of weights, one for each network
        """
        assert len(weights) == len(self.normalized_nets), \
            "%d weights supplied not enough for %d nets" % (len(weights), len(self.normalized_nets))
        combined_network = weights[0]*self.normalized_nets[0]
        for i, w in enumerate(weights):
            combined_network += w*self.normalized_nets[i] 
        return combined_network

    def weight_GMW(self, y, term=None):
        if self.term_weights and term in self.term_weights:
            weights = self.term_weights[term]
            W = self.combine_using_weights(weights)
            process_time = 0
        else:
            W, process_time, weights = weight_GMW(y, self.normalized_nets, self.net_names, term=term) 
        return W, process_time, weights

    def save_net(self, out_file):
        print("Writing %s" % (out_file))
        utils.checkDir(os.path.dirname(out_file))
        if out_file.endswith('.npz'):
            # when the net was loaded, the idx file was already written
            # so no need to write it again
            sp.save_npz(out_file, self.W_SWSN)
        else:
            # convert the adjacency matrix to an edgelist
            G = nx.from_scipy_sparse_matrix(self.W_SWSN)
            idx2node = {i: n for i, n in enumerate(self.nodes)}
            # see also convert_node_labels_to_integers
            G = nx.relabel_nodes(G, idx2node, copy=False)
            delimiter = '\t'
            if out_file.endswith('.csv'):
                delimiter = ','
            nx.write_weighted_edgelist(G, out_file, delimiter=delimiter)


class Sparse_Annotations:
    """
    An object to hold the sparse annotations (including negative examples as -1),
        the list of GO term IDs giving the index of each term, and a mapping from term to index
    """
    # TODO add the DAG matrix
    def __init__(self, ann_matrix, terms, prots):
        self.ann_matrix = ann_matrix
        self.terms = terms
        # used to map from index to term and vice versa
        self.term2idx = {g: i for i, g in enumerate(terms)}
        self.prots = prots
        # used to map from node/prot to the index and vice versa
        self.node2idx = {n: i for i, n in enumerate(prots)}

        #self.eval_ann_matrix = None 
        #if pos_neg_file_eval is not None:
        #    self.add_eval_ann_matrix(pos_neg_file_eval)

    #def add_eval_ann_matrix(self, pos_neg_file_eval):
    #    self.ann_matrix, self.terms, self.eval_ann_matrix = setup_eval_ann(
    #        pos_neg_file_eval, self.ann_matrix, self.terms, self.prots)
    #    # update the term2idx mapping
    #    self.term2idx = {g: i for i, g in enumerate(self.terms)}

    def reshape_to_prots(self, new_prots):
        """ *new_prots*: list of prots to which the cols should be changed (for example, to align to a network)
        """
        print("\treshaping %d prots to %d prots (%d in common)" % (
            len(self.prots), len(new_prots), len(set(self.prots) & set(new_prots))))
        # reshape the matrix cols to the new prots
        # put the prots on the rows to make the switch
        new_ann_mat = sp.lil_matrix((len(new_prots), self.ann_matrix.shape[0]))
        ann_matrix = self.ann_matrix.T.tocsr()
        for i, p in enumerate(new_prots):
            idx = self.node2idx.get(p)
            if idx is not None:
                new_ann_mat[i] = ann_matrix[idx]
        # now transpose back to term rows and prot cols
        self.ann_matrix = new_ann_mat.tocsc().T.tocsr()
        self.prots = new_prots
        # reset the index mapping
        self.node2idx = {n: i for i, n in enumerate(self.prots)}

    def limit_to_terms(self, terms_list):
        """ *terms_list*: list of terms. Data from rows not in this list of terms will be removed
        """
        terms_idx = [self.term2idx[t] for t in terms_list if t in self.term2idx]
        print("\tlimiting data in annotation matrix from %d terms to %d" % (len(self.terms), len(terms_idx)))
        num_pos = len((self.ann_matrix > 0).astype(int).data)
        terms = np.zeros(len(self.terms))
        terms[terms_idx] = 1
        diag = sp.diags(terms)
        self.ann_matrix = diag.dot(self.ann_matrix)
        print("\t%d pos annotations reduced to %d" % (
            num_pos, len((self.ann_matrix > 0).astype(int).data)))

    def limit_to_prots(self, prots):
        """ *prots*: array with 1s at selected prots, 0s at other indices
        """
        diag = sp.diags(prots)
        self.ann_matrix = self.ann_matrix.dot(diag)


def get_net_out_str(net_files, string_net_files=None, string_nets=None):
    #num_networks = len(net_files) + len(string_nets)
    net_files_str = '-'.join(os.path.basename(f).split('.')[0] for f in net_files)+'-' \
                    if len(net_files) > 0 else ""
    # if there is only 1 string network, then write the name instead of the number
    string_nets_str = "" 
    if string_net_files is not None and len(string_net_files) > 0 and \
       string_nets is not None:
        if len(string_nets) == 1:
            string_nets_str = list(string_nets)[0] + '-'
        elif len(string_nets) > 1:
            string_nets_str = "string%d-" % (len(string_nets)) 

    net_str = net_files_str + string_nets_str
    return net_str


def create_sparse_net_file(
        out_pref, net_files=[], string_net_files=[], 
        string_nets=STRING_NETWORKS, string_cutoff=None, forcenet=False):
    if net_files is None:
        net_files = []
    # if there aren't any string net files, then set the string nets to empty
    if len(string_net_files) == 0:
        string_nets = [] 
    # if there are string_net_files, and string_nets is None, set it back to its default
    elif string_nets is None:
        string_nets = STRING_NETWORKS
    string_nets = list(string_nets)
    net_str = get_net_out_str(net_files, string_net_files, string_nets)
    out_pref += net_str
    sparse_nets_file = "%ssparse-nets.mat" % (
        out_pref)
    # the node IDs should be the same for each of the networks,
    # so no need to include the # in the ids file
    node_ids_file = "%snode-ids.txt" % (out_pref)
    net_names_file = "%snet-names.txt" % (out_pref)
    if forcenet is False \
       and os.path.isfile(sparse_nets_file) and os.path.isfile(node_ids_file) \
       and os.path.isfile(net_names_file):
        # read the files
        print("\treading sparse nets from %s" % (sparse_nets_file))
        sparse_networks = list(loadmat(sparse_nets_file)['Networks'][0])
        print("\treading node ids file from %s" % (node_ids_file))
        nodes = utils.readItemList(node_ids_file, 1)
        print("\treading network_names from %s" % (net_names_file))
        network_names = utils.readItemList(net_names_file, 1)

    else:
        print("\tcreating sparse nets and writing to %s" % (sparse_nets_file))
        sparse_networks, network_names, nodes = setup_sparse_networks(
            net_files=net_files, string_net_files=string_net_files, string_nets=string_nets, string_cutoff=string_cutoff)

        # now write them to a file
        write_sparse_net_file(
            sparse_networks, sparse_nets_file, network_names,
            net_names_file, nodes, node_ids_file)

    return sparse_networks, network_names, nodes


def write_sparse_net_file(
        sparse_networks, out_file, network_names,
        net_names_file, nodes, node_ids_file):
    #out_file = "%s/%s-net.mat" % (out_dir, version)
    # save this graph into its own matlab file because it doesn't change by GO category
    print("\twriting sparse networks to %s" % (out_file))
    mat_networks = np.zeros(len(sparse_networks), dtype=np.object)
    for i, net in enumerate(network_names):
        # convert to float otherwise matlab won't parse it correctly
        # see here: https://github.com/scipy/scipy/issues/5028
        mat_networks[i] = sparse_networks[i].astype(float)
    savemat(out_file, {"Networks":mat_networks}, do_compression=True)

    print("\twriting node2idx labels to %s" % (node_ids_file))
    with open(node_ids_file, 'w') as out:
        out.write(''.join(["%s\t%s\n" % (n, i) for i, n in enumerate(nodes)]))

    print("\twriting network names, which can be used to figure out " +
          "which network is at which index in the sparse-nets file, to %s" % (net_names_file))
    with open(net_names_file, 'w') as out:
        out.write(''.join(["%s\n" % (n) for n in network_names]))


def setup_sparse_networks(net_files=[], string_net_files=[], string_nets=[], string_cutoff=None):
    """
    Function to setup networks as sparse matrices 
    *net_files*: list of networks for which to make into a sparse
        matrix. The name of the file will be the name of the sparse matrix
    *string_net_files*: List of string files containing all 14 STRING network columns
    *string_nets*: List of STRING network column names for which to make a sparse matrix. 
    *string_cutoff*: Cutoff to use for the STRING combined network column (last)

    *returns*: List of sparse networks, list of network names, 
        list of proteins in the order they're in in the sparse networks
    """

    network_names = []
    # TODO build the sparse matrices without using networkx
    # I would need to make IDs for all proteins to ensure the IDs and
    # dimensions are the same for all of the matrices
    G = nx.Graph()
    for net_file in tqdm(net_files):
        name = os.path.basename(net_file)
        network_names.append(name)
        tqdm.write("Reading network from %s. Giving the name '%s'" % (net_file, name))
        open_func = gzip.open if '.gz' in net_file else open
        with open_func(net_file, 'r') as f:
            for line in f:
                line = line.decode() if '.gz' in net_file else line
                if line[0] == "#":
                    continue
                #u,v,w = line.rstrip().split('\t')[:3]
                line = line.rstrip().split('\t')
                u,v = line[:2]
                w = line[2] if len(line) > 2 else 1
                G.add_edge(u,v,**{name:float(w)})

    network_names += string_nets
    print("Reading %d STRING networks" % len(string_net_files))
    # for now, group all of the species string networks into a
    # massive network for each of the string_nets specified 
    for string_net_file in tqdm(string_net_files):
        tqdm.write("Reading network from %s" % (string_net_file))
        open_func = gzip.open if '.gz' in string_net_file else open
        with open_func(string_net_file, 'r') as f:
            for line in f:
                line = line.decode() if '.gz' in string_net_file else line
                if line[0] == "#":
                    continue
                #u,v,w = line.rstrip().split('\t')[:3]
                line = line.rstrip().split('\t')
                u,v = line[:2]
                attr_dict = {}
                combined_score = float(line[-1])
                # first check if the combined score is above the cutoff
                if string_cutoff is not None and combined_score < string_cutoff:
                    continue
                for net in string_nets:
                    w = float(line[full_column_names[net]-1])
                    if w > 0:
                        attr_dict[net] = w
                # if the edge already exists, 
                # the old attributes will still be retained
                G.add_edge(u,v,**attr_dict)
    print("\t%d nodes and %d edges" % (G.number_of_nodes(), G.number_of_edges()))

    print("\trelabeling node IDs with integers")
    G, node2idx, idx2node = convert_nodes_to_int(G)
    # keep track of the ordering for later
    nodes = [idx2node[n] for n in sorted(idx2node)]

    print("\tconverting graph to sparse matrices")
    sparse_networks = []
    net_names = []
    for i, net in enumerate(tqdm(network_names)):
        # all of the edges that don't have a weight for the specified network will be given a weight of 1
        # get a subnetwork with the edges that have a weight for this network
        print("\tgetting subnetwork for '%s'" % (net))
        netG = nx.Graph()
        netG.add_weighted_edges_from([(u,v,w) for u,v,w in G.edges(data=net) if w is not None])
        # skip this network if it has no edges, or leave it empty(?)
        if netG.number_of_edges() == 0:
            print("\t0 edges. skipping.")
            continue
        # now convert it to a sparse matrix. The nodelist will make sure they're all the same dimensions
        sparse_matrix = nx.to_scipy_sparse_matrix(netG, nodelist=sorted(idx2node))
        # convert to float otherwise matlab won't parse it correctly
        # see here: https://github.com/scipy/scipy/issues/5028
        sparse_matrix = sparse_matrix.astype(float) 
        sparse_networks.append(sparse_matrix)
        net_names.append(net) 

    return sparse_networks, net_names, nodes


def convert_nodes_to_int(G):
    index = 0 
    node2int = {}
    int2node = {}
    for n in sorted(G.nodes()):
        node2int[n] = index
        int2node[index] = n 
        index += 1
    # see also convert_node_labels_to_integers
    G = nx.relabel_nodes(G,node2int, copy=False)
    return G, node2int, int2node


def create_sparse_ann_and_align_to_net(
        pos_neg_file, sparse_ann_file, net_prots,
        forced=False, verbose=False, **kwargs):
    """
    Wrapper around create_sparse_ann_file that also runs Youngs Negatives (potentially RAM heavy)
    and aligns the ann_matrix to a given network, both of which can be time consuming
    and stores those results to a file
    """ 
    if not kwargs.get('forcenet') and os.path.isfile(sparse_ann_file):
        print("Reading annotation matrix from %s" % (sparse_ann_file))
        loaded_data = np.load(sparse_ann_file, allow_pickle=True)
        ann_matrix = make_csr_from_components(loaded_data['arr_0'])
        terms, prots = loaded_data['arr_1'], loaded_data['arr_2']
        ann_obj = Sparse_Annotations(ann_matrix, terms, prots)
    else:
        ann_matrix, terms, ann_prots = create_sparse_ann_file(pos_neg_file, **kwargs)
        ann_obj = Sparse_Annotations(ann_matrix, terms, ann_prots)
        # align the ann_matrix prots with the prots in the network
        ann_obj.reshape_to_prots(net_prots)

        print("Writing sparse annotations to %s" % (sparse_ann_file))
        os.makedirs(os.path.dirname(sparse_ann_file), exist_ok=True)
        # store all the data in the same file
        ann_matrix_data = get_csr_components(ann_obj.ann_matrix)
        np.savez_compressed(
            sparse_ann_file, ann_matrix_data, ann_obj.terms, ann_obj.prots)
    return ann_obj


def create_sparse_ann_file(
        pos_neg_file, forced=False, verbose=False, **kwargs):
    """
    Store/load the annotation matrix, terms and prots. 
    The DAG and annotation matrix will be aligned, and the prots will not be limitted to a network since the network can change.
    The DAG should be the same DAG that was used to generate the pos_neg_file
    *returns*:
        1) ann_matrix: A matrix with term rows, protein/node columns, and 1,0,-1 for pos,unk,neg values
        2) terms: row labels
        3) prots: column labels
    """
    sparse_ann_file = pos_neg_file + '.npz'

    if forced or not os.path.isfile(sparse_ann_file):
        # load the pos_neg_file first. Should have only one hierarchy (e.g., BP)
        ann_matrix, terms, prots = setup_sparse_annotations(pos_neg_file)

        print("\twriting sparse annotations to %s" % (sparse_ann_file))
        # store all the data in the same file
        ann_matrix_data = get_csr_components(ann_matrix)
        np.savez_compressed(
            sparse_ann_file, ann_matrix_data, terms, prots)
    else:
        print("\nReading annotation matrix from %s" % (sparse_ann_file))
        loaded_data = np.load(sparse_ann_file, allow_pickle=True)
        ann_matrix = make_csr_from_components(loaded_data['arr_0'])
        terms, prots = loaded_data['arr_1'], loaded_data['arr_2']
        #ann_matrix = make_csr_from_components(loaded_data['ann_matrix_data'])
        #terms, prots = loaded_data['terms'], loaded_data['prots']

    return ann_matrix, terms, prots


def setup_sparse_annotations(pos_neg_file):
    """
    
    *returns*: 1) A matrix with term rows, protein/node columns, and 1,0,-1 for pos,unk,neg values
        2) List of terms in the order in which they appear in the matrix
        3) List of prots in the order in which they appear in the matrix
    """
    print("\nSetting up annotation matrix")

    print("Reading positive and negative annotations for each protein from %s" % (pos_neg_file))
    if '-list' in pos_neg_file:
        ann_matrix, terms, prots = read_pos_neg_list_file(pos_neg_file) 
    else:
        ann_matrix, terms, prots = read_pos_neg_table_file(pos_neg_file) 
    num_pos = len((ann_matrix > 0).astype(int).data)
    num_neg = len(ann_matrix.data) - num_pos
    print("\t%d terms, %d prots, %d annotations. %d positives, %d negatives" % (
        ann_matrix.shape[0], ann_matrix.shape[1], len(ann_matrix.data), num_pos, num_neg))

    return ann_matrix, terms, prots


def read_pos_neg_table_file(pos_neg_file):
    """
    Reads a tab-delimited file with prots on rows, terms on columns, and 0,1,-1 as values
    *returns*: 1) A matrix with term rows, protein/node columns, and 1,0,-1 for pos,unk,neg values
        2) List of terms in the order in which they appear in the matrix
        3) List of prots in the order in which they appear in the matrix
    """
    # rather than explicity building the matrix, use the indices to build a coordinate matrix
    # rows are prots, cols are terms
    i_list = []
    j_list = []
    data = []
    terms = []
    prots = []
    i = 0

    # read the file to build the matrix
    open_func = gzip.open if '.gz' in pos_neg_file else open
    with open_func(pos_neg_file, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.decode() if '.gz' in pos_neg_file else line
            if line[0] == '#':
                continue
            line = line.rstrip().split('\t')
            if line_idx == 0:
                # this is the header line
                terms = line[1:]
                continue

            prot, vals = line[0], line[1:]
            prots.append(prot)
            for j, val in enumerate(vals):
                # don't need to store 0 values
                if int(val) != 0:
                    i_list.append(i)
                    j_list.append(j)
                    data.append(int(val))
            i += 1

    # convert it to a sparse matrix 
    print("Building a sparse matrix of annotations")
    ann_matrix = sp.coo_matrix((data, (i_list, j_list)), shape=(len(prots), len(terms)), dtype=float).tocsr()
    ann_matrix = ann_matrix.transpose().tocsr()
    return ann_matrix, terms, prots


# keeping this for backwards compatibility
def read_pos_neg_list_file(pos_neg_file):
    """
    Reads a tab-delimited file with two lines per term. A positives line, and a negatives line
        Each line has 3 columns: term, pos/neg assignment (1 or -1), and a comma-separated list of prots
    *returns*: 1) A matrix with term rows, protein/node columns, and
        1,0,-1 for pos,unk,neg values
        2) List of terms in the order in which they appear in the matrix
        3) List of prots in the order in which they appear in the matrix
    """
    # rather than explicity building the matrix, use the indices to build a coordinate matrix
    # rows are prots, cols are terms
    i_list = []
    j_list = []
    data = []
    terms = []
    prots = []
    # this will track the index of the prots
    node2idx = {}
    i = 0
    j = 0
    # read the file to build the matrix
    open_func = gzip.open if '.gz' in pos_neg_file else open
    with open_func(pos_neg_file, 'r') as f:
        for line in f:
            line = line.decode() if '.gz' in pos_neg_file else line
            if line[0] == '#':
                continue
            term, pos_neg_assignment, curr_prots = line.rstrip().split('\t')[:3]
            curr_idx_list = []
            for prot in curr_prots.split(','):
                prot_idx = node2idx.get(prot)
                if prot_idx is None:
                    prot_idx = i 
                    node2idx[prot] = i
                    prots.append(prot)
                    i += 1
                curr_idx_list.append(prot_idx)
            # the file has two lines per term. A positives line, and a negatives line
            for idx in curr_idx_list:
                i_list.append(idx)
                j_list.append(j)
                data.append(int(pos_neg_assignment))
            if int(pos_neg_assignment) == -1:
                terms.append(term)
                j += 1

    # convert it to a sparse matrix 
    print("Building a sparse matrix of annotations")
    ann_matrix = sp.coo_matrix((data, (i_list, j_list)), shape=(len(prots), len(terms)), dtype=float).tocsr()
    ann_matrix = ann_matrix.transpose()
    return ann_matrix, terms, prots


def weight_GMW(y, normalized_nets, net_names=None, term=None):
    """ TODO DOC
    """
    start_time = time.process_time()
    if term is not None:
        print("\tterm %s: %d positives, %d negatives" % (term, len(np.where(y == 1)[0]), len(np.where(y == -1)[0])))
    alphas, indices = findKernelWeights(y, normalized_nets)
    # print out the computed weights for each network
    if net_names is not None:
        print("\tnetwork weights: %s\n" % (', '.join(
            "%s: %s" % (net_names[x], alphas[i]) for
            i, x in enumerate(indices))))

    weights_list = [0]*len(normalized_nets)
    weights_list[indices[0]] = alphas[0]
    # now add the networks together with the alpha weight applied
    combined_network = alphas[0]*normalized_nets[indices[0]]
    for i in range(1,len(alphas)):
        combined_network += alphas[i]*normalized_nets[indices[i]] 
        weights_list[indices[i]] = alphas[i] 
    total_time = time.process_time() - start_time

    # don't write each term's combined network to a file
    return combined_network, total_time, weights_list


def weight_SWSN(ann_matrix, sparse_nets=None, normalized_nets=None, net_names=None,
                out_file=None, nodes=None, verbose=False):
    """ 
    *normalized_nets*: list of networks stored as scipy sparse matrices. Should already be normalized
    """
    # UPDATED: normalize the networks
    if sparse_nets is not None:
        print("Normalizing the networks")
        normalized_nets = []
        for net in sparse_nets:
            normalized_nets.append(_net_normalize(net))
    elif normalized_nets is None:
        print("No networks given. Nothing to do")
        return None, 0
    if len(normalized_nets) == 1:
        print("Only one network given to weight_SWSN. Nothing to do.")
        total_time = 0
        return sparse_nets[0], total_time
    if verbose:
        print("Removing rows with 0 annotations/positives")
        utils.print_memory_usage()
    # remove rows with 0 annotations/positives
    empty_rows = []
    for i in range(ann_matrix.shape[0]):
        pos, neg = alg_utils.get_term_pos_neg(ann_matrix, i)
        # the combineWeightsSWSN method doesn't seem to
        # work if there's only 1 positive
        if len(pos) <= 1 or len(neg) <= 1:
            empty_rows.append(i)
    # don't modify the original annotation matrix to keep the rows matching the GO ids
    curr_ann_mat = delete_rows_csr(ann_matrix.tocsr(), empty_rows)

    if verbose:
        utils.print_memory_usage()
    print("Weighting networks for %d different GO terms" % (curr_ann_mat.shape[0]))
    print("Running simultaneous weights with specific negatives")
    start_time = time.process_time()
    alpha, indices = combineNetworksSWSN(curr_ann_mat, normalized_nets, verbose=verbose) 
    # print out the computed weights for each network
    if net_names is not None:
        print("network weights:")
        #print("\tnetworks chosen: %s" % (', '.join([net_names[i] for i in indices])))
        weights = defaultdict(int)
        for i in range(len(alpha)):
            weights[net_names[indices[i]]] = alpha[i]
        weights_table = ["%0.3e"%weights[net] for net in net_names]
        print('\t'.join(net_names))
        print('\t'.join(weights_table))

    # now add the networks together with the alpha weight applied
    weights_list = [0]*len(normalized_nets)
    weights_list[indices[0]] = alpha[0]
    combined_network = alpha[0]*normalized_nets[indices[0]]
    for i in range(1,len(alpha)):
        combined_network += alpha[i]*normalized_nets[indices[i]] 
        weights_list[indices[i]] = alpha[i] 
    total_time = time.process_time() - start_time

    if out_file is not None:
        # replace the .txt if present 
        out_file = out_file.replace('.txt', '.npz')
        utils.checkDir(os.path.dirname(out_file))
        print("\twriting combined network to %s" % (out_file))
        sp.save_npz(out_file, combined_network)
        # also write the node ids so it's easier to access
        # TODO figure out a better way to store this
        node2idx_file = out_file + "-node-ids.txt"
        print("\twriting node ids to %s" % (node2idx_file)) 
        with open(node2idx_file, 'w') as out:
            out.write(''.join("%s\t%s\n" % (n, i) for i, n in enumerate(nodes)))

        # write the alpha/weight of the networks as well
        net_weight_file = out_file + "-net-weights.txt"
        print("\twriting network weights to %s" % (net_weight_file)) 
        with open(net_weight_file, 'w') as out:
            out.write(''.join("%s\t%s\n" % (net_names[idx], str(alpha[i])) for i, idx in enumerate(indices)))

    return combined_network, total_time, weights_list


# copied from here: https://stackoverflow.com/a/26504995
def delete_rows_csr(mat, indices):
    """ 
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, sp.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


# this was mostly copied from the deepNF preprocessing script
def _net_normalize(X):
    """ 
    Normalizing networks according to node degrees.
    """
    # normalizing the matrix
    deg = X.sum(axis=1).A.flatten()
    deg = np.divide(1., np.sqrt(deg))
    deg[np.isinf(deg)] = 0 
    # sparse matrix function to make a diagonal matrix
    D = sp.spdiags(deg, 0, X.shape[0], X.shape[1], format="csr")
    X = D.dot(X.dot(D))

    return X


# small utility functions for working with the pieces of
# sparse matrices when saving to or loading from a file
def get_csr_components(A):
    all_data = np.asarray([A.data, A.indices, A.indptr, A.shape], dtype=object)
    return all_data


def make_csr_from_components(all_data):
    return sp.csr_matrix((all_data[0], all_data[1], all_data[2]), shape=all_data[3])
