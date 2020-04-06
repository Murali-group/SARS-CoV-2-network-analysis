
import numpy as np
import src.algorithms.alg_utils as alg_utils
import src.utils.file_utils as utils
import scipy.sparse as sp
# for some reason, this warning prints out way too much. 
# Ignore it for now
import warnings
warnings.simplefilter('ignore', sp.SparseEfficiencyWarning)
from tqdm import trange

__author__ = "Jeff Law"

# citation:
#Youngs, N., Penfold-Brown, D., Drew, K., Shasha, D., & Bonneau, R., Parametric bayesian priors and better choice of negative examples improve protein function prediction, Bioinformatics, 29(9), 1190-1198 (2013).  http://dx.doi.org/10.1093/bioinformatics/btt110
# see http://bonneaulab.bio.nyu.edu/funcprop.html for the original matlab code


def combineNetworksSWSN(y, W, verbose=False):
    """
    Python implementation of the combineNetworksSWSN (Simultaneous Weights with Specific Negatives) Matlab function.
    Used to find appropriate weights for networks with multiple GO term annotations
    *y*: multiple vectors of labels either -1, 0, or 1
    *W*: list of sparse matrices. Each network/kernel is assumed to be
    sparse and symmetric
    
    *returns*: list of weights alpha, and indices of the networks in W for those weights
    """

    num_terms, num_prots = y.shape
    num_networks = len(W)
    # numpy doesn't allow storing arrays of various sizes in an array
    # so just store them in lists and convert to an array later
    omegas = [0]*num_terms
    t_list = [0]*num_terms

    # Calculate the # of times i is positive
    # and the # of times (i,j) is positive
    # build the t vectors for each category
    for i in trange(num_terms, disable=False if verbose else True):
        # TODO the matlab code has the y matrix transformed
        curr_y = y[i].toarray()[0]
        pos_idx = np.where(curr_y > 0)[0]
        neg_idx = np.where(curr_y < 0)[0]
        num_pos = len(pos_idx)
        num_neg = len(neg_idx)
        #print("%d pos, %d neg" % (num_pos, num_neg))
        num_labeled = num_pos + num_neg

        # a few checks on the arguments
        if num_pos == 0 or num_neg == 0:
            print("ERROR: 0 positive and/or 0 negative examples")
            return

        num_unk = len(np.where(curr_y == 0)[0])
        if num_pos + num_neg + num_unk != len(curr_y):
            print("ERROR: entries in y must be either -1, 0, or 1")
            return

        # Jeff: get the values in the upper right triangle of the positive locations
        # and stacks them in a vector
        # I'm assuming this avoids performing double the computations
        pos_target = (num_neg / float(num_labeled))**2
        neg_target = -(num_neg*num_pos / float(num_labeled**2))
        # TODO I shouldn't need to store these directly,
        # just compute and store the sum of omegasTt 
        pos_t = np.ones(int(num_pos*(num_pos-1)/2)) * pos_target
        neg_t = np.ones(num_neg*num_pos) * neg_target
        t_list[i] = np.append(pos_t, neg_t)

        # TODO same with the omegas.
        # I should be able to just store the omegasTomegas
        # each index of omegas is a matrix with m=#pos+#neg rows and n=# nets columns
        omegas[i] = sp.csc_matrix((len(t_list[i]), num_networks))
        for j in range(num_networks):
            Wpp = W[j][pos_idx,:][:,pos_idx]
            # get just the values above the diagonal
            pos_omega_col = np.asarray(Wpp[np.triu_indices(Wpp.shape[0],1)]).flatten()
            Wpn = W[j][pos_idx,:][:,neg_idx]
            # stack the columns on top of each other
            neg_omega_col = Wpn.toarray().flatten('F')
            # set the column of the matrix to this network 
            omegas[i][:,j] = sp.csc_matrix(np.asarray([np.append(pos_omega_col, neg_omega_col)]).T)
            #omegas[i][:,j] = vstack([csc_matrix(pos_omega_col.T), neg_omega_col])
        #omegas[i] = omegas[i].tocsr()

    viableIndices = np.arange(num_networks)
    done = False
    # solve for the minimum of the objective function obtained by adding up all
    # of the distinct omegas and t vectors across categories. Iterate until
    # we have a satisfactory solution with no negative weights
    while not done:
        if verbose:
            print("\t%d networks" % (len(viableIndices)))
        omegaTomega = np.zeros((len(viableIndices)+1, len(viableIndices)+1))
        omegaTt = np.zeros(len(viableIndices)+1)

        # calculate the sum of all of the omegas
        for i in range(num_terms):

            omegaTomega[0,0] = omegaTomega[0,0] + 1 / float(omegas[i].shape[0])
            curr_omega = omegas[i][:,viableIndices]

            # first add the bias term
            tmp = np.asarray(curr_omega.sum(axis=0)).flatten() / float(omegas[i].shape[0])
            # add to the first row and first column
            omegaTomega[0][1:] += tmp
            omegaTomega[:,0][1:] += tmp
            # add the transpose * itself
            omegaTomega[1:,1:] += curr_omega.T.dot(curr_omega)

            omegaTt[0] += t_list[i].sum() / float(omegas[i].shape[0])
            omegaTt[1:] += curr_omega.T.dot(t_list[i])

        # drop any columns that end up as zero
        ss = omegaTomega.sum(axis=1)
        # TODO test using epsilon vs actual 0
        #good_idx = np.where(ss > sys.float_info.epsilon * max(ss))
        empty_cols = np.where(ss == 0)[0]
        #omegaTomega = omegaTomega[good_idx:,][:,good_idx]
        #omegaTt = omegaTt[good_idx]
        #viableIndices = viableIndices[good_idx[1:]-1]
        omegaTomega = alg_utils.delete_nodes(omegaTomega, empty_cols)
        omegaTt = np.delete(omegaTt, empty_cols)
        viableIndices = np.delete(viableIndices, empty_cols-1)

        # now solve for the optimal weights
        alphaStar = np.linalg.solve(omegaTomega, omegaTt)

        # drop negative weights, and finish if we have all positive weights or
        # if we have dropped all the weights
        neg_weights = np.where(alphaStar < 0)[0]
        #print(neg_weights)
        # keep the bias term (index of 0) even if it's negative
        neg_weights = np.setdiff1d(neg_weights, np.array([0]))
        #print("\t%d empty columns, %d neg weights" % (len(empty_cols), len(neg_weights)))
        if verbose:
            print("\talpha: %s" % (', '.join(["%0.3e"%x for x in alphaStar])))
        if len(neg_weights) > 0:
            viableIndices = np.delete(viableIndices, neg_weights-1)

        if len(viableIndices) == len(alphaStar)-1 or len(viableIndices) == 0:
            done = True

    if len(viableIndices) != 0:
        # ignore/remove the bias term
        alpha = alphaStar[1:]
        indices = viableIndices
        if verbose:
            print("\t%d matrices chosen. Indices: %s, weights: %s\n" %
                (len(viableIndices), ', '.join([str(x) for x in indices]),
                ', '.join(["%0.3e"%x for x in alpha])))
    else:
        indices = np.arange(num_networks)
        # use a uniform alpha for all the networks which is the
        # average of the # of networks
        alpha = np.asarray([1/float(num_networks)]*num_networks)
        print("\tAll kernels eliminated or empty, " +
              "assigning an average weight for each kernel: %0.3f\n" % (alpha[0]))
    if verbose:
        utils.print_memory_usage()
    return alpha, indices
