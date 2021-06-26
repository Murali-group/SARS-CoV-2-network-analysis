
#from scipy import sparse
import numpy as np
from ..algorithms import alg_utils


def findKernelWeights(y, W):
    """
    Python implementation of the GeneMANIA findKernelWeights Matlab function
    *y*: vector of labels either -1, 0, or 1
    *W*: list of sparse matrices. Each network/kernel is assumed to be
    sparse and symmetric
    
    *returns*: list of weights alpha, and indices of the networks in W for those weights
    """

    # first get the pos and neg indices, and counds
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == -1)[0]
    num_pos = len(pos_idx)
    num_neg = len(neg_idx)
    #num_prots = len(y)
    num_networks = len(W)

    # a few checks on the arguments
    if num_pos == 0 or num_neg == 0:
        print("WARNING: 0 positive and/or 0 negative examples")
        print("\tUsing average kernel")
        return return_avg(num_networks)

    num_unk = len(np.where(y == 0)[0])
    if num_pos + num_neg + num_unk != len(y):
        print("ERROR: entries in y must be either -1, 0, or 1")
        return

    # label bias for the +/+ elements
    pos_const = 2*num_neg / float(num_pos + num_neg) 
    # label bias for the +/- elements
    neg_const = -2*num_pos / float(num_pos + num_neg) 

    # ---- Calculation of the two matrices needed to do the linear regression:
    #  KtK:  the gram matrix containing the dot products of the kernels
    #  KtT:  the dot products of the kernels with the target values
    KtK = np.zeros((num_networks+1, num_networks+1))
    KtT = np.zeros(num_networks+1)

    # temp storage of +/+ non-diagonal affinities
    Wpp = [0]*num_networks
    # temp storage of +/- non-diagonal affinities
    Wpn = [0]*num_networks
    num_pp_elem = num_pos * (num_pos - 1)
    num_pn_elem = 2 * num_pos * num_neg
    # TODO Why is this used as the pos target rather than num_neg^2 
    # as mentioned in the paper? I guess it's because of the label bias
    pp_target = pos_const**2
    pn_target = pos_const * neg_const
    #elem = num_pp_elem + num_pn_elem

    # value of the bias interactions
    biasVal = 1 / float(num_pp_elem + num_pn_elem)

    KtT[0] = biasVal * (pp_target * num_pp_elem + pn_target * num_pn_elem)
    KtK[0,0] = biasVal

    for i in range(num_networks):
        Wpp[i] = W[i][pos_idx,:][:,pos_idx]
        ## set the diagonal to 0
        #Wpp[i].setdiag(np.zeros(num_pos))
        #Wpp[i].eliminate_zeros()
        Wpn[i] = W[i][pos_idx,:][:,neg_idx]

        # sum all the entries in the matrix
        ssWpp = Wpp[i].sum()
        ssWpn = Wpn[i].sum()

        # Note: because the affinity matrices in W are symmetric, all the sums
        # and dot products involving +/- (i,j) elements are the same for the
        # -/+ (j,i) elements.  So, we simply calcuate double any sums and dot
        # products that we calculate over the +/- elements.
        KtT[i+1] = (pp_target * ssWpp) + (2 * pn_target * ssWpn)
        KtK[i+1, 0] = biasVal * (ssWpp + 2 * ssWpn)
        KtK[0, i+1] = KtK[i+1, 0]

        for j in range(i+1):
            KtK[i+1, j+1] = Wpp[i].multiply(Wpp[j]).sum() + 2*Wpn[i].multiply(Wpn[j]).sum()
            # make KtK symmetric
            KtK[j+1, i+1] = KtK[i+1, j+1]

    indices = np.arange(num_networks+1)

    # remove empty columns
    empty_cols = np.where(KtK.sum(axis=1) == 0)
    KtK = alg_utils.delete_nodes(KtK, empty_cols)
    KtT = np.delete(KtT, empty_cols)
    indices = np.delete(indices, empty_cols)

    done = False if len(indices) > 0 else True
    while not done:
        # alpha = (K'*K)^-1 (K'*T)
        try:
            alpha = np.linalg.solve(KtK, KtT)
        except np.linalg.linalg.LinAlgError as e:
            # I get a singular matrix error once in a while. 
            # I'm not sure why the matrix would be singular, but when that happens, just give the networks the same weight
            print("Warning: np.linalg.linalg.LinAlgError in findKernelWeights.py.")
            print(("Exception: %s" %(e)).rstrip())
            indices = []
            break

        # find the locations of the negative weights
        neg_weights = np.where(alpha < 0)[0]
        #print(neg_weights)
        # keep the bias term (index of 0) even if it's negative
        neg_weights = np.setdiff1d(neg_weights, np.array([0]))

        if len(neg_weights) > 0:
            # remove the negative weights from the matrix
            KtK = alg_utils.delete_nodes(KtK, neg_weights)
            KtT = np.delete(KtT, neg_weights)

            # keep track of the indices
            indices = np.delete(indices, neg_weights)
            # if all the networks were deleted, then stop
            if len(indices) == 1:
                indices = indices[1:]
                break
        else:
            done = True
            # ignore/remove the bias term
            indices = [x-1 for x in indices[1:]]
            alpha = alpha[1:]
            print("\t%d matrices chosen. Indices: %s, weights: %s" %
                  (KtK.shape[0]-1, ', '.join([str(x) for x in indices]),
                   ', '.join([str(x) for x in alpha])))

    if len(indices) == 0:
        print("\tAll kernels eliminated or empty, using average kernel")
        return return_avg(num_networks)

    return alpha, indices


def return_avg(num_networks):
    indices = np.arange(num_networks)
    # use a uniform alpha for all the networks which is the
    # average of the # of networks
    alpha = np.asarray([1/float(num_networks)]*num_networks)
    return alpha, indices
