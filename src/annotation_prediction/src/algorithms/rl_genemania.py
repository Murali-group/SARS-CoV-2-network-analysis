# Python implementation of the Regularlized Laplacian (RL) method, and GeneMANIA
# since they are very similar
__author__ = "Jeff Law"

import os
import time
import numpy as np
from scipy.linalg import inv
from scipy.sparse import eye, diags
from scipy.sparse.linalg import spsolve, cg, spilu, LinearOperator

from . import alg_utils


def setup_laplacian(W):
    # first normalize the network
    P = alg_utils._net_normalize(W)
    # take the column sum and set them as the diagonals of a matrix
    deg = np.asarray(P.sum(axis=0)).flatten()
    deg[np.isinf(deg)] = 0
    D = diags(deg)
    L = D - P
    return L


def runGeneMANIA(L, y, alpha=1, tol=1e-05, Milu=None, verbose=False):
    """
    *L*: Laplacian of the original network
    *y*: vector of positive and negative assignments. 
         If y does not contain negatives, will be run as GeneManiaPlus, also known as Regularized Laplacian (RL). 
    *alpha*: parameter between 0 and 1 to control the influence of neighbors in the network.
        0 would ignore the network completely, and nodes would get their original score.
    *tol*: Conjugate Gradient tolerance for convergance

    *returns*: scores array, process_time, wall_time
        # of iterations needed to converge
    """
    y = y.copy()
    # setup the y vector with the value for the unknowns
    num_pos = len(np.where(y == 1)[0])
    num_neg = len(np.where(y == -1)[0])
    if num_pos == 0:
        print("WARNING: No positive examples given. Skipping.")
        return np.zeros(len(y)), 0,0,0
    # if there are no negative examples, 
    # then leave the unknown examples at 0
    if num_neg == 0:
        pass
    # otherwise, set the unknown examples to the value k
    else:
        # taken from the GeneMANIA paper
        k = (num_pos - num_neg) / float(num_pos + num_neg)
        y[np.where(y == 0)[0]] = k

    # this measures the amount of time taken by all processors
    # and Conjugate Gradient is paralelized
    start_process_time = time.process_time()
    # this measures the amount of time that has passed
    start_wall_time = time.time()
    M = eye(L.shape[0]) + alpha*L

    # keep track of the number of iterations
    num_iters = 0
    def callback(xk):
        # keep the reference to the variable within the callback function (Python 3)
        nonlocal num_iters
        num_iters += 1

    # use scipy's conjugate gradient solver
    f, info = cg(M, y, tol=tol, M=Milu, callback=callback)
    process_time = time.process_time() - start_process_time
    wall_time = time.time() - start_wall_time
    if verbose:
        print("Solved GeneMANIA using conjugate gradient (%0.2f sec, %0.2f sec process_time). Info: %s, k=%0.6f, iters=%d" % (
            wall_time, process_time, str(info), k, num_iters))

    return f, process_time, wall_time, num_iters


def get_diffusion_matrix(W, alpha=1.0, diff_mat_file=None):
    """
    Generate the diffusion/propagation matrix of a network by taking the inverse of the laplaciain,
    also known as the Regularized Laplacian (RL)
    *Note that the result is a dense matrix*

    *W*: scipy sparse matrix representation of a network
    *alpha*: value of alpha for propagation
    *diff_mat_file*: path/to/file to store the RL (example: "%sdiffusion-mat-a%s.npy" % (net_obj.out_pref, str(alpha).replace('.','_')))
    """
    if diff_mat_file is not None and os.path.isfile(diff_mat_file):
        # read in the diffusion mat file
        print("Reading %s" % (diff_mat_file))
        return np.load(diff_mat_file)

    # now get the laplacian
    L = setup_laplacian(W)
    # the equation is (I + a*L)s = y
    # we want to solve for (I + a*L)^-1
    M = eye(L.shape[0]) + alpha*L 
    print("computing the inverse of (I + a*L) as the diffusion matrix, for alpha=%s" % (alpha))
    # first convert the scipy sparse matrix to a numpy matrix
    M_full = M.A
    # now take the inverse
    M_inv = inv(M_full)

    # write to file so this doesn't have to be recomputed
    if diff_mat_file is not None:
        print("Writing to %s" % (diff_mat_file))
        np.save(diff_mat_file, M_inv)

    return M_inv


def get_pred_main_contributors(
        pred_scores, M_inv, pos_idx,
        cutoff=0.001, k=332, W=None, **kwargs):  #clustering=False):
    """
    Get the main contributors for each top prediction. 
    Now normalize each row, and get all nodes with a value > some cutoff
    *pred_scores*: prediction scores from running RL / GeneMANIA
    *M_inv*: inverse of the laplacian
    *pos_idx*: indexes of positive examples
    *cutoff*: if a positive example contributes to the fraction of propagation score of a given node > *cutoff*,
        it will be a main contributor for that node
    *k*: number of top predictions to consider
    *W*: original network. If passed in, will compute the fraction of top contributing nodes that are neighbors
        and return a list along with the top contributing pos nodes for each prediction

    *returns*: dictionary with node idx as keys, and an array of main contributors as the values
    """
    # 
    pred_scores[pos_idx] = 0
    top_k_pred_idx = np.argsort(pred_scores)[::-1][:k]
    # get the diffusion values from the positive nodes to the top predictions 
    pos_to_top_dfsn = M_inv[pos_idx,:][:,top_k_pred_idx] 
    # normalize by the column to get the fraction of diffusion from each pos node
    pos_to_k_dfsn_norm = (pos_to_top_dfsn*np.divide(1,pos_to_top_dfsn.sum(axis=0)))

    # plot those values if specified
    if kwargs.get('plot_file'):
        plot_frac_dfsn_from_pos(top_k_pred_idx, pos_to_k_dfsn_norm, kwargs['plot_file'], cutoff=cutoff, **kwargs)

    main_contributors = {}
    fracs_top_nbrs = {}
    nodes_pos_nbr_dfsn = {} 
    for i, n in enumerate(top_k_pred_idx):
        # get the diffusion values from pos nodes for this node
        # if not clustering:
        pos_above_cutoff = np.where(pos_to_k_dfsn_norm[:,i] > cutoff)[0]
        # else:
        #     s1 = pd.Series(pos_to_k_dfsn_norm[:][:,i])
        #     s1 = s1.sort_values(ascending=False).reset_index()
        #     kmeans = KMeans(n_clusters=2, random_state=0).fit(s1[0].values.reshape(-1,1))
        #     split_point = [i for i, x in enumerate(kmeans.labels_) if x == 1][-1]
        #     split_point2 = [i for i, x in enumerate(kmeans.labels_) if x == 0][-1]
        #     split_point = min([split_point, split_point2])
        #     pos_above_cutoff = s1['index'][:split_point]
        main_contributors[n] = pos_above_cutoff

        if W is not None:
            # now get the edges of this node and see if they overlap with the top pos node influencers
            # extract the row of network to get the neighbors
            row = W[n,:]
            nbrs = (row > 0).nonzero()[1]
            top_pos_node_idx = [pos_idx[k] for k in pos_above_cutoff]
            if len(top_pos_node_idx) == 0:
                #frac_top_nbr = -1
                frac_top_nbr = 2
            else:
                # measure the fraction of top pos nodes that are also neighbors
                frac_top_nbr = len(set(top_pos_node_idx) & set(nbrs)) / len(top_pos_node_idx)
            fracs_top_nbrs[n] = frac_top_nbr

            # get the neighbors that are pos nodes
            # convert to the current indexes
            nbrs = set(nbrs)
            pos_nbrs = [j for j, kn in enumerate(pos_idx) if kn in nbrs]
            # and measure the fraction of diffusion that comes from the pos nodes
            # get the diffusion values from pos nodes for this node
            pos_nbr_dfsn = pos_to_k_dfsn_norm[pos_nbrs,:][:,i].sum()
            nodes_pos_nbr_dfsn[n] = pos_nbr_dfsn

    if W is not None:
        return main_contributors, fracs_top_nbrs, nodes_pos_nbr_dfsn
    else:
        return main_contributors


def plot_frac_dfsn_from_pos(top_k_pred_idx, pos_to_k_dfsn_norm, out_file, cutoff=0.01, **kwargs):
    import pandas as pd
    import matplotlib
    # Use this to save files remotely. 
    matplotlib.use('Agg')  
    import matplotlib.pyplot as plt

    for i in range(len(top_k_pred_idx)):
        #s1 = pd.Series(np.log10(pos_to_k_dfsn_norm[:][:,i]))
        # try just the top 50 pos
        s1 = pd.Series(np.log10(pos_to_k_dfsn_norm[:50][:,i]))
        s1 = s1.sort_values(ascending=False).reset_index()[0]
    #     kmeans = KMeans(n_clusters=2, random_state=0).fit(s1.values.reshape(-1,1))
    #     split_point = [i for i, x in enumerate(kmeans.labels_) if x == 1][-1]
        ax = s1.plot(alpha=0.2)
    #     ax.axvline(split_point, linestyle='--', alpha=0.2)
    #     break

    ax.axhline(np.log10(cutoff), linestyle='--')

    plt.ylabel("log normalized diffusion value")
    plt.xlabel("node # (sorted by diffusion value)")
    if kwargs.get('alpha'):
        plt.title("alpha=%s"%kwargs['alpha'])
    print("writing diffusion score curves to %s" % (out_file))
    plt.savefig(out_file)
    plt.close()
