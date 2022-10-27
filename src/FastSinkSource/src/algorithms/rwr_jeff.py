# Python implementation of SinkSource

import time
import numpy as np
import os
from scipy.linalg  import inv
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve, cg, spilu, LinearOperator
from tqdm import tqdm
import sys
from . import alg_utils


def RWR(P, y, max_iters=1000, eps=0.0001, a=0.95, verbose=False):
    """ Power iteration until all of the scores of the unknowns have converged
    *max_iters*: maximum number of iterations
    *eps*: Epsilon convergence cutoff. If the maximum node score change from one iteration to the next is < *eps*, then stop
    *a*: alpha parameter
    *y*: restart vector

    *returns*: scores array, process_time, wall_time
        # of iterations needed to converge
    """
    # UPDATE 2018-12-21: just start at zero so the time taken measured here is correct
    s = np.zeros(len(y))
    prev_s = s.copy()

    wall_time = time.time()
    # this measures the amount of time taken by all processors
    # only available in Python 3
    process_time = time.process_time()
    for iters in range(1, max_iters+1):
        update_time = time.time()
        # s^(i+1) = aPs^(i) + (1-a)y
        # s = a*csr_matrix.dot(P, prev_s) + (1-a)*y
        s = (1-a)*csr_matrix.dot(P, prev_s) + a*y

        if eps != 0:
            max_d = (s - prev_s).max()
            if verbose:
                print("\t\titer %d; %0.4f sec to update scores, max score change: %0.6e" % (iters, time.time() - update_time, max_d))
            if max_d <= eps:
                # converged!
                break
            prev_s = s.copy()
        else:
            prev_s = s
            #if verbose:
            #    print("\t\titer %d; %0.4f sec to update scores" % (iters, time.time() - update_time))

    wall_time = time.time() - wall_time
    process_time = time.process_time() - process_time
    #if verbose:
    #    print("SinkSource converged after %d iterations (%0.3f wall time (sec), %0.3f process time)" % (iters, wall_time, process_time))

    return s, process_time, wall_time, iters


def LazyRW(P, y, a=0.95, tol=1e-05, verbose=False):
    """
    *y*: vector of positive and negative assignments. 
         If y does not contain negatives, will be run as GeneManiaPlus, also known as Regularized Laplacian (RL). 
    *alpha*: parameter between 0 and 1 to control the influence of neighbors in the network.
        0 would ignore the network completely, and nodes would get their original score.
    *tol*: Conjugate Gradient tolerance for convergance

    *returns*: scores array, process_time, wall_time
        # of iterations needed to converge
    """
    # this measures the amount of time taken by all processors
    # and Conjugate Gradient is paralelized
    start_process_time = time.process_time()
    # this measures the amount of time that has passed
    start_wall_time = time.time()
    # I - aP
    M = eye(P.shape[0]) - a*P

    # keep track of the number of iterations
    num_iters = 0
    def callback(xk):
        # keep the reference to the variable within the callback function (Python 3)
        nonlocal num_iters
        num_iters += 1

    # use scipy's conjugate gradient solver
    s, info = cg(M, y, tol=tol, callback=callback)
    process_time = time.process_time() - start_process_time
    wall_time = time.time() - start_wall_time
    if verbose:
        print("Solved LazyRW using conjugate gradient (%0.2f sec, %0.2f sec process_time). Info: %s, iters=%d" % (
            wall_time, process_time, str(info), num_iters))

    return s, process_time, wall_time, num_iters

def get_diffusion_matrix(W, total_pos, alpha=0.5, diff_mat_file=None, force_run=False):
    """
    Generate the diffusion/propagation matrix of a network by taking the inverse of the laplaciain,
    also known as the Regularized Laplacian (RL)
    *Note that the result is a dense matrix*

    *W*: scipy sparse matrix representation of a network
    *alpha*: value of alpha for propagation
    *diff_mat_file*: path/to/file to store the RL (example: "%sdiffusion-mat-a%s.npy" % (net_obj.out_pref, str(alpha).replace('.','_')))
    """
    if diff_mat_file is not None and os.path.isfile(diff_mat_file) and force_run==False:
        # read in the diffusion mat file
        print("Reading %s" % (diff_mat_file))
        return np.load(diff_mat_file)

    # Transition matrix, X  = D^{-1}*W
    X = alg_utils._net_normalize(W, norm='full')  #normalize such that cols are sources, rows are targets
    # X = alg_utils.normalizeGraphEdgeWeights(W)
    #score = alpha{(I-(1-alpha)X)^-1} * y

    #Nure: as of 04/26/2022 using Jeffs rwr. He did col normlization i.e. in normlized weight matrix,
    # source=column,
    # target=row. That's why we don't need X.tranpose() anymore which we needed for Blessy's code.
    # X=X.transpose().multiply(1 - alpha)
    X = X.multiply(1 - alpha)

    X = eye(X.shape[0])-X
    X_full = X.A
    # # now take the inverse
    X_inv = inv(X_full)
    X_inv= X_inv*(alpha/float(total_pos))


    # write to file so this doesn't have to be recomputed
    if diff_mat_file is not None:
        print("Writing to %s" % (diff_mat_file))
        np.save(diff_mat_file, X_inv)
    del X, X_full
    return X_inv

def get_fluid_flow_matrix(W, fluid_flow_mat_file_M=None, fluid_flow_mat_file_R=None, force_run=False):
    """
    Generate the diffusion/propagation matrix of a network by taking the inverse of the laplaciain,
    also known as the Regularized Laplacian (RL)
    *Note that the result is a dense matrix*

    *W*: scipy sparse matrix representation of a network
    *alpha*: value of alpha for propagation
    *diff_mat_file*: path/to/file to store the RL (example: "%sdiffusion-mat-a%s.npy" % (net_obj.out_pref, str(alpha)
    .replace('.','_')))
    """

    if fluid_flow_mat_file_M is not None and os.path.isfile(fluid_flow_mat_file_M) and (force_run==False):
        # read in the diffusion mat file
        print("Hello! Reading %s" % (fluid_flow_mat_file_M))
        M_log = np.load(fluid_flow_mat_file_M)
    else:
        M = alg_utils._net_normalize(W, norm='full')
        M=M.A

        M_log = np.absolute(np.log10(M))

        # log converts 0 elements into infinity. infinity weight==no edge.
        # so effectively we can convert infinity -> 0 and then convert this array into networkx Graph
        # thus the edges with infinity weight will won't appear in netx graph
        # this way conversion to netx will be time efficient
        M_log[np.isinf(M_log)] = 0
        print("Writing to %s" % (fluid_flow_mat_file_M))
        np.save(fluid_flow_mat_file_M, M_log)

    if fluid_flow_mat_file_R is not None and os.path.isfile(fluid_flow_mat_file_R)and (force_run==False):
        # read in the diffusion mat file
        print("Reading %s" % (fluid_flow_mat_file_R))
        R = np.load(fluid_flow_mat_file_R)
    else:
        R = eye(M.shape[0])-M
        R = inv(R)
        print("Writing to %s" % (fluid_flow_mat_file_R))
        np.save(fluid_flow_mat_file_R, R)
    return M_log, R

