# Python implementation of GeneMANIA

import time
import src.algorithms.alg_utils as alg_utils
import numpy as np
from scipy.sparse import eye, diags
from scipy.sparse.linalg import spsolve, cg, spilu, LinearOperator


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
         If y does not contain negatives, will be run as GeneManiaPlus. 
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
        print("Solved GeneMANIA using conjugate gradient (%0.2f sec, %0.2f sec process_time). Info: %s, k=%0.2f, iters=%d" % (
            wall_time, process_time, str(info), k, num_iters))

    return f, process_time, wall_time, num_iters
