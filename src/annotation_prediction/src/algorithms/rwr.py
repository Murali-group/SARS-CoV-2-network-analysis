# Python sparse matrix implementation of RWR and LazyRW

import time
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import cg


def RWR(P, y, max_iters=1000, eps=0.0001, a=0.95, verbose=False):
    """
    Random Walk with Restarts power iteration until all node scores have converged

    *P*: Network as a scipy sparse matrix. Should already be normalized
    *y*: vector of restart probabilities per node. Should sum to 1
    *max_iters*: maximum number of iterations
    *eps*: Epsilon convergence cutoff. If the maximum node score change
        from one iteration to the next is < *eps*, then stop
    *a*: alpha parameter, where 1-a is the restart probability

    *returns*: scores array, process_time, wall_time,
        # of iterations needed to converge
    """
    s = np.zeros(len(y))
    prev_s = s.copy()

    wall_time = time.time()
    # this measures the amount of time taken by all processes
    process_time = time.process_time()
    for iters in range(1, max_iters + 1):
        update_time = time.time()
        # s^(i+1) = aPs^(i) + (1-a)y
        s = a * csr_matrix.dot(P, prev_s) + (1 - a) * y

        if eps != 0:
            max_d = (s - prev_s).max()
            if verbose:
                print("\t\titer %d; %0.4f sec to update scores, max score change: %0.6e" % (
                    iters, time.time() - update_time, max_d))
            if max_d <= eps:
                # converged!
                break
            prev_s = s.copy()
        else:
            prev_s = s

    wall_time = time.time() - wall_time
    process_time = time.process_time() - process_time

    return s, process_time, wall_time, iters


def LazyRW(P, y, a=0.95, tol=1e-05, verbose=False):
    """
    Lazy Random Walk implementation

    *P*: Network as a scipy sparse matrix. Should already be normalized
    *y*: vector with 1's at each node from which to start the random walk
    *alpha*: parameter between 0 and 1 to control the influence of neighbors in the network.
        0 would ignore the network completely, and nodes would get their original score.
    *tol*: Conjugate Gradient tolerance for convergance

    *returns*: scores array, process_time, wall_time
        # of iterations needed to converge
    """
    # this measures the amount of time taken by all processes
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

