'''
An implementation of the standard pagerank algorithm, using iterative convergence.

Citation for this work:
    Poirel, C. L., Rodrigues, R. R., Chen, K. C., Tyson, J. J., & Murali, T. M.
    (2013). Top-down network analysis to drive bottom-up modeling of physiological
    processes. Journal of Computational Biology, 20(5), 409-418.

Relevant reference:
    Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank citation
    ranking: Bringing order to the web.

This code is authored by:
Nicholas Sharp: nsharp3@vt.edu
Anna Ritz: annaritz@vt.edu
Christopher L. Poirel: chris.poirel@gmail.com
T. M. Murali: tmmurali@cs.vt.edu
"""
'''

# Imports
# import networkx as nx
import sys
from scipy import sparse as sp
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

import os
from scipy.sparse import csr_matrix, find
from optparse import OptionParser, OptionGroup
from scipy.linalg import inv
from scipy.sparse import eye, diags
from . import alg_utils
from scipy.io import mmwrite, mmread
import pickle
'''
Run the RWR algorithm.

Inputs:
    + net           - Network as a scipy sparse matrix. Should already be normalized
    + weights       - (Optional) Initial weights for the graph. These are used both
                      to initialize the algorithm and to provide destination probatilities
                      during the teleportation step. If not given, uniform weights are used.
    + q             - The teleportation probability for the PageRank algorithm
                      (default = 0.5)
    + eps           - A RELATIVE convergence threshold for iteration (default = 0.01)
    + maxIters      - The maximum number of iterations to perform (default = 500)
    + verbose       - Print extra information (default = False)
    + weightName    - What property key holds the weights in the graph (default = weight)

Outputs:
    + currVisitProb - A dictionary of the final node probabilities at the end of the
      iteration process.

'''

def create_and_save_transition_mtx_zerodegnodes(net, N, alpha, out_dir, force_write=True):
    trans_mat_file = out_dir + 'q_'+str(alpha)+'_transition.mtx'
    zero_deg_node_file = out_dir +'q_'+str(alpha)+ '_zero_deg_node.pickle'

    if (not os.path.exists(trans_mat_file)) or (not os.path.exists(zero_deg_node_file)) or force_write:
        # Cache out-degree of all nodes to speed up the update loop
        outDeg = csr_matrix((N, 1), dtype=float)
        zeroDegNodes = set()

        for i in range(N):
            outDeg[i, 0] = 1.0 * net.getrow(i).sum()  # weighted out degree of every node
            if outDeg[i, 0] == 0:
                zeroDegNodes.add(i)
        # print("Number of zero degree nodes = ", len(zeroDegNodes))

        # Create the transition matrix
        # (from_idx, to_idx) = net.nonzero()
        # print("Number of edges in sparse matrix = ", len(from_idx))

        # net[u,v] = Walking in from an edge + Teleporting from source node + Teleporting from dangling node
        #          = (1-q)*net[u,v]/(outDeg[u]) + (q)*(incomingTeleProb[v] + zSum)

        # Walking in from neighbouring edge:
        e = net.multiply(1 - alpha).multiply(outDeg.power(-1))  # (1-q)*net[u,v]/(outDeg[u])

        # #Nure: 10/17/2022 No need to nomalize again(i.e. divide by outdegree) if run_obj.P was passed to rwr.
        # e = net.multiply(1 - alpha)  # (1-q)*net[u,v]/(outDeg[u])


        # Compute transition matrix X
        X = e
        # print("Shape of transition matrix X = ", X.get_shape())  # N X N

        #####SAVE X and zerodegnodes
        print('type of transition matrix: ', type(X))
        os.makedirs(os.path.dirname(trans_mat_file), exist_ok=True)
        mmwrite(trans_mat_file, X)

        with open(zero_deg_node_file, 'wb') as f:
            pickle.dump(zeroDegNodes, f)

        print('saves transition matrix and zerodegnodes\n')
    # READ from files
    X = mmread(trans_mat_file)
    with open(zero_deg_node_file,'rb') as f:
        zeroDegNodes = pickle.load(f)


    return X, zeroDegNodes

def rwr(P, weights={}, alpha=0.5, eps=0.01, maxIters=500, verbose=False, weightName='weight'):

    N = P.get_shape()[0]
    n_pos = len(weights.keys())

    y = np.zeros(N)
    for i in list(weights.keys()):
        y[i] = 1

    I = eye(N)
    M_inv = inv(I - (1-alpha)*P.A)
    scores = alpha/n_pos*np.matmul(M_inv, y)

    # Create a dictionary of final scores (keeping it consistent with the return type in PageRank.py)
    finalScores = {}
    for i in range(N):
        finalScores[i] = scores[i]

    return finalScores



def compute_two_loss_terms(y_true, y_pred, alpha, W):
    '''
    This function computes the two terms in the quadratic loss or energy function in PageRank algorithm
    presented in Ref: Christopher L. Poirel, Reconciling differential gene expression data with
    molecular interaction networks, Bioinformatics, Volume 29, Issue 5, 1 March 2013, Pages 622–629
    '''

    #first term = alpha. ||(y_pred-y_true).(deg_root_inv)||^2 where deg_root_inv(i) = 1/(sqrt(deg(i)))
    deg = np.asarray(W.sum(axis=0)).flatten()
    deg_root_inv = np.divide(1., np.sqrt(deg))
    loss_term1 = 2* alpha * (norm(np.multiply((y_pred-y_true), deg_root_inv),ord = 2))**2 #multiplying with extra 2

    #one implementation for the second term
    # #second term = (1-alpha) (y_pred_norm*L*y_pred_norm_transpose), where y_pred_norm(i) = (y_pred(i)/(deg(i))
    # # L = D-W, D = un-normalized degree matrix, W= un-nomalized weight matrix
    # y_pred_norm = np.divide(y_pred, deg).reshape(-1, len(y_pred))
    # deg[np.isinf(deg)] = 0
    # D = sp.diags(deg)
    # L = (D-W).toarray()
    # loss_term2_1 = (1-alpha)*np.matmul(np.matmul(y_pred_norm, L), y_pred_norm.transpose())[0,0]

    #another implementation for the second term
    loss_term2 = 0
    y_pred_norm = np.divide(y_pred, deg)
    rows, cols, vals = find(W)
    for i in range(len(rows)):
        a = vals[i]
        b = y_pred_norm[rows[i]]
        c = y_pred_norm[cols[i]]
        loss_term2 += vals[i]*((y_pred_norm[rows[i]]-y_pred_norm[cols[i]])**2)

    loss_term2 *= (1-alpha)

    return loss_term1, loss_term2

