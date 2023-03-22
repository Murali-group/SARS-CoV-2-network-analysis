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
from scipy.sparse import csr_matrix, csc_matrix, find
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

def create_transition_mtx_zerodegnodes(net, N, alpha):
    '''
    The following function will create a transition matrix considering
    targets along the rows and sources along the columns.
    '''

    outDeg = csr_matrix((1, N), dtype=float)
    zeroDegNodes = set()
    for i in range(N):
        #NURE: till 12/30 the following was implemented. But the net was default
        # symmetric till this point.
        # outDeg[i, 0] = 1.0 * net.getrow(i).sum()  # weighted out degree of every node
        #NURE: from 12/30. As for directed network we have source of an edge along columns,
        # for computing outdegree we need to sum column wise.
        outDeg[0, i] = 1.0 * net.getcol(i).sum()  # weighted out degree of every node

        if outDeg[0, i] == 0:
            zeroDegNodes.add(i)
    # Walking in from neighbouring edge:
    #Nure: Checked throughly the following statement to make sure that the degree normalization is done column wise.
    e = net.multiply(1 - alpha).multiply(outDeg.power(-1))  # (1-q)*net[u,v]/(outDeg[u])
    return e, zeroDegNodes

def rwr(net, weights={}, alpha=0.5, eps=0.01, maxIters=500, verbose=False, weightName='weight'):
    N = net.get_shape()[0]
    # print("Shape of network = ", net.get_shape())
    # print("Number of teleportation weights = ", len(weights))
    # print("Number of nodes in network = ", N)

    ###### Create transition matrix ###################
    X, zeroDegNodes = create_transition_mtx_zerodegnodes(net, N, alpha)
    incomingTeleProb = {}  # The node weights when the algorithm begins, also used as teleport-to probabilities

    # Find the incoming teleportation probability for each node, which is also used as the initial probabilities in
    # the graph. If no node weights are passed in, use a uniform distribution.
    totalWeight = sum([w for v, w in weights.items()])

    # TODO: handle no incoming weights

    # If weights are given, apply two transformations
    #   - Add a small incoming teleportation probability to every node to ensure that the graph is strongly connected
    #   - Normalize the weights to sum to one: these are now probabilities.
    # Find the smallest non-zero weight in the graph.

    minPosWeight = 1.0
    for v, weight in weights.items():
        if weight == 0:
            continue
        minPosWeight = min(minPosWeight, 1.0 * weight / totalWeight)

    # The epsilon used as the added incoming teleportation probabilty is 1/1000000 the size of the smallest weight given
    # so that it does not impact results.
    smallWeight = minPosWeight / (10 ** 6)

    for i in range(N):
        weight = weights.get(i, 0.0)
        incomingTeleProb[i] = 1.0 * (weight + smallWeight) / (totalWeight + smallWeight * N)

    # Sparse matrices to store the probability scores of the nodes
    currVisitProb = csr_matrix([list(incomingTeleProb.values())], dtype=float)  # currVisitProb: 1 X N
    # prevVisitProb must converted to N X 1 to multiply with Transition Matrix (N X N) and yield new currVisitProb(N X 1)
    prevVisitProb = currVisitProb.transpose()  # prevVisitProb: N X 1


    # Teleporting from source node:
    # currVisitProb holds values of incomingTeleProb
    t = currVisitProb.multiply(alpha).transpose()  # (q)*(incomingTeleProb[v]
    # print("Shape of matrix t = ", t.shape)  # N X 1

    iters = 0
    finished = False

    while not finished:
        iters += 1

        # X: N X N ; prevVisitProb: N X 1 ; Thus, X.transpose() * prevVisitProb: N X 1
        #Nure: till 12/30, in X along rows I had sources, and columns I had targets.
        # currVisitProb = (X.transpose() * prevVisitProb)

        #Nure: from 12/30, in X along rows I have sources, and columns I have targets. So using
        # X directly instead of X.transpose()
        currVisitProb = (X * prevVisitProb)
        currVisitProb = currVisitProb + t  # N X 1


        # Teleporting from dangling node
        # In the basic formulation, nodes with degree zero ("dangling
        # nodes") have no weight to send a random walker if it does not
        # teleport. We consider a walker on one of these nodes to
        # teleport with uniform probability to any node. Here we compute
        # the probability that a walker will teleport to each node by
        # this process.
        zSum = sum([prevVisitProb[x, 0] for x in zeroDegNodes])/N  # scalar
        currVisitProb = currVisitProb + csc_matrix(np.full(currVisitProb.shape,
                        ((1 - alpha) * zSum)))  # the scalar (1-q)*zSum will get broadcasted and added to every element of currVisitProb

        # currVisitProb = currVisitProb + ((1 - alpha) * zSum)  # the scalar (1-q)*zSum will get broadcasted and added to every element of currVisitProb

        # Keep track of the maximum RELATIVE difference between this
        # iteration and the previous to test for convergence
        maxDiff = (abs(prevVisitProb - currVisitProb) / currVisitProb).max()

        # Print statistics on the iteration
        # print("\tIteration %d, max difference %f" % (iters, maxDiff))
        if maxDiff < eps:
            print("RWR converged after %d iterations, max difference %f" % (iters, maxDiff))

        # Give a warning if termination happens by the iteration cap,
        # which generally should not be expected.
        if iters >= maxIters:
            print("WARNING: RWR terminated because max iterations (%d) was reached." % (maxIters))

        # Test for termination, either due to convergence or exceeding the iteration cap
        finished = (maxDiff < eps) or (iters >= maxIters)
        # Update prevVistProb
        prevVisitProb = currVisitProb


        # break
    # Create a dictionary of final scores (keeping it consistent with the return type in PageRank.py)
    finalScores = {}
    for i in range(N):
        finalScores[i] = currVisitProb[i, 0]  #dim(currVisitProb)=N*1.
        # so, take the value of the first col    which is the only col as well
    return finalScores

def compute_two_loss_terms(y_true, y_pred, alpha, W, is_directed=False):
    '''
    This function computes the two terms in the quadratic loss or energy function in PageRank algorithm
    presented in Ref: Christopher L. Poirel, Reconciling differential gene expression data with
    molecular interaction networks, Bioinformatics, Volume 29, Issue 5, 1 March 2013, Pages 622–629
    '''

    if not is_directed:
        #first term = alpha. ||(y_pred-y_true).(deg_root_inv)||^2 where deg_root_inv(i) = 1/(sqrt(deg(i)))
        #But we are multiplying the first term with 2 i.e. first term = 2* alpha. ||(y_pred-y_true).(deg_root_inv)||^2,
        # and we are doing it because in our second term we compute each edge twice and
        # as a result the second term
        # is twice than what we should compute.
        # And to balance them out we need the first term multiplied by 2.

        deg = np.asarray(W.sum(axis=0)).flatten()
        deg_root_inv = np.divide(1., np.sqrt(deg))
        deg_root_inv[np.isinf(deg_root_inv)] = 0 #TODO: is this the right way?
        # if degree of a node is 0, then inv(deg)=inf, we convert it to 0 as we don't
        # want that degree to have any impact on the calculated loss term

        loss_term1_wo_alpha =2* (norm(np.multiply((y_pred-y_true), deg_root_inv),ord = 2))**2
        loss_term1 = alpha * loss_term1_wo_alpha

        #one implementation for the second term
        # #second term = (1-alpha) (y_pred_norm*L*y_pred_norm_transpose), where y_pred_norm(i) = (y_pred(i)/(deg(i))
        # # L = D-W, D = un-normalized degree matrix, W= un-nomalized weight matrix
        # y_pred_norm = np.divide(y_pred, deg).reshape(-1, len(y_pred))
        # deg[np.isinf(deg)] = 0
        # D = sp.diags(deg)
        # L = (D-W).toarray()
        # loss_term2 = (1-alpha)*np.matmul(np.matmul(y_pred_norm, L), y_pred_norm.transpose())[0,0]
        #another implementation for the second term
        loss_term2_wo_alpha = 0
        y_pred_norm = np.divide(y_pred, deg)
        rows, cols, vals = find(W)
        for i in range(len(rows)):
            l = vals[i]*((y_pred_norm[rows[i]]-y_pred_norm[cols[i]])**2)
            loss_term2_wo_alpha += l

        loss_term2 = (1-alpha)*loss_term2_wo_alpha

    else:
        out_deg = np.asarray(W.sum(axis=0)).flatten()
        in_deg = np.asarray(W.sum(axis=1)).flatten()
        in_deg_root_inv = np.divide(1., np.sqrt(in_deg))
        in_deg_root_inv[np.isinf(in_deg_root_inv)] = 0  # TODO: is this the right way?
        # if degree of a node is 0, then inv(deg)=inf, we convert it to 0 as we don't
        # want that degree to have any impact on the calculated loss term
        loss_term1_wo_alpha =2* (norm(np.multiply((y_pred - y_true), in_deg_root_inv), ord=2)) ** 2
        loss_term1 = alpha * loss_term1_wo_alpha

        loss_term2_wo_alpha = 0
        y_pred_out_deg_norm = np.divide(y_pred, out_deg)
        y_pred_in_deg_norm = np.divide(y_pred, in_deg)

        rows, cols, vals = find(W)

        #DEBUG PURPOSE
        loss_2_per_source={p:0 for p in range(W.shape[0])}
        loss_2_per_target={p:0 for p in range(W.shape[0])}

        for i in range(len(rows)):
            l = vals[i] * ((y_pred_in_deg_norm[rows[i]] - y_pred_out_deg_norm[cols[i]]) ** 2)
            loss_2_per_source[cols[i]]+=l
            loss_2_per_target[rows[i]]+=l
            loss_term2_wo_alpha += l

        loss_term2 =(1 - alpha)*loss_term2_wo_alpha

        # #######DEBUG PURPOSE
        # loss_2_per_target = dict(sorted(loss_2_per_target.items(), key=lambda x: x[1], reverse=True))
        # out_deg_zero = np.where(out_deg == 0)[0]
        # loss_2_per_target = set(list(loss_2_per_target.keys())[0:100])
        # print(len(loss_2_per_target.difference(set(out_deg_zero))))
        # ##########

    print('loss without alpha multiplied for alpha: ', alpha, ' ', round(loss_term1_wo_alpha,4),\
          ' ', round(loss_term2_wo_alpha,4))
    print('loss for alpha: ', alpha, ' ', round(loss_term1,4), ' ', round(loss_term2,4))
    return loss_term1, loss_term2


def get_diffusion_matrix(W, alpha=0.5, diff_mat_file=None, force_run=False):
    """
    Generate the diffusion/propagation matrix of a network
    """
    if diff_mat_file is not None and os.path.isfile(diff_mat_file) and force_run==False:
        # read in the diffusion mat file
        print("Reading %s" % (diff_mat_file))
        return np.load(diff_mat_file)

    # Transition matrix, X  = D^{-1}*W
    X = alg_utils._net_normalize(W, norm='full')  #normalize such that cols are sources, rows are targets
    #Nure: as of 04/26/2022 using Jeffs rwr. He did col normlization i.e. in normlized weight matrix, source=column,
    # target=row. That's why we don't need X.tranpose() anymore which we needed for Blessy's code.
    # X=X.transpose().multiply(1 - alpha)
    X = X.multiply(1 - alpha)
    X = eye(X.shape[0])-X
    X_full = X.A
    # # now take the inverse
    X_inv = inv(X_full)

    # write to file so this doesn't have to be recomputed
    if diff_mat_file is not None:
        print("Writing to %s" % (diff_mat_file))
        os.makedirs(os.path.dirname(diff_mat_file), exist_ok=True)
        np.save(diff_mat_file, X_inv)
    del X, X_full
    return X_inv

def get_M(W, alpha):
    M = alg_utils._net_normalize(W, norm='full')
    M = M.multiply(1 - alpha) # So every value in M is < 1.
    return M.A
def get_fluid_flow_matrix(W, alpha, fluid_flow_mat_file_M=None, fluid_flow_mat_file_R=None, force_run=False):
    """

    """

    if os.path.isfile(fluid_flow_mat_file_M) and os.path.isfile(fluid_flow_mat_file_R) and (force_run==False):
        # read in the diffusion mat file
        print("Hello! Reading %s" % (fluid_flow_mat_file_M))
        M_log = np.load(fluid_flow_mat_file_M)

        print("Reading %s" % (fluid_flow_mat_file_R))
        R = np.load(fluid_flow_mat_file_R)
    else:

        M = get_M(W, alpha)
        # M=M.A
        assert (M <= 1).all(), print('greater than 1 values in M')
        eps = 0.000001

        for i in range(M.shape[1]):
            s = np.sum(M[:, i])
            assert (s-1)<=eps, print('problem in norm', s)

        M_log = np.absolute(np.log10(M))

        # log converts 0 elements into infinity. infinity weight==no edge.
        # so effectively we can convert infinity -> 0 and then convert this array into networkx Graph
        # thus the edges with infinity weight will won't appear in netx graph
        # this way conversion to netx will be time efficient
        M_log[np.isinf(M_log)] = 0
        print("Writing to %s" % (fluid_flow_mat_file_M))
        os.makedirs(os.path.dirname(fluid_flow_mat_file_M), exist_ok=True)
        np.save(fluid_flow_mat_file_M, M_log)

        #
        # # Now compute R, where we divide every column(contr from a certain source to all other nodes) of M_inv with
        # #corresponding source's pred score (i.e. probability of reaching the source at steady state)
        # pred_score_mat = np.diag(np.divide(1.,all_pred_scores))
        # R = (pred_score_mat.dot(M_inv.T)).T

        R = inv(eye(M.shape[0])-M)

        print("Writing to %s" % (fluid_flow_mat_file_R))
        os.makedirs(os.path.dirname(fluid_flow_mat_file_R), exist_ok=True)
        np.save(fluid_flow_mat_file_R, R)
        del M
    return M_log, R

