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
import networkx as nx
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from optparse import OptionParser, OptionGroup

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


def rwr(net, weights={}, q=0.5, eps=0.01, maxIters=500, verbose=False, weightName='weight'):
    N = net.get_shape()[0]
    print("Shape of network = ", net.get_shape())
    print("Number of teleportation weights = ", len(weights))
    print("Number of nodes in network = ", N)

    incomingTeleProb = {}  # The node weights when the algorithm begins, also used as teleport-to probabilities
    prevVisitProb = {}  # The visitation probability in the previous iterations
    currVisitProb = {}  # The visitation probability in the current iteration

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

    # Cache out-degree of all nodes to speed up the update loop
    outDeg = csr_matrix((N, 1), dtype=float)
    zeroDegNodes = set()

    for i in range(N):
        outDeg[i, 0] = 1.0 * net.getrow(i).sum()  # weighted out degree of every node
        if outDeg[i, 0] == 0:
            zeroDegNodes.add(i)
    print("Number of zero degree nodes = ", len(zeroDegNodes))

    # Create the transition matrix
    (from_idx, to_idx) = net.nonzero()
    print("Number of edges in sparse matrix = ", len(from_idx))

    # net[u,v] = Walking in from an edge + Teleporting from source node + Teleporting from dangling node
    #          = (1-q)*net[u,v]/(outDeg[u]) + (q)*(incomingTeleProb[v] + zSum)

    # Walking in from neighbouring edge:
    e = net.multiply(1 - q).multiply(outDeg.power(-1))  # (1-q)*net[u,v]/(outDeg[u])
    print("Shape of matrix e = ", e.get_shape())  # N X N

    # Teleporting from source node:
    # currVisitProb holds values of incomingTeleProb
    t = currVisitProb.multiply(q).transpose()  # (q)*(incomingTeleProb[v]
    print("Shape of matrix t = ", t.shape)  # N X 1

    # Compute transition matrix X
    X = e
    print("Shape of transition matrix X = ", X.get_shape())  # N X N

    iters = 0
    finished = False
    while not finished:
        iters += 1

        # X: N X N ; prevVisitProb: N X 1 ; Thus, X.transpose() * prevVisitProb: N X 1
        currVisitProb = (X.transpose() * prevVisitProb)
        currVisitProb = currVisitProb + t  # N X 1

        # Teleporting from dangling node
        # In the basic formulation, nodes with degree zero ("dangling
        # nodes") have no weight to send a random walker if it does not
        # teleport. We consider a walker on one of these nodes to
        # teleport with uniform probability to any node. Here we compute
        # the probability that a walker will teleport to each node by
        # this process.
        zSum = sum([prevVisitProb[x, 0] for x in zeroDegNodes]) / N  # scalar
        currVisitProb = currVisitProb + ((1 - q) * zSum)  # the scalar (1-q)*zSum will get broadcasted and added to every element of currVisitProb

        # Keep track of the maximum RELATIVE difference between this
        # iteration and the previous to test for convergence
        maxDiff = (abs(prevVisitProb - currVisitProb) / currVisitProb).max()

        # Print statistics on the iteration
        print("\tIteration %d, max difference %f" % (iters, maxDiff))
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
        finalScores[i] = currVisitProb[i, 0]

    return finalScores
