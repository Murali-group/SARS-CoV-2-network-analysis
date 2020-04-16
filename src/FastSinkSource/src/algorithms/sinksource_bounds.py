
# function to efficiently and accurately compute the top-k sinksource scores
# Algorithm and proofs adapted from:
# Zhang et al. Fast Inbound Top-K Query for Random Walk with Restart, KDD, 2015

import sys
import time
import numpy as np
from scipy.sparse import csr_matrix
#from scipy.stats import kendalltau  #, spearmanr, weightedtau
from . import alg_utils
#import src.evaluate.eval_utils as eval_utils


class SinkSourceBounds:

    def __init__(self, P, positives, negatives=None, a=0.8, max_iters=1000, 
                 nodes_to_rank=None, ranks_to_compare=None, verbose=False):
        """
        By default, require that the ranks of all nodes be fixed using their UB and LB

        *P*: Row-normalized sparse-matrix representation of the graph
        *nodes_to_rank*: set of nodes to rank in relation to each other
        *ranks_to_compare*: A list of nodes where the index of the node in the list is the rank of that node.
            For example, if node 20 was ranked first and node 50 was ranked second, the list would have [20, 50]
            Used to compare with the current ranking after each iteration.
        *max_iters*: Maximum number of iterations to run power iteration

        *returns*: The scores for all nodes
        """
        self.P = P
        self.positives = positives
        self.negatives = negatives
        self.a = a
        self.nodes_to_rank = nodes_to_rank
        self.ranks_to_compare = ranks_to_compare
        #self.scores_to_compare = scores_to_compare
        self.max_iters = max_iters
        self.verbose = verbose

    def runSinkSourceBounds(self):
        self.num_nodes = self.P.shape[0]
        # f: initial vector f of amount of score received from positive nodes
        # need to remove the non-reachable nodes here, since the bounds will not converge for them
        self.P, self.f, self.node2idx, self.idx2node = alg_utils.setup_fixed_scores(
            self.P, self.positives, self.negatives, a=self.a, 
            remove_nonreachable=True, verbose=self.verbose)
        if len(self.f) == 0:
            print("WARNING: no unknown nodes were reachable from a positive (P matrix and f vector empty after removing nonreachable nodes).")
            print("Setting all scores to 0")
            scores_arr = np.zeros(self.num_nodes)
            return scores_arr
        if self.nodes_to_rank is not None:
            # some of them could've been unreachable, so remove those and fix the mapping
            self.nodes_to_rank = set(self.node2idx[n] \
                    for n in self.nodes_to_rank if n in self.node2idx)

        if self.verbose:
            if self.negatives is not None:
                print("\t%d positives, %d negatives, %d unknowns, a=%s"
                        % (len(self.positives), len(self.negatives), self.P.shape[0], str(self.a)))
            else:
                print("\t%d positives, %d unknowns, a=%s"
                        % (len(self.positives), self.P.shape[0], str(self.a)))

        all_LBs = self._SinkSourceBounds()

        # set the default score to 0 for all nodes as some of them may be unreachable from positives
        scores_arr = np.zeros(self.num_nodes)
        indices = [self.idx2node[n] for n in range(len(all_LBs))]
        scores_arr[indices] = all_LBs
        #self.scores_to_compare = all_LBs

        if self.verbose:
            print("SinkSourceBounds finished after %d iterations (%0.3f total sec, %0.3f sec to update)"
                  % (self.num_iters, self.total_time, self.total_update_time))

        return scores_arr

    def _SinkSourceBounds(self):
        """
        *returns*: The current scores for all nodes
        """
        # TODO check to make sure t > 0, s > 0, k > 0, 0 < a < 1 and such
        unranked_nodes, LBs, prev_LBs, UBs = self.initialize_sets()
        if self.nodes_to_rank is not None:
            unranked_nodes = self.nodes_to_rank

        # the infinity norm is simply the maximum value in the vector
        max_f = self.f.max()
        if self.verbose:
            print("\tmax_f: %0.4f" % (max_f))

        self.num_iters = 0
        # total number of computations performed during the update function
        self.total_comp = 0
        # amount of time taken during the update function
        self.total_update_time = 0
        start_time = time.process_time()
        # also keep track of the max score change after each iteration
        self.max_d_list = []
        # keep track of the UB after each iteration
        self.UB_list = []
        # keep track of how fast the nodes are ranked
        self.num_unranked_list = []
        self.kendalltau_list = []
        ## keep track of fmax, avgp, auprc, auroc at each iteration
        #self.eval_stats_list = []
        #self.spearmanr_list = []
        # keep track of the biggest # of nodes with continuously overlapping upper or lower bounds
        max_unranked_stretch = len(unranked_nodes)
        self.max_unranked_stretch_list = [] 
        # keep track of the maximum difference of the current scores to the final score
        self.max_d_compare_ranks_list = []
        ## also keep track of how many nodes have a fixed ranking from the top of the list
        #num_ranked_from_top = 0 
        #self.num_ranked_from_top_list = [] 

        # iterate until all node rankings are fixed
        while len(unranked_nodes) > 0:
            if self.verbose:
                print("\tnum_iters: %d, |unranked_nodes|: %d, max_unranked_stretch: %d" % (
                    self.num_iters, len(unranked_nodes), max_unranked_stretch))
            if self.num_iters > self.max_iters:
                if self.verbose:
                    print("\thit the max # iters: %d. Stopping." % (self.max_iters))
                break
            self.num_iters += 1
            # keep track of how long it takes to update the bounds at each iteration
            curr_time = time.process_time()

            # power iteration
            LBs = self.a*csr_matrix.dot(self.P, prev_LBs) + self.f

            update_time = time.process_time() - curr_time
            self.total_update_time += update_time
            max_d = (LBs - prev_LBs).max()
            prev_LBs = LBs.copy()
            UB = self.computeUBs(max_f, self.a, self.num_iters)
            #if self.scores_to_compare is not None:
            #    max_d_compare_ranks = (self.scores_to_compare - LBs).max()

            if self.verbose:
                #if self.scores_to_compare is not None:
                #    print("\t\t%0.4f sec to update scores. max_d: %0.2e, UB: %0.2e, max_d_compare_ranks: %0.2e" % (update_time, max_d, UB, max_d_compare_ranks))
                #else:
                print("\t\t%0.4f sec to update scores. max_d: %0.2e, UB: %0.2e" % (update_time, max_d, UB))

            # Find the nodes whose rankings are not yet fixed.
            # If the UB is still > 1, then no node rankings are fixed yet.
            if UB < 1:
                UBs = LBs + UB
                unranked_nodes, max_unranked_stretch = self.check_fixed_rankings(
                        LBs, UBs, unranked_nodes=unranked_nodes) 

            self.max_unranked_stretch_list.append(max_unranked_stretch)
            self.max_d_list.append(max_d) 
            #if self.scores_to_compare is not None:
            #    self.max_d_compare_ranks_list.append(max_d_compare_ranks) 
            self.UB_list.append(UB) 
            self.num_unranked_list.append(len(unranked_nodes))
            if self.ranks_to_compare is not None:
                # also append a measure of the similarity between the current ranking and the rank to compare with
                # get the current node ranks
                scores = {self.idx2node[n]:LBs[n] for n in range(len(LBs))}
                nodes_with_ranks = set(self.ranks_to_compare)
                nodes_to_rank = set(scores.keys()) & nodes_with_ranks
                # check to make sure we have a rank for all of the nodes
                if len(nodes_to_rank) != len(nodes_with_ranks):
                    print("ERROR: some nodes do not have a ranking")
                    print("\t%d nodes_to_rank, %d ranks_to_compare" % (len(nodes_to_rank), len(nodes_with_ranks)))
                    sys.exit()
                # builds a dictionary of the node as the key and the current rank as the value
                # e.g., {50: 0, 20: 1, ...}
                curr_ranks = {n: i for i, n in enumerate(sorted(nodes_to_rank, key=scores.get, reverse=True))}
                # if I sort using ranks_to_compare directly, then for the first couple iterations when many nodes are tied at 0, 
                # will be left in the order they were in (i.e., matching the correct/fixed ordering)
                #curr_ranks = {n: i for i, n in enumerate(sorted(self.ranks_to_compare, key=scores.get, reverse=True))}
                # get the current rank of the nodes in the order of the ranks_to_compare 
                # for example, if the ranks_to_compare has 20 at 0 and 50 at 1, and the current rank is 50: 0, 20: 1,
                # then compare_ranks will be [1, 0]
                compare_ranks = [curr_ranks[n] for n in self.ranks_to_compare]
                # compare the two rankings
                # for example: curr rank: [1,0], orig rank: [0,1] 
                self.kendalltau_list.append(kendalltau(compare_ranks, range(len(self.ranks_to_compare)))[0])

        self.total_time = time.process_time() - start_time
        self.total_comp += len(self.P.data)*self.num_iters
        return LBs

    def computeUBs(self, max_f, a, i):
        if a == 1:
            return 1
        else:
            additional_score = (a**(i) * max_f) / (1-a)

        return additional_score

    def initialize_sets(self):
        unranked_nodes = set(np.arange(self.P.shape[0]).astype(int))

        # set the initial lower bound (LB) of each node to f or 0
        LBs = self.f.copy()
        # dictionary of LBs at the previous iteration
        prev_LBs = np.zeros(len(LBs))
        # dictionary of Upper Bonds for each node
        UBs = np.ones(self.P.shape[0])

        return unranked_nodes, LBs, prev_LBs, UBs

    def get_stats(self):
        """
        Returns the total time, time to update scores, # of iterations, # of computations (estimated), 
        the max_d at each iteration, and the initial size of the graph.
        """
        return self.total_time, self.total_update_time, self.num_iters, self.total_comp

    def check_fixed_rankings(self, LBs, UBs, unranked_nodes):
        """
        Check which nodes among the unranked_nodes set have a fixed ranking.  For a given node u, if the interval spanning the LB and UB 
        does not overlap with any other node's interval, then u's ranking is fixed.  We compute this as follows: 
            Sort the nodes by LB, then for each index k, check if k's LB > k-1's UB, and if k's UB < k+1's LB. 
            We also compute the number of overlapping nodes. 
        *unranked_nodes*: a set of nodes for which the ranking is not fixed. 
        """
        # find all of the nodes whose rankings are not yet fixed
        not_fixed_nodes = set()
        # sorting this tuple is about 6x slower than using numpy's argsort on the LBs. 
        # If the # nodes we're sorting is < 1/6 the total # nodes, then it will be faster. 
        # For a=0.99, BP EXPC LOSO, it takes about 500 iterations to get the UB down to where many of the rankings are fixed.
        # Since it takes about 1500 iterations on average to fix the node rankings, it is faster to sort the tuple.
        all_scores = []
        # if the node's score is still 0, then its ranking is not fixed yet. Skip those
        zero_nodes = set()
        for n in unranked_nodes:
            LB_n = LBs[n]
            if LB_n > 0:
                all_scores.append((n, LB_n))
            else:
                zero_nodes.add(n) 
        all_scores_sorted = sorted(all_scores, key=lambda x: (x[1]))
        # also keep track of the # of nodes in a row that have overlapping upper or lower bounds
        # for now just keep track of the biggest
        max_unranked_stretch = len(zero_nodes)
        # For every node, check if the next node's LB+UB > the curr node's LB. If so, the node is not yet fixed.
        # 
        # Compute as follows: For a given index i, starting at 0, keep incrementing k until the node LB at i+k > UB_i
        # All of those nodes are not fixed. Next, check if i+k is distinct from i+k-1, and if not, then i+k is not fixed either
        # Finally, set i=i+k and repeat. O(|V|)
        i = 0
        while i+1 < len(all_scores_sorted):
            n_i = all_scores_sorted[i][0]
            UB_i = UBs[n_i]
            k = i
            # check if the next node's LB < UB_i. If it is, then both nodes are not fixed
            while k+1 < len(all_scores_sorted): 
                n_k, LB_k = all_scores_sorted[k+1]
                if LB_k > UB_i:
                    break
                not_fixed_nodes.add(n_k)
                #print("i+1: %d not fixed" % (i+1))
                k += 1
            if k != i:
                #print("i: %d not fixed" % (curr_i))
                not_fixed_nodes.add(n_i)
                if k - i > max_unranked_stretch:
                    max_unranked_stretch = k - i
                # now check if the interval for node k overlaps with k-1
                UB_k_minus_1 = UBs[all_scores_sorted[k-1][0]]
                n_k, LB_k = all_scores_sorted[k] 
                # if it does, then we know k is not fixed either.
                if LB_k <= UB_k_minus_1:
                    not_fixed_nodes.add(n_k)
                i += k
                continue
            # if i does not overlap with any other nodes, then continue to the next node
            i += 1

        not_fixed_nodes |= zero_nodes
        return not_fixed_nodes, max_unranked_stretch


