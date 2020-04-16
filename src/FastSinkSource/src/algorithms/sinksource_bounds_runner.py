
import sys, os
import time
from scipy import sparse as sp
import numpy as np
from tqdm import tqdm, trange
import fcntl
from . import fastsinksource_runner as fss_runner
from . import sinksource_bounds as ss_bounds
from . import alg_utils


def setupInputs(run_obj):
    # setup is the same as for fastsinksource
    fss_runner.setupInputs(run_obj)

    return


# setup the params_str used in the output file
def setup_params_str(
        weight_str, params, name="sinksource_bounds"):
    # TODO update with additional parameters
    # ss_lambda affects the network that all these methods use
    ss_lambda = params.get('lambda', 0)
    a, maxi = params['alpha'], params['max_iters']
    rank_pos_neg = params.get('rank_pos_neg')
    params_str = "%s-l%s-a%s-maxi%s" % (
        weight_str, ss_lambda, str_(a), str_(maxi))
    params_str += "-rank-pos-neg" if rank_pos_neg else ""

    return params_str


def get_alg_type():
    return "term-based"


# write the ranks file
def setupOutputs(run_obj, taxon=None, **kwargs):
    ranks_file = run_obj.out_pref + "-ranks.txt"
    # TODO apply this forced before any of the taxon 
    if not os.path.isfile(ranks_file):
        print("Writing rank stats to %s" % (ranks_file))
        append = False
    else:
        print("Appending rank stats to %s." % (ranks_file))
        append = True
    with open(ranks_file, 'a' if append else 'w') as out:
        # lock it to avoid scripts trying to write at the same time
        fcntl.flock(out, fcntl.LOCK_EX)
        if append is False:
            if taxon is not None:
                out.write("#term\ttaxon\tnum_pos\titer\tkendalltau\tnum_unranked\tmax_unr_stretch\tmax_d\tUB\n")
            else:
                out.write("#term\tnum_pos\titer\tkendalltau\tnum_unranked\tmax_unr_stretch\tmax_d\tUB\n")
        for term, rank_stats in run_obj.term_rank_stats.items():
            if taxon is not None:
                term += "\t"+taxon
            out.write(''.join("%s\t%s\n" % (term, stats) for stats in rank_stats))
        fcntl.flock(out, fcntl.LOCK_UN)


def run(run_obj):
    """
    Function to run FastSinkSource, FastSinkSourcePlus, Local and LocalPlus
    *terms_to_run*: terms for which to run the method. 
        Must be a subset of the terms present in the ann_obj
    """
    params_results = run_obj.params_results
    # make sure the term_scores matrix is reset
    # because if it isn't empty, overwriting the stored scores seems to be time consuming
    term_scores = sp.lil_matrix(run_obj.ann_matrix.shape, dtype=np.float)
    term_rank_stats = {}
    P, alg, params = run_obj.P, run_obj.name, run_obj.params
    print("Running %s with these parameters: %s" % (alg, params))

    # run on each term individually
    for term in tqdm(run_obj.terms_to_run):
        idx = run_obj.ann_obj.term2idx[term]
        # get the row corresponding to the current terms annotations 
        y = run_obj.ann_matrix[idx,:]
        positives = (y > 0).nonzero()[1]
        negatives = (y < 0).nonzero()[1]
        # if this method uses positive examples only, then remove the negative examples
        if alg in ["sinksourceplus_bounds"]:
            negatives = None

        if run_obj.net_obj.weight_gmw is True:
            start_time = time.process_time()
            # weight the network for each term individually
            W, process_time = run_obj.net_obj.weight_GMW(y.toarray()[0], term)
            P = alg_utils.normalizeGraphEdgeWeights(W, ss_lambda=params.get('lambda', None))
            params_results['%s_weight_time'%(alg)] += time.process_time() - start_time

        a, max_iters = params['alpha'], params['max_iters']
        compare_ranks = params['compare_ranks']
        # 'rank_pos_neg' will be the test/left-out ann matrix 
        # from which we can get the left-out pos/neg for this term
        rank_pos_neg = params['rank_pos_neg']
        if sp.issparse(rank_pos_neg):
            pos, neg = alg_utils.get_term_pos_neg(rank_pos_neg, idx)
            pos_neg_nodes = set(pos) | set(neg)
        elif rank_pos_neg is True:
            print("ERROR: rank_pos_neg must be the test_ann_mat")
            sys.exit()

        # now actually run the algorithm
        ss_obj = ss_bounds.SinkSourceBounds(
            P, positives, negatives=negatives, max_iters=max_iters,
            a=a, nodes_to_rank=pos_neg_nodes,
            verbose=run_obj.kwargs.get('verbose', False))

        scores_arr = ss_obj.runSinkSourceBounds()
        process_time, update_time, iters, comp = ss_obj.get_stats()

        if run_obj.kwargs.get('verbose', False) is True:
            tqdm.write("\t%s converged after %d iterations " % (alg, iters) +
                    "(%0.4f sec) for %s" % (process_time, term))

        if compare_ranks:
            # compute the kendall's tau at each iteration compared to the final ranking from the previous run.
            tqdm.write("\tRepeating the run, but comparing the ranks from the previous run at each iteration")
            # keep only the nodes with a non-zero score
            scores = {n: s for n, s in enumerate(scores_arr) if s > 0} 
            # ranks is a list containing the ranked order of nodes.
            # The node with the highest score is first, the lowest is last
            if rank_pos_neg is not None:
                ranks = [n for n in sorted(set(scores.keys()) & pos_neg_nodes, key=scores.get, reverse=True)]
            else:
                ranks = [n for n in sorted(scores, key=scores.get, reverse=True)]

            # left off top-k for now
            #ranks = ranks[:k] if self.rank_topk is True else ranks
            ss_obj = ss_bounds.SinkSourceBounds(
                P, positives, negatives=negatives, max_iters=max_iters,
                a=a, nodes_to_rank=pos_neg_nodes, ranks_to_compare=ranks,
                verbose=run_obj.kwargs.get('verbose', False))
            ss_obj.runSinkSourceBounds()

            # leave out precision and recall
            #rank_stats = ["%d\t%d\t%0.4e\t%d\t%d\t%0.2e\t%0.2e\t%0.4f\t%0.4f\t%0.4f" % (
            rank_stats = ["%d\t%d\t%0.4e\t%d\t%d\t%0.2e\t%0.2e" % (
                len(positives), i+1, ss_obj.kendalltau_list[i], ss_obj.num_unranked_list[i],
                ss_obj.max_unranked_stretch_list[i], ss_obj.max_d_list[i], ss_obj.UB_list[i],
                #ss_obj.eval_stats_list[i][0], ss_obj.eval_stats_list[i][1],
                #ss_obj.eval_stats_list[i][2]
                )
                                for i in range(ss_obj.num_iters)]
            term_rank_stats[term] = rank_stats

        # limit the scores to the target nodes
        if len(run_obj.target_prots) != len(scores_arr):
            mask = np.ones(len(scores_arr), np.bool)
            mask[run_obj.target_prots] = 0
            scores_arr[mask] = 0
        term_scores[idx] = scores_arr
        # make sure 0s are removed
        #term_scores.eliminate_zeros()

        # also keep track of the time it takes for each of the parameter sets
        alg_name = "%s%s" % (alg, run_obj.params_str)
        #params_results["%s_wall_time"%alg_name] += wall_time
        params_results["%s_process_time"%alg_name] += process_time
        params_results["%s_update_time"%alg_name] += update_time

    run_obj.term_scores = term_scores.tocsr()
    run_obj.params_results = params_results
    run_obj.term_rank_stats = term_rank_stats
    return


def str_(s):
    return str(s).replace('.','_')
