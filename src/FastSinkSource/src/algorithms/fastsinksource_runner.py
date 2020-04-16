
import time
from tqdm import tqdm, trange
from scipy import sparse as sp
import numpy as np
from . import fastsinksource
from . import alg_utils


def setupInputs(run_obj):
    # extract the variables out of the annotation object
    run_obj.ann_matrix = run_obj.ann_obj.ann_matrix
    run_obj.terms = run_obj.ann_obj.terms

    if run_obj.net_obj.weight_swsn:
        # TODO if the net obj already has the W_SWSN object, then use that instead
        W, process_time = run_obj.net_obj.weight_SWSN(run_obj.ann_matrix)
        run_obj.P = alg_utils.normalizeGraphEdgeWeights(W, ss_lambda=run_obj.params.get('lambda', None))
        run_obj.params_results['%s_weight_time'%(run_obj.name)] += process_time
    elif run_obj.net_obj.weight_gmw:
        # this will be handled on a term by term basis
        run_obj.P = None
    else:
        run_obj.P = alg_utils.normalizeGraphEdgeWeights(run_obj.net_obj.W, ss_lambda=run_obj.params.get('lambda', None))

    return


# setup the params_str used in the output file
def setup_params_str(weight_str, params, name="fastsinksource"):
    # ss_lambda affects the network that all these methods use
    ss_lambda = params.get('lambda', 0)
    params_str = "%s-l%s" % (weight_str, ss_lambda)
    if name.lower() not in ["local", "localplus"]:
        a, eps, maxi = params['alpha'], params['eps'], params['max_iters']
        tol = "-tol%s" % (str_(params['tol'])) if 'tol' in params else ''
        solver = "-%s" % (params['solver']) if 'solver' in params else ''
        params_str += "-a%s-eps%s-maxi%s%s%s" % ( 
            str_(a), str_(eps), str_(maxi), tol, solver)

    return params_str


def get_alg_type():
    return "term-based"


def setupOutputFile(run_obj):
    return


# nothing to do here
def setupOutputs(run_obj, **kwargs):
    return


def run(run_obj):
    """
    Function to run FastSinkSource, FastSinkSourcePlus, Local and LocalPlus
    *terms_to_run*: terms for which to run the method. 
        Must be a subset of the terms present in the ann_obj
    """
    params_results = run_obj.params_results
    P, alg, params = run_obj.P, run_obj.name, run_obj.params

    #if 'solver' in params:
    # make sure the term_scores matrix is reset
    # because if it isn't empty, overwriting the stored scores seems to be time consuming
    term_scores = sp.lil_matrix(run_obj.ann_matrix.shape, dtype=np.float)
    print("Running %s with these parameters: %s" % (alg, params))
    if len(run_obj.target_prots) != len(run_obj.net_obj.nodes):
        print("\tstoring scores for only %d target prots" % (len(run_obj.target_prots)))

    # run FastSinkSource on each term individually
    #for i in trange(run_obj.ann_matrix.shape[0]):
    #term = run_obj.terms[i]
    for term in tqdm(run_obj.terms_to_run):
        idx = run_obj.ann_obj.term2idx[term]
        # get the row corresponding to the current terms annotations 
        y = run_obj.ann_matrix[idx,:]
        positives = (y > 0).nonzero()[1]
        negatives = (y < 0).nonzero()[1]
        # if this method uses positive examples only, then remove the negative examples
        if alg in ["fastsinksourceplus", "sinksourceplus", "localplus"]:
            negatives = None

        if run_obj.net_obj.weight_gmw is True:
            start_time = time.process_time()
            # weight the network for each term individually
            W,_,_ = run_obj.net_obj.weight_GMW(y.toarray()[0], term)
            P = alg_utils.normalizeGraphEdgeWeights(W, ss_lambda=params.get('lambda', None))
            params_results['%s_weight_time'%(alg)] += time.process_time() - start_time

        # now actually run the algorithm
        if alg in ["fastsinksource", "fastsinksourceplus", "sinksource", "sinksourceplus"]:
            a, eps, max_iters = params['alpha'], float(params['eps']), params['max_iters']
            # if a solver is given, it will be used. Otherwise it will use regular power iteration 
            solver = params.get('solver')
            tol = float(params['tol']) if 'tol' in params else 1e-5
            scores, process_time, wall_time, iters = fastsinksource.runFastSinkSource(
                P, positives, negatives=negatives, max_iters=max_iters,
                eps=eps, a=a, tol=tol, solver=solver, verbose=run_obj.kwargs.get('verbose', False))
        elif alg in ["local", "localplus"]:
            scores, process_time, wall_time = fastsinksource.runLocal(
                P, positives, negatives=negatives)
            iters = 1

        if run_obj.kwargs.get('verbose', False) is True:
            tqdm.write("\t%s converged after %d iterations " % (alg, iters) +
                    "(%0.4f sec) for %s" % (process_time, term))

        # limit the scores to the target nodes
        if len(run_obj.target_prots) != len(scores):
            #print("\tstoring results for %d target prots" % (len(run_obj.target_prots)))
            mask = np.ones(len(scores), np.bool)
            mask[run_obj.target_prots] = False
            scores[mask] = 0
        # 0s are not explicitly stored in lil matrix
        term_scores[idx] = scores

        # also keep track of the time it takes for each of the parameter sets
        alg_name = "%s%s" % (alg, run_obj.params_str)
        params_results["%s_wall_time"%alg_name] += wall_time
        params_results["%s_process_time"%alg_name] += process_time

    run_obj.term_scores = term_scores
    run_obj.params_results = params_results
    return


def str_(s):
    return str(s).replace('.','_')
