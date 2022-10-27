from .alg_utils import str_
from . import alg_utils
from scipy import sparse
from . import rwr_nure as rwr
import numpy as np
# import networkx as nx


# extract the variables out of the annotation object
def setupInputs(run_obj):
    run_obj.ann_matrix = run_obj.ann_obj.ann_matrix
    run_obj.terms = run_obj.ann_obj.terms
    run_obj.P = alg_utils._net_normalize(run_obj.net_obj.W,norm='full')

    return


# run the PageRank_Matrix algorithm
def run(run_obj):
    # netx_graphs = run_obj.net_obj.netx_graphs
    # print('netx_graphs:', type(netx_graphs),'\n',type(netx_graphs[0]))
    params = run_obj.params

    term_scores = sparse.lil_matrix(run_obj.ann_matrix.shape, dtype=float)
    print("Running %s with these parameters: %s" % (run_obj.name, params))

    for term in run_obj.terms_to_run:
        idx = run_obj.ann_obj.term2idx[term]
        # get the row corresponding to the current terms annotations
        y = run_obj.ann_matrix[idx, :]
        positives = (y > 0).nonzero()[1]

        positive_weights = {}
        # assign a default weight = 1 for all positive/source nodes
        for i in positives:
            positive_weights[i] = 1

        scores_map = rwr.rwr(run_obj.P, weights=positive_weights,
                                       alpha=params.get('alpha'), eps=params.get('eps'), maxIters=params.get('max_iters'),
                                       verbose=run_obj.verbose)
        # PageRank returns a dictionary of node_id:score mapping.
        # Convert it an array by using the index of array to track the nodes assuming the order is preserved.

        #Nure: commented out following statements as PageRank_Matrix already outputs a dictionary for scores
        # scores = np.zeros(len(final_probability_scores))
        # for id, probability_score in final_probability_scores.items():
        #     scores[id] = probability_score
        term_scores[idx] = np.array(list((scores_map.values())))

    #Nure: comment out following statement and put scores_map.values directly as run_obj.term_scores.
    # run_obj.term_scores = term_scores

    run_obj.term_scores = term_scores

    return


# setup the params_str used in the output file
def setup_params_str(weight_str, params, name="rwr"):
    alpha = params.get('alpha', 0.5)
    eps, maxi = params.get('eps', 0.01), params.get('max_iters', 500)
    params_str = "%s-alpha%s-eps%s-maxi%s" % (weight_str, str_(alpha), str_(eps), str_(maxi))
    return params_str


def get_alg_type():
    return "term-based"


# nothing to do here
def setupOutputs(run_obj, **kwargs):
    return
