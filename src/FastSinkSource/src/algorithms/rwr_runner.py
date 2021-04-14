from .alg_utils import str_
from scipy import sparse
from . import PageRank
import numpy as np

def setupInputs(run_obj):
    print("RWR: setupInputs")
    run_obj.ann_matrix = run_obj.ann_obj.ann_matrix
    run_obj.terms = run_obj.ann_obj.terms
    return


# run the method
def run(run_obj):
    sparse_netx_graphs = run_obj.net_obj.sparse_netx_graphs
    params = run_obj.params
    print("params>>>>")
    print(params)
    print("Number of networks: %s" % len(sparse_netx_graphs))
    print("Running +RWR+ with these parameters: %s" % (params))

    term_scores = sparse.lil_matrix(run_obj.ann_matrix.shape, dtype=float)
    for term in run_obj.terms_to_run:
        idx = run_obj.ann_obj.term2idx[term]
        # get the row corresponding to the current terms annotations
        y = run_obj.ann_matrix[idx, :]
        positives = (y > 0).nonzero()[1]
        print("Annotation Matrix :: ")
        print(run_obj.ann_matrix.shape)

        print("Number of positives :: %s" % len(positives))
        positive_weights = {}
        for i in positives:
            positive_weights[i]=1

        print("Number of positive_weights :: %s" % len(positive_weights))
        print('Starting RWR')
        finalProbs = PageRank.pagerank(sparse_netx_graphs[0], weights=positive_weights, q=params.get('q'), eps=params.get('eps'), maxIters=params.get('max_iters'), verbose=True)
        print('Number of finalProbs :: %s' % len(finalProbs))
        scores = np.zeros(len(finalProbs))
        for id,score in finalProbs.items():
            scores[id] = score
        term_scores[idx] = scores
        print("Completed RWR")

    run_obj.term_scores = term_scores
    return


# if the method is not in Python and was called elsewhere (e.g., R),
# then parse the outputs of the method
def setupOutputs(run_obj):
    return


# setup the params_str used in the output file
def setup_params_str(weight_str, params, name):
    print("setupParamsStr")
    print("Weight_str")
    print(weight_str)
    print("params")
    print(params)
    print("name")
    print(name)
    q = params.get('q', 0.5)
    eps, maxi = params.get('eps', 0.01), params.get('max_iters', 500)
    params_str = "-q%s-eps%s-maxi%s" % (str_(q), str_(eps), str_(maxi))
    return params_str


def get_alg_type():
    return 'RWR'