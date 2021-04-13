from .alg_utils import str_
#import sys, os
#sys.path.append(os.path.abspath(os.path.join('..', '..', '..', 'PathLinker')))
import PageRank

def setupInputs(run_obj):
    print("RWR: setupInputs")
    run_obj.ann_matrix = run_obj.ann_obj.ann_matrix
    run_obj.terms = run_obj.ann_obj.terms
    return


# run the method
def run(run_obj):
    sparse_netx_graphs = run_obj.net_obj.sparse_netx_graphs
    params = run_obj.params

    print("Number of networks: %s" % (sparse_netx_graphs.length))
    print("Running +RWR+ with these parameters: %s" % (params))

    finalProbs = PageRank.pagerank(sparse_netx_graphs[0], #weights=teleProbs,
            q=params.q, eps=params.eps, maxIters=params.max_iters, verbose=True)
    print("Completed RWR")
    print(finalProbs)
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