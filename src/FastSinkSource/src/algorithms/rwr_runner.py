import json
from src.PathLinker import PageRank

def setupInputs(run_obj):
    print("setupInputs")
    print("run_obj")

    print("run_obj.ann_obj.ann_matrix>")
    print(run_obj.ann_obj.ann_matrix)
    print("run_obj.ann_obj.terms>")
    print(run_obj.ann_obj.terms)
    print("run_obj.net_obj.W")
    print(run_obj.net_obj.W)
    return


# run the method
def run(run_obj):
    print("run")
    print("run_obj")
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


def get_alg_type():
    return