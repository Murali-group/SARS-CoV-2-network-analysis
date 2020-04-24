
import sys
from collections import defaultdict
import numpy as np
from scipy import sparse as sp

# my local imports
from . import fastsinksource_runner as fastsinksource
from . import sinksource_bounds_runner as ss_bounds
from . import genemania_runner as genemania
from . import svm_runner as svm
from . import logistic_regression_runner as logistic_regression


LibMapper = {
    'sinksource': fastsinksource,
    'sinksourceplus': fastsinksource,
    'fastsinksource': fastsinksource,
    'fastsinksourceplus': fastsinksource,
    'sinksource_bounds': ss_bounds,
    'sinksourceplus_bounds': ss_bounds,
    'local': fastsinksource,
    'localplus': fastsinksource,
    'genemania': genemania,
    'genemaniaplus': genemania,
    'svm': svm,
    'logistic_regression': logistic_regression,
}


class Runner(object):
    '''
    A runnable analysis to be incorporated into the pipeline
    *kwargs*: Checked for out_pref, verbose, and forecealg options.
        Can be used to pass additional variables needed by the runner
    '''
    def __init__(self, name, net_obj, ann_obj,
                 out_dir, params, **kwargs):
        self.name = name
        self.net_obj = net_obj
        self.ann_obj = ann_obj
        self.out_dir = "%s/%s/" % (out_dir, name)
        params.pop('should_run', None)  # remove the should_run parameter
        self.params = params
        self.kwargs = kwargs
        self.verbose = kwargs.get('verbose', False) 
        self.forced = kwargs.get('forcealg', False) 
        # for term-based algorithms, can limit the terms for which they will be run
        self.terms_to_run = kwargs.get('terms_to_run', ann_obj.terms)
        # also can limit the nodes for which scores are stored with this
        self.target_prots = kwargs.get('target_nodes', np.arange(len(ann_obj.prots)))

        # track measures about each run (e.g., running time)
        self.params_results = defaultdict(int) 
        # store the node scores for each GO term in a sparse matrix
        # using lil matrix so 0s are automatically not stored
        self.term_scores = sp.lil_matrix(ann_obj.ann_matrix.shape, dtype=np.float)

        # keep track of the evaluation settings and weighting method for adding to the output
        eval_str = get_eval_str(name, **kwargs)
        # if cross valudation is run, then the eval str will be set there, so don't include it here
        if kwargs.get('cross_validation_folds') is not None:
            eval_str = "" 
        self.setupParamsStr(eval_str+net_obj.weight_str, params, name)
        default_out_pref = "%s/pred-scores%s%s" % (
            self.out_dir, self.params_str, self.kwargs.get('postfix',''))
        self.out_pref = kwargs.get('out_pref', default_out_pref)

        # for the supervised classification methods, train_mat and test_mat are set during cross validation  
        # leave them as None so they are ignored during prediction mode
        self.train_mat = None
        self.test_mat = None


    # if the algorithm is not inmplemented in Python (e.g., MATLAB, R)
    # use this function to setup files and such
    def setupInputs(self):
        return LibMapper[self.name].setupInputs(self)

    # run the method
    def run(self):
        return LibMapper[self.name].run(self)

    # if the method is not in Python and was called elsewhere (e.g., R), 
    # then parse the outputs of the method
    def setupOutputs(self, **kwargs):
        return LibMapper[self.name].setupOutputs(self, **kwargs)

    # setup the params_str used in the output file
    def setupParamsStr(self, weight_str, params, name):
        self.params_str = LibMapper[self.name].setup_params_str(weight_str, params, name)

    def get_alg_type(self):
        return LibMapper[self.name].get_alg_type()


def get_eval_str(name, **kwargs):
    # if we sampled negative examples when running this method, add that to the output string
    neg_factor = kwargs.get('sample_neg_examples_factor')
    num_reps = kwargs.get('num_reps', 1)
    eval_str = "" 
    # TODO make a better way of indicating an alg needs negative examples than just having 'plus' in the name
    if 'plus' not in name and neg_factor is not None:
        # make sure its treated as an integer if it is one
        neg_factor = int(neg_factor) if int(neg_factor) == neg_factor else neg_factor
        eval_str = "-rep%s-nf%s" % (num_reps, neg_factor)
    return eval_str


def get_weight_str(dataset):
    """
    *dataset*: dictionary of datset settings and file paths. Used to get the 'net_settings': 'weight_method' (e.g., 'swsn' or 'gmw')
    """
    unweighted = dataset['net_settings'].get('unweighted', False) if 'net_settings' in dataset else False
    weight_method = ""
    # if the weight method is to simply add the networks together, I did not include anything in the output file
    if dataset.get('multi_net',False) is True and \
       dataset.get('net_settings') is not None and dataset['net_settings'].get('weight_method','add') != 'add': 
        weight_method = "-"+dataset['net_settings']['weight_method']
    weight_str = '%s%s' % (
        '-unw' if unweighted else '', weight_method)
    return weight_str


def get_runner_params_str(name, params, dataset=None, weight_str="", **kwargs):
    """
    Get the params string for a runner without actually creating the runner object
    *dataset*: dictionary of datset settings and file paths. Used to get the 'net_settings': 'weight_method' (e.g., 'swsn' or 'gmw')
    """
    # get the weight str used when writing output files
    if dataset is not None:
        weight_str = get_weight_str(dataset) 
    eval_str = get_eval_str(name, **kwargs) 
    settings_str = eval_str + weight_str

    return LibMapper[name].setup_params_str(settings_str, params, name=name)

