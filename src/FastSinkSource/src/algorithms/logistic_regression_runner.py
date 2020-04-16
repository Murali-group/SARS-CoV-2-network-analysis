
from tqdm import tqdm, trange
#from rpy2.robjects import *
#from rpy2 import robjects as ro
import numpy as np
from scipy import sparse
import time
from . import alg_utils
from . import logistic_regression as logReg


def setupInputs(run_obj):
    run_obj.ann_matrix = run_obj.ann_obj.ann_matrix
    run_obj.terms = run_obj.ann_obj.terms
    run_obj.prots = run_obj.ann_obj.prots
    run_obj.termidx = run_obj.ann_obj.term2idx
    run_obj.protidx = run_obj.ann_obj.node2idx

    if run_obj.net_obj.weight_swsn:
        W, process_time = run_obj.net_obj.weight_SWSN(run_obj.ann_matrix)
        run_obj.params_results['%s_weight_time'%(run_obj.name)] += process_time
    elif run_obj.net_obj.weight_gmw:
        # this will be handled on a term by term basis
        run_obj.P = None
    else:
        W = run_obj.net_obj.W
        run_obj.P = alg_utils._net_normalize(W)
    '''
    # if influence matrix is to be used, then obtain the influence matrix of W
    if run_obj.net_obj.influence_mat:
        run_obj.P = alg_utils.influenceMatrix(W, ss_lambda=run_obj.params.get('lambda', None))
    else:
    '''

    return


def setup_params_str(weight_str, params, name):
    iters = params['max_iters']
    return "{}-{}-maxi{}".format(weight_str, name, str_(iters))


def setupOutputs(run_obj):
    return


def run(run_obj):
    """
    This script performs logistic regression by building a classifier for each term in the ontology
    """

    params_results = run_obj.params_results
    P, alg, params = run_obj.P, run_obj.name, run_obj.params

    # get the labels matrix and transpose it to have label names as columns
    ann_mat = run_obj.ann_matrix
    max_iters = params['max_iters']
    print("Running %s with these parameters: %s" % (alg, params))
    # see if train and test annotation matrices from the cross validation pipeline exist
        # if not, set train and test to the original annotation matrix itself
    if run_obj.train_mat is not None and run_obj.test_mat is not None:
        #print("Performing cross validation")
        run_obj.cv = True
        train_mat = run_obj.train_mat
        test_mat = run_obj.test_mat
    else:
        run_obj.cv = False
        train_mat = ann_mat
        test_mat = ann_mat

    # stores the scores for all the terms
    scores = sparse.lil_matrix(ann_mat.shape, dtype=np.float)        #   dim: term x genes

    for term in tqdm(run_obj.terms_to_run):    
        idx = run_obj.termidx[term]

        if run_obj.net_obj.weight_gmw is True:
            # get the row corresponding to the current terms annotations 
            y = run_obj.ann_matrix[idx,:]
            start_time = time.process_time()
            # weight the network for each term individually
            W,_,_ = run_obj.net_obj.weight_GMW(y.toarray()[0], term)
            P = alg_utils.normalizeGraphEdgeWeights(W, ss_lambda=params.get('lambda', None))
            params_results['%s_weight_time'%(alg)] += time.process_time() - start_time

        # compute the train gene indices of the annotations for the given label
        train_pos, train_neg = alg_utils.get_term_pos_neg(train_mat,idx)
        train_set = sorted(list(set(train_pos)|set(train_neg)))

        if len(train_pos)==0:
            print("Skipping term, 0 positive examples")
            continue

        if run_obj.cv:
            # if cross validation, then obtain the test gene set on which classifier should be tested
            test_pos, test_neg = alg_utils.get_term_pos_neg(test_mat, idx)
            test_set = set(test_pos) | set(test_neg)
            test_set = sorted(list(test_set))
        else:
            # set all unlabeled genes to the test set
            test_set = sorted(list(set(range(P.shape[0])) - set(train_set)))

        # obtain the feature vector only for the genes in the training set
        X_train = P[train_set, :]
        # obtain the feature vector only for the genes in the testing set
        X_test = P[test_set, :]
        # obtain the labels matrix corresponding to genes in the training set
        y_train = train_mat.transpose()[train_set, :]
        y_train = sparse.lil_matrix(y_train) 

        # get the column of training data for the given label 
        lab = y_train[:,idx].toarray().flatten()

        # now train the model on the constructed training data and the column of labels
        clf = logReg.training(X_train, lab, max_iters)

        # make predictions on the constructed training set
        predict = logReg.testing(clf, X_test)
        predict = predict.tolist()

        # get the current scores for the given label l
        curr_score = scores[idx].toarray().flatten()
        # for the test indices of the current label, set the scores
        curr_score[test_set] = predict
        curr_score[train_pos] = 1
        # add the scores produced by predicting on the current label of test set to a combined score matrix
        scores[idx] = curr_score

    run_obj.term_scores = scores
    run_obj.params_results = params_results


def str_(s):
    return str(s).replace('.','_')

