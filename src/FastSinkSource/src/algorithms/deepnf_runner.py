
from tqdm import tqdm, trange
#from rpy2.robjects import *
#from rpy2 import robjects as ro
import numpy as np
from scipy import sparse
import time

from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

from . import alg_utils
from . import logistic_regression_runner
from .deepNF import net_embedding as deepnf


def setupInputs(run_obj):
    run_obj.ann_matrix = run_obj.ann_obj.ann_matrix
    run_obj.terms = run_obj.ann_obj.terms
    run_obj.prots = run_obj.ann_obj.prots
    run_obj.termidx = run_obj.ann_obj.term2idx
    run_obj.protidx = run_obj.ann_obj.node2idx
    params = run_obj.params

    if hasattr(run_obj, 'P'):
        print("deepNF features already loaded")
    else:
        out_pref = run_obj.net_obj.out_pref
        models_path = "%sdeepnf-embeddings/" % out_pref
        results_path = "%sdeepnf-embeddings/" % out_pref

        # if there's just a single network, then use an autoencoder
        # otherwise, use the multimodal autoencoder 
        model_type = "mda" if run_obj.net_obj.multi_net else "ae"
        params['model_type'] = model_type
        # need to convert the input networks to non-sparse numpy matrices
        print("Converting sparse matrices to full matrices for deepNF")
        if run_obj.net_obj.multi_net:
            nets = [net.A for net in run_obj.net_obj.sparse_networks]
        else:
            nets = [run_obj.net_obj.W.A]

        print("Generating deepNF embeddings (model_type: %s) with these parameters:" % (model_type))
        print(params)
        # if the embeddings already exist, then load them
        # generate the deepnf embeddings
        run_obj.features = deepnf.build_net_embeddings(
            model_type, models_path, results_path, params['hidden_dims'], nets,
            params['epochs'], params['batch_size'], params['noise_factor'])

        # set the "network" as the features
        run_obj.P = run_obj.features

    # make sure these aren't run later
    run_obj.net_obj.weight_gmw = False 
    run_obj.net_obj.weight_swsn = False 
    '''
    if run_obj.net_obj.weight_swsn:
        W, process_time = run_obj.net_obj.weight_SWSN(run_obj.ann_matrix)
        run_obj.params_results['%s_weight_time'%(run_obj.name)] += process_time
    elif run_obj.net_obj.weight_gmw:
        # this will be handled on a term by term basis
        run_obj.P = None
    else:
        W = run_obj.net_obj.W
        run_obj.P = alg_utils._net_normalize(W)
    # if influence matrix is to be used, then obtain the influence matrix of W
    if run_obj.net_obj.influence_mat:
        run_obj.P = alg_utils.influenceMatrix(W, ss_lambda=run_obj.params.get('lambda', None))
    else:
    '''

    return


def setup_params_str(weight_str, params, name):
    #penalty = params.get('penalty', 'l2')
    reg_strength = params.get('C', 1.0)

    params_str = "dims{}-ep{}-bs{}-nf{}-C{}".format(
        #params['model_type'],
        '-'.join(list(map(str, params['hidden_dims']))), params['epochs'],
        params['batch_size'], str_(params['noise_factor']), str_(reg_strength))
    if params.get('gamma') and params['gamma'] != 0:
        params_str += '-g{}'.format(params['gamma'])
    return params_str


def setupOutputs(run_obj):
    return


def run(run_obj):
    # temporarilty switch to svm
    #name = run_obj.name
    #run_obj.name = 'svm'
    #logistic_regression_runner.run(run_obj)
    #run_obj.name = name
    P = run_obj.P
    alg = run_obj.name
    params = run_obj.params
    C, gamma = float(params.get('C',1)), float(params.get('gamma',1))

    # get the labels matrix and transpose it to have label names as columns
    ann_mat = run_obj.ann_matrix
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
    scores = sparse.lil_matrix(ann_mat.shape, dtype=np.float)        # dim: term x genes
    for term in run_obj.terms_to_run:
        idx = run_obj.termidx[term]
        # compute the train gene indices of the annotations for the given label
        train_pos, train_neg = alg_utils.get_term_pos_neg(train_mat,idx)
        train_set = sorted(list(set(train_pos) | set(train_neg)))

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
        labels = y_train[:,idx].toarray().flatten()

        # if alg == "logistic_regression":
        #     # now train the model on the constructed training data and the column of labels
        #     clf = logReg.training(X_train, lab, **params)
        #     # make predictions on the constructed training set
        #     predict = logReg.testing(clf, X_test)
        # elif alg == "svm":
        #     # now train the model on the constructed training data and the column of labels
        #     clf = svm.training(X_train, lab, **params)
        #     # make predictions on the constructed training set
        #     predict = svm.testing(clf, X_test)

        # The original deepNF implementation does nested cross-validation (see deepNF/validation.py).
        # Since we're using randomly sampled negative examples to train the classifier, we don't want to overfit to the negatives
        K_rbf = kernel_func(X_train, param=gamma)
        K_rbf_test = kernel_func(X_test, X_train, param=gamma)

        #clf = SVC(C=C, kernel='precomputed', probability=True)
        clf = SVC(C=C, kernel='precomputed')
        clf.fit(K_rbf, labels)

        # decision_function essentially computes the distance from the hyperplane
        predict = clf.decision_function(K_rbf_test)

        # get the current scores for the given label l
        curr_score = scores[idx].toarray().flatten()
        # for the test indices of the current label, set the scores
        curr_score[test_set] = predict
        curr_score[train_pos] = 1
        # add the scores produced by predicting on the current label of test set to a combined score matrix
        scores[idx] = curr_score

    run_obj.term_scores = scores
    #run_obj.params_results = params_results


def kernel_func(X, Y=None, param=0):
    if param != 0:
        K = rbf_kernel(X, Y, gamma=param)
    else:
        K = linear_kernel(X, Y)

    return K


def str_(s):
    return str(s).replace('.','_')

