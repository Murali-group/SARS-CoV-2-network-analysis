
from sklearn.svm import LinearSVC
#from sklearn.svm import SVC
#from sklearn.calibration import CalibratedClassifierCV
#from scikit.learn.svm.sparse import SVC


def training(P, labels, penalty='l2', loss='squared_hinge', tol=0.0001, C=1.0, max_iters=1000, **kwargs):
    '''
    *P* : features corresponding to the samples in the training set
    *labels*: labels corresponding to the samples in the training set

    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html for details of the parameters
    '''

    clf = LinearSVC(penalty=penalty, loss=loss, tol=float(tol), C=float(C), max_iter=int(max_iters))
    clf.fit(P, labels)

    return clf


def testing(clf, P):
    # decision_function essentially computes the distance from the hyperplane
    predict = clf.decision_function(P)
    return predict
