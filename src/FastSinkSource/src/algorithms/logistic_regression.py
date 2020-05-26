
from sklearn.linear_model import LogisticRegression


def training(P, labels, penalty='l2', tol=0.0001, C=1.0, max_iter=100, **kwargs):
    '''
    *P* : features corresponding to the samples in the training set
    *labels*: labels corresponding to the samples in the training set

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html for details of the parameters
    '''
    logReg_clf = LogisticRegression(
        penalty=penalty, tol=float(tol), C=float(C), max_iter=int(max_iter))
    logReg_clf.fit(P, labels)

    return logReg_clf


def testing(clf, P):
    predict = clf.predict_proba(P)[:,1]
    return predict
