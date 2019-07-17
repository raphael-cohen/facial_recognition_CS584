import pickle

import numpy as np

import pandas as pd
import scipy
from sklearn import metrics
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# from sklearn.cross_validation import StratifiedShuffleSplit


def main():

    data = np.array(pickle.load(open('X.pkl', 'rb')))
    labels = np.array(pickle.load(open('y_bush_vs_others.pkl', 'rb')))
    labels = np.array([ 1 if x == 0 else 0 for x in labels ])
    print(np.sum(labels))
    # weights = n_samples / (n_classes * np.bincount(y))
    # labels = labels.reshape((data.shape[0],))
    print(labels.shape)
    classifiers = {
        "LDA" : LinearDiscriminantAnalysis,
        "QDA" : QuadraticDiscriminantAnalysis,
        "LR" : LogisticRegression,
        "BNB" : BernoulliNB,
        "KNN" : KNeighborsClassifier,
        "SVM" : SVC
    }
    # nb_positive = np.bincount(labels)
    # print(nb_positive)

    #
    # X_train, X_test, y_train, y_test = train_test_split(
    #     data, labels, test_size=1.0 / 3.0, random_state=2518, stratify = labels)

    # KNN(X_train, y_train, X_test, y_test, "bush_models")
    # SVM(X_train, y_train, X_test, y_test, kernel = "linear", proba = True, folder = "bush_models")
    # LOGISTIC(X_train, y_train, X_test, y_test, "bush_models")
    # LDA(X_train, y_train, X_test, y_test, "bush_models")
    # QDA(X_train, y_train, X_test, y_test, "bush_models")
    # BNB(X_train, y_train, X_test, y_test, "bush_models")


def to_pkl(filename, sep):
    data = pd.read_csv(filename, sep=' ')
    pickle.dump(data, open(filename + '.pkl', 'wb'))

#
# def classifier(clf, X_train, y_train, X_test, y_test):
#
#     # clf = KNeighborsClassifier(n_neighbors=7, n_jobs=7)
#
#     clf.fit(X_train, y_train)
#
#     joblib.dump(neigh, 'knn_k7.joblib')
#
#     print(neigh.score(X_test, y_test))


def KNN(X_train, y_train, X_test, y_test, folder = "bush_models"):

    neigh = KNeighborsClassifier(n_neighbors=7, n_jobs=7)

    neigh.fit(X_train, y_train)

    joblib.dump(neigh, folder+'/knn_k7.joblib')

    print(neigh.score(X_test, y_test))


def SVM(X_train, y_train, X_test, y_test, kernel = "linear", proba = False, folder = "bush_models"):

    svm = SVC(kernel=kernel, probability=proba)  # , C=0.025)
    # neigh = KNeighborsClassifier(n_neighbors=7, n_jobs=7)
    svm.fit(X_train, y_train)

    joblib.dump(svm, folder+"/svm_"+kernel+"_kernel.joblib")

    print(svm.score(X_test, y_test))


def LOGISTIC(X_train, y_train, X_test, y_test, weights={0: 1, 1: 1}, folder = "bush_models"):

    lr = LogisticRegression(random_state=2518, solver='liblinear',
                            multi_class='ovr')
    lr = lr.fit(X_train, y_train)

    joblib.dump(lr, folder+'/lr.joblib')

    print(lr.score(X_test, y_test))


def LDA(X_train, y_train, X_test, y_test, n_components = None, weights={0: 1, 1: 1}, folder = "bush_models"):

    lda = LinearDiscriminantAnalysis(n_components = n_components)
    lda = lda.fit(X_train, y_train)

    joblib.dump(lda, folder+'/lda.joblib')

    print(lda.score(X_test, y_test))


def QDA(X_train, y_train, X_test, y_test, weights={0: 1, 1: 1}, folder = "bush_models"):

    qda = QuadraticDiscriminantAnalysis()
    qda = qda.fit(X_train, y_train)

    joblib.dump(qda, folder+'/qda.joblib')

    print(qda.score(X_test, y_test))


#Not good
def BNB(X_train, y_train, X_test, y_test, weights={0: 1, 1: 1}, alpha = 1.0, folder = "bush_models"):

    bnb = BernoulliNB(alpha = alpha)

    bnb = bnb.fit(X_train, y_train)

    joblib.dump(bnb, folder+"/"+str(alpha)+'_bnb.joblib')

    print(bnb.score(X_test, y_test))



#
# def DT(X,y):
#     DecisionTreeClassifier(max_depth=5),
#
# def RF(X,y):
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

if __name__ == '__main__':
    main()
