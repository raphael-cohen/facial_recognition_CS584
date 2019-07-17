import multiprocessing as mp
import pickle

import numpy as np

import pandas as pd
import scipy
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def main():

    local_path = "/export/home/rcohen6/courses/ML/project/"

    data = np.array(pickle.load(open(local_path + 'X.pkl', 'rb')))

    labels = np.array(pickle.load(
        open(local_path + 'y_bush_vs_others.pkl', 'rb'))).ravel()

    labels2 = np.array(pickle.load(
        open(local_path + 'y_williams_vs_others.pkl', 'rb'))).ravel()

    # Run KNN with all parameters and store the results with pkl
    # mutli_knn(data, labels, labels2)

    # Run SVM with all parameters and store the results with pkl
    # multi_svm(data, labels, labels2)
    grid_search_svm(data, labels, labels2)


def process_cv_knn(X, y, nn, name, d):

    print("neighbors = ", nn)
    n_split = 3
    knn = KNeighborsClassifier(n_neighbors=nn)

    cv_results = cross_validate(knn, X, y, cv=StratifiedKFold(n_splits=n_split, shuffle=True, random_state=2518),
                                scoring=('precision', 'recall', 'f1'), return_train_score=False, n_jobs=n_split, verbose=10)
    print(cv_results)

    d[nn] = cv_results


def process_cv_svm(X, y, kernel, name, d):

    print("kernel = ", kernel)
    gamma_k = ['rbf', 'poly', 'sigmoid']

    gamma = ["auto"]
    degree = [3]

    if kernel in gamma_k:
        gamma = ["auto", "scale"]
        if kernel == 'poly':
            degree = [1, 2, 3, 4]

    n_split = 3
    C = [10**(-3), 1.0, 10, 10**(3), 10**(5)]
    res = []

    for g in gamma:
        for d in degree:
            for c in C:
                print([name,kernel,g,d,c])
                svc = SVC(kernel=kernel, C=c, gamma=g, degree=d)
                cv_results = cross_validate(svc, X, y, cv=StratifiedKFold(n_splits=n_split, shuffle=True, random_state=2518),
                                            scoring=('precision', 'recall', 'f1'), return_train_score=False, n_jobs=n_split, verbose=10)

                cv_results["parameters"] = [kernel, gamma, c, d, name]

                res.append(cv_results)

    d[kernel] = res

    print(res)


def mutli_knn(data, labels, labels2):

    nn = [1, 3, 5]
    local_path = "/export/home/rcohen6/courses/ML/project/phase1/"
    workers = []

    with mp.Manager() as manager:

        d_bush = manager.dict()
        d_williams = manager.dict()

        for n in nn:

            p1 = mp.Process(
                target=process_cv_knn, args=(data, labels, n, "bush", d_bush,))
            p2 = mp.Process(
                target=process_cv_knn, args=(data, labels2, n, "williams", d_williams,))

            workers.append(p1)
            workers.append(p2)

            p1.start()
            p2.start()

        for worker in workers:
            worker.join()

        pickle.dump(dict(d_bush), open(local_path + "bush_knn.pkl", "wb"))
        pickle.dump(dict(d_williams), open(
            local_path + "williams_knn.pkl", "wb"))
        print("OK")


def multi_svm(data, labels, labels2):

    local_path = "/export/home/rcohen6/courses/ML/project/phase1/"
    workers = []
    kernels = ['rbf', 'linear', 'poly', 'sigmoid']

    name_b = local_path + "bush_svm.pkl"
    name_w = local_path + "williams_svm.pkl"

    with mp.Manager() as manager:

        d_bush = manager.dict()
        d_williams = manager.dict()

        for k in kernels:

            p1 = mp.Process(
                target=process_cv_svm, args=(data, labels, k, "bush", d_bush,))
            p2 = mp.Process(
                target=process_cv_svm, args=(data, labels2, k, "williams", d_williams,))

            workers.append(p1)
            workers.append(p2)

            p1.start()
            p2.start()

        for worker in workers:
            worker.join()

        pickle.dump(dict(d_bush), open(name_b, "wb"))
        pickle.dump(dict(d_williams), open(name_w, "wb"))
        print("OK")


def grid_search_svm(data, labels, labels2):
    parameters = {'kernel':('rbf', 'linear', 'poly', 'sigmoid'),
                  'C':[10**(-3), 1.0, 10, 10**(3), 10**(5)], 'gamma':('auto','scale'),
                  'degree':[1,2,3,4]}

    svc = SVC()

    clf = GridSearchCV(svc, parameters, cv=3, n_jobs=-1, verbose=10)
    clf.fit(data, labels2)

    try:
        pickle.dump(dict(clf), open("/export/home/rcohen6/courses/ML/project/phase1/williams_grid_svc.pkl", "wb"))
    except Exception as e:
        print("1 pas ok")

    try:
        pickle.dump(clf, open("/export/home/rcohen6/courses/ML/project/phase1/williams_grid_svc2.pkl", "wb"))
    except Exception as e:
        print("2 pas ok")

    try:
        pickle.dump(clf.cv_results_, open("/export/home/rcohen6/courses/ML/project/phase1/williams_grid_svc3.pkl", "wb"))
    except Exception as e:
        print("3 pas ok")


if __name__ == '__main__':
    main()
