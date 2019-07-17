import multiprocessing
import pickle
# import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from pprint import pprint
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


import numpy as np

def main():

    # # other()
    # classic("bush_svm.bak.pkl")
    w = pickle.load(open("williams.pickle", 'rb'))
    b = pickle.load(open("bush.pickle", 'rb'))
    print(w)
    print(b)
    # key_order = ["fit_time", "score_time",
    #              "test_f1", "test_precision", "test_recall"]
    #
    # # local_path = "/export/home/rcohen6/courses/ML/project/phase1/"
    # # res = pickle.load(open(local_path + "bush_svm.pkl", 'rb'))
    # res = pickle.load(open("williams_grid_svc2.pkl", 'rb'))
    # print(res.cv_results_['rank_test_score'])



def classic(file):

    key_order = ["fit_time", "score_time",
                 "test_f1", "test_precision", "test_recall"]
    kernels = ['rbf', 'linear', 'poly', 'sigmoid']

    # local_path = "/export/home/rcohen6/courses/ML/project/phase1/"
    # res = pickle.load(open(local_path + "bush_svm.pkl", 'rb'))
    knn_res = pickle.load(open("bush_knn.pkl", 'rb'))
    # print(knn_res)
    res = pickle.load(open("bush_svm.bak.pkl", 'rb'))
    # print(res['rbf'][3])

    bush = [np.mean(knn_res[1]['test_f1']), np.mean(knn_res[3]['test_f1']), np.mean(knn_res[5]['test_f1']), np.mean(res['rbf'][3]['test_f1'])]
    print(bush)

    pickle.dump(bush, open("bush.pkl", "wb"))

    knn_res = pickle.load(open("williams_knn.pkl", 'rb'))
    res = pickle.load(open("williams_svm.bak.pkl", 'rb'))

    williams = [np.mean(knn_res[1]['test_f1']), np.mean(knn_res[3]['test_f1']), np.mean(knn_res[5]['test_f1']), np.mean(res['rbf'][3]['test_f1'])]
    print(williams)

    pickle.dump(williams, open("williams.pkl", "wb"))



def grid(file):
    print("bob")


def other():

    n_split = 3
    data = np.array(pickle.load(open('/media/Onedrive/IIT/Courses/CS584 Machine_Learning/project/X.pkl', 'rb')))

    labels = np.array(pickle.load(
        open('/media/Onedrive/IIT/Courses/CS584 Machine_Learning/project/y_bush_vs_others.pkl', 'rb')))
    print(labels)
    #
    # labels2 = np.array(pickle.load(
    #     open('y_williams_vs_others.pkl', 'rb')))
    #
    svc = SVC(kernel='rbf', C=10**5, gamma='scale')
    cv_results = cross_validate(svc, data, labels, cv=StratifiedKFold(n_splits=n_split, shuffle=True, random_state=2518),
                                scoring=('precision', 'recall', 'f1'), return_train_score=False, n_jobs=n_split, verbose=10)

    print(cv_results)

    try:
        pickle.dump(cv_results, open("rbf_105_scale_bush.pkl", "wb"))
    except Exception as e:
        print("2 pas ok")

if __name__ == '__main__':
    main()
