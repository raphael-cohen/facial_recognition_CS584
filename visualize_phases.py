import multiprocessing
import pickle
# import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from pprint import pprint

import numpy as np


def main():

    local_path = "/media/Onedrive/IIT/Courses/CS584 Machine_Learning/project/phase2/"#"/export/home/rcohen6/courses/ML/project/phase1/"
    # visualize1_svm()
    # res = pickle.load(open(local_path + "bush_linear_grid_svc3_1024serv2.pkl", 'rb'))
    # print(res.keys())
    # print(res['params'])
    # print(res['param_C'])
    # # print(res['param_kernel'])
    # print(res['rank_test_f1'])
    #
    # print(res['mean_test_f1'])
    # visualize1_knn(local_path + "williams_knn_pca_dim1024.pkl")
    # visualize2()
    # pickle.dump([0.134310134310134, 0.50536821], open(local_path + "williams.pkl", "wb"))
    # pickle.dump([0.146593511576159, 0.51976755], open(local_path + "bush.pkl", "wb"))
    w = pickle.load(open(local_path + "williams.pkl", 'rb'))
    b = pickle.load(open(local_path + "bush.pkl", 'rb'))
    print(w)
    print(b)


def visualize1_svm():

    key_order = ["fit_time", "score_time",
                 "test_f1", "test_precision", "test_recall"]


    # res = pickle.load(open(local_path + "bush_svm.pkl", 'rb'))
    res = pickle.load(open(local_path + "williams_svm.pkl", 'rb'))
    C = [10**(-3), 1.0, 10**(3), 10**(5), 10**(8)]
    for k in res.keys():
        print("\nkernel : " + k)
        for r, c in zip(res[k], C):
            print("\nc: " + str(c) + "\n")
            for k1 in key_order:#r.keys():
                print(k1 + ": ", r[k1])


def visualize1_knn(file):

    key_order = ["fit_time", "score_time",
                 "test_f1", "test_precision", "test_recall"]

    # local_path = "/export/home/rcohen6/courses/ML/project/phase1/"
    # res = pickle.load(open(local_path + "bush_knn.pkl", 'rb'))
    res = pickle.load(open(file, 'rb'))

    nn = [1, 3, 5]
    for k in res.keys():
        print("\nkernel : " + str(k))
        # for k1 in res[k].keys():
        for k1 in key_order:
            print(k1 + ": ", res[k][k1])

    print("moyenne pour le meilleur : \t'{0}'".format(np.mean(res[1]['test_f1'])))
        # print(res[k])

        # for r, c in zip(res[k], C):
        #     print("\nc: " + str(c) + "\n")
        #     for k1 in r.keys():
        #         print(k1+": ",r[k1])


def visualize2():
    local_path = "/export/home/rcohen6/courses/ML/project/phase2/"
    files = [f for f in listdir(local_path) if isfile(join(local_path, f))]

    for pca_dim in [1024, 512, 256, 128, 64]:
        for clf in ["knn", "svm"]:
            for n in ["bush", "williams"]:
                fs = [f for f in files if (clf in f and n in f)]
                for f in fs:
                    print_phase2(local_path, f)


def print_phase1(path, f):

    _, classifier, name, param, _ = f.replace("_", ".").split(".")

    res = pickle.load(open(path + f, 'rb'))
    if classifier == "knn":
        print("\nclassifier type: ", classifier, "\nname: ",
              name, "\nn_neighbors: ", param, '\n')
    else:
        print("\nclassifier type: ", classifier,
              "\nname: ", name, "\nkernel: ", param, '\n')
    for key in res.keys():
        print(key, res[key])


def print_phase2(path, f):

    _, classifier, name, pca_dm, param, _ = f.replace("_", ".").split(".")

    res = pickle.load(open(path + f, 'rb'))
    if classifier == "knn":
        print("\nclassifier type: ", classifier, "\nname: ", name,
              "\npca_dim: ", pca_dm, "\nn_neighbors: ", param, '\n')
    else:
        print("\nclassifier type: ", classifier, "\nname: ", name,
              "\npca_dim: ", pca_dm, "\nkernel: ", param, '\n')
    for key in res.keys():
        print(key, res[key])


if __name__ == '__main__':
    main()
