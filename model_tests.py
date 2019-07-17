import matplotlib.pyplot as plt
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
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

from experiment_bush import *



def main():

    data = np.array(pickle.load(open('X.pkl', 'rb')))
    labels = np.array(pickle.load(open('y_bush_vs_others.pkl', 'rb')))
    labels = np.array([ 1 if x == 0 else 0 for x in labels ])
    # weights = n_samples / (n_classes * np.bincount(y))
    labels = labels.reshape((data.shape[0],))
    print(labels.shape)
    classifiers = {
        "LDA" : LinearDiscriminantAnalysis,
        "QDA" : QuadraticDiscriminantAnalysis,
        "LR" : LogisticRegression,
        "BNB" : BernoulliNB,
        "KNN" : KNeighborsClassifier,
        "SVM" : SVC
    }

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=1.0 / 3.0, random_state=2518, stratify = labels)


    clf = joblib.load('bush_models/svm_linear_kernel.joblib')
    # clf = joblib.load('svm_linear_kernel.joblib')
    clf2 = joblib.load('bush_models/lr.joblib')
    lda_clf = joblib.load('bush_models/lda.joblib')
    # plot_roc(clf, X_test, y_test)
    # plot_lda(clf, X_test, y_test)
    # plot_pca(clf, X_test, y_test)
    plot_general(lda_clf, clf, clf2, X_test, y_test)
    # print(clf.score(X_test, y_test))


def plot_roc(clf, X_test, y_test):

    threshold = 0.1
    prob = clf.predict_proba(X_test)
    # predictions = [0 if x[0] > 0.3 else 1 for x in prob]
    # predictions = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    threshold = 0.000000001
    predictions = proba[:,0] >= threshold #clf.predict(X_test)

    fpr = sum(y_test)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
    print(thresholds)
    cm = metrics.confusion_matrix(y_test, predictions)
    print("total negative:\n", cm[1,1]+cm[1,0])
    print("confustion matrix:\n",cm)
    print("f1 score:\n",metrics.f1_score(y_test, predictions))
    print("precision:\n", metrics.precision_score(y_test, predictions))
    print("recall:\n", metrics.recall_score(y_test, predictions))
    print("cohen kappa score:\n", metrics.cohen_kappa_score(y_test, predictions))
    print("true positive rate:\n", cm[0,0]/(cm[0,0]+cm[0,1]))

    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_lda(clf, X_test, y_test):

    cmap = [('Qualitative', [
            'Pastel1', 'Pastel2'])]

    X_test_transformed = clf.transform(X_test).reshape(X_test.shape[0])
    print(np.where(y_test == 1))
    fourierT = np.fft.fft(X_test_transformed).real
    proba = clf.predict_proba(X_test)
    threshold = 0.5
    decision = proba[:,0] >= threshold #clf.predict(X_test)
    color_pred = np.where(decision ==1, "red", "blue")
    color_real = np.where(y_test ==1, "blue", "red")
    # print()Oct 22, 2018 1:32 PM

    # fourierT_true =np.fft.fft(X_test_transformed).real
    # fourierT_false = np.fft.fft(X_test_transformed).real
    plt.subplot(3,1,1)
    plt.scatter(y = fourierT.real, x = X_test_transformed, c = color_real)#y_test+2)
    plt.subplot(3,1,2)
    plt.scatter(y = fourierT, x = X_test_transformed, c = color_pred)

    threshold = 0.0001
    decision = proba[:,0] >= threshold #clf.predict(X_test)
    color_pred = np.where(decision ==1, "red", "blue")
    plt.subplot(3,1,3)
    plt.scatter(y = fourierT, x = X_test_transformed, c = color_pred)

    print(X_test_transformed.shape)
    # plt.scatter(y = np.zeros(X_test_transformed.shape[0]), x = X_test_transformed[:], c=y_test, alpha = 0.8, marker='+', s = 200)
    plt.show()


def plot_pca(clf, X_test, y_test):

    proba = clf.predict_proba(X_test)
    threshold = 0.5
    # decision = proba[:,0] >= threshold #clf.predict(X_test)
    decision = clf.predict(X_test)
    color_pred = np.where(decision ==1, "blue", "red")
    color_real = np.where(y_test ==1, "blue", "red")


    pca = PCA(n_components=2)
    pca.fit_transform(X_test)
    X_test_transformed = pca.fit_transform(X_test)

    plt.subplot(2,1,1)
    plt.scatter(x = X_test_transformed[:,0], y = X_test_transformed[:,1], c = color_real)#y_test+2)
    plt.subplot(2,1,2)
    plt.scatter(x = X_test_transformed[:,0], y = X_test_transformed[:,1], c = color_pred)

    plt.show()
    # print(np.where(y_test == 1))

def plot_general(lda_clf, clf, clf2, X_test, y_test):

        X_test_transformed = lda_clf.transform(X_test).reshape(X_test.shape[0])
        print(np.where(y_test == 1))
        fourierT = np.fft.fft(X_test_transformed).real
        # proba = clf.predict_proba(X_test)
        # threshold = 0.5
        # decision = proba[:,0] >= threshold #clf.predict(X_test)
        decision = clf.predict(X_test)
        color_pred = np.where(decision ==1, "blue", "red")
        color_real = np.where(y_test ==1, "blue", "red")

        plt.subplot(4,1,1)
        plt.scatter(y = fourierT, x = X_test_transformed, c = color_real, alpha = 0.5, marker="x")
        plt.title('real')

        plt.subplot(4,1,2)
        plt.scatter(y = fourierT, x = X_test_transformed, c = color_pred, alpha = 0.5, marker="x")
        plt.title('QDA')

        plt.subplot(4,1,3)
        decision = lda_clf.predict(X_test)
        color_pred = np.where(decision ==1, "blue", "red")
        plt.scatter(y = fourierT, x = X_test_transformed, c = color_pred, alpha = 0.5, marker="x")
        plt.title('LDA')

        plt.subplot(4,1,4)
        decision = clf2.predict(X_test)
        color_pred = np.where(decision ==1, "blue", "red")
        plt.scatter(y = fourierT, x = X_test_transformed, c = color_pred, alpha = 0.5, marker="x")
        plt.title('LR')

        plt.savefig('foo.png')

        # savefig('foo.png')
        # color_pred = np.where(decision ==1, "red", "blue")
        plt.show()

if __name__ == '__main__':
    main()
