from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dropout
from keras import backend as K
from keras import regularizers
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import to_categorical
from keras import optimizers
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from keras_sequential_ascii import sequential_model_to_ascii_printout


def main():

    name = 'bush'

    data = np.array(pkl.load(open('../../X.pkl', 'rb')))
    labels = np.array(pkl.load(open('../../y_{0}_vs_others.pkl'.format(name), 'rb'))).flatten()
    data = data.reshape((len(data), 64, 64, 1))

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=1. / 3, random_state=2518, stratify = labels, shuffle = True)

    print(x_train.shape)
    print(y_train.shape)

    load_m(x_train, x_test, y_train, y_test, name)
    # results = pkl.load(open('{0}.pickle'.format(name), 'rb'))

    # print(results)


def load_m(x_train, x_test, y_train, y_test, name):
    model = load_model('../../phase4/{0}.model'.format(name))
    # model = load_model('{0}.model'.format(name))
    results = []
    # sequential_model_to_ascii_printout(model)
    for x, y in zip([x_train, x_test], [y_train, y_test]):

        pred = np.where(model.predict(x) >= 0.5, 1, 0)
        cm = confusion_matrix(y_true=y, y_pred=pred)
        print(cm)
        tn, fp, fn, tp = cm.ravel()
        print("\nTN\tFP\tFN\tTP\n{0}\t{1}\t{2}\t{3}\n".format(tn, fp, fn, tp))
        prec = tp/(tp+fp)
        recall = tp/(tp+fn)
        acc = (tp+tn)/(tp+tn+fn+fp)
        F1 = 2*prec*recall/(prec+recall)
        results.append(F1)
        print("F1-Score\t\tPrecision\t\tRecall\t\tAccuracy\n{0}\t{1}\t{2}\t{3}\n".format(F1, prec, recall, acc))
    #
    #
    # with open("{0}.pickle".format(name), 'wb') as file_pi:
    #     pkl.dump(results, file_pi)

if __name__ == '__main__':
    main()
