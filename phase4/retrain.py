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





def main():

    name = 'williams'

    data = np.array(pkl.load(open('../X.pkl', 'rb')))
    labels = np.array(pkl.load(open('../y_{0}_vs_others.pkl'.format(name), 'rb'))).flatten()
    data = data.reshape((len(data), 64, 64, 1))

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=1. / 3, random_state=2518, stratify = labels, shuffle = True)

    print(x_train.shape)
    print(y_train.shape)

    num_positive = np.sum(labels)
    # model = load_model('initial-model-bush.model_predict.model')#'{0}.model'.format(name))
    model = load_model('initial-model-williams.model')
    # print(results)
    class_weight = {0: 1., 1: len(labels)/num_positive*3}
    # class_weight = {0:1, 1:1}
    print(class_weight)

    model = load_model('williams10_predict.model')
    for i in range(5):
        print("Iteration: {0}".format(i))
        model.fit(x_train, y_train, validation_data=(
            x_test, y_test), shuffle=True, epochs=20, batch_size=64, class_weight=class_weight)

        load_m(model, x_train, x_test, y_train, y_test, name )

    model.save("{0}10_predict.model".format(name))

def load_m(model, x_train, x_test, y_train, y_test, name):
    # model = load_model('{0}.model'.format(name))
    results = []
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
    # with open("{0}.pickle".format(name), 'wb') as file_pi:
    #     pkl.dump(results, file_pi)



def recall(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))



if __name__ == '__main__':
    main()
