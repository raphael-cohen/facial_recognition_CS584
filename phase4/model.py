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
from keras.utils.vis_utils import plot_model
from keras_sequential_ascii import sequential_model_to_ascii_printout
from os import listdir
import os
from os.path import isfile, join
import matplotlib.pyplot as plt
import cv2
#pip install keras_sequential_ascii


def main():

    name = 'williams'

    # data = np.array(pkl.load(open('../X.pkl', 'rb')))
    npath = 'np_faces'
    onlyfiles = [ p for p in listdir('np_faces') if isfile(join('np_faces',p)) ]

    total = len(onlyfiles)
    total_img = 0
    # for n, f in enumerate(onlyfiles):
    #
    #     imgs = np.load(join(npath,f))
    #     total_img += imgs.shape[0]
    #     if n%20 == 0:
    #         print("{0}/{1}\n{2}".format(n,total, total_img))
        #What is xxx ?
    total_img = 63564 #beurk but no choice
    print(total_img)
    data = np.empty((total_img, 64,64,1))
    names = np.empty(total_img, dtype=object)
    start = 0
    sp = 0
    for n, f in enumerate(onlyfiles):

        imgs = np.load(join(npath,f)).astype(np.float32)
        nimg = imgs.shape[0]
        space = np.prod(imgs.shape)
        # print(imgs)

        np.put(data, range(sp, sp+space), imgs)
        np.put(names, range(start, start+nimg), [f]*nimg)
        start += nimg
        sp += space
        if n%10 == 0:
            print("{0}/{1}".format(n,total))

    print(names.shape)
    # celebrities = np.array(['Anthony_Hopkins.npy', 'Burt_Reynolds.npy',
    #                         'Jack_Nicholson.npy', 'John_Cusack.npy',
    #                         'Jeffrey_Tambor.npy', 'Leslie_Neilsen.npy',
    #                         'Mark_Wahlberg.npy', 'Richard_E._Grant.npy'])
    celebrities = np.array(['Lourdes_Benedicto.npy', 'Lisa_Bonet.npy',
                            'Samuel_L._Jackson.npy', 'Tatyana_M._Ali.npy',
                            'Tempestt_Bledsoe.npy', 'Wanda_De_Jesus.npy',
                            'Shannon_Kane.npy','Jasmine_Guy.npy'])
    # print(celebrities.shape)

    labels = np.zeros(names.shape[0])

    for cel in celebrities:
        print(cel)
        labels += np.where(names == cel, 1, 0)

    # print(names[0:30])
    # print(labels)
    print(sum(labels))
    print(data.shape)
    print(data[0].shape)
    #
    # plt.imshow((data[1]).reshape((64,64)))
    # plt.show()
    # print(data[0])
    # labels = np.array(pkl.load(open('../y_bush_vs_others.pkl', 'rb'))).flatten()
    # labels = np.array(pkl.load(open('../y_{0}_vs_others.pkl'.format(name), 'rb'))).flatten()
    # one = np.ones(500)
    # labels = np.hstack((one, np.zeros(data.shape[0]-500)))
    # print(np.asarray(data[0] ,dtype=np.float32))
    # for d
    # print((data[0]).reshape(64,64,1).shape)

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=1. / 3, random_state=2518, stratify = labels, shuffle = True)

    num_positive = np.sum(labels)


    class_weight = {0: 1., 1: len(labels)/num_positive*2}
    # class_weight = {0:1, 1:1}
    print(class_weight)

    model = conv_predict_model(acthidden='tanh', actoutput='sigmoid')

    opt = optimizers.adadelta()
    # opt = optimizers.nadam()
    # Revenir là dessus
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    #
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])#, precision,  recall, f1])
    model.fit(x_train, y_train, validation_data=(
        x_test, y_test), shuffle=True, epochs=350, batch_size=256, class_weight=class_weight)

    # sequential_model_to_ascii_printout(model)
#400
    name ='initial-model-williams'
    model.save("{0}.model".format(name))
    #
    # with open("{0}_history.pkl".format(name), 'wb') as file_pi:
    #     pkl.dump(model.history, file_pi)


def conv_predict_model(acthidden='tanh', actoutput='sigmoid'):

    input_img = Input(shape=(64, 64, 1))

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation=acthidden, padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation=acthidden, padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(32, (3, 3), activation=acthidden, padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation=acthidden, padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation=acthidden, padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(40, activation=acthidden))#, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(20, activation=acthidden))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation=acthidden))
    model.add(Dense(1, activation=actoutput))

    return model


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
