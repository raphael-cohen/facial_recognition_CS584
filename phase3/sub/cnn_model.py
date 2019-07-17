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
#pip install keras_sequential_ascii


def main():

    name = 'williams'

    data = np.array(pkl.load(open('../X.pkl', 'rb')))
    # labels = np.array(pkl.load(open('../y_bush_vs_others.pkl', 'rb'))).flatten()
    labels = np.array(pkl.load(open('../y_{0}_vs_others.pkl'.format(name), 'rb'))).flatten()
    data = data.reshape((len(data), 64, 64, 1))

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=1. / 3, random_state=2518, stratify = labels, shuffle = True)

    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    print(data.shape)
    print(labels.shape)

    num_positive = np.sum(labels)


    class_weight = {0: 1., 1: len(labels)/num_positive*2}
    print(class_weight)

    model = conv_predict_model(acthidden='tanh', actoutput='sigmoid')

    opt = optimizers.adadelta()
    # Revenir là dessus
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    #
    # model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    # model.fit(x_train, y_train, validation_data=(
    #     x_test, y_test), shuffle=True, epochs=500, batch_size=16, class_weight=class_weight)

    # sequential_model_to_ascii_printout(model)
#400

    model.save("{0}_predict.model".format(name))

    with open("{0}_history.pkl".format(name), 'wb') as file_pi:
        pkl.dump(model.history, file_pi)


def conv_predict_model(acthidden='tanh', actoutput='sigmoid'):

    input_img = Input(shape=(64, 64, 1))

    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation=acthidden, padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(8, (3, 3), activation=acthidden, padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(8, (3, 3), activation=acthidden, padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(20, activation=acthidden))#, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation=actoutput))

    return model


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


if __name__ == '__main__':
    main()
