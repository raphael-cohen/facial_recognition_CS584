import pickle

import numpy as np

import matplotlib.pyplot as plt


def main():

    data = np.array(pickle.load(open('X.pkl', 'rb')))
    labels = np.array(pickle.load(open('y_bush_vs_others.pkl', 'rb')))
    labels = np.array([1 if x == 0 else 0 for x in labels])
    # weights = n_samples / (n_classes * np.bincount(y))
    labels = labels.reshape((data.shape[0],))

    indexes = np.array(np.where(labels == 0)).flatten()

    n_indexes = np.array(np.where(labels == 1)).flatten()
    # n_indexes = np.where(labels == 1).flatten()
    # print(indexes)
    # datay = np.where()
    img1 = data[indexes[2]].reshape((64, 64))
    print(np.max(img1))
    fft2 = np.array(np.fft.fft2(img1))
    fft = np.array(abs(np.fft.fft(data[indexes[2]]).real))
    print(np.max(fft))
    fft = (fft) / np.max(fft)
    print(np.max(fft))
    # plt.imshow(fft2.real)
    plt.imshow((fft.reshape((64, 64))))
    # plt.scatter(fft.real, fft.imag)
    # plt.imshow(img1)

    plt.show()
    #
    # plt.subplot(4,1,1)
    # fft = 100*np.fft.fft2(data[n_indexes[0]].reshape((64,64))
    # # plt.imshow(data[n_indexes[0]].reshape((64,64)))
    # plt.plot(fft)
    # plt.subplot(4,1,2)
    # fft = np.fft.fft2(data[n_indexes[1]].reshape((64,64)))
    # plt.plot(fft)
    # plt.subplot(4,1,3)
    # fft = np.fft.fft2(data[n_indexes[2]].reshape((64,64)))
    # plt.plot(fft)
    # plt.subplot(4,1,4)
    # fft = np.fft.fft2(data[indexes[0]].reshape((64,64)))
    # plt.plot(fft)
    # plt.show()
    # print(data[1])


if __name__ == '__main__':
    main()
