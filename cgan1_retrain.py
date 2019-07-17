import numpy as np

#%matplotlib inline
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.layers import Input, Flatten, Embedding, multiply, Dropout
from keras.layers import Concatenate, GaussianNoise,Activation
from keras.optimizers import Adam
from keras.utils import np_utils, to_categorical
from keras import initializers
from keras import backend as K
from keras.models import load_model


# load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()




num_classes = len(np.unique(y_train))
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']




if K.image_data_format() == 'channels_first':
   X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
   X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)
   input_shape = (3, 32, 32)
else:
   X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
   X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
   input_shape = (32, 32, 3)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

# the generator is using tanh activation, for which we need to preprocess
# the image data into the range between -1 and 1.

X_train = np.float32(X_train)
X_train = (X_train / 255 - 0.5) * 2
X_train = np.clip(X_train, -1, 1)

X_test = np.float32(X_test)
X_test = (X_test / 255 - 0.5) * 2
X_test = np.clip(X_test, -1, 1)

print('X_train reshape:', X_train.shape)
print('X_test reshape:', X_test.shape)


print(X_train[0].shape)


# latent space dimension
z = Input(shape=(100,))

# classes
labels = Input(shape=(10,))


generator = load_model('generator.model')

discriminator = load_model('discriminator.model')

d_g = load_model('cgan.model')

label = Input(shape=(10,), name='label')
z = Input(shape=(100,), name='z')

fake_img = generator([z, label])
validity = discriminator([fake_img, label])





epochs = 100
batch_size = 32
smooth = 0.1
latent_dim = 100

real = np.ones(shape=(batch_size, 1))
fake = np.zeros(shape=(batch_size, 1))

d_loss = []
d_g_loss = []
exp_replay = []

for e in range(epochs + 1):
    for i in range(len(X_train) // batch_size):

        # Train Discriminator weights
        discriminator.trainable = True

        # Real samples
        X_batch = X_train[i*batch_size:(i+1)*batch_size]
        real_labels = to_categorical(y_train[i*batch_size:(i+1)*batch_size].reshape(-1, 1), num_classes=10)

        d_loss_real = discriminator.train_on_batch(x=[X_batch, real_labels],
                                                   y=real * (1 - smooth))

        # Fake Samples
        z = np.random.normal(loc=0, scale=1, size=(batch_size, latent_dim))
        random_labels = to_categorical(np.random.randint(0, 10, batch_size).reshape(-1, 1), num_classes=10)
        X_fake = generator.predict_on_batch([z, random_labels])
        generated_images = X_fake
        gene_labels = random_labels
        d_loss_fake = discriminator.train_on_batch(x=[X_fake, random_labels], y=fake)

        # Discriminator loss
        d_loss_batch = 0.5 * (d_loss_real[0] + d_loss_fake[0])

        # Train Generator weights
        discriminator.trainable = False

        z = np.random.normal(loc=0, scale=1, size=(batch_size, latent_dim))
        random_labels = to_categorical(np.random.randint(0, 10, batch_size).reshape(-1, 1), num_classes=10)
        d_g_loss_batch = d_g.train_on_batch(x=[z, random_labels], y=real)

        print(
            'epoch = %d/%d, batch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (e + 1, epochs, i, len(X_train) // batch_size, d_loss_batch, d_g_loss_batch[0]),
            100*' ',
            end='\r'
        )

    d_loss.append(d_loss_batch)
    d_g_loss.append(d_g_loss_batch[0])
    print('epoch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (e + 1, epochs, d_loss[-1], d_g_loss[-1]), 100*' ')

    #Avoid mode collapse
    ### Experience replay
    # Store a random point for experience replay
    # r_idx = np.random.randint(batch_size-1)
    # r_idx = np.random.randint(10)
    # exp_replay.append([generated_images[r_idx], labels[r_idx], gene_labels[r_idx]])
    #
    # #If we have enough points, do experience replay
    # if len(exp_replay) == batch_size:
    #   generated_images = np.array([p[0] for p in exp_replay])
    #   labels = np.array([p[1] for p in exp_replay])
    #   gene_labels = np.array([p[2] for p in exp_replay])
    #   expprep_loss_gene = discriminator.train_on_batch([generated_images, labels], gene_labels)
    #   exp_replay = []
    #   break


    if e % 98 == 0:
        samples = 10
        z = np.random.normal(loc=0, scale=1, size=(samples, latent_dim))
        labels = to_categorical(np.arange(0, 10).reshape(-1, 1), num_classes=10)

        x_fake = generator.predict([z, labels])
        x_fake = np.clip(x_fake, -1, 1)
        x_fake = (x_fake + 1) * 127
        x_fake = np.round(x_fake).astype('uint8')

        d_g.save("cgan.model")
        discriminator.save("discriminator.model")
        generator.save("generator.model")

        # d_g.save("cganER.model")
        # discriminator.save("discriminatorER.model")
        # generator.save("generatorER.model")

        for k in range(samples):
            plt.subplot(2, 5, k + 1, xticks=[], yticks=[])
            plt.imshow(x_fake[k])
            plt.title(class_names[k])

        plt.tight_layout()
        plt.show()



# plotting the metrics
plt.plot(d_loss)
plt.plot(d_g_loss)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Discriminator', 'Adversarial'], loc='center right')
plt.show()
