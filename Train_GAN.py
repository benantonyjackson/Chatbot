import random

from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from matplotlib import pyplot
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal

from Trained_CNN_Wrapper import TrainedModel
from tensorflow.keras.models import load_model

import numpy as np

import cv2
from PIL import Image
from numpy import asarray
from os import listdir

from tensorflow.keras import backend as K
import tensorflow.keras
import tensorflow as tf


from tensorflow.python.client import device_lib


"""
Title: Keras - Python Deep Learning Neural Network API
Author: Jason Brownlee
Date: 27.12.2020
Code version: 
Availability:
https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/
"""

"""
Title: Keras - Python Deep Learning Neural Network API
Author: Jordan Bird
Date: 07.01.2021
Code version: No version number available 
Availability: 
https://github.com/jordan-bird/art-DCGAN-Keras
"""


# gan structure https://cpang4.github.io/gan/

# Best practice initialiser for GANS
# initialWeights = RandomNormal(mean=0.0, stddev=0.02, seed=None)


# define the standalone discriminator model
def define_discriminator(in_shape=(256,256,3)):
    hidden_nodes = 256
    model = Sequential()
    model.add(Conv2D(hidden_nodes, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(hidden_nodes, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(BatchNormalization())
    model.add(Conv2D(hidden_nodes, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(BatchNormalization())
    model.add(Conv2D(hidden_nodes, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(BatchNormalization())
    model.add(Conv2D(hidden_nodes, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    # opt = Adam(lr=0.0002, beta_1=0.5)
    opt = Adam(lr=0.00005, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# define the standalone generator model
def define_generator(latent_dim, n_images, input_height=32, input_width=32):
    model = Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))
    # upsample to 8x8
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    # upsample to 64x64
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    # model.add(Conv2DTranspose(3, (3, 3), activation='tanh', padding='same'))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(BatchNormalization())

    # upsample to 128x128
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    # upsample to 256x256
    # model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu'))
    # model.add(LeakyReLU(alpha=0.2))

    # model.add(Conv2DTranspose(3, (3, 3), activation='tanh', padding='same'))
    # model.add(LeakyReLU(alpha=0.2))

    # output layer
    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


# load and prepare cifar10 training images
def load_real_samples(input_width = 128, input_height = 128):
    # # load cifar10 dataset
    # (trainX, _), (_, _) = load_data()
    # # convert from unsigned ints to floats
    # X = trainX.astype('float32')
    # # scale from [0,255] to [-1,1]
    # X = (X - 127.5) / 127.5
    # return X

    train_path = 'dataset/train'

    classes = ['bear_gan']

    # classes = ['bears', 'wolves', 'dogs', 'swords', 'African Elephant', 'Amur Leopard', 'Arctic Fox', 'Black Rhino',
    #            'Black Spider Monkey', 'Bluefin Tuna', 'Chimpanzee', 'European Rabbit', 'Orangutan']
    images = []

    for i in range(0, len(classes)):
        dirs = listdir(train_path + "/" + classes[i])
        for image_name in dirs:
            # https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
            # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
            try:
                image = (cv2.imread(train_path + "/" + classes[i] + "/" + image_name, 1))
                resized_image = cv2.resize(image, (input_width, input_height), interpolation=cv2.INTER_AREA)
                # img_with_noise = sp_noise(resized_image, 0.01)
                scaled_image = (resized_image - 127.5) / 127.5
                images.append(scaled_image)
                #
                # X = (scaled_image + 1) / 2.0
                # # plot the result
                # pyplot.imshow(X)
                # pyplot.show()
            except Exception as e:
                pass

    print("Images loaded")
    retImage = np.array(images)
    return retImage


# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    # y = ones((n_samples, 1))

    y = np.full((n_samples, 1), 0.9)

    return X, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y


# create and save a plot of generated images
def save_plot(examples, epoch, n=7):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i])
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch + 1)
    pyplot.savefig(filename)
    pyplot.close()


# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    # save plot
    save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)
    print("Saved to " + filename)


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=32):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)

    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            # if (i + 1) % 100 == 0:
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))
        # evaluate the model performance, sometimes
        if (i + 1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


if __name__ == "__main__":
    number_of_images = 256

    w = 128
    h = 128

    # size of the latent space
    latent_dim = 72000
    # create the discriminator
    d_model = define_discriminator(in_shape=(w,h,3))
    # create the generator
    g_model = define_generator(latent_dim, n_images=number_of_images, input_height=h, input_width=w)
    # g_model = load_model('generator_model_020.h5')
    # create the gan
    gan_model = define_gan(g_model, d_model)
    # load image data
    dataset = load_real_samples(input_width=w, input_height=h)
    # train model
    train(g_model, d_model, gan_model, dataset, latent_dim, n_batch=4, n_epochs=1000)
    print("Done")
