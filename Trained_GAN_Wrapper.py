from numpy import asarray
from matplotlib import pyplot
from numpy.random import randn
from numpy.random import randint

import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model


"""
Title: Keras - Python Deep Learning Neural Network API
Author: Jason Brownlee
Date: 27.12.2020
Code version: 
Availability:
https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/
"""


class TrainedGan:
    def __init__(self, model_dir='bear-gan-model.h5'):
        self.model = load_model(model_dir, compile=False)

    def generate_image(self, latent_space):
        # Generates an array of random noise
        vector = asarray([[(random.randint(-1000, 1000) / 1000) for _ in range(latent_space)]])
        # generate image
        return self.model.predict(vector)

    def generate_and_display_image(self, latent_space=60000):
        X = self.generate_image(latent_space=latent_space)
        # scale from [-1,1] to [0,1]
        X = (X + 1) / 2.0
        # plot the result
        pyplot.imshow(X[0, :, :])
        pyplot.show()


if __name__ == "__main__":
    tg = TrainedGan()
    for i in range(0, 15):
        tg.generate_and_display_image()
