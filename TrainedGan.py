from tensorflow.keras.models import load_model
from numpy import asarray
from matplotlib import pyplot
from numpy.random import randn
from numpy.random import randint

import random

class TrainedGan:

    def __init__(self, model_dir='generator_model_100.h5'):
        self.model = load_model(model_dir)
        load_model(model_dir)

    def load_model(self, model_dir):
        # load model
        return load_model(model_dir)

    def generate_image(self):
        vector = asarray([[(random.randint(-1000, 1000) / 1000) for _ in range(100)]])
        # generate image
        return self.model.predict(vector)

    def generate_and_display_image(self):
        X = self.generate_image()
        # scale from [-1,1] to [0,1]
        X = (X + 1) / 2.0
        # plot the result
        pyplot.imshow(X[0, :, :])
        pyplot.show()


if __name__ == "__main__":
    tg = TrainedGan()
    for i in range(0, 15):
        tg.generate_and_display_image()