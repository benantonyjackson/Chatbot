from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# Credit for CNN stuff https://www.youtube.com/playlist?list=PLZbbT5o_s2xrL4F90oKfloWM7ExXT_U_b


class TrainedModel:

    def __init__(self, path="model.h5"):
        self.path = path
        self.model = load_model(path)
        self.classes = ['bears', 'Dogs', 'Wolves']
        self.test_path = 'dataset/test'

    def test_against_testing_set(self):

        test_batches = ImageDataGenerator().flow_from_directory(self.test_path, classes=self.classes,
                                                                     target_size=(224, 224))

        test_img, test_lables = next(test_batches)

        print(test_img)
        print(test_lables)

        predictions = self.model.predict_generator(test_batches, steps=1, verbose=0)
        print(predictions)


if __name__ == "__main__":
    tm = TrainedModel()

    tm.test_against_testing_set()