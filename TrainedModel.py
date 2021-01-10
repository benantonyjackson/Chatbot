# Disables tenosorflow console output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from os import listdir
from tensorflow.keras.applications.resnet50 import decode_predictions


import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import load_model

""" 
Title: Keras - Python Deep Learning Neural Network API
Author: Mandy
Date: 21.12.2020
Code version: v1.0.3
Availability: 
https://www.youtube.com/playlist?list=PLZbbT5o_s2xrL4F90oKfloWM7ExXT_U_b
https://deeplizard.com/learn/playlist/PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL
https://deeplizard.com/resources
"""


class TrainedModel:

    def __init__(self, path="model.h5"):
        self.path = path
        self.model = load_model(path)
        self.classes = ['bears', 'wolves']
        self.test_path = 'dataset/test'

    def test_against_testing_set(self):
        for cls in self.classes:
            dirs = listdir(self.test_path + "/" + cls)
            for directory in dirs:
                predicted_class = self.predict_local_image(self.test_path + "/" + cls + "/" + directory)
                print(predicted_class + ": " + directory)

    def predict_local_image(self, path):
        # https://stackoverflow.com/questions/43469281/how-to-predict-input-image-using-trained-model-in-keras
        img = image.load_img(path, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        predictions = self.model.predict(images, batch_size=10)
        return self.classes[np.argmax(predictions[0])]


if __name__ == "__main__":
    tm = TrainedModel()
    tm.test_against_testing_set()
