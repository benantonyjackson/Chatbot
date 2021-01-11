import numpy as np
# Disables tenosorflow console output
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os
from os import listdir
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.preprocessing import image
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
    def __init__(self, path="model.h5", classes=['bears', 'wolves', 'dogs', 'horses'], test_path='dataset/test'):
        self.path = path
        self.model = load_model(path)
        self.classes = classes
        self.test_path = test_path

    # This function is only for my own testing
    def test_against_testing_set(self):
        for cls in self.classes:
            dirs = listdir(self.test_path + "/" + cls)
            for directory in dirs:
                predicted_class = self.predict_local_image(self.test_path + "/" + cls + "/" + directory)
                print(predicted_class + ": " + self.test_path + "/" + cls + "/" + directory)

    def predict_local_image(self, path):
        img = image.load_img(path, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        predictions = self.model.predict(x)
        # https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
        return self.classes[np.argmax(predictions[0])]


if __name__ == "__main__":
    tm = TrainedModel()
    tm.test_against_testing_set()
