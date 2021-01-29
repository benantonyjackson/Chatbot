import numpy as np
# Disables tenosorflow console output
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

""" 
Title: tf.keras.preprocessing.image.load_img 
Author: Tensor flow documentation
Date: 21.12.2020
Availability: 
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/load_img
"""


class TrainedModel:
    def __init__(self, path="model.h5", classes=['bears', 'wolves', 'dogs', 'horses'], test_path='samples'):
        self.path = path
        self.model = load_model(path)
        self.classes = classes
        self.test_path = test_path

    def predict_local_image(self, path):
        img = image.load_img(path, target_size=(256, 256))

        x = image.img_to_array(img)
        x = np.array([x])

        predictions = self.model.predict(x)
        return self.classes[np.argmax(predictions[0])]
