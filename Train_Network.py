import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

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

train_path = 'dataset/train'
valid_path = 'dataset/validate'

input_width = 256
input_height = 256

classes = ['bears', 'wolves']

train_batches = ImageDataGenerator().flow_from_directory(train_path, classes=classes,
                                                         target_size=(input_width, input_height))
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, classes=classes,
                                                         target_size=(input_width, input_height))


model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(input_width, input_height, 3)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(256, activation="relu"),
    layers.Dense(len(classes), activation="softmax"),
])

model.compile(Adam(lr=.0001), loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_batches, validation_data=valid_batches, epochs=15, verbose=2)

model.save("model.h5")
