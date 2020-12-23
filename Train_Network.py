import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


# Credit for CNN stuff https://www.youtube.com/playlist?list=PLZbbT5o_s2xrL4F90oKfloWM7ExXT_U_b
train_path = 'dataset/train'
test_path = 'dataset/test'
valid_path = 'dataset/validate'

classes = ['bears', 'swords', 'wolves']

train_batches = ImageDataGenerator().flow_from_directory(train_path, classes=classes, target_size=(224, 224))
test_batches = ImageDataGenerator().flow_from_directory(test_path, classes=classes, target_size=(224, 224))
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, classes=classes, target_size=(224, 224))


model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(len(classes), activation="softmax"),
])

model.compile(Adam(lr=.0001), loss="categorical_crossentropy", metrics=["accuracy"])

model.fit_generator(train_batches, validation_data=valid_batches, epochs=15, verbose=2)

model.save("model.h5")
