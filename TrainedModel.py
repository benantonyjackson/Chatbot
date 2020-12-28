from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from os import listdir

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import load_model

# Credit for CNN stuff https://www.youtube.com/playlist?list=PLZbbT5o_s2xrL4F90oKfloWM7ExXT_U_b


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
                self.predict_local_image(self.test_path + "/" + cls + "/" + directory)

        # test_batches = ImageDataGenerator().flow_from_directory(self.test_path, classes=self.classes,
        #                                                              target_size=(224, 224))
        #
        # test_img, test_lables = next(test_batches)
        #
        # # print(test_img)
        # # print(test_lables)
        #
        # predictions = self.model.predict_generator(test_batches, steps=1, verbose=0)
        # print(predictions)

    def predict_local_image(self, path):
        # https://stackoverflow.com/questions/43469281/how-to-predict-input-image-using-trained-model-in-keras
        img = image.load_img(path, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = self.model.predict_classes(images, batch_size=10)
        print(self.classes[classes[0]] + ": " + path)


if __name__ == "__main__":
    tm = TrainedModel()
    tm.test_against_testing_set()
    tm.predict_local_image("Figure_1.png")
    # tm.predict_local_image("dataset/test/bears/_094a0547.jpg")
    # tm.predict_local_image("dataset/test/bears/_83938024_83938023.jpg")
    # tm.predict_local_image("dataset/test/bears/_102722447_sophiebear1.jpg")
    # tm.predict_local_image("dataset/test/bears/_103171237_hi048246414.jpg")
    # tm.predict_local_image("dataset/test/wolves/_106348479_mediaitem106348478.jpg")
    # tm.predict_local_image("dataset/test/wolves/_109106949_meandkalani_photographeradelebarclay2.jpg")
    # tm.predict_local_image("dataset/test/wolves/5carpathianwolf_365181.jpg")
    # tm.predict_local_image("dataset/test/wolves/09e95736-bfac-11e9-8f25-9b5536624008_image_hires_063817.jpg")



    # tm.predict_local_image("dataset/test/swords/03_sword_polishing_night_03.jpg")
    # tm.predict_local_image("dataset/test/swords/4c528a1b.jpg")
    # tm.predict_local_image("dataset/test/swords/3hnds.jpeg")
    # tm.predict_local_image("dataset/test/swords/06_weapon.jpg")
