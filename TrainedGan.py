from tensorflow.keras.models import load_model
from numpy import asarray
from matplotlib import pyplot
from numpy.random import randn
from numpy.random import randint

import random
# load model
model = load_model('generator_model_090.h5')
# all 0s
for i in range(0, 15):
    vector = asarray([[(random.randint(-1000, 1000) / 1000) for _ in range(100)]])
    # generate image
    X = model.predict(vector)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot the result
    pyplot.imshow(X[0, :, :])
    pyplot.show()