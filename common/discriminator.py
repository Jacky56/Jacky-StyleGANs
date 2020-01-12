import numpy as np
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
import keras.backend as K
import tensorflow as tf
from keras.utils import plot_model
from common.kblocks import *


def default_discriminator():
    image = Input([64,64,3], name='image')
    vgg_encoder = latent_W_VGG_encoder(image)
    layer = Dropout(0.50)(vgg_encoder)
    layer = Dense(units=512, activation='relu')(layer)
    layer = Dropout(0.50)(layer)
    layer = Dense(units=512, activation='relu')(layer)
    layer = Dropout(0.50)(layer)
    out = Dense(units=1, activation='sigmoid')(layer)
    model = Model(inputs=[image], outputs=[out])

    return model


# model = discriminator()
# plot_model(model, "./../diagram/discriminator.png")
# model.summary()
