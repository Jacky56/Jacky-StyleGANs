import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from keras.layers import *
from keras.models import *
import keras.backend as K
import tensorflow as tf
from keras.utils import plot_model


from common.kblocks import *




inputs = Input([64,64,3], name='input')
layer = latent_W_VGG_encoder(inputs)
outputs = Dense(units=40, activation='sigmoid')(layer)

model = Model(inputs=[inputs], outputs=[outputs])


model.summary()

plot_model(model, './../diagram/VGG_encoder.png')
