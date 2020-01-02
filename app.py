import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from keras.layers import *
from keras.models import *
from keras.datasets import mnist
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from common.kblocks import *

const_ph = K.constant(1.0, dtype=tf.float32,shape=[1, 4, 4, 1], name="constant")
scale, bias = Style_Gate(lantent_vector, num_filters)
noise_channel = Noise_gate(noise, num_filters, input.shape[1])
layer = Conv2D(filters=3,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')

sigma = multiply([layer.weights[0], scale])
sigma = K.pow(sigma, 2)
sigma = K.sum(sigma, axis=[0, 1, 2], keepdims=True)
sigma = K.pow(sigma, -0.5)
# sigma = Reshape([1, 1, num_filters])(sigma)
layer1 = Lambda(lambda x: x * sigma)(layer1)

bias_sig = multiply([layer.weights[1], scale])
bias_sig = K.pow(bias_sig, 2)
bias_sig = K.sum(bias_sig, axis=[0], keepdims=True)
bias_sig = K.pow(bias_sig, -0.5)
layer1 = Lambda(lambda x: x + bias_sig)(layer1)

layer1 = layer(const_ph)

model = Model(outputs=layer1)

print(layer1)