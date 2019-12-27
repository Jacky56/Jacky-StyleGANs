import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from keras.layers import *
from keras.models import *
from keras.datasets import mnist
from keras.optimizers import Adam
import keras.backend as K


def generator_block(input, num_filters, noise, lantent_vector):
    scale, bias = Style_Gate(lantent_vector, num_filters)
    noise_channel = Noise_gate(noise, num_filters, input.output_shape[1])

    layer = Conv2D(filters=num_filters,
                   kernel_size=(3,3),
                   activation='relu',
                   padding='same')(input)

    layer = add([noise_channel, layer])

    layer = Lambda(AdaIN)(layer, scale, bias)

    return layer

def discriminator_block



def Style_Gate(lantent_vector, num_filters):
    #latent w parsed into 2 FC, scale and bias
    scale = Dense(units=num_filters)(lantent_vector)
    bias = Dense(units=num_filters)(lantent_vector)

    scale = Reshape([-1, 1, 1, num_filters])(scale)
    bias = Reshape([-1, 1, 1, num_filters])(bias)

    return scale, bias

def Noise_gate(Noise, num_filters, filter_size):
    #crops the middle of the image
    noise_channel = Cropping2D(filter_size)(Noise)
    #projection layer
    noise_channel = Conv2D(filters=num_filters,
                           kernel_size=(1,1),
                           padding='same')(noise_channel)
    return noise_channel

def AdaIN(x, scale, bias):
    #standardising each channel of instance
    x_mean = K.mean(x, [1, 2], True)
    x_std = K.std(x, [1, 2], True) + 1e-7
    x = (x - x_mean) / x_std

    #apply feature space from latent w to image
    return (x * scale) + bias

