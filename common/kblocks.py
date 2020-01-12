import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from keras.layers import *
from keras.models import *
import keras.backend as K


def generator_block(input, num_filters, noise, lantent_vector):
    scale, bias = Style_Gate(lantent_vector, num_filters)
    noise_channel = Noise_Gate(noise, num_filters, input.shape[1])

    layer = Conv2D(filters=num_filters,
                   kernel_size=(3,3),
                   activation=LeakyReLU(alpha=0.1),
                   padding='same')(input)

    layer = add([noise_channel, layer])

    layer = Lambda(AdaIN)([layer, scale, bias])

    return layer

def generator_block_v2(input, num_filters, noise, lantent_vector):
    scale, bias = Style_Gate(lantent_vector, num_filters)
    noise_channel = Noise_Gate(noise, input.shape[-1], input.shape[1])

    input = add([noise_channel, input])

    layer = Conv2D(filters=num_filters,
                   kernel_size=(3,3),
                   activation='relu',
                   use_bias=False,
                   padding='same')

    layer1 = layer(input)

    # layer1 = Lambda(lambda x: x - layer.weights[1])(layer1)

    sigma = multiply([layer.weights[0], scale])
    sigma = K.pow(sigma, 2)
    sigma = K.sum(sigma, axis=[0, 1, 2], keepdims=True)
    sigma = K.pow(sigma + 1e-7, -0.5)
    sigma = Reshape([1, 1, num_filters])(sigma)

    layer1 = Lambda(lambda x: x * sigma)(layer1)
    layer1 = Lambda(lambda x: x + bias)(layer1)

    return layer1


def Style_Gate(lantent_vector, num_filters):
    #latent w parsed into 2 FC, scale and bias
    scale = Dense(units=num_filters)(lantent_vector)
    bias = Dense(units=num_filters)(lantent_vector)

    scale = Reshape([1, 1, num_filters])(scale)
    bias = Reshape([1, 1, num_filters])(bias)

    return scale, bias

def Noise_Gate(Noise, num_filters, image_size):
    num_filters = int(num_filters)
    image_size = int(image_size)

    #crops the middle of the image
    crop_size = int((Noise.shape[1] - image_size) // 2)
    noise_channel = Cropping2D(cropping=crop_size)(Noise)
    #projection layer
    noise_channel = Conv2D(filters=num_filters,
                           kernel_size=(1, 1),
                           padding='same')(noise_channel)
    return noise_channel

def AdaIN(inputs):
    x, scale, bias = inputs
    #standardising each channel of instance
    x_mean = K.mean(x, [1, 2], keepdims=True)

    x_std = K.std(x, [1, 2], keepdims=True) + 1e-7
    x = (x - x_mean) / x_std
    #apply feature space from latent w to image
    return (x * scale) + bias


def generate_image_noise(batch_size, image_size):
    return K.random_normal(shape=[batch_size, image_size, image_size, 1])

def generate_latent_Z(vector_size):
    return K.random_normal(shape=[vector_size, 1])

def latent_W(latent_Z, units=256):
    layer = Dense(units=units, activation='relu')(latent_Z)
    layer = Dense(units=units, activation='relu')(layer)
    layer = Dense(units=units, activation='relu')(layer)
    layer = Dense(units=units, activation='relu')(layer)
    layer = Dense(units=units, activation='relu')(layer)
    layer = Dense(units=units, activation='relu')(layer)
    layer = Dense(units=units, activation='relu')(layer)
    layer = Dense(units=units, activation='relu')(layer)
    return layer

# 64x64x3
def latent_W_VGG_encoder(image):

    projection64 = Conv2D(filters=16,
                          kernel_size=(1,1),
                          activation='relu',
                          padding='valid')(image)
    down32 = MaxPooling2D()(projection64)
    # 32x32x16

    vgg16 = VGG_module(down32, 32)
    # 16x16x32

    vgg8 = VGG_module(vgg16, 64)
    #8x8x64

    vgg4 = VGG_module(vgg8, 128)
    #4x4x128

    vgg2 = VGG_module(vgg4, 256)
    # 2x2x256

    vgg1 = Conv2D(filters=256,
                  kernel_size=(2,2),
                  activation='relu',
                  padding='valid')(vgg2)
    down1 = MaxPooling2D()(vgg2)
    residual1 = add([vgg1, down1])
    norm1 = BatchNormalization()(residual1)
    # 1x1x256

    layer = Flatten()(norm1)
    # 1x256

    layer = Dense(units=512,
                  activation='relu',
                  name='latent_vector')(layer)
    # 1x512

    return layer

def get_latent_vector_VGG_encoder(model):
    while model.layers != None:
        if model.layers[-1].name != 'latent_vector':
            model.layers.pop()
        else:
            return model
    return None




def VGG_module(image, num_channels, kernel_size=(3,3)):
    conv = Conv_block(image, num_channels, kernel_size)
    projection = Conv2D(filters=num_channels,
                          kernel_size=(1,1),
                          activation='relu',
                          padding='same')(image)
    residual = add([conv, projection])
    downsample = MaxPooling2D()(residual)
    norm = BatchNormalization()(downsample)

    return norm

def Conv_block(input, num_filters, kernal_size=(3,3)):
    layer = Conv2D(filters=num_filters,
                   kernel_size=kernal_size,
                   activation='relu',
                   padding='same')(input)
    layer = Conv2D(filters=num_filters,
                   kernel_size=kernal_size,
                   activation='relu',
                   padding='same')(layer)

    return layer

def to_rgb(inputs):
    layer = Conv2D(filters=3,
           kernel_size=(1, 1),
           activation='tanh',
           padding='same')(inputs)

    return layer

