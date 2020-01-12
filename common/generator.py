import numpy as np
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
import keras.backend as K
import tensorflow as tf
from keras.utils import plot_model
from common.kblocks import *


def set_latent_enconder(model):
    model = get_latent_vector_VGG_encoder(model)
    for layer in model.layers:
        if layer.name != 'latent_vector':
            layer.trainable = False
    return model

def generator(latent_model):

    const_ph = Input([4, 4, 1], name="constant")
    noise_ph = Input([64, 64, 1], name="noise")

    latent_z_ph1 = Input([64,64,3], name="image1")
    latent_z_ph2 = Input([64, 64, 3], name="image2")
    latent_vector1 = latent_model(latent_z_ph1)
    latent_vector2 = latent_model(latent_z_ph2)

    # latent_z_ph = Input([256], name="latent_z")
    # latent_vector = latent_W(latent_z_ph)

    block_4 = generator_block(const_ph, 256, noise_ph, latent_vector1)
    block_4 = generator_block(block_4, 256, noise_ph, latent_vector1)

    rbg_final = to_rgb(block_4)
    rbg_final = UpSampling2D(size=(2, 2), interpolation='bilinear')(rbg_final)

    up_8 = UpSampling2D(size=(2, 2), interpolation='bilinear')(block_4)

    block_8 = generator_block(up_8, 128, noise_ph, latent_vector1)
    block_8 = generator_block(block_8, 128, noise_ph, latent_vector1)

    rbg_8 = to_rgb(block_8)
    rbg_final = add([rbg_final, rbg_8])
    rbg_final = UpSampling2D(size=(2, 2), interpolation='bilinear')(rbg_final)

    up_16 = UpSampling2D(size=(2, 2), interpolation='bilinear')(block_8)

    block_16 = generator_block(up_16, 64, noise_ph, latent_vector1)
    block_16 = generator_block(block_16, 64, noise_ph, latent_vector2)

    rbg_16 = to_rgb(block_16)
    rbg_final = add([rbg_final, rbg_16])
    rbg_final = UpSampling2D(size=(2, 2), interpolation='bilinear')(rbg_final)

    up_32 = UpSampling2D(size=(2, 2), interpolation='bilinear')(block_16)

    block_32 = generator_block(up_32, 32, noise_ph, latent_vector2)
    block_32 = generator_block(block_32, 32, noise_ph, latent_vector2)

    rbg_32 = to_rgb(block_32)
    rbg_final = add([rbg_final, rbg_32])
    rbg_final = UpSampling2D(size=(2, 2), interpolation='bilinear')(rbg_final)

    up_64 = UpSampling2D(size=(2, 2), interpolation='bilinear')(block_32)

    block_64 = generator_block(up_64, 16, noise_ph, latent_vector2)
    block_64 = generator_block(block_64, 16, noise_ph, latent_vector2)

    rbg_64 = to_rgb(block_64)
    rbg_final = add([rbg_final, rbg_64], name="generator_out")

    rbg_final_final = Activation('sigmoid')(rbg_final)


    model = Model(inputs=[const_ph, noise_ph, latent_z_ph1, latent_z_ph2], outputs=rbg_final_final)

    return model



def default_gen(dir):
    latent_model = set_latent_enconder(load_model(dir))

    model = generator(latent_model)
    return model

#
#
# plot_model(model,"./../diagram/generator.png")
# model.summary()
# # #
# # #
# # #
# # # a = model.predict(
# # #     {
# # #         "constant": np.ones([1,4,4,1]),
# # #         "noise": np.random.normal(0,1, [1,64,64,1]),
# # #         "latent_z": np.ones([1,64,64,3])
# # #     }
# # # )
# #
# model.save_weights('./../models/generator.h5')
# model.load_weights('./../models/generator.h5')

