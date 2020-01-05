import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from keras.layers import *
from keras.models import *
import keras.backend as K
import tensorflow as tf
from keras.utils import plot_model


from common.kblocks import *



def generator():

    const_ph = Input([4, 4, 1], name="constant")
    noise_ph = Input([64, 64, 1], name="noise")

    latent_z_ph = Input([64,64,3], name="latent_z")
    latent_vector = latent_W_VGG_encoder(latent_z_ph)

    # latent_z_ph = Input([256], name="latent_z")
    # latent_vector = latent_W(latent_z_ph)

    block_4 = generator_block_v2(const_ph, 256, noise_ph, latent_vector)
    block_4 = generator_block_v2(block_4, 256, noise_ph, latent_vector)

    up_8 = UpSampling2D(size=(2, 2), interpolation='bilinear')(block_4)

    block_8 = generator_block_v2(up_8, 128, noise_ph, latent_vector)
    block_8 = generator_block_v2(block_8, 128, noise_ph, latent_vector)

    up_16 = UpSampling2D(size=(2, 2), interpolation='bilinear')(block_8)

    block_16 = generator_block_v2(up_16, 64, noise_ph, latent_vector)
    block_16 = generator_block_v2(block_16, 64, noise_ph, latent_vector)

    up_32 = UpSampling2D(size=(2, 2), interpolation='bilinear')(block_16)

    block_32 = generator_block_v2(up_32, 32, noise_ph, latent_vector)
    block_32 = generator_block_v2(block_32, 32, noise_ph, latent_vector)

    up_64 = UpSampling2D(size=(2, 2), interpolation='bilinear')(block_32)

    block_64 = generator_block_v2(up_64, 32, noise_ph, latent_vector)
    block_64 = generator_block_v2(block_64, 32, noise_ph, latent_vector)

    rbg_64 = to_rgb(block_64)

    model = Model(inputs=[const_ph, noise_ph, latent_z_ph], outputs=rbg_64)

    return model



model = generator()



# plot_model(model,"./../diagram/generator.png")
#
#
#
# a = model.predict(
#     {
#         "constant": np.ones([1,4,4,1]),
#         "noise": np.random.normal(0,1, [1,64,64,1]),
#         "latent_z": np.ones([1,64,64,3])
#     }
# )

model.save_weights('./../models/test.h5')
model.load_weights('./../models/test.h5')

