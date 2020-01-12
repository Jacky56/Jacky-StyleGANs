import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd
import cv2

from keras.layers import *
from keras.models import *
from keras.datasets import mnist
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from common.kblocks import *
from make_datasets import *
from keras.callbacks import TensorBoard as tb
from make_datasets.modify_metadata import *
from common.train import Train
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from common.generator import *
from common.discriminator import *

def custom_loss(y_true, y_pred):
    return tf.losses.sigmoid_cross_entropy(y_true, y_pred, label_smoothing=0.1)

def build_discriminator_trainer(discriminator, generator):
    for layer in generator.layers:
        layer.trainable = False
    for layer in discriminator.layers:
        layer.trainable = True

    const_ph = Input([4, 4, 1], name="constant")
    noise_ph = Input([64, 64, 1], name="noise")
    latent_z_ph1 = Input([64, 64, 3], name="image1")
    latent_z_ph2 = Input([64, 64, 3], name="image2")

    real_image = Input([64, 64, 3], name="discriminator_image")

    output_generator = generator([const_ph, noise_ph, latent_z_ph1, latent_z_ph2])
    output_fake = discriminator(output_generator)

    output_real = discriminator(real_image)

    model_discriminator = Model(inputs=[const_ph,
                                        noise_ph,
                                        latent_z_ph1,
                                        latent_z_ph2,
                                        real_image],
                                outputs=[output_real, output_fake])

    model_discriminator.compile(optimizer=Adam(0.0004, 0.5), loss=[custom_loss, custom_loss], metrics=['binary_accuracy', 'mse', custom_loss])

    return model_discriminator


def build_gnerator_trainer(discriminator, generator):
    for layer in generator.layers:
        layer.trainable = True
    for layer in discriminator.layers:
        layer.trainable = False

    const_ph = Input([4, 4, 1], name="constant")
    noise_ph = Input([64, 64, 1], name="noise")
    latent_z_ph1 = Input([64, 64, 3], name="image1")
    latent_z_ph2 = Input([64, 64, 3], name="image2")

    output_generator = generator([const_ph, noise_ph, latent_z_ph1, latent_z_ph2])
    output_discriminator = discriminator(output_generator)

    model_generator = Model(inputs=[const_ph , noise_ph, latent_z_ph1, latent_z_ph2], outputs=[output_discriminator])
    model_generator.compile(optimizer=Adam(0.0001, 0.5), loss='binary_crossentropy', metrics=['binary_accuracy', 'mse', custom_loss])

    return model_generator


global_discriminator = default_discriminator()
global_generator = default_gen('./models/VGG_encoder.h5')

dicriminator_name = 'discriminator_weights.h5'
generator_name = 'generator_weights.h5'
try:
    global_discriminator.load_weights('./models/{}'.format(dicriminator_name))
except:
    print("cannot find discriminator weights, creating new h5")

try:
    global_generator.load_weights('./models/{}'.format(generator_name))
except:
    print("cannot find generator weights, creating new h5")

metadata = pd.read_csv('./source/custom_set/metadata/attr.txt', sep='\s+')
base_src = './source/custom_set/images/'
tb_dis = tb(log_dir='./logs/{}'.format(dicriminator_name))
tb_gen = tb(log_dir='./logs/{}'.format(generator_name))
model_discriminator = build_discriminator_trainer(global_discriminator, global_generator)
model_generator = build_gnerator_trainer(global_discriminator, global_generator)


model_discriminator.summary()
model_generator.summary()

cp_dis = ModelCheckpoint('./models/{}'.format('full_dis.h5'),
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=True,
                             period=10)

cp_gen = ModelCheckpoint('./models/{}'.format('full_gen.h5'),
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=True,
                             period=10)
image_size = 64
batch_size = 126

trainer_discriminator = Train()
trainer_discriminator.set_base_data_path(base_src)
trainer_discriminator.set_checkpoint(cp_dis)
trainer_discriminator.set_metadata(metadata)
trainer_discriminator.set_model(model_discriminator)
trainer_discriminator.set_tensorboard(cp_gen)

trainer_generator = Train()
trainer_generator.set_base_data_path(base_src)
trainer_generator.set_checkpoint(cp_dis)
trainer_generator.set_metadata(metadata)
trainer_generator.set_model(model_generator)
trainer_generator.set_tensorboard(cp_gen)

for i in range(1000):
    trainer_discriminator.train_dis_GANs(image_size, batch_size, sample_size=250)
    trainer_generator.train_gen_GANs(image_size, batch_size, sample_size=4000)

    if i % 10 == 0:
        global_discriminator.save_weights('./models/{}'.format(dicriminator_name))
        global_generator.save_weights('./models/{}'.format(generator_name))
        print('discriminator and generator saved.')

