import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd
import cv2

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import *
from keras.models import *
from keras.datasets import mnist
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from common.kblocks import *
from make_datasets import *
from make_datasets.modify_metadata import *
import random

class Train:
    def __init__(self):
        self.base_data_path = None
        self.model = None
        self.checkpoint = None
        self.metadata = None
        self.tensorboard = None

    def set_base_data_path(self, dir):
        self.base_data_path = dir

    def set_metadata(self, df):
        self.metadata = df

    def set_model(self, model):
        self.model = model

    def set_checkpoint(self, checkpoint):
        self.checkpoint = checkpoint

    def set_tensorboard(self, tensorboard):
        self.tensorboard = tensorboard

    def train(self, image_size, batch_size, epochs=1):

        feature_names = [feature for feature in self.metadata.columns if feature != 'filename']
        base_data_path_image = '{}/{}/'.format(self.base_data_path, image_size)

        train_datagen = ImageDataGenerator(
            rotation_range=20,
            rescale=1. / 255.,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        valid_datagen = ImageDataGenerator(rescale=1. / 255)

        train_df = self.metadata
        valid_df = self.metadata.sample(frac=0.3)

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=base_data_path_image,
            x_col='filename',
            y_col=feature_names,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='raw'
        )

        valid_generator = valid_datagen.flow_from_dataframe(
            dataframe=valid_df,
            directory=base_data_path_image,
            x_col='filename',
            y_col=feature_names,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='raw'
        )

        self.model.fit_generator(
            train_generator,
            epochs=epochs,
            steps_per_epoch=len(train_generator),
            validation_data=valid_generator,
            validation_steps=len(valid_generator),
            max_queue_size=1,
            shuffle=True,
            # verbose=1,
            callbacks=[self.checkpoint, self.tensorboard])

    def train_dis_GANs(self, image_size, max_image_size,  batch_size, epochs=1, sample_size=100):
        return self.model.fit_generator(
            self.setup_data_dis_GANS(image_size, max_image_size, batch_size, sample_size),
            epochs=epochs,
            steps_per_epoch=sample_size//batch_size,
            max_queue_size=1,
            shuffle=True,
            callbacks=[self.checkpoint, self.tensorboard])

    def train_gen_GANs(self, image_size,max_image_size, batch_size, epochs=1, sample_size=100):
        self.model.fit_generator(
            self.setup_data_gen_GANS(image_size, max_image_size, batch_size, sample_size),
            epochs=epochs,
            steps_per_epoch=sample_size//batch_size,
            max_queue_size=1,
            shuffle=True,
            callbacks=[self.checkpoint, self.tensorboard])

    def setup_data_dis_GANS(self, image_size, max_image_size, batch_size, sample_size):
        base_data_path_image = '{}/{}/'.format(self.base_data_path, image_size)

        train_df = []
        for i in range(3):
            train_df.append(self.metadata.sample(sample_size))

        noise = np.random.uniform(0, 1, [sample_size, max_image_size, max_image_size, 1])

        imagegen = ImageDataGenerator(
            rescale=1. / 255.,
            # rotation_range=15,
            # width_shift_range=0.05,
            # height_shift_range=0.05,
            # shear_range=0.05,
            # zoom_range=0.15,
            horizontal_flip=True,
            fill_mode='nearest')

        variablegen = ImageDataGenerator()

        genN = variablegen.flow(noise, batch_size=batch_size)


        genX1 = imagegen.flow_from_dataframe(
                dataframe=train_df[0],
                directory=base_data_path_image,
                x_col='filename',
                target_size=(max_image_size, max_image_size),
                batch_size=batch_size,
                interpolation='bilinear',
                class_mode=None)

        genX2 = imagegen.flow_from_dataframe(
                dataframe=train_df[1],
                directory=base_data_path_image,
                x_col='filename',
                target_size=(max_image_size, max_image_size),
                batch_size=batch_size,
                interpolation='bilinear',
                class_mode=None)

        genX3 = imagegen.flow_from_dataframe(
                dataframe=train_df[2],
                directory=base_data_path_image,
                x_col='filename',
                target_size=(max_image_size, max_image_size),
                batch_size=batch_size,
                interpolation='bilinear',
                class_mode=None)

        while True:
            noise = genN.next()
            img1 = genX1.next()
            img2 = genX2.next()
            img3 = genX3.next()
            yield [noise, img1, img2, img3], [np.ones([noise.shape[0]]), np.zeros([noise.shape[0]])]

    def setup_data_gen_GANS(self, image_size, max_image_size, batch_size, sample_size):
        base_data_path_image = '{}/{}/'.format(self.base_data_path, image_size)

        train_df = []
        for i in range(2):
            train_df.append(self.metadata.sample(sample_size))

        noise = np.random.uniform(0, 1, [sample_size, max_image_size, max_image_size, 1])

        imagegen = ImageDataGenerator(
            rescale=1. / 255.,
            # rotation_range=15,
            # width_shift_range=0.05,
            # height_shift_range=0.05,
            # shear_range=0.05,
            # zoom_range=0.15,
            horizontal_flip=True,
            fill_mode='nearest')

        variablegen = ImageDataGenerator()

        genN = variablegen.flow(noise, batch_size=batch_size)

        genX1 = imagegen.flow_from_dataframe(
            dataframe=train_df[0],
            directory=base_data_path_image,
            x_col='filename',
            target_size=(max_image_size, max_image_size),
            batch_size=batch_size,
            interpolation='bilinear',
            class_mode=None)

        genX2 = imagegen.flow_from_dataframe(
            dataframe=train_df[1],
            directory=base_data_path_image,
            x_col='filename',
            target_size=(max_image_size, max_image_size),
            batch_size=batch_size,
            interpolation='bilinear',
            class_mode=None)

        while True:
            noise = genN.next()
            img1 = genX1.next()
            img2 = genX2.next()

            if random.random() > 0.5:
                img_out = img2
            else:
                img_out = img1

            yield [noise, img1, img2], [np.ones([noise.shape[0]]), img_out]

