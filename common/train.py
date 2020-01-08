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
            class_mode='categorical'
        )

        valid_generator = valid_datagen.flow_from_dataframe(
            dataframe=valid_df,
            directory=base_data_path_image,
            x_col='filename',
            y_col=feature_names,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical'
        )

        # x_train, y_train = self.getSample(image_size, batch_size)
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


    def getSample(self, image_size, batch_size):
        sampled_metadata = self.metadata.sample(batch_size)
        image_set = []

        for filename in sampled_metadata['filename']:
            image_set.append(cv2.imread('{}/{}/{}'.format(self.base_data_path, image_size, filename)))

        sampled_labels = sampled_metadata.drop(['filename'], axis=1)
        sampled_images = np.array(image_set)
        sampled_images = sampled_images / 255.0

        return sampled_images, sampled_labels
