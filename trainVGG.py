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

model_name = 'VGG_encoder.h5'
metadata = pd.read_csv('./source/custom_set/metadata/attr.txt', sep='\s+')
base_src = './source/custom_set/images/'
tesnorboard = tb(log_dir='./logs/{}'.format(model_name))
model = load_model('./models/{}'.format(model_name))
checkpoint = ModelCheckpoint('./models/{}'.format(model_name), monitor='val_accuracy', verbose=1, save_best_only=False, period=10)

image_size = 64
batch_size = 256

trainer = Train()
trainer.set_base_data_path(base_src)
trainer.set_checkpoint(checkpoint)
trainer.set_metadata(metadata)
trainer.set_model(model)
trainer.set_tensorboard(tesnorboard)

trainer.train(image_size, batch_size, epochs=150)

