import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd
from keras.layers import *
from keras.models import *
import keras.backend as K
import tensorflow as tf
from keras.utils import plot_model


from common.kblocks import *


filtered = pd.read_csv('./../source/custom_set/metadata/attr.txt', sep='\s+')
labels = filtered.drop(['filename'], axis=1)



inputs = Input([64,64,3], name='input')
layer = latent_W_VGG_encoder(inputs)
layer = Dropout(0.5)(layer)
outputs = Dense(units=labels.shape[1], activation='sigmoid')(layer)
model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adadelta', loss='cosine_proximity', metrics=['binary_accuracy', 'mse'])

model.summary()

model.save('./../models/VGG_encoder.h5')
