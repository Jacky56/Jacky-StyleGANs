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

from common.generator import default_gen
from common.kblocks import *
from make_datasets import *
from make_datasets.modify_metadata import *

# cols = ['Blurry', '5_o_Clock_Shadow', 'Sideburns', 'Wearing_Necktie', 'Wearing_Hat', 'Wearing_Necklace']
# meta_data = pd.read_csv('source/Anno/list_attr_celeba.txt', sep='\s+')
#
# filtered = drop_rows_columns(meta_data, cols, -1)
#
# filtered.to_csv('./source/custom_set/metadata/attr.txt', sep=' ')
# filtered = pd.read_csv('source/custom_set/metadata/attr.txt', sep='\s+')
#
#
#
# base_src = './source/img_align_celeba_png.7z/img_align_celeba_png/'
base_target = './source/custom_set/images/'
# sizes = [4, 8, 16, 32, 64]

# for size in sizes:
#     for img in filtered['filename']:
#         src = base_src + img
#         print(src)
#         target = '{}/{}/{}'.format(base_target, size, img)
#         build_image_data(src, (size, size), target)


global_generator = default_gen('./models/VGG_encoder.h5')
generator_name = 'generator_weights.h5'
try:
    global_generator.load_weights('./models/{}'.format(generator_name))
except:
    print("cannot find discriminator weights, creating new h5")


a = cv2.imread(base_target + '64/000001.png').reshape([1,64,64,3])
b = cv2.imread(base_target + '64/000002.png').reshape([1,64,64,3])

c = np.ones([1,4,4,1])
n = np.random.normal(0,1,[1,64,64,1])

# model = Model(inputs=[const_ph, noise_ph, latent_z_ph1, latent_z_ph2], outputs=rbg_final)


p = global_generator.predict([c,n,a,b])

img = p.reshape([64,64,3])
img *= 255.0/img.max()
img = img.astype(np.uint8)
img = np.array(img, np.int32)
plt.imshow(img)
plt.show()
