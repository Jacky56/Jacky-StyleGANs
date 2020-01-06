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
from make_datasets.modify_metadata import *

# cols = ['Blurry', '5_o_Clock_Shadow', 'Sideburns', 'Wearing_Necktie', 'Wearing_Hat', 'Wearing_Necklace']
# meta_data = pd.read_csv('source/Anno/list_attr_celeba.txt', sep='\s+')
#
# filtered = drop_rows_columns(meta_data, cols, -1)
#
# filtered.to_csv('./source/custom_set/metadata/attr.txt', sep=' ')
filtered = pd.read_csv('source/custom_set/metadata/attr.txt', sep='\s+')



base_src = './source/img_align_celeba_png.7z/img_align_celeba_png/'
base_target = './source/custom_set/images/'
sizes = [4, 8, 16, 32, 64]

# for size in sizes:
#     for img in filtered['filename']:
#         src = base_src + img
#         print(src)
#         target = '{}/{}/{}'.format(base_target, size, img)
#         build_image_data(src, (size, size), target)


a = cv2.imread(base_target + '64/000001.png')
a = np.reshape(a, [1,64,64,3])
model = load_model('models/VGG_encoder.h5')

print(model.predict(a))
