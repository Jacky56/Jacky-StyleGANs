import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd
import cv2
import ntpath

def drop_rows_columns(df, col_names, val):
    filtered = df.copy()
    for col in col_names:
        filtered = filtered.loc[df[col] == val]

    filtered = filtered.drop(col_names, axis=1)

    return filtered

def drop_cols(df, col_names):
    filtered = df.drop(col_names, axis=1)
    return filtered

def build_image_data(src, size, target):
    img = cv2.imread(src)
    resized = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(target, resized)
    print('file:{} size:{} target: {} done'.format(ntpath.basename(src), size, target))


