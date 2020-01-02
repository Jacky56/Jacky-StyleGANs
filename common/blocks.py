import tensorflow as tf
import numpy as np
import pandas as pd





def fully_connected(input, weight, bias):
    activation = tf.matmul(input, weight) + bias
    return activation


def generator_block(x, filters, lantent_vector, name):

    activation = tf.nn.conv2d(input=x,
                 filters=filters,
                 strides=[1, 1, 1, 1],
                 padding="SAME",
                 name=name)

    return activation




def AdaIN(x, y):

    x_mean = tf.reduce_mean(x,[1,2],True)
    x_std = tf.math.reduce_std(c, [1, 2], True)
    x = (x - x_mean) / x_std

    y_mean = tf.reduce_mean(y)
    y_std = tf.math.reduce_std(y)

    return (x*y_std) + y_mean



c = np.arange(16)
c = c.reshape((1,4,4,1))
f = np.arange(8)
f = f.reshape((2,2,c.shape[-1],2))

c = tf.convert_to_tensor(c, dtype=tf.float32)
f = tf.convert_to_tensor(f, dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    activation = generator_block(c,f,"name")
    a = sess.run(activation) + 1000
    print(a)


