# -*- coding:utf-8
"""
@project:faster_rcnn+_note
@author:xiezheng
@file:test1.py
"""
import numpy as np
import tensorflow as tf
import keras.backend as K
a = tf.random_normal([3,3,4],mean=0,stddev=1)
b = tf.random_normal([3,3,4],mean=0,stddev=1)
less_than_zero = tf.less(a,0)
label_a = tf.cast(less_than_zero,tf.int32)
# label = tf.where()

# positive = tf.gather_nd(a,label)
with tf.Session() as sess:
    sess.run(label_a)
    print(a.eval())
    print('less_than_zero is :',less_than_zero.eval())
    print('label_a is :',label_a.eval())
    # a = sess.run(positive)
    # print(a)