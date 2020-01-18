# -*- coding:utf-8
"""
@project:faster_rcnn+_note
@author:xiezheng
@file:test2.py
"""
import tensorflow as tf
# samples = tf.random_normal([2,4],mean=0,stddev=1)
samples = tf.constant([[1,2,3,4],
                       [5,6,7,8]],tf.int32)
a = tf.constant([[False,True,False,True],
             [False,False,True,True]])
b = tf.cast(a,tf.int32)
c = tf.where(b)
d = tf.gather_nd(samples,c)
with tf.Session() as sess:
    print(sess.run(samples))
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(d))