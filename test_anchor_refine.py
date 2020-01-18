# -*- coding:utf-8
"""
@project:faster_rcnn+_note
@author:xiezheng
@file:test_anchor_refine.py
"""
import tensorflow as tf

a = tf.constant([[[1,2,3,4],[4,3,2,1],[3,2,1,5]],
                 [[1,2,3,4],[4,3,2,1], [3,2,1,5]]])
b = tf.constant([[[1,2,3,4],[4,3,2,1],[3,2,1,5]],
                 [[1,2,3,4],[4,3,2,1],[3,2,1,5]]])
with tf.Session() as tf:
    print(a.shape)
    print(a[:,1].eval())
    print(a[:,2].eval())





