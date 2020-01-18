# -*- coding:utf-8
"""
@project:faster_rcnn+_note
@author:xiezheng
@file:test_nms.py
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
image = cv2.imread('test.jpg')
image = cv2.resize(image,(2000,1000))
# image = image[:,:,::-1]
# cv2.imshow('1',mat=image)
plt.imshow(image)
axc = plt.gca()
colors = ['r','blue','black','pink']
boxes = [[200,300,300,400],[800,200,400,400],[1200,700,400,340],[1500,600,400,800]]
for i,c in enumerate(colors):
    box = boxes[i]
    rec = patches.Rectangle((box[0],box[1]),box[2],box[3],facecolor='none',edgecolor=str(c))
    axc.add_patch(rec)
plt.show()

nor_boxes = np.array(boxes)/1000
sores = [0.7,0.6,0.5,0.1]
count = 4
th = 0.1
idx = tf.image.non_max_suppression(nor_boxes,sores,count,th)
sess = tf.Session()
ids = sess.run(idx)
print(ids)
axc = plt.gca()
plt.imshow(image)
axc = plt.gca()
colors = ['r','black','black','black']

for i,id in enumerate(ids):
    id = int(id)
    box = boxes[id]
    rec = patches.Rectangle((box[0], box[1]), box[2], box[3], facecolor='none', edgecolor=colors[i])
    axc.add_patch(rec)
plt.show()
