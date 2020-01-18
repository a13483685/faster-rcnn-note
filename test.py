# -*- coding:utf-8
"""
@project:faster_rcnn+_note
@author:xiezheng
@file:test.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch

size_X = 16
size_Y = 16

scales = [1,2,4] #框大小
ratios = [0.5,1,2] #缩放倍率

rpn_stride = 8
scales,ratios = np.meshgrid(scales,ratios)
scales,ratios = scales.flatten(),ratios.flatten()
scaleY = scales * np.sqrt(ratios)
scaleX = scales / np.sqrt(ratios)

def anchor_gen(size_X,size_Y,rpn_stride,scales,ratios):
    # 锚点就是对应原图感受野的中心，抽取特征信息相当于把锚点周围錨框的特征信息给抽取出去了
    # 之后根据这些特征信息来计算优劣
    shift_x = np.arange(0,size_X) * rpn_stride #特征图对应原图的位置
    shift_y = np.arange(0,size_Y) * rpn_stride
    shift_x,shift_y = np.meshgrid(shift_x,shift_y)
    center_x,anchor_X = np.meshgrid(shift_x,scaleX)
    center_y,anchor_Y = np.meshgrid(shift_y,scaleY)
    anchor_size = np.stack([anchor_Y,anchor_X],axis=2).reshape(-1,2)
    anchor_center = np.stack([center_y,center_x],axis=2).reshape(-1,2)
    #左上定点的坐标和右下定点的坐标
    boxes = np.concatenate([anchor_center - 0.5*anchor_size,anchor_center + 0.5*anchor_size],axis=1)
    return boxes

anchor = anchor_gen(size_X,size_Y,rpn_stride,scales,ratios)
print(anchor.shape)

axs = plt.axes()
img = np.ones((128,128,3))
plt.imshow(img)
for i in range(anchor.shape[0]):
    boxes = anchor[i]
    rec = patch.Rectangle((boxes[0],boxes[1]),boxes[2]-boxes[0],boxes[3]-boxes[1],edgecolor='r',facecolor='none')
    axs.add_patch(rec)
plt.show()
