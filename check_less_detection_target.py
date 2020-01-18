# -*- coding:utf-8
"""
@project:faster_rcnn+_note
@author:xiezheng
@file:check_less_detection_target.py
"""
import tensorflow as tf
import numpy as np
import keras.backend as K
import keras.engine as KE
import keras.layersm as KL

def batch_slice(inputs,graph_fn,batch_size,names=None):
    if not isinstance(inputs,list):
        inputs = [inputs]
    outputs = []
    for i in range(batch_size):
        input_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*input_slice)
        if not isinstance(output_slice,(tuple,list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    if names is None:
        names = [None]*len(outputs)

    result = [tf.stack(o,axis=0,name=n) for o,n in zip(outputs,names)]
    if len(result) == 1:
        return result[0]
    return result

def box_refinement_graph(boxes,gt_boxes):
    '''

    :param boxes: proposal层推荐过来的
    :param gt_boxes: 真实的boundingbox
    :return:
    '''
    boxes = tf.cast(boxes,tf.float32)
    gt_boxes = tf.cast(gt_boxes,tf.float32)
    heigh = boxes[...,2] - boxes[...,0]
    width = boxes[...,3] - boxes[...,1]
    center_y = boxes[...,0] + heigh*0.5
    center_x = boxes[...,1] + width*0.5

    #计算缩放量

    gt_h = gt_boxes[...,2] - gt_boxes[...,0]
    gt_w = gt_boxes[...,3] - gt_boxes[...,1]
    gt_center_y = gt_boxes[...,0] + 0.5*gt_h
    gt_center_x = gt_boxes[...,1] + 0.5*gt_w
    #偏移量
    dy = (gt_center_y - center_y)/heigh
    dx = (gt_center_x - center_x)/width
    #伸缩量
    dh = tf.log(gt_h/heigh)
    dw = tf.log(gt_w/width)
    delta = tf.stack([dy,dx,dh,dw],axis=1)
    return delta

def overlaps_graph(boxes1,boxes2):
    '''
    计算2组框的iou
    加入box1有16（N）组框，box2有10(M)组框最后就是一个16行10列(N行M列)的矩阵
    :param boxes1:propsal推荐的框
    :param boxes2:真实的boundingbox
    :return:
    '''
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1,1),[1,1,tf.shape(boxes2)[0]]),[-1,4])
    b2 = tf.reshape(boxes2,[tf.shape(boxes1)[0],1])

    b1_y1,b1_y2,b1_x1,b1_x2 = tf.split(b1,4,axis=1)
    b2_y1, b2_y2, b2_x1, b2_x2 = tf.split(b2,4,axis=1)

    y1 = tf.maximum(b1_y1,b2_y1)
    x1 = tf.maximum(b1_x1,b2_x1)
    y2 = tf.minimum(b2_y2 ,b1_y2)
    x2 = tf.minimum(b2_x2,b1_x2)

    intersection = tf.maximum((x2-x1),0) * tf.maximum((y2-y1),0)
    union = ((b1_x2 - b1_x1)*(b1_y2 - b1_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1)) - intersection
    iou = union/intersection
    overlap = tf.reshape(iou,[tf.shape(boxes1)[0],tf.shape(boxes2)[0]])
    return overlap

def trim_zeros_graph(boxes,name=None):
    '''
    假如这个box里面有一些padding，16个propsal假如只有前面5个是有意义的
    后面11个都是padding成0的。那么就可以将0的部分去掉，并且把坐标位置取出来，主要是去除0元素的功能
    :param boxes:
    :param name:
    :return:
    '''
    none_zero = tf.cast(tf.reduce_sum(tf.abs(boxes),axis=1),tf.bool)
    boxes = tf.boolean_mask(boxes,none_zero,name=name)
    return boxes,none_zero

def detection_target_graph(propsals,gt_class_id,gt_bboxes,config):
    '''
    在一个数据上面是如何进行处理的
    :param propsals:上一层propsals推荐得到的区域
    :param gt_class_id:真实的class id
    :param gt_bboxes:真实的boundingbox
    :param config:
    :return:
    '''
    # 将非0的部分提取出来
    # gt_class_id,gt_bboxes这些实际上在数据里面都是有padding的，我们把数据组成一个个padding的话，
    # 要组成一个个batch的话，每一个数据都是一样的，比如说在一个batch中一共有32张图，第一张图里面有5个目标
    # 第二种图里面有10个目标，第三张图里面有6个目标，不可能说每个数据都不一样，比如说一个是5一个是6一个是10
    # 这样组成不了一个batch，就取一个比较大的值，比如说100，只要不超过100就ok了，然后100多出来的部分就padding
    # 成0。gt_class_id,gt_bboxes里面必然是有padding的，将非0的部分取出来
    propsals,_ = trim_zeros_graph(propsals,name='trim_propsals')
    boxes, none_zero = trim_zeros_graph(gt_bboxes,name='trim_boxes')
    gt_class_id = tf.boolean_mask(gt_class_id,none_zero)

    overlaps = overlaps_graph(propsals,gt_bboxes)#计算propsals与真实boundingbox之间的交并比
    # [16,10] 也就是N行M列的矩阵
    max_iouArg = tf.reduce_max(overlaps,axis=1) #取每一行的最大值
    max_iouGT = tf.argmax(overlaps,axis=0)#取得每一行最大值的坐标，如果没有的话大于iou的话就取一个最大值

    positive_mask = (max_iouArg > 0.5)#正负iou的阈值设置为了0.5
    positive_idxs = tf.where(positive_mask)[:,0]
    negtive_idxs = tf.where(max_iouArg < 0.5)[:,0]
    # positive的总是是要定下来的
    num_positive = int(config.num_proposals_train * config.num_proposals_ratio)
    positive_idxs = tf.random_shuffle(positive_idxs)[:num_positive]
    positive_idxs = tf.unique(positive_idxs)[0]

    num_positive = tf.shape(positive_idxs)[0]
    r = 1 / config.num_proposals_ratio