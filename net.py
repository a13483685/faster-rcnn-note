# -*- coding:utf-8
"""
@project:faster_rcnn+_note
@author:xiezheng
@file:net.py
"""
import keras.layers as KL
import tensorflow as tf
from keras.models import Model
from keras import optimizers
import keras.backend as K
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def block(filters ,block_id):
    if block_id != 0:
        stride = 1
    else:
        stride =2
    def f(x):
        origin = x
        x = KL.Conv2D(filters=filters,kernel_size=(1,1),strides=stride,padding='same')(x)
        x = KL.BatchNormalization(axis=3)(x)
        x = KL.Activation('relu')(x)
        x = KL.Conv2D(filters=filters,kernel_size=(3,3),padding='same')(x)
        x = KL.BatchNormalization(axis=3)(x)
        x = KL.Activation('relu')(x)
        x = KL.Conv2D(filters=4*filters,kernel_size=(1,1))(x)
        x = KL.BatchNormalization(axis=3)(x)
        if block_id == 0:
            shortcut = KL.Conv2D(filters=4*filters,kernel_size=(1,1),strides=stride,padding='same')(origin)
            shortcut = KL.BatchNormalization(axis=3)(shortcut)
        else:
            shortcut=x
        x = KL.add([x,shortcut])
        x = KL.Activation('relu')(x)
        return x
    return f

def resNet_featureExtrator(input):
    x = KL.Conv2D(64,(3,3),padding='same')(input)
    x = KL.BatchNormalization(axis=3)(x)
    x = KL.Activation('relu')(x)

    fiters = 64
    blocks = [3,6,4]
    for i,blocks_num in enumerate(blocks):
        for block_id in range(blocks_num):
            x = block(filters=fiters,block_id=block_id)(x)
        fiters*=2
    return x

def rpn_net(inputs,k):
    # inputs shape is :(?, 8, 8, 1024)
    shared_map=KL.Conv2D(256,kernel_size=(3,3),padding='same',activation='relu')(inputs)
    rpn_class = KL.Conv2D(2*k,kernel_size=(1,1))(shared_map)
    rpn_class = KL.Lambda(lambda x:tf.reshape(x,[tf.shape(x)[0],-1,2]))(rpn_class)
    rpn_class = KL.Activation('linear')(rpn_class)
    rpn_prob = KL.Activation('softmax')(rpn_class)

    y = KL.Conv2D(4*k,(1,1))(shared_map)
    y = KL.Activation('linear')(y)
    rpn_box = KL.Lambda(lambda x:tf.reshape(x,[tf.shape(x)[0],-1,4]))(y)
    return rpn_class,rpn_prob,rpn_box

#以下是定义2个损失函数
#计算分类loss
def rpn_class_loss(rpn_match,rpn_class_logists):
    # rpn_match [None,576,1]
    print('rpn_match.shape is :{}',rpn_match.shape)
    print('rpn_class_logists.shape is :{}',rpn_class_logists.shape)
    rpn_match = tf.squeeze(rpn_match,-1)
    indices = tf.where(tf.not_equal(rpn_match,0))#取出坐标 -1,1
    #将-1转化为0,
    anchor_class = tf.cast(tf.equal(rpn_match,1),tf.int32)
    # prediction,将预测值中的对应的indices label为-1,1的元素取出来
    rpn_class_logists = tf.gather_nd(rpn_class_logists,indices)
    anchor_class = tf.gather_nd(anchor_class,indices)
    loss = K.sparse_categorical_crossentropy(output=rpn_class_logists,target=anchor_class,from_logits=True)
    loss = K.switch(tf.size(loss)>0,K.mean(loss),tf.constant(0.0))
    return loss

def batch_pack_graph(x,counts,num_rows):
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i,:counts[i]])
    return tf.concat(outputs,axis=0)


#计算回归loss
#只有标签为1的才需要修正
#rpn_bbox计算结果 [None,576,4]
#target_bbox是真实的结果
def rpn_box_loss(target_bbox,rpn_match,rpn_bbox):
    # print('-----------------------')
    print('rpn_bbox before shape is :{}'.format(rpn_bbox.shape))
    rpn_match = tf.squeeze(rpn_match,-1)# [None,576]
    indices = tf.where(K.equal(rpn_match,1))
    rpn_bbox = tf.gather_nd(rpn_bbox,indices)
    print('rpn_bbox shape is :{}'.format(rpn_bbox.shape))
    #真实的计算结果
    batch_count = K.sum(tf.cast(tf.equal(rpn_match,1),tf.int32),axis=1)#rpn_match当中每一行有多少个label为1（正錨框）的总数，一行代表一个特征图
    target_bbox = batch_pack_graph(target_bbox,batch_count,20)
    print('target_bbox shape is {0},rpn_bbox shape is {1}'.format(target_bbox.shape,rpn_bbox.shape))
    diff = K.abs(target_bbox-rpn_bbox)
    less_than_one = tf.cast(K.less(diff,1.0),tf.float32)
    loss = (less_than_one*0.5*diff**2) + (1-less_than_one)* (diff-0.5)
    loss = K.switch(tf.size(loss) > 0,K.mean(loss),tf.constant(0.0))
    return loss
#定义输入
input_image = KL.Input(shape=[64,64,3],dtype=tf.float32)
#真实的boundingbox
input_bbox = KL.Input(shape=[None,4],dtype=tf.float32)
#形状的类别：三角形，圆形，正方形
input_class_ids =KL.Input(shape=[None],dtype=tf.int32)
#目标
input_rpn_match = KL.Input(shape=(None,1),dtype=tf.int32)
input_rpn_bbox = KL.Input(shape=(None,4),dtype=tf.float32)
feature_map = resNet_featureExtrator(input_image)
print('faturemap shape is :{}'.format(feature_map.shape))
rpn_class,rpn_prob,rpn_box = rpn_net(feature_map,9)
loss_rpn_match = KL.Lambda(lambda x:rpn_class_loss(*x),name='loss_rpn_match')([input_rpn_match,rpn_class])
loss_rpn_bbox = KL.Lambda(lambda x:rpn_box_loss(*x),name='loss_rpn_bbox')([input_rpn_bbox,input_rpn_match,rpn_box])
model = Model([input_image,input_bbox,input_class_ids,input_rpn_match,input_rpn_bbox],
              [rpn_class,rpn_prob,rpn_box,loss_rpn_match,loss_rpn_bbox])

loss_lay1 = model.get_layer('loss_rpn_match').output
loss_lay2 = model.get_layer('loss_rpn_bbox').output

model.add_loss(loss_lay1)
model.add_loss(loss_lay2)



sgd = optimizers.SGD(lr=0.00005,momentum=0.9)
# model.compile(loss=[None]*len(model.output),optimizer=optimizers.SGD(lr=0.00005,momentum=0.9))
model.compile(loss=[None]*5,optimizer=sgd)
model.metrics_names.append('loss_rpn_match')
model.metrics_tensors.append(tf.reduce_mean(loss_lay1,keep_dims=True))
model.metrics_names.append('loss_rpn_bbox')
model.metrics_tensors.append(tf.reduce_mean(loss_lay2,keep_dims=True))
# model.summary()
from keras.utils.vis_utils import plot_model
plot_model(model,'model.jpg',show_shapes=True,show_layer_names=True)


from config import Config
from utils import shapeData as dataSet
config = Config()
dataSet = dataSet([64,64],config=config)


def data_Gen(dataset, num_batch, batch_size, config):
    for _ in range(num_batch):
        images = []
        bboxes = []
        class_ids = []
        rpn_matchs = []
        rpn_bboxes = []
        for i in range(batch_size):
            image, bbox, class_id, rpn_match, rpn_bbox, _ = data = dataset.load_data()
            pad_num = config.max_gt_obj - bbox.shape[0]
            pad_box = np.zeros((pad_num, 4))
            pad_ids = np.zeros((pad_num, 1))
            bbox = np.concatenate([bbox, pad_box], axis=0)
            class_id = np.concatenate([class_id, pad_ids], axis=0)

            images.append(image)
            bboxes.append(bbox)
            class_ids.append(class_id)
            rpn_matchs.append(rpn_match)
            rpn_bboxes.append(rpn_bbox)
        # print('----------------------------------------------')
        images = np.concatenate(images, 0).reshape(batch_size, config.image_size[0], config.image_size[1], 3)
        bboxes = np.concatenate(bboxes, 0).reshape(batch_size, -1, 4)
        class_ids = np.concatenate(class_ids, 0).reshape(batch_size, -1)
        rpn_matchs = np.concatenate(rpn_matchs, 0).reshape(batch_size, -1, 1)
        # print('rpn_match shape is :{}'.format(rpn_match.shape))
        rpn_bboxes = np.concatenate(rpn_bboxes, 0).reshape(batch_size, -1, 4)
        # print('rpn_bboxes shape is :{}'.format(rpn_bboxes.shape))
        yield [images, bboxes, class_ids, rpn_matchs, rpn_bboxes], []

model.load_weights('model.h5')
dataGen = data_Gen(dataSet,35000,20,config)
# model.fit_generator(dataGen,steps_per_epoch=20,epochs=800)
# model.save_weights('model.h5')

# print(model.summary())
# print(model.outputs)

#proposal的小函数

def anchor_refinement(boxes,delta):
    # boxes:[20,100,4]
    # delta:   [100,4]

    boxes = tf.cast(boxes,tf.float32)
    w = boxes[:,3] - boxes[:,1] #[20,4]
    h = boxes[:,2] - boxes[:,0] #[20,4]
    center_x = boxes[:,1] + w/2 #原始的center_x [20,4]
    center_y = boxes[:,0] + h/2 #原始的center_y [20,4]
    center_y += delta[:,0] * h #x,y 也是依据宽高来进行缩放的，为了保证平移不变性？
    center_x += delta[:,1] * w
    h *=tf.exp(delta[:,2])
    w *=tf.exp(delta[:,3])
    y1 = center_y - h/2
    x1 = center_x - w/2
    y2 = center_y + h/2
    x2 = center_x + w/2
    boxes = tf.stack([y1,x1,y2,x2],axis=1)#这里要注意concat和stack的区别，concat不会改变维度，在原有的维度上拼接
    return boxes

def boxes_clip(boxes,window):
    wy1,wx1,wy2,wx2 = tf.split(window,4)
    y1,x1,y2,x2 = tf.split(boxes,4,axis=1)
    y1 = tf.maximum(tf.minimum(y1,wy2),wy1)
    x1 = tf.maximum(tf.minimum(x1,wx2),wx1)
    y2 = tf.maximum(tf.minimum(y2,wy2),wy1)
    x2 = tf.maximum(tf.minimum(x2,wx2),wx1)
    cliped = tf.concat([y1,x1,y2,x2],axis=1)
    cliped.set_shape([cliped.shape[0],4])
    return cliped

def batch_slice(inputs,graph_fn,batch_size):
    if not isinstance(inputs,list):
        inputs = [inputs]
    output = []
    for i in range(batch_size):
        input_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*input_slice)
        if not isinstance(output_slice,(list,tuple)):
            output_slice = [output_slice]
        output.append(output_slice)
    output = list(zip(*output))
    result = [tf.stack(o,axis=0) for o in output]
    if len(result) == 1:
        result = result[0]
    return result

import keras.engine as KE
class propsal(KE.Layer):
    ''''''
    def __init__(self,propsal_count,nms_thresh,anchors,batch_size,config=None,**kwargs):
        '''

        :param propsal_count:推荐给后面的层的个数是一定的
        :param nms_thresh:nms阈值
        :param anchors:标准錨框[y1,x1,y2,x2]
        :param batch_size:
        :param config:
        :param kwargs:
        '''
        super(propsal,self).__init__(**kwargs)
        self.propsal_count =propsal_count
        self.nms_thresh = nms_thresh
        self.anchors = anchors #定义的标准錨框
        self.batch_size = batch_size
        self.config = config
    def call(self, inputs, **kwargs):
        probs = inputs[0][:,:,1] #前景概率
        deltas = inputs[1] #偏移值
        deltas =deltas*np.reshape(self.config.RPN_BBOX_STD_DEV,(1,1,4))
        # 取得得分最高的一批錨框的编号
        prenms_num = min(100,self.anchors.shape[0])#最少取出100个錨框
        idxs = tf.nn.top_k(probs,prenms_num).indices #得到得分最高的anchors
        probs = batch_slice([probs,idxs],lambda x,y:tf.gather(x,y),self.batch_size)
        deltas = batch_slice([deltas,idxs],lambda x,y:tf.gather(x,y),self.batch_size)
        print('before process anchor size is : {}'.format(self.anchors.shape))
        print('---------------------------------------------')
        anchors = batch_slice([idxs],lambda x:tf.gather(self.anchors,x),self.batch_size)
        print('---------------------------------------------')
        print('anchors shape is :{},deltas shape is :{}'.format(anchors.shape,tf.shape(deltas)))
        refined_box = batch_slice([anchors,deltas],lambda x,y:anchor_refinement(x,y),self.batch_size) #修正后的anchors
        H,W =self.config.image_size[:2]
        windows = np.array([0,0,H,W]).astype(np.float32)
        cliped_boxes = batch_slice([refined_box],lambda x:boxes_clip(x,windows),self.batch_size)
        normalized_boxes = cliped_boxes/np.array([H,W,H,W])#按照窗口的高度来进行归一化，防止到了后面的层，坐标变化
        #推荐给后层的数量不够怎么办？还要进行padding
            #score:得分
        def nms(normalized_boxes,scores):
            idxs = tf.image.non_max_suppression(normalized_boxes,scores,self.propsal_count,self.nms_thresh)
            box = tf.gather(normalized_boxes,idxs)
            padding_num = tf.maximum((self.propsal_count-tf.shape(box)[0]),0)
            box = tf.pad(box,[(0,padding_num),(0,0)])
            return box
        propsal_ = batch_slice([normalized_boxes,probs],nms,self.batch_size)
        return propsal_
    def compute_output_shape(self, input_shape):
        return (None,self.propsal_count,4)

test_data = next(dataGen)[0]
images = test_data[0]
bboxes = test_data[1]
class_ids = test_data[2]
rpn_matchs = test_data[3]
rpn_bboxes = test_data[4]
rpn_class,rpn_prob,rpn_bboxes,_,_ = model.predict([images, bboxes, class_ids, rpn_matchs, rpn_bboxes])
rpn_class = tf.convert_to_tensor(rpn_class)
rpn_prob = tf.convert_to_tensor(rpn_prob)#rpn分类得分
rpn_bboxes = tf.convert_to_tensor(rpn_bboxes)#每个anchor算出来的修正量
print('rpn_class is :',rpn_class)
print('rpn_prob is :',rpn_prob)
print('rpn_bboxes is :',rpn_bboxes)
import utils
anchors = utils.anchor_gen([8,8],ratios=config.ratios,scales=config.scales,rpn_stride=config.rpn_stride,anchor_stride=config.anchor_stride)
#推荐之前是100个，推荐之后经过mns之后是16个
#[20,16,4]
# [batchsize,anchors数量，坐标值]
propsals = propsal(propsal_count=16,nms_thresh=0.7,anchors=anchors,batch_size=20,config=config)([rpn_prob,rpn_bboxes])

sess = tf.Session()
propsals_ = sess.run(propsals) * 64#在原图的大小上操作

import random
ix = random.sample(range(20),1)[0]
propsal_ = propsals_[ix]
img = images[ix]
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.imshow(img)
axs = plt.gca()
for i in range(propsal_.shape[0]):
    box = propsal_[i]
    rec = patches.Rectangle((box[0],box[1]),box[2] - box[0],box[3] - box[1],facecolor='none',edgecolor='r')
    axs.add_patch(rec)
plt.show()