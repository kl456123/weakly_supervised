#!/usr/bin/env python
# encoding: utf-8

import sys
# sys.path.append('..')
from libs.net import Net
from libs.tools import *
from libs.layers import Layers
from libs.points import *
import numpy as np


import caffe
import os
from skimage import transform, io, img_as_float
import numpy as np


model_def = 'model/darknet19/darknet19_deploy.prototxt'
model_weights = 'model/darknet19/darknet19_deploy.caffemodel'


def test_vggnet():
    num_classes = 1000
    net = Net(model_def, model_weights, num_classes, deploy=True)
    image_file = 'image/dog.jpg'
    mean_file = 'model/ilsvrc_2012_mean.npy'
    # mu = np.load(mean_file)
    input_shape = net._net.blobs['data'].data.shape

    transformed_im = preprocess(image_file,
                                input_shape=input_shape,
                                mean_file=mean_file,
                                scale='255',
                                format='RGB')
    output = net._net.forward(data=np.asarray([transformed_im]))
    prob = output['result']
    idx = np.argmax(prob[0])

    synsets = open('model/ilsvrc_synsets.txt').readlines()
    image_label = synsets[idx].split(',')[0].strip()
    image_label = image_label[image_label.index(
        ' ') + 1:]
    print image_label
    cls_idx = idx
    layers = Layers(net)
    prior = 0
    kernel_weight = np.zeros((num_classes,))
    kernel_weight[cls_idx] = 1
    pos = [np.nan, np.nan, np.nan]
    start_points = [Point_2D(pos, kernel_weight, prior)]
    start_layer_name = 'pool19'
    points, scores = layers.backward(
        start_points, start_layer_name, isFilter=True, debug=True)
    print points[0]
    points = threshold_system(points,
                              scores,
                              input_shape=input_shape[2:],
                              reserve_num_ratio=1,
                              reserve_num=50000,
                              reserve_scores_ratio=0,
                              max_num=50000)

    vis_square(net._net.blobs['data'].data[net.img_idx].transpose(
        1, 2, 0)[np.newaxis, ...])
    vis_activated_point(
        net._net, points, net.img_idx, 0.5)


caffe_init()

test_vggnet()
