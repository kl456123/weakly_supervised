#!/usr/bin/env python
# encoding: utf-8

import sys
# sys.path.append('..')
from lib.net import Net
from lib.tools import *
from lib.part_object_scores import Part_Object_Scores
import numpy as np

import caffe
import os
from skimage import transform, io, img_as_float
import numpy as np


model_def = 'model/vgg16/VGG_ILSVRC_16_layers_deploy.prototxt'
model_weights = 'model/vgg16/VGG_ILSVRC_16_layers.caffemodel'


def test_fullyconnectedlayer(filtered_kernel_idx_weight_pairs):
    pass


def test_vggnet():
    num_classes = 1000
    net = Net(model_def, model_weights, num_classes, deploy=True)
    image_file = 'image/cat.jpg'
    mean_file = 'model/ilsvrc_2012_mean.npy'
    # mu = np.load(mean_file)

    input_shape = net._net.blobs['data'].data.shape

    transformed_im = preprocess(image_file,
                                input_shape=input_shape,
                                mean_file=mean_file,
                                scale='255',
                                format='RGB')
    output = net._net.forward(data=np.asarray([transformed_im]))
    prob = output['prob']
    idx = np.argmax(prob[0])

    synsets = open('model/ilsvrc_synsets.txt').readlines()
    image_label = synsets[idx].split(',')[0].strip()
    image_label = image_label[image_label.index(
        ' ') + 1:]
    print image_label
    cls_idx = idx
    ####################################
    ####################################
    ####################################
    layer_name = 'fc8'
    center = []
    center.append(cls_idx)
    top_blob_name = net._net.top_names[layer_name][0]
    center += list(
        get_max_xy(net._net.blobs[top_blob_name].data[net.img_idx, cls_idx]))

    spatial_position = center[1:]
    start_prior_score = 0

    part_object_scores = Part_Object_Scores(
        layer_name, start_prior_score, spatial_position, net)
    last_filtered_kernel_idx_weight_pairs = ([cls_idx], [1])
    part_object_scores.init(last_filtered_kernel_idx_weight_pairs)
    print part_object_scores.get_score_map()
    # start from part_object_scores
    queue = top_down(part_object_scores,
                     net,
                     threshold_policy='float',
                     min_receptive_field=0,
                     threshold=0)
    # print queue[0].get_raw_spatial_position()
    print (get_scores_sum(queue))
    filtered_queue = threshold_system(
        queue, input_shape=input_shape, reserve_num=200, reserve_ratio=1, display=True)
    all_raw_spatial_position = get_all_raw_spatial_position(filtered_queue)
    # visualization all points
    vis_square(net._net.blobs['data'].data[net.img_idx].transpose(
        1, 2, 0)[np.newaxis, ...])
    vis_activated_point(
        net._net, all_raw_spatial_position, net.img_idx, np.array([0, 0, 0]))


caffe_init()

test_vggnet()
