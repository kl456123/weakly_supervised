#!/usr/bin/env python
# encoding: utf-8
import sys
# sys.path.append('..')
from lib.net import Net
from lib.tools import *
from lib.part_object_scores import Part_Object_Scores
import numpy as np


def test_all():
    model_def = 'model/nin_max_big/train_val.prototxt'
    model_weights = 'model/nin_max_big/cifar10_nin_iter_120000.caffemodel'
    num_classes = 10

    net3 = Net(model_def, model_weights, num_classes)

    net3.img_idx = 0
    assert net3.get_label() == net3.get_prediction(), 'prediction is error'
    cls_idx = net3.get_label()
    # print net3._net.blobs['data'].data[net3.img_idx]
    # global config
    net3.display()
    input_shape = (32, 32)

    ###############################
    # ########## first level#########
    ###############################
    layer_name = 'cccp6'
    center = []
    center.append(cls_idx)
    top_blob_name = net3._net.top_names[layer_name][0]
    center += list(
        get_max_xy(net3._net.blobs[top_blob_name].data[net3.img_idx, cls_idx]))

    spatial_position = center[1:]
    start_prior_score = 0

    part_object_scores = Part_Object_Scores(
        layer_name, start_prior_score, spatial_position, net3)
    last_filtered_kernel_idx_weight_pairs = ([cls_idx], [1])
    part_object_scores.init(last_filtered_kernel_idx_weight_pairs)
    print part_object_scores.get_score()
    # print part_object_scores.get_score()

    # start from part_object_scores
    queue = top_down(part_object_scores,
                     net3,
                     threshold_policy='float',
                     min_receptive_field=0,
                     threshold=0)
    # print queue[0].get_raw_spatial_position()
    print (get_scores_sum(queue))
    filtered_queue = threshold_system(
        queue, input_shape=input_shape, reserve_num=200, reserve_ratio=1, display=True)
    all_raw_spatial_position = get_all_raw_spatial_position(filtered_queue)
    # visualization all points
    vis_square(net3._net.blobs['data'].data[net3.img_idx].transpose(
        1, 2, 0)[np.newaxis, ...])
    vis_activated_point(
        net3._net, all_raw_spatial_position, net3.img_idx, 0.5)


def test_demo(image_file):
    import os
    model_def = 'model/nin_ave_big/deploy.prototxt'
    model_weights = 'model/nin_ave_big/cifar10_nin__iter_120000.caffemodel'
    mean_file = 'model/mean.npy'

    model_def = os.path.abspath(model_def)
    model_weights = os.path.abspath(model_weights)
    num_classes = 10

    net3 = Net(model_def, model_weights, num_classes, deploy=True)
    # transform input data
    transformer = caffe.io.Transformer(
        {'data': net3._net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    # transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255)
    mu = np.load(mean_file)
    mu = mu.mean(1).mean(1)

    transformer.set_mean('data', mu)
    net3._net.blobs['data'].reshape(1, 3, 32, 32)
    input_shape = (32, 32)
    im = caffe.io.load_image(image_file)
    transformed_im = transformer.preprocess('data', im)
    output_layer_name = 'pool3'

    # plt.imshow(im)
    # plt.show()
    net3._net.blobs['data'].data[...] = transformed_im
    output = net3._net.forward()
    # print net3._net.blobs['data'].data[0]
    # stop()
    cls_idx = net3._net.blobs[output_layer_name].data[0].argmax()
    print cls_idx
    cls_idx = 0

    layer_name = 'cccp6'
    center = []
    center.append(cls_idx)
    top_blob_name = net3._net.top_names[layer_name][0]
    center += list(
        get_max_xy(net3._net.blobs[top_blob_name].data[net3.img_idx, cls_idx]))

    spatial_position = center[1:]
    start_prior_score = 0

    part_object_scores = Part_Object_Scores(
        layer_name, start_prior_score, spatial_position, net3)
    last_filtered_kernel_idx_weight_pairs = ([cls_idx], [1])
    part_object_scores.init(last_filtered_kernel_idx_weight_pairs)
    print part_object_scores.get_score_map()
    # start from part_object_scores
    queue = top_down(part_object_scores,
                     net3,
                     threshold_policy='float',
                     min_receptive_field=0,
                     threshold=0)
    # print queue[0].get_raw_spatial_position()
    print (get_scores_sum(queue))
    filtered_queue = threshold_system(
        queue, input_shape=input_shape, reserve_num=200, reserve_ratio=1, display=True)
    all_raw_spatial_position = get_all_raw_spatial_position(filtered_queue)
    # visualization all points
    vis_square(net3._net.blobs['data'].data[net3.img_idx].transpose(
        1, 2, 0)[np.newaxis, ...])
    vis_activated_point(
        net3._net, all_raw_spatial_position, net3.img_idx, np.array([0, 0, 0]))


caffe_init()
# test_net3()
# test_part_object_scores()
# test_all()
test_demo('image/dog.jpg')
