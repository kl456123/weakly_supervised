#!/usr/bin/env python
# encoding: utf-8
import sys
sys.path.append('/home/jinshan/Documents/papercode/weakly_supervised/')
from libs.net import Net
from libs.tools import *
import numpy as np
from libs.layers import Layers
from libs.points import *
from libs.caffe_helper import get_max_xy
import torch as T


def test_part(class_idx, part_pos):
    model_def = 'model/nin_max_bigger/nin_max_bigger_train_val.prototxt'
    model_weights = 'model/nin_max_bigger/cifar10_nin_iter_120000.caffemodel'
    # model_def = 'model/nin_max_big/train_val.prototxt'
    # model_weights = 'model/nin_max_big/cifar10_nin_iter_120000.caffemodel'
    num_classes = 10

    net3 = Net(model_def, model_weights, num_classes)
    net3.forward(7)
    label = net3._net.blobs['label'].data
    im_list = np.where(label == class_idx)[0]
    for img_idx in im_list:
        net3.img_idx = img_idx

        assert net3.get_label() == net3.get_prediction(), 'prediction is error'
        cls_idx = net3.get_label()  # print cls_idx
        # global config
        # net3.display()
        layers = Layers(net3)
        kernel_weight = T.zeros((num_classes,)).cuda()
        kernel_weight[cls_idx] = 1
        start_layer_name = 'pool3'
        pos = [np.nan, np.nan, np.nan]
        prior = 0
        start_points = [Point_2D(pos, kernel_weight, prior)]
        points, scores = layers.backward(
            start_points,
            start_layer_name,
            isFilter=True,
            debug=True,
            test_flag=1,
            test_part_pos=part_pos)
        if scores is None:
            continue

        shape_2D = (32, 32)
        points = threshold_system(points,
                                  scores,
                                  input_shape=shape_2D,
                                  reserve_num_ratio=1,
                                  reserve_num=30,
                                  reserve_scores_ratio=0.9)
        # img_path = ''
        # visualize(img_path, pos_2D, diag_percent=0.1, image_label='cat')
        # print_point(points[0])

        # visualization all points
        vis_square(net3._net.blobs['data'].data[net3.img_idx].transpose(
            1, 2, 0)[np.newaxis, ...])
        # stop()
        vis_activated_point(
            net3._net, points, net3.img_idx, 0, save=True)


def test_all():
    model_def = 'model/nin_max_bigger/nin_max_bigger_train_val.prototxt'
    model_weights = 'model/nin_max_bigger/cifar10_nin_iter_120000.caffemodel'
    # model_def = 'model/nin_max_big/train_val.prototxt'
    # model_weights = 'model/nin_max_big/cifar10_nin_iter_120000.caffemodel'
    num_classes = 10

    net3 = Net(model_def, model_weights, num_classes)
    net3.forward(7)
    # np.where(label==cls)
    class_idx = 2
    label = net3._net.blobs['label'].data
    im_list = np.where(label == class_idx)[0]
    net3.img_idx = im_list[-3]
    assert net3.get_label() == net3.get_prediction(), 'prediction is error'
    cls_idx = net3.get_label()
    print(cls_idx)
    # global config
    # net3.display()
    layers = Layers(net3)
    kernel_weight = T.zeros((num_classes,)).cuda()
    kernel_weight[cls_idx] = 1
    start_layer_name = 'pool3'
    pos = [np.nan, np.nan, np.nan]
    prior = 0
    start_points = [Point_2D(pos, kernel_weight, prior)]
    points, scores = layers.backward(
        start_points,
        start_layer_name, isFilter=True, debug=True)
    shape_2D = (32, 32)
    points = threshold_system(points,
                              scores,
                              input_shape=shape_2D,
                              reserve_num_ratio=1,
                              reserve_num=30,
                              reserve_scores_ratio=0.9)

    # visualization all points
    vis_square(net3._net.blobs['data'].data[net3.img_idx].transpose(
        1, 2, 0)[np.newaxis, ...])
    # stop()
    vis_activated_point(
        net3._net, points, net3.img_idx, 0, save=True)


def test_demo(image_file):
    model_def = 'model/nin_max_bigger/deploy.prototxt'
    model_weights = 'model/nin_max_bigger/cifar10_nin_iter_120000.caffemodel'
    mean_file = 'model/mean.npy'

    num_classes = 10
    # classes_name = []
    net3 = Net(model_def, model_weights, num_classes, num_preforward=0)
    input_shape = net3._net.blobs['data'].data.shape

    transformed_im = preprocess(image_file,
                                input_shape=input_shape,
                                mean_file=mean_file,
                                scale='255',
                                format='RGB')
    net3._net.forward(data=np.asarray([transformed_im]))
    # prob = output['pool3']
    # idx = np.argmax(prob[0])
    prior = 0
    net3.img_idx = 0
    cls_idx = net3.get_prediction()
    print(cls_idx)
    # global config
    net3.display()
    input_shape = (32, 32)
    layers = Layers(net3)
    kernel_weight = T.zeros((num_classes,)).cuda()
    kernel_weight[cls_idx] = 1
    start_layer_name = 'pool3'
    pos = [np.nan, np.nan, np.nan]
    start_points = [Point_2D(pos, kernel_weight, prior)]
    # center = []
    # center.append(cls_idx)
    # top_blob_name = net3._net.top_names[start_layer_name][0]
    # center += list(
    # get_max_xy(net3._net.blobs[top_blob_name].data[net3.img_idx, cls_idx]))
    # start_points = [Point_3D(center, 1)]
    points, scores = layers.backward(
        start_points, start_layer_name, isFilter=True, debug=True)

    shape_2D = (32, 32)
    points = threshold_system(points,
                              scores,
                              input_shape=shape_2D,
                              reserve_num_ratio=1,
                              reserve_num=30,
                              reserve_scores_ratio=1)
    # img_path = ''
    # visualize(img_path, pos_2D, diag_percent=0.1, image_label='cat')
    # print_point(points[0])

    # visualization all points
    vis_square(net3._net.blobs['data'].data[net3.img_idx].transpose(
        1, 2, 0)[np.newaxis, ...])
    # stop()
    vis_activated_point(
        net3._net, points, net3.img_idx, 0.5)


caffe_init()
test_all()
# test_demo('dirty_data.png')
# test_part(2, (1,1))
