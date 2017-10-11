#!/usr/bin/env python
# encoding: utf-8
import sys
# sys.path.append('..')
from libs.net import Net
from libs.tools import *
import numpy as np
from libs.layers import Layers
from libs.points import *
from libs.utils import *


def test_all():
    model_def = 'model/nin_max_bigger/nin_max_bigger_train_val.prototxt'
    model_weights = 'model/nin_max_bigger/cifar10_nin_iter_120000.caffemodel'
    num_classes = 10
    classes_name = []

    net3 = Net(model_def, model_weights, num_classes)

    net3.img_idx = 36
    assert net3.get_label() == net3.get_prediction(), 'prediction is error'
    cls_idx = net3.get_label()
    # global config
    net3.display()
    input_shape = (32, 32)
    layers = Layers(net3)
    kernel_weight = np.zeros((num_classes,))
    kernel_weight[cls_idx] = 1
    start_layer_name = 'cccp6'
    center = []
    center.append(cls_idx)
    top_blob_name = net3._net.top_names[start_layer_name][0]
    center += list(
        get_max_xy(net3._net.blobs[top_blob_name].data[net3.img_idx, cls_idx]))
    start_points = [Point_3D(center, 1)]
    points, scores = layers.backward(
        start_points, start_layer_name, isFilter=True, debug=True)

    shape_2D = (32, 32)
    points = threshold_system(points,
                              scores,
                              input_shape=shape_2D,
                              reserve_num_ratio=0.7,
                              reserve_num=400,
                              reserve_scores_ratio=0.1)
    # img_path = ''
    # visualize(img_path, pos_2D, diag_percent=0.1, image_label='cat')
    # print_point(points[0])

    # visualization all points
    vis_square(net3._net.blobs['data'].data[net3.img_idx].transpose(
        1, 2, 0)[np.newaxis, ...])
    vis_activated_point(
        net3._net, points, net3.img_idx, 0.5)


caffe_init()
test_all()
