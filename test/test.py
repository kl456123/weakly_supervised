#!/usr/bin/env python
# encoding: utf-8
from __future__ import division
# import sys
# sys.path.append('/home/jinshan/Documents/papercode/weakly_supervised/')
from libs.net import Net
from libs.tools import *
import numpy as np
from libs.layers import Layers
from libs.points import *
from libs.caffe_helper import get_max_xy

import matplotlib.pyplot as plt
import torch as T


def init_record(num_classes, num_parts):
    res = []
    for i in range(num_classes):
        parts = []
        for j in range(num_parts):
            parts.append([])
        res.append(parts)
    return res


def get_part_score(forward_time=1):
    model_def = 'model/nin_max_bigger/nin_max_bigger_train_val.prototxt'
    model_weights = 'model/nin_max_bigger/cifar10_nin_iter_120000.caffemodel'
    # model_def = 'model/nin_max_big/train_val.prototxt'
    # model_weights = 'model/nin_max_big/cifar10_nin_iter_120000.caffemodel'
    num_classes = 10
    num_parts = 9
    res = init_record(num_classes, num_parts)

    net = Net(model_def, model_weights, num_classes)
    for i in range(forward_time):
        net.forward()
        labels = net._net.blobs['label'].data
    # im_list = np.where(label == class_idx)[0]
        for img_idx, label in enumerate(list(labels)):
            net.img_idx = img_idx

            # assert net.get_label() == net.get_prediction(), 'prediction is error'
            if(not net.get_label() == net.get_prediction()):
                continue
            cls_idx = net.get_label()  # print cls_idx
            # global config
            # net3.display()
            layers = Layers(net)
            kernel_weight = T.zeros((num_classes,)).cuda()
            kernel_weight[cls_idx] = 1
            start_layer_name = 'pool3'
            pos = [np.nan, np.nan, np.nan]
            prior = 0
            start_points = [Point_2D(pos, kernel_weight, prior)]
            score_map = layers.backward(
                start_points,
                start_layer_name,
                isFilter=True,
                debug=False,
                test_part=True,
                log=False)
            score_map = score_map.cpu().numpy().ravel()
            for part_idx, score in enumerate(list(score_map)):
                res[cls_idx][part_idx].append(score)
    return res


def test_negative(neg_layer_name=None):
    model_def = 'model/nin_max_bigger/nin_max_bigger_train_val.prototxt'
    model_weights = 'model/nin_max_bigger/cifar10_nin_non_negative.caffemodel'
    # model_def = 'model/nin_max_big/train_val.prototxt'
    # model_weights = 'model/nin_max_big/cifar10_nin_iter_120000.caffemodel'
    num_classes = 10

    net = Net(model_def, model_weights, num_classes)

    print('normal mode')
    eval_net(net, forward_time=100)

    # filter negative
    # net.filter_negative(neg_layer_name)
    # print('no negative mode')
    # eval_net(net, forward_time=100)
    # net.save()


def eval_net(net, forward_time=1):
    print 'start testing... '
    correct = 0
    _sum = forward_time * 100
    for i in range(forward_time):
        net.forward()
        correct += sum(net.get_all_label() == net.get_all_prediction())
    print('accuracy: %f' % (correct / _sum))


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


def analysis(record):
    for cls_idx, cls_record in enumerate(record):
        cls_record = np.array(cls_record)
        cls_record_sum = cls_record.sum(axis=1)
        print('class index is %d' % (cls_idx))
        print(cls_record_sum)


def plot(record, num_classes=10):
    for cls_idx in range(num_classes):
        cls_record = record[cls_idx]
        cls_record = np.array(cls_record)
        num = cls_record.shape[1]
        num_parts = cls_record.shape[0]
        x = np.arange(num)
        for i in range(num_parts):
            plt.plot(x, cls_record[i])
        plt.show()


caffe_init()
test_negative('cccp6')
# test_all()
# res = get_part_score(100)
# analysis(res)
# plot(record, num_classes)
# test_demo('dirty_data.png')
# test_part(2, (1,1))
