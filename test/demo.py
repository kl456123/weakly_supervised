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
model_def = 'tiny_yolo_voc_deploy.prototxt'
model_weights = 'tiny_yolo_voc.caffemodel'


def test_darknet():
    num_classes = 21
    net3 = Net(model_def, model_weights, num_classes, deploy=True)
    image_file = 'dog2.jpg'
    input_shape = net3._net.blobs['data'].data.shape
    transformered_im = preprocess(image_file, input_shape)
    output = net3._net.forward(data=np.asarray([transformered_im]))
    # all_start_spatial_position = [(4, 7), (7, 7)]
    # all_anchor_idx = [1, 3]
    # all_class_idx = [14, 12]
    all_start_spatial_position = [(7, 7)]
    all_anchor_idx = [3]
    all_class_idx = [11]

    # print net3.get_receptive_field('conv4')
    ############################
    # ###### test the first #####
    ############################
    start_layer_name = 'conv9'
    idx = 0
    kernel_idx = all_anchor_idx[idx] * 25 + 5 + all_class_idx[idx]
    start_spatial_position = all_start_spatial_position[idx]
    start_kernel_idx_weight_pairs = ([kernel_idx], [1])
    start_part_object_scores = Part_Object_Scores(
        start_layer_name, start_spatial_position, net3)
    start_part_object_scores.init(start_kernel_idx_weight_pairs)
    print start_part_object_scores.get_score_map()
    # stop()
    queue, bias_sum = top_down(start_part_object_scores,
                               net3,
                               threshold_policy='float',
                               min_receptive_field=30,
                               threshold=None)
    # sum of all scores should be equal to start score
    print (get_scores_sum(queue) + bias_sum)
    # TODO we should exchange the order of two process
    # and we should note that for different task we have concentrate on different points
    # import pdb
    # pdb.set_trace()
    filtered_queue = threshold_system(
        queue, input_shape=input_shape, reserve_num=300, reserve_ratio=0.1, display=True)
    # filtered_queue = filter_repeat_part_object_scores(queue, input_shape[-2:])
    # print 'filtered_queue: {:d}'.format(len(filtered_queue))

    # filtered_queue = filter_ratio(filtered_queue, 1, reserve_num=30)

    all_raw_spatial_position = get_all_raw_spatial_position(filtered_queue)
    # filtered_raw_spatial_position = filter_repeat(
    # all_raw_spatial_position, input_shape[-2:])
    # print 'remain activated points: {:d}'.format(len(all_raw_spatial_position))
    # visualization all points
    vis_square(net3._net.blobs['data'].data[net3.img_idx].transpose(
        1, 2, 0)[np.newaxis, ...])
    vis_activated_point(
        net3._net, all_raw_spatial_position, net3.img_idx, [1, 0, 0])


def threshold_system(queue, **kwargs):
    # read args
    input_shape = kwargs['input_shape']
    reserve_num = kwargs['reserve_num']
    reserve_ratio = kwargs['reserve_ratio']
    display = kwargs['display']

    print 'input number: {:d}'.format(len(queue))
    filtered_queue = filter_repeat_part_object_scores(queue, input_shape[-2:])
    print 'remain number after filtering repeat points: {:d}'.format(len(filtered_queue))

    filtered_queue = filter_ratio(filtered_queue, reserve_ratio, reserve_num)
    print 'remain number after reserving the top scores: {:d}'.format(len(filtered_queue))
    return filtered_queue


def top_down_debug(top_part_object_scores,
                   net,
                   min_receptive_field=10,
                   threshold=0,
                   threshold_policy='float',
                   display=True):
    queue = []
    res = []
    bias_sum = 0
    queue.append(top_part_object_scores)
    current_layer_name = ''
    while len(queue) != 0:
        part_object_scores = queue[0]
        if display and part_object_scores._layer_name != current_layer_name:
            if current_layer_name != '':
                print 'layer_name: {:s},\treceptive_field:{:d},\t number: {:d}'\
                    .format(current_layer_name, part_object_scores.receptive_field, len(queue))
            current_layer_name = part_object_scores._layer_name

        # if part_object_scores.receptive_field < min_receptive_field:
            # break
        # if part_object_scores.get_children_layer_name() is None:
            # break
        # if part_object_scores._layer_name == 'pool6':
            # stop()
        del queue[0]
        score_map = part_object_scores.get_score_map()
        keep = threshold_filter(score_map, threshold, threshold_policy)
        if len(keep[0]) == 0 \
                or part_object_scores.get_children_receptive_field() < min_receptive_field\
                or part_object_scores.get_children_layer_name() is None:
            res.append(part_object_scores)
            continue
        # stop()
        bias_sum += score_map[1]
        all_next_part_object_scores = handler_all_children(
            part_object_scores.get_filtered_kernel_idx_weight_pairs(),
            part_object_scores.get_child_spatial_position(),
            part_object_scores.get_children_layer_name(),
            net)
#         queue += filter_part_object_scores(
        # all_next_part_object_scores, keep[0])
        queue.append(all_next_part_object_scores[0])
    return res, bias_sum


caffe_init()
test_darknet()
