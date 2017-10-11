#!/usr/bin/env python
# encoding: utf-8

from libs.tools import _get, stop
from libs.points import *
import numpy as np


class Layers(object):
    def __init__(self, net):
        self.net = net

# output point is 2D
    def conv(self, layer_name, points, isFilter=True):
        layer_info = self.net.get_layer_info(layer_name)
        K = layer_info['kernel_size']
        P = _get(layer_info['pad'], 0)
        S = _get(layer_info['stride'], 1)
        D = _get(layer_info['dilation'], 1)
        dim = points[0].dim
        weight, bias = self.net.get_param_data(layer_name)

        res = []

        for point in points:
            pos = point.pos
            children_pos = point.get_children_pos(K, S, D, P)
            bottom_name = self.net._net.bottom_names[layer_name][0]
            shape = self.net._net.blobs[bottom_name].data.shape
            data = self.net.get_block_data(bottom_name, children_pos, P)
            data = data.reshape((-1, K, K))
            if dim == 2:
                weighted_weight = point.weight.reshape((-1, 1, 1, 1)) * weight
                weighted_weight = weighted_weight.sum(axis=0)
                weighted_bias = point.weight * bias
                weighted_bias = weighted_bias.sum(axis=0)
            elif dim == 3:
                weighted_weight = point.weight * weight[int(pos[0])]
                weighted_bias = point.weight * bias[int(pos[0])]
                weighted_bias = weighted_bias.sum(axis=0)

            conv = weighted_weight * data
            c, h, w = conv.shape
            kdim = h * w
            per_prior = (weighted_bias + point.prior) / kdim
            score_map = conv.sum(axis=0) + per_prior
            if isFilter:
                keep = score_map > 0
            else:
                keep = score_map > -np.inf
            score_map = score_map[keep].ravel()
            keep_weight = weighted_weight[:, keep].transpose((1, 0))
            keep_pos = children_pos[keep.ravel()]
            num_keep = keep_pos.shape[0]
            for i in range(num_keep):
                per_pos = keep_pos[i]
                per_weight = keep_weight[i]
                if not check_is_pad(per_pos, shape[2:]):
                    res.append(Point_2D(per_pos,
                                        per_weight,
                                        per_prior, score_map[i]))

        return res

    def relu(self, layer_name, points, isFilter=True):
        layer_info = self.net.get_layer_info(layer_name)
        negative_slope = _get(layer_info['negative_slope'], 0)
        dim = points[0].dim
        bottom_name = self.net.get_bottom_name(layer_name)
        data = self.net.get_block_data(bottom_name, None, 0)
        # data = self.net._net.blobs[bottom_name].data[self.net.img_idx]
        res = []
        for point in points:
            pos = point.pos
            if dim == 2:
                if len(data.shape) == 1:
                    point.weight[np.where(data <= 0)] *= negative_slope
                else:
                    point.weight[np.where(data[:, int(pos[1]), int(pos[2])] <= 0)
                                 ] *= negative_slope
            else:
                if isFilter and data[int(pos[0]), int(pos[1]), int(pos[2])] <= 0:
                    continue
                    # point.weight *= negative_slope
            res.append(point)
        return res

    def pool(self, layer_name, points, isFilter=True):
        layer_info = self.net.get_layer_info(layer_name)
        K = layer_info['kernel_size']
        S = _get(layer_info.get('stride'), 1)
        P = _get(layer_info.get('pad'), 0)
        # output point is 3D
        skip_merge = False
        if len(points) == 1:
            skip_merge = True
        bottom_points_3D = []
        bottom_name = self.net._net.bottom_names[layer_name][0]
        # all point_2D must be converted to 3D before passing through pooling layer
        top_points_3D = []
        FC = 0
        if np.isnan(points[0].pos[1]):
            FC = 1
        if points[0].dim == 2:
            for point in points:
                if FC:
                    top_name = self.net._net.top_names[layer_name][0]
                    data = self.net._net.blobs[top_name].data[self.net.img_idx]
                    c, h, w = data.shape
                    num = c * h * w
                    positions_3D = np.array(np.unravel_index(
                        np.arange(num), data.shape)).T
                else:
                    c = len(point.weight)
                    positions_3D = [(i, int(point.pos[1]), int(
                        point.pos[2])) for i in range(c)]
                points_3D = convert_2Dto3D(point, positions_3D)
                top_points_3D += points_3D

        pooling_blob_shape = self.net._net.blobs[bottom_name].data.shape[1:]
        for point in top_points_3D:
            assert point.dim == 3, 'all points must be converted to 3D first'
            # 3D
            # just one child
            pos = point.pos
            c = int(pos[0])
            h = int(pos[1])
            w = int(pos[2])
            if check_is_pad(pos, pooling_blob_shape):
                # just used for case that pad!=0
                continue
            children_pos = point.get_children_pos(K, S, D=1, P=P)
            data = self.net.get_block_data(
                bottom_name, children_pos, 0)
            idx = np.argmax(data[c], axis=0)
            if isFilter and data[c, idx] * point.weight <= 0:
                # filter
                continue
            x, y = np.unravel_index(idx, (K, K))
            point_3D = Point_3D(
                (c, x + h * S, y + w * S), point.weight, point.prior)
            bottom_points_3D.append(point_3D)

        if not skip_merge:
            bottom_points_3D = merge_points(
                bottom_points_3D, pooling_blob_shape)
        return bottom_points_3D

    def fc(self, layer_name, points):
        # output point is 2D,just like 1x1 conv
        # now just 2D input point is supported
        assert len(points) == 1, 'just one 2D point'
        point_2D = points[0]
        weight, bias = self.net.get_param_data(layer_name)
        weighted_weight = point_2D.weight[..., np.newaxis] * weight
        weighted_weight = weighted_weight.sum(axis=0)
        weighted_bias = point_2D.weight * bias
        weighted_bias = weighted_bias.sum(axis=0)
        prior = point_2D.prior + weighted_bias
        return [Point_2D([np.nan, np.nan, np.nan], weighted_weight, prior)]

    def backward(self,
                 points,
                 start_layer_name,
                 log=True,
                 isFilter=True,
                 debug=False):
        all_layer_sequence = self.net.all_layer_sequence
        start_flag = 0
        scores = None
        for layer_name in reversed(all_layer_sequence):
            if start_layer_name == layer_name:
                start_flag = 1
            if not start_flag:
                continue
            layer_info = self.net.get_layer_info(layer_name)
            layer_type = layer_info['type']
            if layer_type == 'fc':
                points = self.fc(layer_name, points)
            elif layer_type == 'conv':
                points = self.conv(layer_name, points, isFilter)
            elif layer_type == 'pooling':
                points = self.pool(layer_name, points, isFilter)
            elif layer_type == 'relu':
                points = self.relu(layer_name, points, isFilter)
            if log:
                print 'layer_name: {:s},\tnumber: {:d}'\
                    .format(layer_name, len(points))

            if debug:
                scores = self.get_points_scores(points, layer_name)
                print 'scores: {:.2f}'.\
                    format(scores.sum(axis=0))
        if scores is None:
            scores = self.get_points_scores(points, layer_name)
        return points, scores

    def get_points_scores(self, points, layer_name):
        bottom_name = self.net._net.bottom_names[layer_name][0]
        data = self.net._net.blobs[bottom_name].data[self.net.img_idx]
        dim = points[0].dim
        res = []
        for point in points:
            pos = point.pos
            if dim == 2:
                stop()
                if len(data.shape) == 1:
                    score = point.weight * data[0]
                else:
                    score = point.weight * data[:, int(pos[1]), int(pos[2])]
                score = score.sum(axis=0) + point.prior
                # if score < 0:
                # stop()
            else:
                score = point.weight * \
                    data[int(pos[0]), int(pos[1]), int(pos[2])]
            res.append(score)
        res = np.array(res)
        return res


def check_is_pad(pos, shape):
    # pos = (c,h,w)
    if pos[1] < 0 or pos[1] >= shape[0]:
        return True
    if pos[2] < 0 or pos[2] >= shape[1]:
        return True
    return False
