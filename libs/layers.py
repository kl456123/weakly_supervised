#!/usr/bin/env python
# encoding: utf-8

from libs.tools import *
from libs.points import *


class Layers(object):
    def __init__(self, net):
        self.net = net

# output point is 2D
    def conv(self, layer_name, points):
        layer_info = self.net.get_layer_info(layer_name)
        K = layer_info['kernel_size']
        P = _get(layer_info['pad'], 0)
        S = _get(laye_info['stride'], 1)
        D = _get(layer_info['dilation'], 1)
        dim = points[0].dim
        weight, bias = self.net.get_param_data(layer_name)

        res = []

        for point in points:
            pos = point.pos
            children_pos = point.get_children_pos(K, S, D)
            bottom_name = self.net._net.bottom_names[layer_name][0]
            shape = self.net._net.blobs[bottom_name].data.shape
            data = self.net.get_block_data(bottom_name, children_pos, P)
            if dim == 2:
                weighted_weight = point.weight * weight
                weighted_weight = weighted_weight.sum(axis=0)
                weighted_bias = point.weight * bias
                weighted_bias = weighted_bias.sum(axis=0)
            elif dim == 3:
                weighted_weight = point.weight * weight[pos[0]]
                weighted_bias = point.weight * bias[pos[0]]

            conv = weighted_weight * data
            c, h, w = conv.shape
            kdim = h * w
            score_map = conv.sum(axis=0) + bias + point.prior
            keep = score_map > 0
            keep_weight = weighted_weight[:, keep]
            keep_pos = children_pos[keep]
            per_prior = (bias + point.prior) / kdim
            for per_pos, per_weight in keep_pos, keep_weight:
                if not check_is_pad(per_pos, shape, P):
                    res.append(Point_2D(per_pos, per_weight, per_prior))

        return res

    def relu(self, layer_name, points):
        layer_info = self.net.get_layer_info(layer_name)
        negative_slope = _get(layer_info['negative_slope'], 0)
        dim = points[0].dim
        bottom_name = self.net.get_bottom_name(layer_name)
        data = self.net._net.blob[bottom_name].data[self.net.img_idx]
        for point in points:
            pos = point.pos
            weight = point.weight
            if dim == 2:
                weight[np.where(data[:, pos[1], pos[2]] < 0)] *= negative_slope
            else:
                weight[np.where(data[pos[0], pos[1], pos[2]] < 0)
                       ] *= negative_slope
        return points

    def pool(self, layer_name, points):
        layer_info = self.net.get_layer_info(layer_name)
        K = layer_info['kernel_size']
        S = _get(layer_info['stride'], 1)
        # output point is 3D
        dim = points[0].dim
        all_points_3D = []
        bottom_name = self.net._net.bottom_names[layer_name][0]
        for point in points:
            # default dilation =1,pad = 0 for pooling layer
            children_pos = point.get_children_pos(K, S, 1)
            # shape = self.net._net.blobs[bottom_name].data.shape
            data = self.net.get_block_data(bottom_name, children_pos, 0)
            c, h, w = data.shape
            mask = data.reshape(c, -1).argmax(axis=1)
            mask = np.unravel_index(mask, (h, w))
            if dim == 2:
                positions_3D = zip(np.arange(c), mask[0], mask[1])
                children_val = data[np.arange(c), mask[0], mask[1]]
                # contribution of each point_3D,so we filter negative value
                scores = children_val * point.weight
                keep = scores > 0
                points_3D = convert_2Dto3D(point, positions_3D)
                # filter
                points_3D = [points_3D[idx] for idx in keep[0]]
                all_points_3D.append(point_3D)
            else:
                raise NotImplementedError
        pooling_blob_shape = self.net._net.blobs[bottom_name]
        all_points_3D = merge_points(all_points_3D, pooling_blob_shape)
        return all_points_3D

    def fc(self, layer_name, points):
        # output point is 2D,just like 1x1 conv
        # now just 2D input point is supported
        assert len(points) == 1, 'just one 2D point'
        point_2D = points[0]
        weight, bias = self.net.get_param_data(layer_name)
        weighted_weight = point_2D.weight * weight
        weighted_weight = weighted_weight.sum(axis=0)
        weighted_bias = point_2D.weight * bias
        weighted_bias = weighted_bias.sum(axis=0)
        prior = point_2D.prior + weighted_bias
        return Point_2D([None, 0, 0], weighted_weight, prior)

    def backward(self, points):
        all_layer_sequence = self.net.all_layer_sequence
        for layer_name in all_layer_sequence:
            layer_info = self.net.get_layer_info(layer_name)
            layer_type = layer_info['type']
            if layer_type == 'fc':
                points = self.fc(layer_name, points)
            elif layer_type == 'conv':
                points = self.conv(layer_name, points)
            elif layer_type == 'pooling':
                points = self.pool(layer_name, points)
            elif layer_type == 'relu':
                points = self.relu(layer_name, points)
        return points


def check_is_pad(pos, shape, P):
    if pos[0] < P or pos[0] > shape[0]:
        return True
    if pos[1] < P or pos[1] > shape[1]:
        return True
    return False
