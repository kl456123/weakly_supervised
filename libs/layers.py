#!/usr/bin/env python
# encoding: utf-8

from libs.tools import _get, stop
from libs.points import *
import numpy as np


class Layers(object):
    def __init__(self, net):
        self.net = net


# output point is 2D
    def conv(self,
             layer_name,
             points,
             isFilter=True,
             test_part_pos=None):
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
            score_sum = score_map.sum()
            threshold = score_sum / 9
            if isFilter:
                keep = score_map > threshold
            else:
                keep = score_map > -np.inf
            if test_part_pos is not None:
                if keep[test_part_pos]:
                    keep[...] = False
                    part_pos = test_part_pos
                    keep[part_pos] = True
                else:
                    return []
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
        # res = []
        for point in points:
            pos = point.pos
            if dim == 2:
                if len(data.shape) == 1:
                    point.weight[np.where(data <= 0)] *= negative_slope
                else:
                    point.weight[np.where(data[:, int(pos[1]), int(pos[2])] <= 0)
                                 ] *= negative_slope
            else:
                if data[int(pos[0]), int(pos[1]), int(pos[2])] <= 0:
                    # continue
                    point.weight *= negative_slope
            # res.append(point)
        return points

    def pool_faster(self, layer_name, points, isFilter=True):
        layer_info = self.net.get_layer_info(layer_name)
        K = layer_info['kernel_size']
        S = _get(layer_info.get('stride'), 1)
        P = _get(layer_info.get('pad'), 0)
        dim = points[0].dim
        FC = 0
        if np.isnan(points[0].pos[1]):
            FC = 1
        bottom_name = self.net._net.bottom_names[layer_name][0]
        bottom_data = self.net._net.blobs[bottom_name].data[self.net.img_idx]
        top_name = self.net._net.top_names[layer_name][0]
        top_data = self.net._net.blobs[top_name].data[self.net.img_idx]

        if K == 0:
            K = bottom_data.shape[1]
        top_shape = top_data.shape
        mask = get_pool_mask(bottom_data, K, S, P)
        weight = np.zeros(mask.shape[:-1])
        prior = np.zeros_like(weight)
        res = []
        num = points[0].weight.size
        assert dim == 2, 'dim ==3 is not supported'
        for point in points:
            #         if dim == 2:
                # if FC:
                    # pass
                # else:
                    # pass
            # else:
                # pass
            pos = point.pos
            per_prior = point.prior / num
            for i in range(num):

                if FC:

                    x, y, z = np.unravel_index(i, top_shape)
                    data_val = top_data[x, y, z]
                    if isFilter and point.weight[i] * data_val <= 0:
                        continue
                    weight[x, y, z] += point.weight[i]
                    prior[x, y, z] += per_prior
                else:
                    data_val = top_data[i, int(pos[1]), int(pos[2])]
                    if isFilter and point.weight[i] * data_val <= 0:
                        continue
                    weight[i, int(pos[1]), int(pos[2])] += point.weight[i]
                    prior[i, int(pos[1]), int(pos[2])] += per_prior
                # res.append(
                # Point_3D(children_pos[i], point.weight[i], per_prior))
        keep = list(np.array(np.nonzero(weight)).T)
        for c, h, w in keep:
            res.append(
                Point_3D(mask[c, h, w], weight[c, h, w], prior[c, h, w]))

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
        import time
        start_pool = time.time()
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
        convert_time = time.time()
        print('convert_time :{:.5f}'.format(convert_time - start_pool))

        pooling_blob_shape = self.net._net.blobs[bottom_name].data.shape[1:]
        for point in top_points_3D:
            if isFilter and point.weight <= 0:
                # filter
                continue
            assert point.dim == 3, 'all points must be converted to 3D first'
            # 3D
            # just one child
            pos = point.pos
            c = int(pos[0])
            h = int(pos[1])
            w = int(pos[2])
            # if check_is_pad(pos, pooling_blob_shape):
            # # just used for case that pad!=0
            # continue
            children_pos = point.get_children_pos(K, S, D=1, P=P)
            data = self.net.get_block_data(
                bottom_name, children_pos, 0)
            idx = np.argmax(data[c], axis=0)

            # x, y = np.unravel_index(idx, (K, K))
            point.pos = children_pos[idx]
            # point_3D = Point_3D(
            # children_pos[idx], point.weight, point.prior)
            bottom_points_3D.append(point)
        filter_time = time.time()
        print('filter time:{:.5f}'.format(filter_time - convert_time))

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
                 debug=False,
                 test_flag=0,
                 test_part_pos=None):
        import time
        max_num = 50000
        all_layer_sequence = self.net.all_layer_sequence
        start_flag = 0
        scores = None
        backward_start_time = time.time()
        raw_size = self.net._net.blobs['data'].data.shape[2:]
        # test_flag = 1
        for layer_name in reversed(all_layer_sequence):
            start_num = len(points)

            if start_layer_name == layer_name:
                start_flag = 1
            if not start_flag:
                continue
            layer_info = self.net.get_layer_info(layer_name)
            bottom_name = self.net._net.bottom_names[layer_name][0]
            feat_size = self.net._net.blobs[bottom_name].data.shape[2:]
            layer_type = layer_info['type']
            # if layer_name == 'conv5_3':
            start_time = time.time()
            if layer_type == 'fc':
                points = self.fc(layer_name, points)
            elif layer_type == 'conv':
                if test_flag:
                    test_flag -= 1
                else:
                    test_part_pos = None
                points = self.conv(layer_name,
                                          points,
                                          isFilter,
                                          test_part_pos)
            elif layer_type == 'pooling':
                # self.visualization_vggnet(layer_name, points)
                points = self.pool_faster(layer_name, points, isFilter)
            elif layer_type == 'relu':
                points = self.relu(layer_name, points, isFilter)
            dura_time = time.time() - start_time
            if len(points) == 0:
                return points, None
            scores = get_points_scores(self.net, points, layer_name)
            if debug:
                print('scores: {:.2f}'.
                      format(scores.sum(axis=0)))
            # if scores <= 0:
                # stop()
                # return [], scores
            points = helper_cluster_points(points, feat_size, threshold=0)
            if log:
                num = len(points)
                print('INFO: {:s},\tnumber: {:d}\tdura_time: {:.4f},\ttime_per_point: {:.4f}'
                      .format(layer_name, num, dura_time, dura_time / start_num))
                # print 'receptive_field: {:.2f}'.format(self.net.get_rece)
                # if num > max_num:
                # print 'stop early'
                # # stop()
                # break
        #     points, scores = post_filter_points(
            # points,
            # scores,
            # max_num=3000)
        # if scores is None:
        # points = merge_points_2D_better(points, feat_size)
        backward_dura_time = time.time() - backward_start_time
        print('backward duration: {:.3f}'.format(backward_dura_time))
        scores = get_points_scores(self.net, points, layer_name)
        mapping_points(points, feat_size, raw_size)
        return points, scores

    def visualization_vggnet(self, layer_name, points):
        import matplotlib.pyplot as plt
        assert len(points) == 1, 'top layer must be fc'
        point = points[0]
        top_name = self.net._net.top_names[layer_name][0]
        top_data = self.net._net.blobs[top_name].data[self.net.img_idx]
        weight = point.weight.reshape(top_data.shape)
        CAM = (top_data * weight).sum(axis=0)
        plt.imshow(CAM)
        plt.show()


def get_points_scores(net, points, layer_name):
    bottom_name = net._net.bottom_names[layer_name][0]
    data = net._net.blobs[bottom_name].data[net.img_idx]
    dim = points[0].dim
    res = []
    FC = False
    if np.isnan(points[0].pos[1]):
        FC = True
    for point in points:
        pos = point.pos
        if dim == 2:
            if FC:
                score = point.weight * data.ravel()
            else:
                score = point.weight * data[:, int(pos[1]), int(pos[2])]
            score = score.sum(axis=0) + point.prior
        else:
            score = point.weight * \
                data[int(pos[0]), int(pos[1]), int(pos[2])] +\
                point.prior
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


def get_pool_mask(data, K, S, P=0):
    c, h, w = data.shape

    w_next = int((w - K + 2 * P) / S + 1)
    res = np.zeros((c, w_next, w_next, 3))
    for i in range(w_next):
        for j in range(w_next):
            x = (S * i, S * i + K)
            y = (S * j, S * j + K)
            if P == 0:
                mask = np.unravel_index(
                    np.argmax(data[:, x[0]:x[1], y[0]:y[1]].reshape((c, -1)), axis=1), (K, K))
                res[:, i, j] = np.array(
                    [np.arange(c), mask[0] + x[0], mask[1] + y[0]]).T
            else:
                raise NotImplementedError
    return res


def arr2hash(arr):
    return ','.join(str(i) for i in arr)


def hash2arr(hash_str):
    arr = hash_str.split(',')
    if arr[0] == 'None':
        c = None
    else:
        c = float(arr[0])
    x = float(arr[1])
    y = float(arr[2])
    return (c, x, y)


def get_children_pos(pos, K, S=1, D=1, P=0, filter_pad=False):
    K = (K - 1) * (D - 1) + K
    H = np.arange(S * pos[1], S * pos[1] + K, D) - P
    W = np.arange(S * pos[2], S * pos[2] + K, D) - P
    if filter_pad:
        H = H[H > 0]
        W = W[W > 0]
    res = []
    for h in H:
        for w in W:
            res.append([pos[0], h, w])
    return np.array(res)



