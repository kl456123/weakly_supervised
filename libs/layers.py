#!/usr/bin/env python
# encoding: utf-8

from libs.points import *
from libs.common_helper import *
from libs.pytorch_helper import tensor, clear

import numpy as np
import torch as T


class Layers(object):
    def __init__(self, net):
        self.net = net

    def get_part_score(self,
                       layer_name,
                       points):
        assert len(points) == 1, 'the len of point is 1'
        point = points[0]

        layer_info = self.net.get_layer_info(layer_name)
        K = layer_info['kernel_size']
        P = get(layer_info['pad'], 0)
        S = get(layer_info['stride'], 1)
        D = get(layer_info['dilation'], 1)
        weight, bias = self.net.get_param_data(layer_name)
        weight = tensor(weight)
        bias = tensor(bias)
        # n = weight.shape[0]
        c = weight.size()[1]
        # weight_shape = weight.shape[1:]
        bottom_name = self.net.get_bottom_name(layer_name)
        pos = point.pos

        weighted_weight = point.weight * weight[int(pos[0])]

        children_pos = get_children_pos(pos, K, S, D, P)
        data = tensor(self.net.get_block_data(
            bottom_name, children_pos, P))
        conv = data.view((c,K,K))
        conv.mul_(weighted_weight)
        score_map = conv.sum(dim=0)
        return score_map


    def conv_faster_3D(self,
                       layer_name,
                       points,
                       isFilter=True,
                       test_part_pos=None):
        layer_info = self.net.get_layer_info(layer_name)
        K = layer_info['kernel_size']
        P = get(layer_info['pad'], 0)
        S = get(layer_info['stride'], 1)
        D = get(layer_info['dilation'], 1)
        weight, bias = self.net.get_param_data(layer_name)
        weight = tensor(weight)
        bias = tensor(bias)
        # n = weight.shape[0]
        c = weight.shape[1]
        # weight_shape = weight.shape[1:]
        res = []
        all_weighted_weight = []
        all_prior = []
        all_data = []
        all_pos = []
        kdim = K * K
        bottom_name = self.net.get_bottom_name(layer_name)
        bottom_shape = self.net.get_blob_shape_by_blob_name(bottom_name)
        for point in points:
            pos = point.pos
            all_prior.append(
                (point.prior + bias[int(pos[0])] * point.weight) / kdim)
            weighted_weight = point.weight * weight[int(pos[0])]
            all_weighted_weight.append(weighted_weight)
            children_pos = get_children_pos(pos, K, S, D, P)
            all_pos.append(children_pos)
            data = tensor(self.net.get_block_data(
                bottom_name, children_pos, P))
            all_data.append(data)
        all_pos = np.concatenate(all_pos)
        all_conv = T.stack(all_data)
        all_weighted_weight = T.stack(all_weighted_weight)
        all_prior = tensor(all_prior)
        all_conv = all_conv.view((-1, c, K, K))
        all_conv.mul_(all_weighted_weight)
        all_score_map = all_conv.sum(
            dim=1) + all_prior[:, None, None]
        all_score_sum = all_score_map.view((-1, kdim)).sum(dim=1)
        threshold = all_score_sum / 9
        if isFilter:
            all_keep = all_score_map > threshold.view((-1, 1, 1))
        else:
            all_keep = all_score_map > float('-inf')

        if test_part_pos is not None:
            if all_keep[0, test_part_pos[0], test_part_pos[1]]:
                all_keep[0, :, :] = False
                part_pos = test_part_pos
                all_keep[0, part_pos[0], part_pos[1]] = True
            else:
                return []
        all_score_map = all_score_map.view(-1)
        all_weighted_weight = all_weighted_weight.permute(
            0, 2, 3, 1).contiguous().view((-1, c))
        all_keep = all_keep.view(-1)
        num = all_keep.size()[0]
        for i in range(num):
            if all_keep[i]:
                per_pos = all_pos[i]
                per_weight = all_weighted_weight[i]
                per_prior = all_prior[i // kdim]
                if not check_is_pad(per_pos, bottom_shape[-2:]):
                    res.append(Point_2D(per_pos,
                                        per_weight,
                                        per_prior,
                                        all_score_map[i]))
        return res

    def conv_faster(self,
                    layer_name,
                    points,
                    isFilter=True,
                    test_part_pos=None):
        layer_info = self.net.get_layer_info(layer_name)
        K = layer_info['kernel_size']
        P = get(layer_info['pad'], 0)
        S = get(layer_info['stride'], 1)
        D = get(layer_info['dilation'], 1)
        dim = points[0].dim
        weight, bias = self.net.get_param_data(layer_name)
        n = weight.shape[0]
        c = weight.shape[1]
        weight_shape = weight.shape[1:]
        res = []
        all_weight = []
        all_prior = []
        kdim = K * K
        bottom_name = self.net.get_bottom_name(layer_name)
        bottom_shape = self.net.get_blob_shape_by_blob_name(bottom_name)
        for point in points:
            pos = point.pos
            all_prior.append(point.prior)
            if dim == 2:
                all_weight.append(point.weight)
            elif dim == 3:
                temp_weight = T.zeros(n).cuda()
                temp_weight[int(pos[0])] = point.weight
                all_weight.append(temp_weight)

        weight = tensor(weight)
        bias = tensor(bias)
        all_weight = tensor(all_weight)
        all_prior = tensor(all_prior)
        weighted_weight = all_weight.mm(weight.view((weight.shape[0], -1)))
        weighted_weight = weighted_weight.view(
            (weighted_weight.shape[0],) + weight_shape)
        weighted_bias = all_weight.mv(bias)
        all_prior = (weighted_bias + all_prior) / kdim

        ##############################################
        # merge points according to their pos
        ##############################################
        all_children_weight_pos_pairs = {}
        all_children_prior_pos_pairs = {}
        for idx, point in enumerate(points):
            pos = point.pos
            w = weighted_weight[idx]
            p = all_prior[idx]
            hash_pos = arr2hash(pos)
            if hash_pos not in all_children_weight_pos_pairs:
                all_children_weight_pos_pairs[hash_pos] = []
                all_children_prior_pos_pairs[hash_pos] = []
            all_children_weight_pos_pairs[hash_pos].append(w)
            all_children_prior_pos_pairs[hash_pos].append(p)

        for hash_pos in all_children_weight_pos_pairs:
            pos = hash2arr(hash_pos)
            children_pos = get_children_pos(pos, K, S, D, P)
            children_weight = T.stack(all_children_weight_pos_pairs[hash_pos])
            data = tensor(self.net.get_block_data(
                bottom_name, children_pos, P))
            data = data.view((-1, K, K))
            children_conv = data[None, :, :, :] * children_weight
            # children_conv = children_weight.view((-1, K, K)).mm(
            # data.view((data.shape[0], -1)))
            # children_conv.view_((-1, K, K))
            all_per_prior = tensor(all_children_prior_pos_pairs[hash_pos])
            all_score_map = children_conv.sum(
                dim=1) + all_per_prior[:, None, None]
            tmp = all_score_map.view((-1, kdim))
            # sum all of positive number
            # tmp[tmp < 0] = 0
            all_score_sum = tmp.sum(dim=1)
            threshold = all_score_sum / 9
            if isFilter:
                all_keep = all_score_map > threshold.view((-1, 1, 1))
            else:
                all_keep = all_score_map > float('-inf')

            if test_part_pos is not None:
                if all_keep[0, test_part_pos[0], test_part_pos[1]]:
                    all_keep[0, :, :] = False
                    part_pos = test_part_pos
                    all_keep[0, part_pos[0], part_pos[1]] = True
                else:
                    return []

            all_score_map = all_score_map.view(-1)
            all_keep_weight = children_weight.permute(
                0, 2, 3, 1).contiguous().view((-1, c))
            all_keep = all_keep.view(-1)
            num = all_keep.shape[0]
            for i in range(num):
                if all_keep[i]:
                    per_pos = children_pos[i % kdim]
                    per_weight = all_keep_weight[i]
                    per_prior = all_per_prior[i // kdim]
                    if not check_is_pad(per_pos, bottom_shape[-2:]):
                        res.append(Point_2D(per_pos,
                                            per_weight,
                                            per_prior,
                                            all_score_map[i]))
        return res


# output point is 2D
    def conv(self,
             layer_name,
             points,
             isFilter=True,
             test_part_pos=None):
        layer_info = self.net.get_layer_info(layer_name)
        K = layer_info['kernel_size']
        P = get(layer_info['pad'], 0)
        S = get(layer_info['stride'], 1)
        D = get(layer_info['dilation'], 1)
        dim = points[0].dim
        weight, bias = self.net.get_param_data(layer_name)

        res = []

        for point in points:
            pos = point.pos
            children_pos = point.get_children_pos(K, S, D, P)

            bottom_name = self.net.get_bottom_name(layer_name)
            shape = self.net.get_blob_shape_by_blob_name(bottom_name)
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
        negative_slope = get(layer_info['negative_slope'], 0)
        dim = points[0].dim
        bottom_name = self.net.get_bottom_name(layer_name)
        data = tensor(self.net.get_block_data(bottom_name, None, 0))
        for point in points:
            pos = point.pos
            if dim == 2:
                if len(data.size()) == 1:
                    point.weight[data <= 0] *= negative_slope
                else:
                    point.weight[data[:, int(pos[1]), int(pos[2])] <= 0
                                 ] *= negative_slope
            else:
                if data[int(pos[0]), int(pos[1]), int(pos[2])] <= 0:
                    point.weight *= negative_slope
        return points

    def pool_faster(self, layer_name, points, isFilter=True):
        layer_info = self.net.get_layer_info(layer_name)
        K = layer_info['kernel_size']
        S = get(layer_info.get('stride'), 1)
        P = get(layer_info.get('pad'), 0)
        dim = points[0].dim
        FC = 0
        if np.isnan(points[0].pos[1]):
            FC = 1

        bottom_name = self.net.get_bottom_name(layer_name)
        bottom_data = self.net.get_blobs_data_by_name(bottom_name)
        top_name = self.net.get_top_name(layer_name)
        top_data = self.net.get_blobs_data_by_name(top_name)

        if K == 0:
            K = bottom_data.shape[1]
        top_shape = top_data.shape
        mask = tensor(get_pool_mask(bottom_data, K, S, P))
        weight = T.zeros(mask.size()[:-1]).cuda()
        prior = T.zeros(weight.size()).cuda()
        res = []
        num = points[0].weight.size()[0]
        assert dim == 2, 'dim ==3 is not supported'
        top_data = tensor(top_data)
        for point in points:
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
        keep = list(T.nonzero(weight))
        for c, h, w in keep:
            res.append(
                Point_3D(mask[c, h, w], weight[c, h, w], prior[c, h, w]))

        return res

    def fc(self, layer_name, points):
        # output point is 2D,just like 1x1 conv
        # now just 2D input point is supported
        assert len(points) == 1, 'just one 2D point'
        point_2D = points[0]
        weight, bias = self.net.get_param_data(layer_name)
        weight = tensor(weight)
        bias = tensor(bias)
        weighted_weight = point_2D.weight[:, None] * weight
        weighted_weight = weighted_weight.sum(dim=0)
        weighted_bias = point_2D.weight * bias
        weighted_bias = weighted_bias.sum(dim=0)
        prior = point_2D.prior + weighted_bias
        return [Point_2D([np.nan, np.nan, np.nan], weighted_weight, prior[0])]

    def backward(self,
                 points,
                 start_layer_name,
                 log=True,
                 isFilter=True,
                 debug=False,
                 test_flag=0,
                 test_part_pos=None,
                 test_part=False):
        import time
        # max_num = 50000
        all_layer_sequence = self.net.all_layer_sequence
        start_flag = 0
        scores = None
        backward_start_time = time.time()
        raw_size = self.net._net.blobs['data'].data.shape[2:]
        # test_flag = 1
        not_first_pool = 1
        for layer_idx, layer_name in enumerate(reversed(all_layer_sequence)):
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
                dim = points[0].dim
                if test_part:
                    return self.get_part_score(layer_name,points)
                if dim == 2:
                    points = self.conv_faster(layer_name,
                                              points,
                                              isFilter,
                                              test_part_pos)
                else:
                    points = self.conv_faster_3D(layer_name,
                                                 points,
                                                 isFilter,
                                                 test_part_pos)
            elif layer_type == 'pooling':
                # self.visualization_vggnet(layer_name, points)
                points = self.pool_faster(layer_name, points, isFilter)
            elif layer_type == 'relu':
                points = self.relu(layer_name, points, isFilter)

            # empty cache to reduce usage of GPU memory
            # clear()
            dura_time = time.time() - start_time
            if len(points) == 0:
                return points, None
            scores = get_points_scores(self.net, points, layer_name)
            if debug:
                print('scores: {:s}'.
                      format(scores.sum(dim=0)))
            if layer_type == 'pooling' and not_first_pool:
                points, scores = post_filter_points(
                    points,
                    scores,
                    reserve_scores_ratio=0.8,
                    max_num=200000)
                not_first_pool = 1
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

        # if scores is None:
        # points = merge_points_2D_better(points, feat_size)
        backward_dura_time = time.time() - backward_start_time
        print('backward duration: {:.3f}'.format(backward_dura_time))
        scores = get_points_scores(self.net, points, layer_name).sum(dim=1)
        mapping_points(points, feat_size, raw_size)
        return points, scores

    def visualization_vggnet(self, layer_name, points):
        import matplotlib.pyplot as plt
        assert len(points) == 1, 'top layer must be fc'
        point = points[0]
        top_name = self.net._net.top_names[layer_name][0]
        top_data = self.net._net.blobs[top_name].data[self.net.img_idx]
        weight = point.weight.reshape(top_data.shape)
        CAM = (top_data * weight).sum(dim=0)
        plt.imshow(CAM)
        plt.show()


def get_points_scores(net, points, layer_name):
    bottom_name = net._net.bottom_names[layer_name][0]
    data = tensor(net._net.blobs[bottom_name].data[net.img_idx])
    dim = points[0].dim
    res = []
    FC = False
    if np.isnan(points[0].pos[1]):
        FC = True
    for point in points:
        pos = point.pos
        if dim == 2:
            if FC:
                score = point.weight * data.view(-1)
            else:
                score = point.weight * data[:, int(pos[1]), int(pos[2])]
            score = score.sum(dim=0) + point.prior
        else:
            score = point.weight * \
                data[int(pos[0]), int(pos[1]), int(pos[2])] +\
                point.prior
        res.append(score)
    res = tensor(res)
    return res


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
