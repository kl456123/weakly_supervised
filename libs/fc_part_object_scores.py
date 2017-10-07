#!/usr/bin/env python
# encoding: utf-8

from tools import *
import numpy as np

# false when out of interval(include boundary)


# part object in conv layer output
# position is 3D


class FC_Part_Object(object):
    def __init__(self,
                 layer_name,
                 prior_score,
                 net):
        # easy attribution
        self.layer_name = layer_name
        self.layer_type = 'fc'
        self.children_layer_name = None
        self.raw_spatial_pos = None
        self.kernel = None
        self.net = net
        self.neg_slope = None

        self._prior_score = prior_score
        # note that score map is tuple
        self._score_map = None
        self._children_pos = None
        self._rec_field = np.inf

        self._children_kernel_weight = None

    # all get function
    def get_prior_score(self):
        return self._prior_score

    def get_children_kernel_weight(self):
        return self._children_kernel_weight

    def get_mutable_kernel(self):
        return self.kernel[0].copy(), self.kernel[1].copy()

    def get_score(self):
        return self._score_map[0].sum() + self._score_map[1]

    def get_score_map(self):
        return self._score_map

    def get_weight(self):
        return self.kernel[0]

    def get_bias(self):
        return self.kernel[1]

    def __gen_neg_slope(self):
        layer_info = self.net.get_layer_info(self.layer_name)
        neg_slope = layer_info.get('negative_slope')
        # default value is 1, just pass it
        self.neg_slope = neg_slope if neg_slope is not None else 1

    def __gen_children_layer_name(self):
        self.children_layer_name = self.net.get_children_layer_name(
            self.layer_name)

    def __gen_kernel(self, kernel_weight):
        layer_info = self.net.get_layer_info(self.layer_name)
        bn = _get(layer_info.get('bn'))

        if bn:
                # equalitive kernel
            weight, bias = self.net.get_bn_conv_param_data(
                self.layer_name)
        else:
            weight, bias = self.net.get_param_data(self.layer_name)
        # weight out_c,in_c
        # kernel_weight out_c
        weighted_weight = weight * kernel_weight
        weighted_bias = bias * kernel_weight

        self.kernel = [weighted_weight.sum(axis=0),
                       weighted_bias.sum(axis=0)]

    def __gen_score_map_and_children_kernel_weight(self):
        layer_name = self.layer_name
        weight, bias = self.get_mutable_kernel()
        assert weight.ndim == 1, 'weight\'s shape is wrong'
        in_c, = weight.shape
        bottom_blob_name = self.net._net.bottom_names[layer_name][0]
        data = self.net.get_data(
            bottom_blob_name,
            None,
            (0, in_c),
            layer_type=self.layer_type)

        assert weight.shape == data.shape, 'weight and data must have the same shape'

        conv = data * weight

        self._score_map = (conv.sum(axis=0), bias + self.get_prior_score())
        # weight[np.where(data < 0)] *= self.neg_slope

        # just one child
        self._children_kernel_weight = weight[np.newaxis, ...]

    def convert_to_spatial(self, chw, to_layer='pooling'):
        children_kernel_weight = self._children_kernel_weight.reshape(chw)
        keep = children_kernel_weight > 0
        self._children_pos = zip(keep)
        if to_layer == 'pooling':
            self._children_kernel_weight = children_kernel_weight[keep]
        else:
            raise NotImplemented

    def init(self, kernel_weight):
        self.__gen_children_layer_name()
        self.__gen_neg_slope()
        self.__gen_kernel(kernel_weight)
        self.__gen_score_map_and_children_kernel_weight()

    # def visualize_children(self, pixel_val=0.5):
        # raw_spatial_pos = help_out_map_ins(
        # self.pos[1:], self.layer_name, self.net)
        # bottom_name = self.net._net.bottom_names[self.layer_name][0]
        # help_vis_max_activated_point(
        # self.net, bottom_name, self.get_, pixel_val)

    def draw_detection(self, display):
        data = self.net._net.blobs['data'].data[self.net.img_idx].copy()
        return help_draw_detection(
            data, self.raw_spatial_pos, self.rec_field, display)
