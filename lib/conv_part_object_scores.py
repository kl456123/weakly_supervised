#!/usr/bin/env python
# encoding: utf-8

from tools import *
import numpy as np

# false when out of interval(include boundary)


def _check_in(x, l, r):
    if x > r or x < l:
        return False
    return True


def _get(value, default):
    return value if value is not None else default

# part object in conv layer output
# position is 3D


class Conv_Part_Object(object):
    def __init__(self,
                 layer_name,
                 prior_score,
                 pos,
                 net):
        # easy attribution
        self.layer_name = layer_name
        self.layer_type = 'conv'
        self.is_pad = None
        self.pos = pos
        self.children_layer_name = None
        self.raw_spatial_pos = None
        self.kernel = None
        self.net = net
        self.neg_slope = None

        self._prior_score = prior_score
        # note that score map is tuple
        self._score_map = None
        self._children_pos = None

        self.rec_field = None
        self._children_rec_field = None
        self._all_children_kernel_weight = None

    # all get function
    def get_prior_score(self):
        return self._prior_score

    def get_all_children_kernel_weight(self):
        return self._all_children_kernel_weight

    def get_children_pos(self):
        return self._children_pos

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

    def __gen_is_pad(self):
        top_name = self.net._net.top_names[self.layer_name][0]
        h, w = self.net._net.blobs[top_name].data.shape[2:]
        top_blob_info = self.net.get_blob_info_by_layer_name(self._layer_name)
        if top_blob_info is None:
            # no parent layer exist
            pad = 0
        else:
            pad = top_blob_info['pad']
        assert _check_in(self.pos[1], -pad, h + pad),\
            'position is out of feat map'
        assert _check_in(self.pos[2], -pad, w + pad),\
            'position is out of feat map'
        if h <= self.pos[1] or w <= self.pos[1]\
                or self.pos[2] < 0 or self.pos[2] < 0:
            self.is_pad = True
        else:
            self.is_pad = False

    def __gen_neg_slope(self):
        layer_info = self.net.get_layer_info(self.layer_name)
        neg_slope = layer_info.get('negative_slope')
        # default value is 1, just pass it
        self.neg_slope = neg_slope if neg_slope is not None else 1

    def __gen_rece_field(self):
        # top_name = self._net._net.top_names[self._layer_name][0]
        if self._layer_type == 'fc':
            self.receptive_field = np.inf
        else:
            self.receptive_field = self._net.get_receptive_field(
                self._layer_name)

    def __gen_children_rece_field(self):
        children_layer_name = self.get_children_layer_name()
        self.children_receptive_field = self.net.get_receptive_field(
            children_layer_name)

    def __gen_children_layer_name(self):
        self._children_layer_name = self._net.get_children_layer_name(
            self._layer_name)

    def __gen_raw_spatial_pos(self):
        top_name = self._net._net.top_names[self._layer_name][0]
        feat_map_size = self._net._net.blobs[top_name].data.shape[2:]
        raw_map_size = self._net._net.blobs['data'].data.shape[2:]
        self.raw_spatial_pos = feat_map_raw(
            self.spatial_pos, feat_map_size, raw_map_size)

    def __gen_child_spatial_pos(self):
        self.child_spatial_pos = np.array(help_out_map_ins(
            self.pos,
            self.layer_name,
            self.net))

    def __gen_kernel(self, kernel_weight):
        layer_info = self.net.get_layer_info(self.layer_name)
        bn = _get(layer_info.get('bn'))

        if bn:
                # equalitive kernel
            weight, bias = self.net.get_bn_conv_param_data(
                self.layer_name)
        else:
            weight, bias = self.net.get_param_data(self.layer_name)
        weighted_weight = weight * kernel_weight
        weighted_bias = bias * kernel_weight

        self.kernel = [weighted_weight.sum(axis=0),
                       weighted_bias.sum(axis=0)]

    def __gen_score_map_and_all_children_kernel_weight(self):
        layer_name = self._layer_name
        layer_info = self.net.get_layer_info(layer_name)
        pad = _get(layer_info.get('pad'))
        weight, bias = self.get_mutable_kernel()
        assert weight.ndim == 3, 'weight\'s shape is wrong'
        c, h, w = weight.shape
        kdim = w * h
        bottom_blob_name = self.net._net.bottom_names[layer_name][0]
        data = self.net.get_block_data(bottom_blob_name,
                                       self.children_pos,
                                       pad=pad)

        assert weight.shape == data.shape, 'weight and data must have the same shape'

        conv = data * weight

        self._score_map = (conv.sum(axis=0), bias + self.get_prior_score())
        weight[np.where(data < 0)] *= self.neg_slope

        self._all_children_kernel_weight = weight.reshape(
            (-1, kdim)).transpose((1, 0))

    def init(self, kernel_weight):
        self.__gen_is_pad()
        if self.is_pad:
            return
        self.__gen_children_layer_name()
        self.__gen_child_spatial_pos()
        self.__gen_rece_field()
        self.__gen_raw_spatial_pos()
        self.__gen_kernel(kernel_weight)
        self.__gen_score_map_and_all_children_kernel_weight(self.neg_slope)

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
