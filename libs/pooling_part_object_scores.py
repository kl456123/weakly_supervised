#!/usr/bin/env python
# encoding: utf-8

from tools import *
import numpy as np

# part object in conv layer output
# position is 3D
# no kernel here due to pooling layer is not a construction layer
# It does not have position sensetive attribution


class Pooling_Part_Object(object):
    def __init__(self,
                 layer_name,
                 prior_score,
                 pos,
                 net):
        # easy attribution
        self.layer_name = layer_name
        self.layer_type = 'pooling'
        self.is_pad = None
        self.pos = pos
        self.children_layer_name = None
        self.raw_spatial_pos = None
        # self.kernel = None
        self.net = net
        self.neg_slope = None

        self._prior_score = prior_score
        # note that score map is tuple
        self._score_map = None
        self._children_pos = None

        self._rec_field = None
        self._children_rec_field = None
    # all get function

    def get_prior_score(self):
        return self._prior_score

    def get_score(self):
        return self._score_map[0].sum() + self._score_map[1]

    def get_children_pos(self):
        return self._children_pos

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
            self.spatial_pos[1:], feat_map_size, raw_map_size)

    def __gen_child_pos_and_children_kernel_weight(self, weight, filter_neg=True):
        bottom_blob_name = self.net._net.bottom_names[self.layer_name][0]
        layer_info = self.net.get_layer_info(self.layer_name)
        pad = get(layer_info['pad'])
        spatial_pos = np.array(help_out_map_ins(
            self.pos,
            self.layer_name,
            self.net))
        # one image
        data = self.net.get_block_data(bottom_blob_name, spatial_pos, pad)
        # C H W
        c, h, w = data.shape
        mask = data.reshape(c, -1).argmax(axis=1)
        # 3D
        self._children_pos = zip(np.arange(c), mask[0], mask[1])
        children_val = data[np.arange(c), mask[0], mask[1]]
        children_kernel_weight = np.zeros((c, c))
        # note that weight here refers to weights of selected value in each channel
        weight[children_val < 0] *= self.neg_slope
        children_kernel_weight[(np.arange(c), np.arange(c))] = weight
        self._children_kernel_weight = children_kernel_weight
        if filter_neg:
            # filter all neg score for final score
            tmp = weight * children_val
            keep = tmp > 0
            self._children_pos = self._children_pos[keep]
            self._children_kernel_weight = self._children_kernel_weight[keep]

    def init(self, kernel_weight):
        self.__gen_is_pad()
        if self.is_pad:
            return
        self.__gen_children_layer_name()
        self.__gen_rece_field()
        self.__gen_raw_spatial_pos()
        self.__gen_child_pos_and_children_kernel_weight()

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
