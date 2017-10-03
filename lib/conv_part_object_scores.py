#!/usr/bin/env python
# encoding: utf-8

from tools import *


# false when out of interval(include boundary)
def _check_in(x, l, r):
    if x > r or x < l:
        return False
    return True

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

        self._rec_field = None
        self._children_rec_field = None
        self._all_kernel_weight = []

    # all get function
    def get_prior_score(self):
        return self._prior_score

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
        self.children_receptive_field = self._net.get_receptive_field(
            children_layer_name)

    def __gen_children_layer_name(self):
        self._children_layer_name = self._net.get_children_layer_name(
            self._layer_name)

    def __gen_raw_spatial_pos(self):
        top_name = self._net._net.top_names[self._layer_name][0]
        feat_map_size = self._net._net.blobs[top_name].data.shape[2:]
        raw_map_size = self._net._net.blobs['data'].data.shape[2:]
        self._raw_spatial_position = feat_map_raw(
            self.spatial_position, feat_map_size, raw_map_size)

    def __gen_child_spatial_pos(self):
        if self.children_spatial_pos is None:
            self.child_spatial_position = \
                np.array(help_out_map_ins(
                    self.spatial_position,
                    self._layer_name,
                    self._net))

    def generate_kernel(self, last_filtered_kernel_idx_weight_pairs):

        if self.net.get_layer_info(self._layer_name)['bn']:
            # equalitive kernel
            weight, bias = self.net.get_bn_conv_param_data(
                self._layer_name)
        else:
            weight, bias = self.net.get_param_data(self._layer_name)
        weight_filtered = weight[last_filtered_kernel_idx_weight_pairs[0]]
        bias_filtered = bias[last_filtered_kernel_idx_weight_pairs[0]]
        weighted_weight = weight_filtered * \
            np.array(
                last_filtered_kernel_idx_weight_pairs[1])[..., np.newaxis, np.newaxis, np.newaxis]
        weighted_bias = bias_filtered * \
            last_filtered_kernel_idx_weight_pairs[1]

        self.kernel = [weighted_weight.sum(axis=0),
                       weighted_bias.sum(axis=0)]

    def generate_children_filtered_kernel_idx_weight_pairs(self,
                                                           negative_slope=0):
        layer_name = self._layer_name
        spatial_position = self.spatial_position

        layer_info = self._net.get_layer_info(layer_name)

        weight, bias = self.get_kernel()
        if self._layer_type != 'fc':
            ps = layer_info['ps']
            dilation = layer_info['dilation']
            pad = layer_info['pad']
            # print layer_name, pad
            stride = layer_info['stride']
            # assert spatial_position[0] < 8 and spatial_position[1] < 8, 'error'

            w = weight.shape[-1]
            h = weight.shape[-2]
            c = weight.shape[-3]
        else:
            c = weight.shape[0]
            h = 1
            w = 1
            ps = False
        weight = weight.reshape(c, h, w)
        bottom_blob_name = self._net._net.bottom_names[layer_name][0]
        res = []
        for kernel_spatial_idx, next_spatial_position in enumerate(
                self.child_spatial_position):
            kernel_spatial = (kernel_spatial_idx / w, kernel_spatial_idx % w)
            if ps:
                channels_interval = (c * kernel_spatial_idx,
                                     c * kernel_spatial_idx + c)
            else:
                channels_interval = (0, c)
            data_croped = self._net.get_data(
                bottom_blob_name,
                next_spatial_position,
                channels_interval,
                pad=pad)
            weight_croped = weight[:, kernel_spatial[0], kernel_spatial[1]]
            if negative_slope == 0:
                filtered_idx = np.array(np.where(data_croped > 0)[0])
                filtered_weight = weight_croped[filtered_idx]
            else:
                negative_idx = np.array(np.where(data_croped < 0)[0])
                filtered_idx = np.arange(c)
                filtered_weight = weight_croped.copy()
                filtered_weight[negative_idx] *= negative_slope
            if ps:
                filtered_idx += c * kernel_spatial_idx
            idx_weight_pairs = (filtered_idx, filtered_weight)
            res.append(idx_weight_pairs)
        self._all_kernel_weight = res

    def get_filtered_kernel_idx_weight_pairs(self):
        return self._all_kernel_weight

    def __gen_score_map(self):
        layer_name = self._layer_name
        pos = self.pos
        layer_info = self.net.get_layer_info(layer_name)
        weight, bias = self.kernel
        assert weight.ndim == 3, 'weight\'s shape is wrong'
        c, h, w = weight.shape
        kdim = w * h
        bottom_blob_name = self.net._net.bottom_names[layer_name][0]

    def generate_score_map(self):
        layer_name = self._layer_name
        spatial_position = self.spatial_position
        weight, bias = self.get_kernel()
        # stop()
        w = weight.shape[-1]
        h = weight.shape[-2]
        c = weight.shape[-3]
        weight = weight.reshape(c, h, w)

        # just one blob in the bottom
        ps = layer_info['ps']
        dilation = layer_info['dilation']
        pad = layer_info['pad']
        stride = layer_info['stride']
        data_croped = []

        #  for kernel_spatial_idx in range(kernel_spatial_dim):
        #  kernel_spatial = (kernel_spatial_idx / w, kernel_spatial_idx % w)
        #  next_spatial_position = out_map_in(
        #  spatial_position, kernel_spatial, pad, stride, dilation)
        for kernel_spatial_idx, next_spatial_position in enumerate(self.child_spatial_position):
            if ps:
                channels_interval = (c * kernel_spatial_idx,
                                     c * kernel_spatial_idx + c)
            else:
                # print 'not ps,channel: ', c
                channels_interval = (0, c)
            data_croped.append(self._net.get_data(
                bottom_blob_name,
                next_spatial_position,
                channels_interval,
                pad=pad))
        data_croped = np.concatenate(data_croped).reshape(
            h, w, -1).transpose(2, 0, 1)
        # elementwise multiple
        conved_spatial_feat = data_croped * weight
        self._score_map = (conved_spatial_feat.sum(
            axis=0), bias + self.get_prior_score())

    def init(self, last_filtered_kernel_idx_weight_pairs):
        self.__gen_is_pad()
        if self.get_is_padding():
            return
        # generate child position and filter them according to their score
        self.generate_child_spatial_position()
        # generate corresponding kernel
        self.generate_kernel(last_filtered_kernel_idx_weight_pairs)
        #
        negative_slope = self.get_negative_slope()
        self.generate_children_filtered_kernel_idx_weight_pairs(
            negative_slope)
        self.generate_score_map()
        self.generate_activated_child_idx()
        # stop()
        self.generate_raw_spatial_position()
        self.generate_receptive_field()
        self.generate_children_layer_name()

    def visualize_children(self, pixel_val=0.5):
        raw_spatial_position = help_out_map_ins(
            self.spatial_position, self._layer_name, self._net)
        bottom_name = self._net._net.bottom_names[self._layer_name][0]
        help_vis_max_activated_point(
            self._net, bottom_name, self.get_child_spatial_position(), pixel_val)

    def draw_detection(self, display):
        data = self._net._net.blobs['data'].data[self._net.img_idx].copy()
        return help_draw_detection(
            data, self._raw_spatial_position, self.receptive_field, display)
