#!/usr/bin/env python
# encoding: utf-8
from tools import *


class Part_Object_Scores(object):
    def __init__(self,
                 layer_name,
                 prior_score,
                 spatial_position,
                 net):
        self.prior_score = prior_score
        self._layer_name = layer_name
        self._layer_type = None
        self._children_layer_name = None
        self.spatial_position = spatial_position
        self._raw_spatial_position = None
        # note that score map is tuple
        self._score_map = None
        self._net = net
        self.child_spatial_position = None
        self._num_children = None
        self._negative_slope = None
        self.activated_child_idx = None
        self.receptive_field = None
        self.children_receptive_field = None
        self.kernel = None
        self._is_pad = None

    def get_prior_score(self):
        # return self.prior_score
        return 0

    def get_negative_slope(self):
        if self._negative_slope is not None:
            return self._negative_slope

    def generate_layer_type(self):
        layer_info = self._net.get_layer_info(self._layer_name)
        if 'pooling' in layer_info:
            self._layer_type = 'pooling'
        else:
            self._layer_type = 'conv'
        if 'negative_slope' in layer_info:
            self._negative_slope = layer_info['negative_slope']
        else:
            # default value
            self._negative_slope = 1

    def generate_is_padding(self):
        top_name = self._net._net.top_names[self._layer_name][0]
        h, w = self._net._net.blobs[top_name].data.shape[2:]
        top_blob_info = self._net.get_blob_info_by_layer_name(self._layer_name)
        if top_blob_info is None:
            # no parent layer exist
            pad = 0
        else:
            pad = top_blob_info['pad']
        assert h + \
            pad > self.spatial_position[0], 'position is out of feat map'
        assert w + \
            pad > self.spatial_position[1], 'position is out of feat map'
        if h <= self.spatial_position[0] or w <= self.spatial_position[1]\
                or self.spatial_position[0] < 0 or self.spatial_position[1] < 0:
            self._is_pad = True
        else:
            self._is_pad = False

    def get_is_padding(self):
        if self._is_pad is None:
            self.generate_is_padding()
        return self._is_pad

    def generate_receptive_field(self):
        # top_name = self._net._net.top_names[self._layer_name][0]
        self.receptive_field = self._net.get_receptive_field(self._layer_name)

    def generate_children_receptive_field(self):
        children_layer_name = self.get_children_layer_name()
        self.children_receptive_field = self._net.get_receptive_field(
            children_layer_name)

    def get_children_receptive_field(self):
        if self.children_receptive_field is None:
            self.generate_children_receptive_field()
        return self.children_receptive_field

    def generate_children_layer_name(self):
        self._children_layer_name = self._net.get_children_layer_name(
            self._layer_name)

    def get_children_layer_name(self):
        if self._children_layer_name is None:
            self.generate_children_layer_name()

        return self._children_layer_name

    def generate_raw_spatial_position(self):
        top_name = self._net._net.top_names[self._layer_name][0]
        feat_map_size = self._net._net.blobs[top_name].data.shape[2:]
        raw_map_size = self._net._net.blobs['data'].data.shape[2:]
        self._raw_spatial_position = feat_map_raw(
            self.spatial_position, feat_map_size, raw_map_size)

    def get_raw_spatial_position(self):
        return self._raw_spatial_position

    def generate_child_spatial_position(self):
        if self.child_spatial_position is None:
            self.child_spatial_position = np.array(help_out_map_ins(
                self.spatial_position, self._layer_name, self._net))

    def get_child_spatial_position(self):
        return self.child_spatial_position

    def generate_activated_child_idx(self):
        assert self._score_map is not None, 'please generate score map first'
        self.activated_child_idx = np.where(self._score_map[0].ravel() > 0)[0]

    def get_num_children(self):
        return self._num_children

    def get_activated_child_idx(self, policy='relu'):
        if self.activated_child_idx is None:
            self.generate_activated_child_idx()
        return self.activated_child_idx

    def get_activated_child_position(self):
        if self.child_spatial_position is None:
            self.generate_child_spatial_position()
        return self.child_spatial_position[self.get_activated_child_idx()]

    def get_score(self):
        return self._score_map[0].sum() \
            + self._score_map[1]

    def get_score_map(self):
        if self._score_map is None:
            self.generate_score_map()
        return self._score_map

    def generate_pooling_kernel(self, ps):
        layer_name = self._layer_name
        layer_info = self._net.get_layer_info(layer_name)

        bottom_blob_name = self._net._net.bottom_names[layer_name][0]
        pad = layer_info['pad']
        dilation = layer_info['dilation']
        stride = layer_info['stride']

        spatial_position = self.spatial_position
        kernel_spatial_dim = layer_info['kernel_size'] * \
            layer_info['kernel_size']
        w = layer_info['kernel_size']
        h = w
        c = self._net._net.blobs[bottom_blob_name].data.shape[1]
        data_croped = []

        # for kernel_spatial_idx in range(kernel_spatial_dim):
        # kernel_spatial = (kernel_spatial_idx / w, kernel_spatial_idx % w)
        # next_spatial_position = out_map_in(spatial_position,
        # kernel_spatial,
        # pad,
        # stride,
        # dilation)
        for kernel_spatial_idx, next_spatial_position in enumerate(
                self.child_spatial_position):
            if ps:
                raise NotImplementedError
                channels_interval = (c * kernel_spatial_idx,
                                     c * kernel_spatial_idx + c)
            else:
                channels_interval = (0, c)
            data_croped.append(self._net.get_data(
                bottom_blob_name,
                next_spatial_position,
                channels_interval,
                pad=pad))
        data_croped = np.concatenate(data_croped).reshape(
            h, w, -1).transpose(2, 0, 1)
        kernel = np.zeros_like(data_croped)
        mask = data_croped.reshape(data_croped.shape[0], -1).argmax(axis=1)
        mask = np.unravel_index(mask, data_croped.shape[1:])
        kernel[(np.arange(data_croped.shape[0]), mask[0], mask[1])] = 1
        conv_kernel = convert_pool_kernel_to_conv_kernel(kernel)

        bias = np.zeros(conv_kernel.shape[0])
        return conv_kernel, bias

    def generate_kernel(self, last_filtered_kernel_idx_weight_pairs):
        if self._layer_type == 'pooling':
            # generate equalitive kernel for pooling
            weight, bias = self.generate_pooling_kernel(ps=False)
            self.kernel = (weight, bias)

        else:

            # for different position it has different kernel,
            # so should regernerate kernel before using it
            if self._net.get_layer_info(self._layer_name)['bn']:
                # equalitive kernel
                weight, bias = self._net.get_bn_conv_param_data(
                    self._layer_name)
            else:
                weight, bias = self._net.get_param_data(self._layer_name)
        weight_filtered = weight[last_filtered_kernel_idx_weight_pairs[0]]
        # weight_filtered.shape[1:]
        bias_filtered = bias[last_filtered_kernel_idx_weight_pairs[0]]
        weighted_weight = weight_filtered * \
            np.array(
                last_filtered_kernel_idx_weight_pairs[1])[..., np.newaxis, np.newaxis, np.newaxis]
        weighted_bias = bias_filtered * \
            last_filtered_kernel_idx_weight_pairs[1]
        self.kernel = (weighted_weight.sum(axis=0),
                       weighted_bias.sum(axis=0))

    def get_kernel(self):
        assert self.kernel is not None, 'please generate kernel first'
        return self.kernel

    # def generate_children_filtered_kernel_idx_weight_pairs_pooling(self,
        # last_filtered_kernel_idx_weight_pairs):
        # weight, bias = self.get_kernel()
        # kernel_spatial_dim = weight.shape[-1] * weight.shape[-2]
        # res = []
        # for kernel_spatial_idx in range(kernel_spatial_dim):
        # filtered_idx = last_filtered_kernel_idx_weight_pairs[0]
        # filtered_weight = np.ones((len(filtered_idx),))
        # idx_weight_pairs = (filtered_idx, filtered_weight)
        # res.append(idx_weight_pairs)
        # self._all_filtered_kernel_idx_weight_pairs = res

    def generate_children_filtered_kernel_idx_weight_pairs(self, negative_slope=0):
        layer_name = self._layer_name
        spatial_position = self.spatial_position

        layer_info = self._net.get_layer_info(layer_name)

        weight, bias = self.get_kernel()
        ps = layer_info['ps']
        dilation = layer_info['dilation']
        pad = layer_info['pad']
        # print layer_name, pad
        stride = layer_info['stride']
        # assert spatial_position[0] < 8 and spatial_position[1] < 8, 'error'

        w = weight.shape[-1]
        h = weight.shape[-2]
        c = weight.shape[-3]
        weight = weight.reshape(c, h, w)
        kernel_spatial_dim = w * h
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
        self._all_filtered_kernel_idx_weight_pairs = res

    def get_filtered_kernel_idx_weight_pairs(self):
        if self._all_filtered_kernel_idx_weight_pairs is not None:
            return self._all_filtered_kernel_idx_weight_pairs

    def generate_score_map(self):
        layer_name = self._layer_name
        spatial_position = self.spatial_position
        layer_info = self._net.get_layer_info(layer_name)
        weight, bias = self.get_kernel()
        w = weight.shape[-1]
        h = weight.shape[-2]
        c = weight.shape[-3]
        weight = weight.reshape(c, h, w)

        kernel_spatial_dim = w * h
        # just one blob in the bottom
        bottom_blob_name = self._net._net.bottom_names[layer_name][0]
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
        self.generate_layer_type()
        if self.get_is_padding():
            return
        # generate child position and filter them according to their score
        self.generate_child_spatial_position()
        # generate corresponding kernel
        self.generate_kernel(last_filtered_kernel_idx_weight_pairs)
        #
        negative_slope = self.get_negative_slope()
        # if self._layer_type == 'pooling':
        # self.generate_children_filtered_kernel_idx_weight_pairs_pooling(
        # last_filtered_kernel_idx_weight_pairs)
        # else:
        self.generate_children_filtered_kernel_idx_weight_pairs(
            negative_slope)
        self.generate_score_map()
        self.generate_activated_child_idx()
        self.generate_raw_spatial_position()
        self.generate_receptive_field()
        self.generate_children_layer_name()
        # self.post_process()

    # def post_process(self):
        # # here we check if the next layer is scale layer or not
        # parnet_layer_name = self._net.get_parent_layer_name(self._layer_name)
        # if self._net.check_is_scale_layer(parnet_layer_name):
        # self._net._net.backward_scale_layer(parent_layer_name)

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
