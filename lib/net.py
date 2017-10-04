#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import caffe
import os
from tools import *


class Net(object):
    def __init__(self, model_def, model_weights, num_classes, deploy=False):
        self.model_def = os.path.abspath(model_def)
        self.model_weights = os.path.abspath(model_weights)

        self._net = caffe.Net(model_def, model_weights, caffe.TEST)
        # self._auxiliary_layer_info = None
        # self.auxiliary_layer_sequence=None
        self.generate_layer_info(model_def)
        # for load data into layer
        if not deploy:
            self._net.forward()
        self.batch_size = None
        # defaults image,you can change it
        self.img_idx = 0
        self.num_classes = num_classes
        # self.set_layer_info()
        self.receptive_field_info = {}
        self.generate_layer_sequence()
        self.generate_auxiliary_layer_sequence()
        self.generate_all_receptive_field()
        self.all_layer_sequence = self.layer_sequence +\
            self.auxiliary_layer_sequence

    def generate_auxiliary_layer_sequence(self):
        self.auxiliary_layer_sequence =\
            self._auxiliary_layer_info.keys()

    # def reverse_auxiliary_layer(self, class_id):
        # auxiliary_layer_sequence = self.auxiliary_layer_sequence
        # for layer_name in reversed(auxiliary_layer_sequence):
        # top_blob_name = self._net.top_names[layer_name][0]
        # data = self._net.blobs[top_blob_name].data
        # weight, bias = self.get_param_data(layer_name)
        # class_weight = weight[class_id]

        # def check_is_scale_layer(self, layer_name):
        # if self.get_layer_info(layer_name) is None:
        # return True
        # return False
    def get_block_data(self, blob_name, all_spatial_positions, pad=0):
        data = self._net.blob[blob_name].data[self.img_idx]
        if pad:
            data_paded = np.lib.pad(data, pad, mode='constant')[pad:-pad]
        else:
            data_paded = data
        # data C,H,W
        h = [spatial_position[0] for spatial_position in all_spatial_positions]
        w = [spatial_position[1] for spatial_position in all_spatial_positions]
        c = np.arange(len(h))
        return data_paded[(c, h, w)]

    def get_children_layer_name(self, current_layer_name, include_scale=False):

        if current_layer_name == self.all_layer_sequence[0]:
            return None
        for idx, layer_name in enumerate(self.all_layer_sequence):
            if layer_name == current_layer_name:
                return self.all_layer_sequence[idx - 1]

    def get_bn_conv_param_data(self, conv_layer_name, eps=1e-5):
        # some high parameters
        # this function generates weights and bias for conv followed BN
        # by doing this we can consider all layer as just only one conv layer

        # note that bias is None
        weight, bias = self.get_param_data(conv_layer_name)
        bn_layer_name = generate_layer_name(conv_layer_name, 'bn')
        scale_layer_name = generate_layer_name(conv_layer_name, 'scale')

        # scale layer param
        alpha, beta = self.get_param_data(scale_layer_name)

        # bn layer param
        mean, var, moving_average = self.get_param_data(bn_layer_name)
        if moving_average == 0:
            scale_factor = 0
        else:
            scale_factor = 1.0 / moving_average
        mean = mean * scale_factor
        var = var * scale_factor

        # normalize variance
        var = np.sqrt(var + eps)

        weight = alpha.reshape((-1, 1, 1, 1)) * \
            weight / var.reshape((-1, 1, 1, 1))
        bias = -1.0 * alpha * mean / var + beta
        return weight, bias

    def get_parent_layer_name(self, current_layer_name, include_scale=False):
        if current_layer_name == self.all_layer_sequence[-1]:
            return None
        for idx, layer_name in enumerate(self.all_layer_sequence):
            if layer_name == current_layer_name:
                return self.all_layer_sequence[idx + 1]

    def get_blob_info_by_layer_name(self, current_layer_name):
        # stop()
        # top blob info is in the top of layer
        layer_name = self.get_parent_layer_name(current_layer_name)
        if layer_name is None:
            print 'no parent exist ,may be the top layer'
            return None
        return self.get_layer_info(layer_name)

    def display(self):
        for layer_name, blob in self._net.blobs.iteritems():
            print layer_name + '\t' + str(blob.data.shape)

    def set_img_idx(self, img_idx):
        batch_size = self.get_batch_size()
        assert batch_size > img_idx, 'img_idx is out of range (0,{:d})'\
            .format(batch_size)

        self.img_idx = img_idx

    def get_kernel_spatial_dim(self, layer_name):
        weight = self.get_param_data(layer_name)[0]
        return weight.shape[-2] * weight.shape[-1]

    def generate_layer_info(self, prototxt):
        self._layer_info, self._auxiliary_layer_info = parse_net_proto(
            prototxt)

    def generate_layer_sequence(self):
        # all layer_name that have not layer info are considered as scale layer
        # stop()
        self.layer_sequence = self._layer_info.keys()

    def get_layer_info(self, layer_name):
        if layer_name in self.layer_sequence:
            return self._layer_info[layer_name]
        elif layer_name in self.auxiliary_layer_sequence:
            return self._auxiliary_layer_info[layer_name]
        else:
            print '{:s} is scale layer,return None'.format(layer_name)
            return None

    def get_blobs_name_by_idx(self, id):
        return self._net.blobs.keys()[id]

    def get_blobs_data_by_idx(self, id, one_img=True):
        if one_img:
            return self._net.blobs.values()[id].data[self.img_idx]
        else:
            return self._net.blobs.values()[id].data

    def get_blobs_data_by_name(self, blob_name, one_img=True):
        if one_img:
            return self._net.blobs[blob_name].data[self.img_idx]
        else:
            return self._net.blobs[blob_name].data

    def get_batch_size(self):
        if self._batch_size is None:

            blob_data = self.get_blobs_data_by_idx(0, False)
            self.batch_size = blob_data.shape[0]
        return self.batch_size

    def get_prediction(self, img_idx=None):
        if img_idx is None:
            img_idx = self.img_idx
        pred_blob = 'pool3'
        return self._net.blobs[pred_blob].data[img_idx].argmax()

    def get_label(self, img_idx=None):
        if img_idx is None:
            img_idx = self.img_idx
        return int(self._net.blobs['label'].data[img_idx])

    def get_receptive_field_data(self, blob_name, spatial_scope, pad=0, one_img=True):
        pass

    def get_data(self,
                 blob_name,
                 spatial_position,
                 channels_interval,
                 one_img=True,
                 pad=0,
                 layer_type='conv'):
        #  print spatial_position
        data = self.get_blobs_data_by_name(blob_name, False)
        if layer_type == 'fc':
            if one_img:
                return data[self.img_idx, channels_interval[0]:channels_interval[1]]
            else:
                return data[:, channels_interval[0]:channels_interval[1]]
        # channels = data.shape[-3]
        height = data.shape[-2]
        width = data.shape[-1]
        # import pdb
        # pdb.set_trace()
        # channels_interval[0]:channels_interval[1], :, :]

        # assert channels >= channels_interval[1] and channels < channels_interval[0],\
        # 'channels_interval is out of range'
        assert height + pad > spatial_position[0], 'blob_name{:s} pad{:d} spatial_position[0] {:d} is out of (0,{:d})'\
            .format(blob_name, pad, spatial_position[0], height - 1)
        assert width + pad > spatial_position[1], 'blob_name{:s} pad{:d} spatial_position[1] {:d}is out of (0,{:d})'\
            .format(blob_name, pad, spatial_position[1], width - 1)
        if height <= spatial_position[0] or width <= spatial_position[1]\
                or spatial_position[0] < 0 or spatial_position[1] < 0:
            if one_img:

                return np.zeros((channels_interval[1] - channels_interval[0]))
            else:
                return np.zeros((self.batch_size, channels_interval[1] - channels_interval[0]))
        if one_img:
            return data[self.img_idx,
                        channels_interval[0]:channels_interval[1],
                        spatial_position[0],
                        spatial_position[1]]
        else:
            return data[:,
                        channels_interval[0]:channels_interval[1],
                        spatial_position[0],
                        spatial_position[1]]

    def get_num_classes(self):
        return self.num_classes

    def get_filtered_kernel_idx_weight_pairs(self,
                                             layer_name,
                                             spatial_position,
                                             kernel_idx=None,
                                             last_filtered_kernel_idx_weight_pairs=None):
        layer_info = self.get_layer_info(layer_name)
        if kernel_idx is None:
            kernel_idx = last_filtered_kernel_idx_weight_pairs[0]

        weight, bias = self.get_param_data(layer_name, kernel_idx)
        ps = layer_info['ps']
        dilation = layer_info['dilation']
        pad = layer_info['pad']
        stride = layer_info['stride']

        w = weight.shape[-1]
        h = weight.shape[-2]
        c = weight.shape[-3]
        kernel_spatial_dim = w * h
        bottom_blob_name = self._net.bottom_names[layer_name][0]
        res = []
        for kernel_spatial_idx in range(kernel_spatial_dim):
            kernel_spatial = (kernel_spatial_idx / w, kernel_spatial_idx % w)
            next_spatial_position = out_map_in(spatial_position,
                                               kernel_spatial,
                                               pad,
                                               stride,
                                               dilation)
            if ps:
                channels_interval = (c * kernel_spatial_idx,
                                     c * kernel_spatial_idx + c)
            else:
                channels_interval = (0, -1)
            data_croped = self.get_data(
                bottom_blob_name,
                next_spatial_position,
                channels_interval, pad=pad)
            weight_croped = weight[:, kernel_spatial[0], kernel_spatial[1]]
            # multiple
            #  scores = data_croped * weight_croped
            filtered_idx = np.array(np.where(data_croped > 0))
            filtered_weight = weight_croped[filtered_idx]
            # import pdb
            # pdb.set_trace()

            if ps:
                filtered_idx += c * kernel_spatial_idx
            # add filtered_bias
            idx_weight_pairs = (filtered_idx, filtered_weight)
            res.append(idx_weight_pairs)
        return res

    def get_param_data(self, layer_name, idx=None):
        # print 'layer_name: {:s}'.format(layer_name)
        layer = self.get_layer(layer_name)
        weight = layer.blobs[0].data
        if len(layer.blobs) == 1:
            bias = np.zeros(weight.shape[0:1])
        else:
            bias = layer.blobs[1].data

        if idx is not None:
            if len(layer.blobs) == 3:
                return weight[idx], bias[idx], layer.blobs[2].data[idx]
            else:
                return weight[idx], bias[idx]
        else:
            if len(layer.blobs) == 3:
                return weight, bias, layer.blobs[2].data
            else:
                return weight, bias

    def get_layer(self, layer_name):
        return self._net.layer_dict[layer_name]

    def get_conved_spatial_feat(self, layer_name, spatial_position):
        layer_info = self.get_layer_info(layer_name)
        weight, bias = self.get_param_data(layer_name)
        w = weight.shape[-1]
        h = weight.shape[-2]
        c = weight.shape[-3]
        # weights_sliced
        # if filtered_kernel_idx is None:
        # weights_sliced = weight
        # else:
        # weights_sliced = weight[filtered_kernel_idx]

        kernel_spatial_dim = w * h
        # just one blob in the bottom
        bottom_blob_name = self._net.bottom_names[layer_name][0]
        ps = layer_info['ps']
        dilation = layer_info['dilation']
        pad = layer_info['pad']
        stride = layer_info['stride']
        data_croped = []
        for kernel_spatial_idx in range(kernel_spatial_dim):
            kernel_spatial = (kernel_spatial_idx / w, kernel_spatial_idx % w)
            next_spatial_position = out_map_in(
                spatial_position, kernel_spatial, pad, stride, dilation)
            if ps:
                channels_interval = (c * kernel_spatial_idx,
                                     c * kernel_spatial_idx + c)
            else:
                channels_interval = (0, -1)
            data_croped.append(self.get_data(
                bottom_blob_name,
                next_spatial_position,
                channels_interval, pad=pad))
        # data_croped
        #  import pdb
        #  pdb.set_trace()
        data_croped = np.concatenate(data_croped).reshape(
            h, w, -1).transpose(2, 0, 1)
        # elementwise multiple
        conved_spatial_feat = data_croped * weight
        return conved_spatial_feat.sum(axis=1), bias

#      def get_next_layer_name(self, layer_name):
        #  self._layer_info.keys()[]

    def get_next_position_feat_pairs(self, layer_name, next_layer_name, spatial_position, kernel_idx):
        weight, bias = self.get_param_data(layer_name)
        h = weight.shape[-2]
        w = weight.shape[-1]
        layer_info = self.get_layer_info(layer_name)
        dilation = layer_info['dilation']
        pad = layer_info['pad']
        stride = layer_info['stride']
        all_idx_weight_pairs = self.get_filtered_kernel_idx_weight_pairs(
            layer_name, spatial_position, kernel_idx)
        res = []
        #  import pdb
        #  pdb.set_trace()
        for idx, idx_weight_pairs in enumerate(all_idx_weight_pairs):
            kernel_spatial = (idx / w, idx % w)
            next_spatial_position = out_map_in(
                spatial_position, kernel_spatial, pad, stride, dilation)
            conved_spatial_feat, bias = self.get_conved_spatial_feat(
                next_layer_name, next_spatial_position)
            weighted_feat, weighted_bias = weighted_filted_conved_spatial_feat(
                conved_spatial_feat, bias, idx_weight_pairs)
            res.append((next_spatial_position, (weighted_feat, weighted_bias)))
        return res

    def get_start_position_feat_pair(self, layer_name, start_spatial_position, kernel_idx):
        conved_spatial_feat, bias = self.get_conved_spatial_feat(
            layer_name, start_spatial_position)
        return start_spatial_position, (conved_spatial_feat[kernel_idx], bias[kernel_idx])

    def generate_all_receptive_field(self):
        # temp init
        # self.receptive_field_info['conv5'] = 26
        # self.receptive_field_info['conv4'] = 18
        layer_sequence = self.layer_sequence
        for layer_name in layer_sequence:
            self.generate_receptive_field(layer_name)

    def generate_receptive_field(self, current_layer_name):
        # refer to top blob's name of the layer conscisely
        layer_sequence = self.layer_sequence
        # top_name = self._net.top_names[layer_name][0]
        start_flag = 0
        top_receptive_field = 1
        for layer_name in reversed(layer_sequence):
            if start_flag or layer_name == current_layer_name:
                start_flag = 1
                layer_info = self.get_layer_info(layer_name)
                if layer_info is None:
                    continue
                stride = layer_info['stride']
                dilation = layer_info['dilation']
                kernel_size = layer_info['kernel_size']

                bottom_receptive_field = receptive_top2down(top_receptive_field,
                                                            stride,
                                                            kernel_size,
                                                            dilation)
                top_receptive_field = bottom_receptive_field
        self.receptive_field_info[current_layer_name] = top_receptive_field

    def get_receptive_field(self, layer_name, name_type='layer'):
        if layer_name in self.auxiliary_layer_sequence:
            return np.inf
        if layer_name not in self.receptive_field_info:
            self.generate_receptive_field(layer_name)
        return self.receptive_field_info[layer_name]
