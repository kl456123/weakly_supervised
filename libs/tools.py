#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import caffe
import matplotlib.pyplot as plt
from random import random as rand


def normalize(all_scores, type='relu'):
    if not isinstance(all_scores, np.ndarray):
        all_scores = np.array(all_scores)
    all_scores[np.where(all_scores < 0)] = 0
    _sum = all_scores.sum()
    all_proportion = all_scores.astype(np.float) / _sum

    return all_proportion


def weighted_filted_conved_spatial_feat(conved_spatial_feat, bias, filtered_kernel_idx_weight_pairs):
    #  note that conved_spatial_feat (ou_c,h,w)
    #  filtered_kernel_idx_weight_pairs [(id,weight)....]
    # weighted_feat = np.zeros(conved_spatial_feat.shape[1:])
    # for pair in filtered_kernel_idx_weight_pairs:
    #  import pdb
    #  pdb.set_trace()
    spatial_shape = conved_spatial_feat.shape[1:]
    conved_spatial_feat = conved_spatial_feat.reshape(
        (conved_spatial_feat.shape[0], -1))
    idx = filtered_kernel_idx_weight_pairs[0]
    weight = filtered_kernel_idx_weight_pairs[1]

    weighted_feat = conved_spatial_feat[idx] * weight[..., np.newaxis]
    filtered_bias = bias[filtered_kernel_idx_weight_pairs[0]] * weight

    return weighted_feat.sum(axis=1).reshape(spatial_shape), filtered_bias.sum()


def vis_square(data, display=True):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    # normalize data for display
    plt.figure(figsize=(2, 2))
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant',
                  constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape(
        (n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape(
        (n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    if display:
        plt.imshow(data)
        plt.axis('off')
        plt.show()
    return data


def caffe_init(mode='gpu'):
    if mode == 'gpu':
        caffe.set_mode_gpu()
        caffe.set_device(0)
    else:
        caffe.set_mode_cpu()


def vis_activated_point(net, activated_points, img_idx, pixel_val=0.5, save=False):
    # import pdb
    # pdb.set_trace()
    #     note that activated points is 2D (h,w)
    data = net.blobs['data'].data[img_idx].copy()
    height = data.shape[-2]
    width = data.shape[- 1]
    activated_points = list(activated_points)
    _data = None
    for origial_activated_point in activated_points:
        #         print np.ceil(origial_activated_point[0])
        # import pdb
        # pdb.set_trace()
        if origial_activated_point[0] < 0 or origial_activated_point[0] > height - 1:
            continue
        if origial_activated_point[1] < 0 or origial_activated_point[1] > width - 1:
            continue
        h = origial_activated_point[0].astype(np.int)
        w = origial_activated_point[1].astype(np.int)

        data[:, h, w] = pixel_val
        c = data.shape[0]
        if c == 1:
            _data = data.reshape((h, w))
        else:
            _data = data.transpose(1, 2, 0)[np.newaxis, ...]
    if _data is None:
        print('no activated points!!')
        return
    else:
        dirty_data = vis_square(_data)
        if save:
            plt.imsave('dirty_data.png', dirty_data)


def help_vis_max_activated_point(net, blob_name, next_activated_points, pixel_val=0.5):
    img_idx = net.img_idx
    feat_map_size = net._net.blobs[blob_name].data.shape[2:]
    origial_map_size = net._net.blobs['data'].data.shape[2:]
    origial_activated_points = feat_map_raw(next_activated_points,
                                            feat_map_size,
                                            origial_map_size)
    vis_activated_point(net._net, origial_activated_points, img_idx, pixel_val)


def filter_activated_points(point_score_pairs):
    for idx, pair in enumerate(point_score_pairs):
        if pair[1] == 0:
            # ignore it
            del point_score_pairs[idx]


def out_map_in(spatial_center, kernel_spatial, pad, stride, dilation=1):
    # param: kernel_spatial is 2D referring to spatial position

    out_h_idx = spatial_center[0]
    out_w_idx = spatial_center[1]
    in_h_idx = out_h_idx * stride - pad + kernel_spatial[0] * dilation
    in_w_idx = out_w_idx * stride - pad + kernel_spatial[1] * dilation
    return in_h_idx, in_w_idx


def out_map_ins(spatial_center, kernel_spatial_dim, pad, stride, dilation=1):
    # defaultly the shape of kernel is square
    w = int(np.sqrt(kernel_spatial_dim))
    res = []
    for kernel_spatial_idx in range(kernel_spatial_dim):
        kernel_spatial = (kernel_spatial_idx / w, kernel_spatial_idx % w)
        res.append(out_map_in(spatial_center,
                              kernel_spatial, pad, stride, dilation))
    return res


def help_out_map_ins(spatial_center, layer_name, net):
    layer_info = net.get_layer_info(layer_name)
    pad = layer_info['pad']
    dilation = layer_info['dilation']
    stride = layer_info['stride']
    # if pooling layer
    if 'pooling' in layer_info:
        kernel_size = layer_info['kernel_size']
        kernel_spatial_dim = kernel_size * kernel_size
    else:
        filters, bias = net.get_param_data(layer_name)
        kernel_spatial_dim = filters.shape[-1] * filters.shape[-2]
    return out_map_ins(spatial_center, kernel_spatial_dim, pad, stride, dilation)


def draw_detection(normalized_data, raw_spatial_position, receptive_field, display=True):
    if type(receptive_field) is int:
        receptive_field = (receptive_field, receptive_field)
    assert len(receptive_field) == 2, 'unsupported type of receptive_field'
    left_top = np.zeros((2,))
    left_top[1] = raw_spatial_position[1] - (receptive_field[1] - 1) / 2.0
    left_top[0] = raw_spatial_position[0] - (receptive_field[0] - 1) / 2.0
    color = (rand(), rand(), rand())
    rect = plt.Rectangle((left_top[1], left_top[0]),
                         receptive_field[1],
                         receptive_field[0], fill=False,
                         edgecolor=color, linewidth=1.5)
    if display:
        plt.gca().add_patch(rect)
        plt.imshow(normalized_data)
        plt.show()
    return rect


def draw_all_detection(normalized_data, res):
    plt.close('all')
    plt.imshow(normalized_data)
    for rect in res:
        plt.gca().add_patch(rect)

    plt.axis('off')
    plt.show()


def help_draw_all_detection(data, res):
    normalized_data = vis_square(data, False)
    draw_all_detection(
        normalized_data, res)


def help_draw_detection(data, raw_spatial_position, receptive_field, display=True):
    normalized_data = vis_square(data, False)
    return draw_detection(normalized_data, raw_spatial_position,
                          receptive_field, display)


def receptive_top2down(top_receptive_field, stride, kernel_size, dilation):
    if kernel_size == 0:
        print('warn: may be global pooling is set true\
        so receptive_field is global(just return None)')
        return 0
    kernel_extent = (kernel_size - 1) * (dilation - 1) + kernel_size
    return (top_receptive_field - 1) * stride + kernel_extent


def generate_layer_name(main_layer_name, subfix):
    return main_layer_name + '_' + subfix


def preprocess(image_file, **kwargs):
    input_shape = kwargs.get('input_shape')
    mean_file = kwargs.get('mean_file')
    scale = kwargs.get('scale')
    im_format = kwargs.get('format')
    im = caffe.io.load_image(image_file)
    # RGB format
    transformer = caffe.io.Transformer({
        'data': input_shape
    })
    transformer.set_transpose('data', (2, 0, 1))
    # substract mean
    if mean_file is not None:
        mu = np.load(mean_file)
        mu = mu.mean(1).mean(1)
        transformer.set_mean('data', mu)
    if scale == '255':
        transformer.set_raw_scale('data', 255)
    if im_format == 'BGR':
        transformer.set_channel_swap('data', (2, 1, 0))

    transformered_im = transformer.preprocess('data', im)
    return transformered_im


def default_value(name):
    if name == 'dilation':
        return 1
    elif name == 'pad':
        return 0
    elif name == 'stride':
        return 1
    elif name == 'ps':
        return False
    elif name == 'negative_slope':
        return 0
    else:
        raise NotImplemented


def help_parse(arr, name):
    if isinstance(arr, int) or isinstance(arr, float):
        return arr
    elif isinstance(arr, bool):
        return arr
    length = len(arr)
    assert length < 2, 'length >1 is not supported at present!'
    return default_value(name) if length == 0 else arr[0]


def parse_net_proto(prototxt):
    from caffe.proto import caffe_pb2
    from google.protobuf.text_format import Merge
    from collections import OrderedDict
    net_param = caffe_pb2.NetParameter()
    with open(prototxt, 'r') as f:
        Merge(f.read(), net_param)
    layer_names = []
    all_layer_info = OrderedDict()
    is_bn = 0
    for layer in net_param.layer:
        layer_name = layer.name
        one_layer_info = {}
        # one_layer_info['ps'] = layer.
        if layer.type == 'Pooling':
            one_layer_info['type'] = 'pooling'
            param = layer.pooling_param
            one_layer_info['pooling'] = 'max' if param.pool == 0 else 'ave'
            one_layer_info['kernel_size'] = help_parse(param.kernel_size, None)
            one_layer_info['bn'] = False
            one_layer_info['stride'] = help_parse(param.stride, 'stride')
        elif layer.type == 'Convolution':
            one_layer_info['type'] = 'conv'
            param = layer.convolution_param
            one_layer_info['kernel_size'] = help_parse(param.kernel_size, None)
            one_layer_info['dilation'] = help_parse(param.dilation, 'dilation')
            one_layer_info['bn'] = False
            one_layer_info['pad'] = help_parse(param.pad, 'pad')
            one_layer_info['stride'] = help_parse(param.stride, 'stride')
            one_layer_info['pad'] = help_parse(param.pad, 'pad')
        elif layer.type == 'ReLU':
            one_layer_info['type'] = 'relu'
            param = layer.relu_param
            one_layer_info['negative_slope'] = help_parse(param.negative_slope,
                                                          'negative_slope')
        elif layer.type == 'BatchNorm' or layer.type == 'Scale':
            is_bn += 1
            continue
        elif layer.type == 'InnerProduct':
            param = layer.inner_product_param
            one_layer_info['type'] = 'fc'
            one_layer_info['num_output'] = param.num_output
        else:
            print('skip layer {:s}'.format(layer.name))
            continue
        if is_bn == 2:
            all_layer_info[layer_names[-1]]['bn'] = True
            is_bn = 0
        layer_names.append(layer_name)
        all_layer_info[layer_name] = one_layer_info
    all_layer_info[layer_names[-1]]['bn'] = False
    return all_layer_info


def get_all_receptive_field(layer_sequence, layers_info):
    from collections import OrderedDict
    receptive_field_info = OrderedDict()
    for layer_name in layer_sequence:
        receptive_field_info[layer_name] =\
            get_receptive_field(layer_name,
                                layer_sequence,
                                layers_info)
    return receptive_field_info


def get_receptive_field(current_layer_name, layer_sequence, layers_info):
    # refer to top blob's name of the layer conscisely
    start_flag = 0
    top_receptive_field = 1
    for layer_name in reversed(layer_sequence):
        if start_flag or layer_name == current_layer_name:
            start_flag = 1
            layer_info = layers_info[layer_name]
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
    return top_receptive_field


def display(net):
    for layer_name, blob in net.blobs.iteritems():
        print(layer_name + '\t' + str(blob.data.shape))


def get_featmap_size():
    pass


def calculate_similarity(a, b):
    return (a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b))
