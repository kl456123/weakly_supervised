#!/usr/bin/env python
# encoding: utf-8
import caffe
import sys
# sys.path.append('..')
from lib.net import Net
from lib.tools import *
from lib.part_object_scores import Part_Object_Scores
import numpy as np

import os
model_def = 'tiny_yolo_voc_deploy.prototxt'
model_weights = 'tiny_yolo_voc.caffemodel'


#######################################
#######################################
#######################################
# here we test some method of Net to
# vertify its function

# test get_bn_conv_param_data


def test_get_bn_conv_param_data():
    from scipy import signal

    num_classes = 21
    net3 = Net(model_def, model_weights, num_classes, deploy=True)
    image_file = 'dog.jpg'
    input_shape = net3._net.blobs['data'].data.shape
    transformered_im = preprocess(image_file, input_shape)
    output = net3._net.forward(
        data=np.asarray([transformered_im]))
    top_blob_name = 'conv8_scale'
    bottom_blob_name = 'conv7_scale'
    conv_layer_name = 'conv8'
    bottom_data = net3.get_blobs_data_by_name(bottom_blob_name)
    conved_blob_name = 'conv8'
    conved_data = net3.get_blobs_data_by_name(conved_blob_name)

    equal_weight, equal_bias = net3.get_bn_conv_param_data(conv_layer_name)
    weight, bias = net3.get_param_data(conv_layer_name)
    import pdb
    pdb.set_trace()

    top_data = net3.get_blobs_data_by_name(top_blob_name)


test_get_bn_conv_param_data()
