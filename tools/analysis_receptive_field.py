#!/usr/bin/env python
# encoding: utf-8

from lib.tools import *
from pprint import pprint
import os


# model_def = 'model/nin_max_bigger/nin_max_bigger_train_val.prototxt'
model_def = 'vgg.train_val.prototxt'

model_def = os.path.abspath(model_def)
layer_info = parse_net_proto(model_def)

layer_sequence = layer_info.keys()

receptive_field_info = get_all_receptive_field(layer_sequence,
                                               layer_info)
pprint(receptive_field_info)
