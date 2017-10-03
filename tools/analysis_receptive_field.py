#!/usr/bin/env python
# encoding: utf-8

from lib.tools import *
from pprint import pprint
import os
import sys

if len(sys.argv) != 2:
    print "Usage: python net.prototxt"
    sys.exit()


# model_def = 'model/nin_max_bigger/nin_max_bigger_train_val.prototxt'
model_def = sys.argv[1]

model_def = os.path.abspath(model_def)
layer_info = parse_net_proto(model_def)

layer_sequence = layer_info.keys()

receptive_field_info = get_all_receptive_field(layer_sequence,
                                               layer_info)
pprint(receptive_field_info)
