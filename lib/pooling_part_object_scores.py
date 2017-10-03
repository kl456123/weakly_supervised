#!/usr/bin/env python
# encoding: utf-8

from tools import *


# part object in conv layer output
# position is 3D
# no kernel here due to pooling layer is not a construction layer
# It does not have position sensetive attribution
class Pooling_Part_Object(object):
    def __init__(self,
                 layer_name,
                 prior_score,
                 position,
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
