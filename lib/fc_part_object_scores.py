#!/usr/bin/env python
# encoding: utf-8

from tools import *
import numpy as np

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
        self.layer_type = 'fc'
        # self.is_pad = None
        # self.pos = pos
        self.children_layer_name = None
        # self.raw_spatial_pos = None
        self.net = net
        self.neg_slope = None

        self._prior_score = prior_score
        # note that score map is tuple
        self._score_map = None
        self._children_pos = None

        self._rec_field = np.inf
        self._children_rec_field = None
    # all get function

    def get_prior_score(self):
        return self._prior_score

    def get_score(self):
        return self._score_map[0].sum() + self._score_map[1]
