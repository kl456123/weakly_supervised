#!/usr/bin/env python
# encoding: utf-8
from conv_part_object_scores import Conv_Part_Object
from pooling_part_object_scores import Pooling_Part_Object
from fc_part_object_scores import FC_Part_Object


def handle_part_objects(start_part_objects,
                        start_layer_idx,
                        net,
                        threshold_policy='float',
                        display=True):
    all_layer_sequence = net.all_layer_sequence
    num_layers = len(all_layer_sequence)
    top_part_objects = start_part_objects
    # spatial = 0
    for layer_idx in reversed(range(num_layers)):
        layer_name = all_layer_sequence[layer_idx]
        bottom_part_objects = layer_handler(top_part_objects, layer_name, net)
        top_part_objects = bottom_part_objects
    return top_part_objects


def merge_part_objects(part_objects,shape):
    mark = np.zeros(shape)
    mark[...] = 0
    res = []
    for part_object in part_objects:
        pos = part_obje
        score = part_object_scores.get_score()
        pos = pos.astype(np.int)
        if pos[0] >= shape[0] or pos[1] >= shape[1] or pos[2] >= shape[2]:
            continue
        mark[tuple(pos)] += score



def layer_handler(part_objects, layer_name, net):
    # for part_object in part_objects:
        # # 3D (the first dim is None) # children_pos = part_object.get_children_pos()
        # children_kernel_weight = part_object.get_children_kernel_weight()
    layer_info = net.get_layer_info(layer_name)
    # each layer must have this attr
    layer_type = layer_info['type']
    # layer_type = part_objects[0].layer_type
    # bottom_part_objects = []
    if layer_type == 'conv':
        bottom_part_objects = conv_layer_hander(part_objects, layer_name, net)
    elif layer_type == 'pooling':
        bottom_part_objects = pooling_layer_handler(
            part_objects, layer_name, net)
    elif layer_type == 'fc':
        bottom_part_objects = fc_layer_handler(part_objects, layer_name, net)
    else:
        raise NotImplementedError
    return bottom_part_objects


def conv_layer_hander(part_objects, layer_name, net):
    all_children_part_objects = []
    merge = False
    if part_objects[0].layer_type == 'pooling':
        merge = True
    for part_object in part_objects:
        # 3D (the first dim is None)
        children_pos = part_object.get_children_pos()
        children_kernel_weight = part_object.get_children_kernel_weight()
        prior_score = part_object.get_prior_score()
        for idx, pos in enumerate(children_pos):
            # pos_weight_pair = (pos, children_kernel_weight[idx])
            child_part_object = Conv_Part_Object(
                layer_name,
                prior_score,
                pos,
                net)
            child_part_object.init(children_kernel_weight)
            all_children_part_objects.append(child_part_object)
    if merge:
        all_children_part_objects = merge_part_objects(all_children_part_objects)
    return all_children_part_objects


def pooling_layer_handler(part_objects, children_layer_name, net):
    all_children_part_objects = []
    to_spatial = False
    if part_objects[0].layer_type == 'fc':
        to_spatial = True
    for part_object in part_objects:
        # 3D (the first dim is None)
        if to_spatial:
            top_name = net._net.top_names[children_layer_name][0]
            chw = net._net.blobs[top_name].data.shape
            part_objects.convert_to_spatial(chw)

        children_pos = part_object.get_children_pos()
        children_kernel_weight = part_object.get_children_kernel_weight()
        prior_score = part_object.get_prior_score()
        for idx, pos in enumerate(children_pos):
            # pos_weight_pair = (pos, children_kernel_weight[idx])
            child_part_object = Pooling_Part_Object(
                children_layer_name,
                prior_score,
                pos,
                net)
            child_part_object.init(children_kernel_weight)
            all_children_part_objects.append(child_part_object)
    # note that here we can merge all overide object to reduce computation

    return all_children_part_objects


# def merge(all_children_part_objects, shape):
    # mark = np.zeros(shape)
    # mark[...] = 0
    # res = []
    # for part_object_scores in all_children_part_objects:
    # pos = part_object_scores.pos
    # score = part_object_scores.get_score()
    # pos = pos.astype(np.int)
    # if pos[0] >= shape[0] or pos[1] >= shape[1] or pos[2] >= shape[2]:
    # continue
    # mark[tuple(pos)] += score
    # # res.append(part_object_scores)
    # return res

# def postprocess(all_children_part_objects):
    # pass


def fc_layer_handler(part_objects, children_layer_name, net):
    all_children_part_objects = []
    # if part_objects[0].layer_type=='fc' and
    for part_object in part_objects:
        # if to_spatial:
            # top_name = net._net.top_names[children_layer_name][0]
            # chw = net._net.blobs[top_name].data.shape
            # part_object.convert_to_spatial(chw)
        child_kernel_weight = part_object.get_children_kernel_weight()
        prior_score = part_object.get_prior_score()

        child_part_object = FC_Part_Object(
            children_layer_name,
            prior_score,
            net)
        child_part_object.init(child_kernel_weight)
        all_children_part_objects.append(child_part_object)
    return all_children_part_objects
