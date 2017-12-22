#!/usr/bin/env python
# encoding: utf-8

# from __future__ import division
import caffe
import numpy as np


import matplotlib.pyplot as plt


def filter_negative(net, name=None):
    for layer_name, param in net.params.items():
        if (layer_name is not None) and not (layer_name == name):
            continue
        weight = param[0].data
        bias = param[1].data

        weight[weight < 0] = 0
        bias[bias > 0] = 0


def plot(arr):
    length = len(arr)
    x = np.arange(length)
    plt.plot(x, arr)
    plt.show()


caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('model/nin_max_bigger/nin_max_bigger_solver.prototxt')
solver.net.copy_from('model/nin_max_bigger/cifar10_nin_iter_120000.caffemodel')
# solver.restore('../model/nin_max_bigger/cifar10_nin_iter_120000.solverstate')

# model_def = './model/'
# model_weights =


niter = 1000
test_interval = 20
interval = 10
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval / interval)))


def test_net(test_net):
    print('start testing...')
    acc = 0
    for test_it in range(100):
        solver.test_nets[0].forward()
        acc += solver.test_nets[0].blobs['accuracy'].data
    acc /= 1e2
    print('test acc: %f' % (acc))
    return acc


for it in range(niter // interval):
    filter_negative(solver.net, 'cccp6')
    solver.step(interval)
    # record loss
    train_loss[it] = solver.net.blobs['loss'].data

    if it % test_interval == 0:
        acc = test_net(solver.test_net[0])
        test_acc[it // test_interval] = acc

plot(train_loss)
plot(test_acc)

solver.net.save('model/nin_max_bigger/cifar10_nin_non_negative.caffemodel')
