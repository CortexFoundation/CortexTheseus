import mxnet as mx
from mxnet import ndarray as nd
from mxnet.symbol import _internal
from mxnet import symbol as _sym

from mrt import transformer

import numpy as np
import logging
import time
import os

def test_nd_op(ctx=mx.gpu()):
    nd.waitall()
    np_arr = np.zeros((3, 1, 5, 5))
    nd_arr = nd.array(np_arr, ctx)
    print(nd_arr)
    time.sleep(5)
    print('success')

def test_nd_save(ctx=mx.gpu()):
    a = nd.array([1,2], ctx)
    nd.save(os.path.expanduser("~/data/test_nd_save_data"), a)
    print('success')

def test_net(ctx=mx.gpu()):
    # sym_path = "/home/ryt/data/ssd_512_vgg16_atrous_voc.json"
    # prm_path = "/home/ryt/data/ssd_512_vgg16_atrous_voc.params"
    sym_path = "/home/ryt/data/alexnet.json"
    prm_path = "/home/ryt/data/alexnet.params"
    shape = (16, 3, 224, 224)
    data_np = np.random.uniform(size=shape)
    data = nd.array(data_np)
    model = Model.load(sym_path, prm_path)
    graph = model.to_graph(ctx=ctx)
    print('testing net... ')
    out = graph(data)
    print('success')

    pass

if __name__ == '__main__':
    test_net(mx.gpu())
    # test_nd_op()
    # test_nd_save(ctx=mx.cpu())

