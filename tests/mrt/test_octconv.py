import mxnet as mx
from mxnet import gluon
from mxnet import symbol
from mxnet.contrib import quantization as qm
from mxnet.gluon.model_zoo import vision
from mxnet import ndarray as nd
from mxnet.gluon import HybridBlock, nn
from mxnet.contrib import quantization as mquant

import tvm
from tvm.contrib import graph_runtime
import nnvm

import numpy as np
import logging
import os

import utils
import dataset as ds
import sym_utils as sutils
import sym_pass as spass
import sym_annotate as anno
import sym_calib as calib
import sim_quant_helper as sim
import gluon_zoo as zoo

# import resnet18 as resnet
# import resnet152 as resnet
import resnet50 as resnet

from sym_pass import *

def load_fname(version, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    fname = "./data/a02_resnet-26_alpha-0.250%s%s"%(version, suffix)
    return utils.extend_fname(fname, with_ext)

def test_sym_nnvm(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.nnvm")
    logger.info("=== Log Test NNVM ===")

    target = "llvm"
    tvm_ctx = tvm.context(target, 1)
    mx_ctx = mx.gpu(2)
    inputs_ext = { 'data': {
            'shape': (batch_size, 3, 224, 224),
    } }
    inputs = [mx.sym.var(name) for name in inputs_ext]
    inputs_shape = {k:v['shape'] for k,v in inputs_ext.items()}

    data_iter = load_dataset(batch_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    data_iter_func()

    version = ""
    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    (inputs_ext,) = sim.load_ext(dump_ext)
    sym, params = mx.sym.load(dump_sym), nd.load(dump_params)
    # sim.load_ins_ext(params, inputs_ext)

    # nnvm_sym, _ = nnvm.frontend.from_mxnet(sym)
    # with open('debug_nnvm_sym_after_load_from_mxnet.json', 'w') as fout:
    #    fout.write(nnvm_sym.debug_str())
    dump_sym, dump_params = load_fname(version, "nnvm.compile", False)
    spass.mxnet_to_nnvm(sym, params, inputs_ext, dump_sym, dump_params, target='llvm')

def test_sym_pass(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.sym.pass")

    version = ""
    sym_fname, param_fname = load_fname(version)
    sym, params = mx.sym.load(sym_fname), nd.load(param_fname)
    params = {k.split(':')[1]:v for k, v in params.items()}


    calib_ctx = mx.gpu(2)
    ctx = [mx.gpu(int(i)) for i in "1,2,3,4,5,6,7".split(',') if i.strip()]
    inputs_ext = { 'data': {
            'shape': (batch_size, 3, 224, 224),
    } }
    inputs = [mx.sym.var(name) for name in inputs_ext]

    logger.info("load dataset, symbol and parameters")

    order = sutils.topo_sort(sym)
    for op_head in order:
        if op_head.attr('name') == 'classifier':
            break
    sym = op_head
    net = mx.gluon.nn.SymbolBlock(sym, inputs)
    load_parameters(net, params, ctx=ctx)

    data_iter = ds.load_imagenet_rec(batch_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    for i in range(10):
        if i == 3:
            break
        data, _ = data_iter_func()
    data_iter.reset()

    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    def resnet(data, label):
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
        res = [net.forward(d) for d in data]
        res = nd.concatenate(res)
        acc_top1.update(label, res)
        _, top1 = acc_top1.get()
        acc_top5.update(label, res)
        _, top5 = acc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

    sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
    qsym, qparams, precs, _ = calib.sym_simulate(sym, params, inputs_ext, data, calib_ctx)
    qsym, qparams = calib.sym_realize(qsym, qparams, inputs_ext, precs, "cvm")
    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    sim.save_ext(dump_ext, inputs_ext)
    nd.save(dump_params, qparams)
    open(dump_sym, "w").write(qsym.tojson())

    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    (inputs_ext,) = sim.load_ext(dump_ext)
    net3 = utils.load_model(dump_sym, dump_params, inputs, ctx=ctx)
    qacc_top1 = mx.metric.Accuracy()
    qacc_top5 = mx.metric.TopKAccuracy(5)
    qacc_top1.reset()
    qacc_top5.reset()
    def cvm_quantize(data, label):
        data = sim.load_real_data(data, 'data', inputs_ext)
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
        res = [net3.forward(d) for d in data]
        res = nd.concatenate(res)
        qacc_top1.update(label, res)
        _, top1 = qacc_top1.get()
        qacc_top5.update(label, res)
        _, top5 = qacc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

    utils.multi_validate(resnet, data_iter_func,
            cvm_quantize,
            iter_num=iter_num, logger=logger)
    # multi_eval_accuracy(graph_func, data_iter_func,
    #         gluon_cv, cvm_quantize,
    #         iter_num=iter_num, logger=logger)

def save_data():
    batch_size = 1024
    data_iter = load_dataset(batch_size)
    calib_data = data_iter.next()
    x, _ = quant_helper(calib_data.data[0])
    np.save('/tmp/imagenet.x', x.asnumpy())
    np.save('/tmp/imagenet.y', calib_data.label[0].asnumpy())

if __name__ == "__main__":
    utils.log_init()

    resnet.save_graph(mx.gpu())
    # zoo.save_model('resnet50_v1', 1000)
    # zoo.save_model('resnet18_v1')
    # zoo.save_model('resnet50_v1d_0.86')
    # zoo.save_model('resnet18_v1b_0.89')

    # save_data()

    test_sym_pass(batch_size=160, iter_num=20)
    # test_sym_nnvm(batch_size=1, iter_num=1)


