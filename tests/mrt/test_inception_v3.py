import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn

import tvm
from tvm.contrib import graph_runtime
import nnvm

import sym_calib as calib
import mrt as _mrt
import utils
import dataset as ds
import gluon_zoo as zoo
import sym_pass as spass
import sym_utils as sutils
import sim_quant_helper as sim

import logging
import numpy as np

def get_dump_fname(suffix="quant"):
    return './data/inception_v3.%s.json'%suffix, \
        './data/inception_v3.%s.params'%suffix

def load_fname(version, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    fname = "./data/tf_inception%s%s"%(version, suffix)
    return utils.extend_fname(fname, with_ext)

def test_sym_nnvm(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.nnvm")
    logger.info("=== Log Test NNVM ===")

    version = "v3"
    dump_sym, dump_params, dump_ext = load_fname(version, "mrt", True)
    sym, params = mx.sym.load(dump_sym), nd.load(dump_params)
    (inputs_ext,) = sim.load_ext(dump_ext)
    data_iter = ds.load_imagenet_rec(batch_size, 299)
    data = data_iter.next().data[0]

    _mrt.std_dump(sym, params, inputs_ext, data, "inception_v3")

def test_sym_pass(batch_size=10, iter_num=10, quantize=True):

    logger = logging.getLogger("log.test.sym.pass")

    calib_ctx = mx.gpu(2)
    ctx = [mx.gpu(int(i)) for i in "1,2,3,4".split(',') if i.strip()]
    input_size = 299
    version = "v3"
    h, w = input_size, input_size
    inputs_ext = {
        'data': {
            'shape': (batch_size, 3, h, w),
        }
    }
    inputs = [mx.sym.var(name) for name in inputs_ext]

    logger.info("load dataset, symbol and parameters")
    data_iter = ds.load_imagenet_rec(batch_size, input_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]

    net1 = utils.load_model(*load_fname(version), inputs, ctx=ctx)
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    def inception_v3(data, label):
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
        res = [net1.forward(d) for d in data]
        res = nd.concatenate(res)
        acc_top1.update(label, res)
        _, top1 = acc_top1.get()
        acc_top5.update(label, res)
        _, top5 = acc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

    if quantize:
        sym_file, param_file = load_fname(version)
        sym, params = mx.sym.load(sym_file), nd.load(param_file)
        sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
        data, _ = data_iter_func()
        if True:
            dump_sym, dump_params, dump_ext = load_fname(version, "mrt", True)
            mrt = _mrt.MRT(sym, params, inputs_ext)
            mrt.set_data('data', data)
            mrt.calibrate(ctx=calib_ctx)
            mrt.set_output_prec(8)
            qsym, qparams, inputs_ext = mrt.quantize()
        else:
            dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
            inputs_ext['data']['data'] = data
            th_dict = calib.sym_calibrate(sym, params, inputs_ext, ctx=calib_ctx)
            qsym, qparams, precs, _ = calib.sym_simulate(sym, params, inputs_ext, th_dict)
            qsym, qparams = calib.sym_realize(qsym, qparams, inputs_ext, precs)
        sim.save_ext(dump_ext, inputs_ext)
        nd.save(dump_params, qparams)
        open(dump_sym, "w").write(qsym.tojson())

    dump_sym, dump_params, dump_ext = load_fname(version, "mrt", True)
    (inputs_ext,) = sim.load_ext(dump_ext)
    net2 = utils.load_model(dump_sym, dump_params, inputs, ctx=ctx)
    qacc_top1 = mx.metric.Accuracy()
    qacc_top5 = mx.metric.TopKAccuracy(5)
    qacc_top1.reset()
    qacc_top5.reset()
    def cvm_quantize(data, label):
        data = sim.load_real_data(data, 'data', inputs_ext)
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
        res = [net2.forward(d) for d in data]
        res = nd.concatenate(res)
        qacc_top1.update(label, res)
        _, top1 = qacc_top1.get()
        qacc_top5.update(label, res)
        _, top5 = qacc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

    utils.multi_validate(inception_v3, data_iter_func,
            cvm_quantize,
            iter_num=iter_num, logger=logger)

def validate(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.mx.quantize")

    ctx = [mx.gpu(int(i)) for i in "1,2,3,4,5,6,7".split(',') if i.strip()]
    input_size = 299
    h, w = input_size, input_size
    inputs_ext = { 'data': {
        'shape': (batch_size, 3, h, w),
    }}
    inputs = [mx.sym.var(n) for n in inputs_ext]

    data_iter = ds.load_imagenet_rec(batch_size, input_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    #  data, _ = data_iter_func()

    net1 = utils.load_model(*load_fname("_v3"), inputs, ctx=ctx)
    def graph_func(data):
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
        res = [net1.forward(d) for d in data]
        return nd.concatenate(res)

    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    net2 = utils.load_model(*load_fname("v3"), inputs, ctx=ctx)
    def gluon_cv(data, label):
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
        res = [net2.forward(d) for d in data]
        res = nd.concatenate(res)
        acc_top1.update(label, res)
        _, top1 = acc_top1.get()
        acc_top5.update(label, res)
        _, top5 = acc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

    utils.multi_validate(gluon_cv, data_iter_func,
           iter_num=iter_num, logger=logger)

if __name__ == '__main__':
    utils.log_init()

    # zoo.save_inception_v3()
    # zoo.save_model('inceptionv3', 1000)
    if False:
        data_iter = ds.load_imagenet_rec(4, 299)
        version = "v3"
        while True:
            dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
            (inputs_ext,) = sim.load_ext(dump_ext)
            sym, params = mx.sym.load(dump_sym), nd.load(dump_params)
            data = data_iter.next().data[0]
            data = sim.load_real_data(data, 'data', inputs_ext)
            inputs_ext['data']['data'] = data
            spass.sym_dump_ops(sym, params, inputs_ext,
                    datadir="/data/wlt", ctx=mx.gpu(3))
        exit()

    #  test_sym_nnvm(1, 0)
    test_sym_pass(16, 10)
    # test_sym_pass(160, 1000, quantize=False)
    # test_mxnet_sym(1)
    # validate(700, 100000)
