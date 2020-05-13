import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
from mxnet import gluon

import tvm
from tvm.contrib import graph_runtime
import nnvm

import sym_calib as calib
import utils
import mrt as _mrt
import dataset as ds
import gluon_zoo as zoo
import sym_pass as spass
import sym_utils as sutils
import sim_quant_helper as sim
import cvm_op as cvm

import logging
import numpy as np

version = '1.0'
# version = '_v2_1.0'
def load_fname(version, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    prefix = "./data/mobilenet%s%s"%(version, suffix)
    return utils.extend_fname(prefix, with_ext)

def test_mx_quantize(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.mx.quantize")

    ctx = [mx.gpu(int(i)) for i in "1,3".split(',') if i.strip()]
    inputs_ext = { 'data': {
        'shape': (batch_size, 3, 224, 224),
    }}
    inputs = [mx.sym.var(n) for n in inputs_ext]

    data_iter = ds.load_imagenet_rec(batch_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    data, _ = data_iter_func()

    net1 = utils.load_model(*load_fname(version), inputs, ctx=ctx)
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    def mobilenet(data, label):
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
        res = [net1.forward(d) for d in data]
        res = nd.concatenate(res)
        acc_top1.update(label, res)
        _, top1 = acc_top1.get()
        acc_top5.update(label, res)
        _, top5 = acc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

    calib_ctx = mx.gpu(1)
    sym_fname, param_fname = load_fname(version)
    sym, params = mx.sym.load(sym_fname), nd.load(param_fname)
    sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
    if True:
        if True:
            mrt = _mrt.MRT(sym, params, inputs_ext)
            mrt.set_data('data', data)
            mrt.calibrate()
            # [ 0.0008745864 0.03330660510427334 ] 0.6670066884888368 0.7753906
            mrt.set_threshold("mobilenet0_dense0_weight", 0.67)
            # [ -0.0036011334 0.054821780899052534 ] 1.100036751338784 1.4626989
            mrt.set_threshold("mobilenet0_conv24_batchnorm24_fwd_weight", 1.1)
            # [ 0.013243316 1.7543557133786065 ] 70.18747185088569 94.66275
            mrt.set_threshold("mobilenet0_conv23_batchnorm23_fwd_weight", 35.10)
            # [ -0.0016149869 0.05713169649243355 ] 1.1442489167675376 1.7122083
            mrt.set_threshold("mobilenet0_conv20_batchnorm20_fwd_weight", 1.144)
            # [ -0.0015804865 0.04523811489343643 ] 0.9063427844084799 1.0745146
            mrt.set_threshold("mobilenet0_conv16_batchnorm16_fwd_weight", 0.90)
            # [ 0.4315614 2.447332109723772 ] 49.37820360490254 63.959927
            mrt.set_threshold("mobilenet0_conv2_batchnorm2_fwd", 49.37)
            # [ 0.9770754 1.3392452512468611 ] 27.761980422905516 40.729546
            mrt.set_threshold("mobilenet0_relu2_fwd", 27.76)
            # [ 1.0975745 1.0489919010632773 ] 22.077412493692915 23.784576
            mrt.set_threshold("mobilenet0_relu4_fwd", 22.08)
            # [ 0.9885562 2.360489403014386 ] 48.19834426651407 69.22121
            mrt.set_threshold("mobilenet0_conv5_batchnorm5_fwd", 48.2)
            # [ 0.7895588 1.0544661745870065 ] 21.878882319617176 30.95745
            mrt.set_threshold("mobilenet0_relu17_fwd", 21.88)
            # [ 0.8717863 1.0887600296120434 ] 22.646986888608513 28.265652
            mrt.set_threshold("mobilenet0_relu19_fwd", 22.65)
            # [ 0.35124516 0.6501711574631898 ] 13.354668314135012 20.770807
            mrt.set_threshold("mobilenet0_relu20_fwd", 13.35)
            # [ 0.9378179 1.110470714216975 ] 23.147232155910086 27.886068
            mrt.set_threshold("mobilenet0_relu21_fwd", 23.15)
            # [ 0.36263302 0.6352599878026505 ] 13.067832775738754 17.18809
            mrt.set_threshold("mobilenet0_relu22_fwd", 13.07)
            # [ 0.19875833 0.49999100821358816 ] 10.198578498193196 16.625143
            mrt.set_threshold("mobilenet0_relu24_fwd", 10.2)
            # [ 0.32357717 1.6308352606637138 ] 65.55698759215218 75.84912
            mrt.set_threshold("mobilenet0_conv25_batchnorm25_fwd", 32.94)
            # [ 0.36793178 1.512995992388044 ] 30.62785163096019 49.464615
            mrt.set_threshold("mobilenet0_relu26_fwd", 30.63)
            # [ 18.028658 38.61970520019531 ] 790.4227619171143 805.51886
            mrt.set_threshold("sum0", 790.423)
            mrt.set_output_prec(8)
            qsym, qparams, inputs_ext = mrt.quantize()
        else:
            inputs_ext['data']['data'] = data
            th_dict = calib.sym_calibrate(sym, params, inputs_ext, ctx=calib_ctx)
            qsym, qparams, precs, _ = calib.sym_simulate(sym, params, inputs_ext, th_dict)
            qsym, qparams = calib.sym_realize(qsym, qparams, inputs_ext, precs)
        dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
        sim.save_ext(dump_ext, inputs_ext)
        nd.save(dump_params, qparams)
        open(dump_sym, "w").write(qsym.tojson())

        dump_sym, dump_params = load_fname(version, "nnvm.compile")
        nnvm_sym, nnvm_params = spass.mxnet_to_nnvm(qsym, qparams, inputs_ext)
        spass.cvm_build(nnvm_sym, nnvm_params, inputs_ext, dump_sym, dump_params)

    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
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

    utils.multi_validate(mobilenet, data_iter_func,
            cvm_quantize,
            iter_num=iter_num, logger=logger)
    # utils.multi_eval_accuracy(mobilenet, data_iter_func,
    #         cvm_quantize,
    #         iter_num=iter_num, logger=logger)

def test_sym_nnvm():
    logger = logging.getLogger("log.test.nnvm")
    logger.info("=== Log Test NNVM ===")

    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    sym, params = mx.sym.load(dump_sym), nd.load(dump_params)
    (inputs_ext,) = sim.load_ext(dump_ext)
    data_iter = ds.load_imagenet_rec(1)
    data = data_iter.next().data[0]

    _mrt.std_dump(sym, params, inputs_ext, data, "mobilenet"+version)


if __name__ == '__main__':
    utils.log_init()

    # zoo.save_mobilenet_v2_1_0()
    # zoo.save_mobilenet1_0()
    # zoo.save_model('mobilenetv2_1.0')
    # zoo.save_model('mobilenet1.0')
    # zoo.save_model('mobilenet1.0_int8', 1000)

    test_mx_quantize(16, 100000)
    # test_sym_nnvm()
    # test_sym_nnvm(16, 10)
    # test_performance(16, 10)
