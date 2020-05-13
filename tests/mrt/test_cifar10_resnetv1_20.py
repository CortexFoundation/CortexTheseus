import gluon_zoo as gz
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon

import sym_pass as spass
import dataset as ds
import sym_calib as calib
import sim_quant_helper as sim
import utils
import mrt as _mrt
import logging
import os

version = "20_v1"
gz.save_model("cifar_resnet"+version)
#exit(0)

def load_fname(version, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    prefix = "./data/cifar_resnet%s%s" % (version, suffix)
    return utils.extend_fname(prefix, with_ext=with_ext)

batch_size = 16
input_size = 32
inputs_ext = { 'data': {
    'shape': (batch_size, 3, input_size, input_size)
}}
inputs = [mx.sym.var(n) for n in inputs_ext]
calib_ctx = mx.gpu(2)
ctx = [mx.gpu(int(i)) for i in "1,2,3,4,5".split(',') if i.strip()]

utils.log_init()

val_data = ds.load_cifar10(batch_size, input_size)
data_iter = iter(val_data)
def data_iter_func():
    data, label = next(data_iter)
    return data, label
data, _ = next(data_iter)

sym_file, param_file = load_fname(version)
net1 = utils.load_model(sym_file, param_file, inputs, ctx=ctx)
acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)
acc_top1.reset()
acc_top5.reset()
def squeezenet(data, label):
    data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
    res = [net1.forward(d) for d in data]
    res = nd.concatenate(res)
    acc_top1.update(label, res)
    _, top1 = acc_top1.get()
    acc_top5.update(label, res)
    _, top5 = acc_top5.get()
    return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

# load original model
sym_fname, param_fname = load_fname(version)
sym, params = mx.sym.load(sym_fname), nd.load(param_fname)
print(param_fname)
sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)

# quantize process
mrt = _mrt.MRT(sym, params, inputs_ext)     # initialize
mrt.set_data('data', data)                  # set input data
mrt.calibrate(ctx=calib_ctx)                # calibration
mrt.set_output_prec(8)                      # set output prec, do nothing by default
qsym, qparams, inputs_ext = mrt.quantize()  # quantization

if False:
    # dump quantized model
    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    sim.save_ext(dump_ext, inputs_ext)
    nd.save(dump_params, qparams)
    open(dump_sym, "w").write(qsym.tojson())

    # convert to cvm executor model
    inputs_ext['data']['shape'] = (1, 3, input_size, input_size)
    nnvm_sym, nnvm_params = spass.mxnet_to_nnvm(qsym, qparams, inputs_ext)
    spass.cvm_build(nnvm_sym, nnvm_params, inputs_ext, *load_fname(version, "nnvm"))

# load quantized model for accuracy
net3 = mx.gluon.nn.SymbolBlock(qsym, inputs)
utils.load_parameters(net3, qparams, ctx=ctx)
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


logger = logging.getLogger("log.test.sym.pass")
# compare accuracy between models
utils.multi_validate(squeezenet, data_iter_func,
        cvm_quantize,
        iter_num=10, logger=logger)


import tvm
from tvm.contrib import graph_runtime
import nnvm
# target = "cuda"
# tvm_ctx = tvm.context(target, 2)
# inputs_shape = {k:v['shape'] for k,v in inputs_ext.items()}
# nnvm_sym, _ = nnvm.frontend.from_mxnet(sym)
# nnvm_sym, real_params = spass.nnvm_realize(nnvm_sym, params, inputs_ext)
# use_dtype = "int32"
# for key, value in list(real_params.items()):
#    real_params[key] = tvm.nd.array(value.asnumpy().astype(use_dtype), tvm_ctx)
# with nnvm.compiler.build_config(opt_level=0, runtime="tvm"):
#    deploy_graph, lib, real_params = nnvm.compiler.build(
#        nnvm_sym, target=target, shape=inputs_shape,
#        params=real_params, dtype=use_dtype)
# param_bytes = nnvm.compiler.save_param_dict(real_params)
# module = graph_runtime.create(deploy_graph, lib, tvm_ctx)
# module.load_params(param_bytes)
# def nnvm_real(data):
#     data = sim.load_real_data(data, 'data', inputs_ext)
#     module.run(data=data.asnumpy())
#     return nd.array(module.get_output(0).asnumpy())


#utils.multi_validate(cvm_quantize, data_iter,
#        # cvm_quantize,
#        iter_num=100000)
# utils.multi_eval_accuracy(nnvm_real, data_iter,
#        iter_num=10000)
