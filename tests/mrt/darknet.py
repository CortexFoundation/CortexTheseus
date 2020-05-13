import gluon_zoo as gz
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon

import sym_pass as spass
import dataset as ds
import sym_calib as calib
import sim_quant_helper as sim
import utils

# gz.save_model("darknet53")

def load_fname(version, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    prefix = "./data/darknet%s%s" % (version, suffix)
    return utils.extend_fname(prefix, with_ext=with_ext)

batch_size = 160
input_size = 224
inputs_ext = { 'data': {
    'shape': (batch_size, 3, input_size, input_size)
}}
inputs = [mx.sym.var(n) for n in inputs_ext]
calib_ctx = mx.gpu(2)
ctx = [mx.gpu(int(i)) for i in "1,2,3,4,5,6,7".split(',') if i.strip()]

utils.log_init()

data_iter = ds.load_imagenet_rec(batch_size, input_size)
def data_iter_func():
    data = data_iter.next()
    return data.data[0], data.label[0]
data, _ = data_iter_func()

version = "53"
sym_file, param_file = load_fname(version)
net1 = utils.load_model(sym_file, param_file, inputs, ctx=ctx)
acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)
acc_top1.reset()
acc_top5.reset()
def darknet(data, label):
    data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
    res = [net1.forward(d) for d in data]
    res = nd.concatenate(res)
    acc_top1.update(label, res)
    _, top1 = acc_top1.get()
    acc_top5.update(label, res)
    _, top5 = acc_top5.get()
    return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

sym, params = mx.sym.load(sym_file), nd.load(param_file)
sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
qsym, qparams, precs, _ = calib.sym_simulate(sym, params, inputs_ext, data, calib_ctx)
qsym, qparams = calib.sym_realize(qsym, qparams, inputs_ext, precs, "cvm")
dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
sim.save_ext(dump_ext, inputs_ext)
nd.save(dump_params, qparams)
open(dump_sym, "w").write(qsym.tojson())

dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
sym, params = mx.sym.load(dump_sym), nd.load(dump_params)
(inputs_ext,) = sim.load_ext(dump_ext)
inputs = [mx.sym.var(n) for n in inputs_ext]
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

utils.multi_validate(darknet, data_iter_func,
        cvm_quantize,
        iter_num=100000)
# utils.multi_eval_accuracy(alexnet, data_iter_func,
#        cvm_quantize,
#        iter_num=10000)
