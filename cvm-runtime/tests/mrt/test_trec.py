import gluon_zoo as gz
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
import tvm
from tvm.contrib import graph_runtime
import nnvm
import pickle

import sym_pass as spass
import dataset as ds
import sym_calib as calib
import sim_quant_helper as sim
import ops_generator as opg
import utils
import mrt as _mrt

def load_fname(suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    prefix = "./data/trec%s" % (suffix)
    return utils.extend_fname(prefix, with_ext=with_ext)

batch_size = 16
ctx = mx.gpu()
inputs_ext = { 'data': {
    'shape': (38, batch_size)
}}
inputs = [mx.sym.var(n) for n in inputs_ext]

utils.log_init()

data_iter = ds.load_trec(batch_size)
def data_iter_func():
    return next(data_iter)
data, label = data_iter_func()

sym_file, param_file = load_fname()
net1 = utils.load_model(sym_file, param_file, inputs, ctx=ctx)
def trec(data):
    res = net1(data.as_in_context(ctx))
    return res

sym, params = mx.sym.load(sym_file), nd.load(param_file)
sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
if True:
    mrt = _mrt.MRT(sym, params, inputs_ext)
    mrt.set_data('data', data)
    mrt.calibrate(ctx=ctx)
    mrt.set_input_prec('data', 16)
    mrt.set_fixed('data')
    mrt.set_output_prec(8)
    qsym, qparams, inputs_ext = mrt.quantize()
else:
    inputs_ext['data']['data'] = data
    th_dict = calib.sym_calibrate(sym, params, inputs_ext, ctx=ctx)
    qsym, qparams, _ = calib.pure_int8_quantize(sym, params, inputs_ext, th_dict)
net2 = gluon.nn.SymbolBlock(qsym, inputs)
utils.load_parameters(net2, qparams, ctx=ctx)
def quantize(data):
    data = sim.load_real_data(data, 'data', inputs_ext)
    res = net2(data.as_in_context(ctx))
    return res

quant_sym, quant_params, quant_ext = load_fname("sym.quantize", with_ext=True)
open(quant_sym, "w").write(qsym.tojson())


if False:
    inputs_ext['data']['shape'] = (38, 1)
    data = data[:, 0].reshape(38, 1)
    _mrt.std_dump(qsym, qparams, inputs_ext, data, "trec",
            batch=True, data_dtype="int32", max_num=1000,
            dump_ops=["sentimentnet0_embedding0_fwd"])
    opg.dump_file("take",
            ["/data/std_out/trec/sentimentnet0_embedding0_fwd_0.mrt.dump.in.npy",
             "/data/std_out/trec/sentimentnet0_embedding0_fwd_1.mrt.dump.in.npy"],
            ["/data/std_out/trec/sentimentnet0_embedding0_fwd_0.mrt.dump.out.npy"],
            "/data/std_out/trec/sentimentnet0_embedding0_fwd.attr")
    exit()

if False:
    while True:
        data, _ = next(data_iter)
        data = sim.load_real_data(data, 'data', inputs_ext)
        inputs_ext['data']['data'] = data
        spass.sym_dump_ops(qsym, qparams, inputs_ext,
                ctx=mx.gpu(3))
    exit()

utils.multi_eval_accuracy(trec, data_iter_func,
        quantize,
        iter_num=1000)

