import gluon_zoo as gz
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon

import sym_pass as spass
import dataset as ds
import sym_calib as calib
import sim_quant_helper as sim
import ops_generator as opg
import utils
import mrt as _mrt

batch_size = 16
input_size = 28
inputs_ext = { 'data': {
    'shape': (batch_size, 1, input_size, input_size),
}}
inputs = [mx.sym.var(n) for n in inputs_ext]
ctx = mx.gpu(4)

utils.log_init()

val_data = ds.load_quickdraw10(batch_size)
data_iter = iter(val_data)
def data_iter_func():
    data, label = next(data_iter)
    return data, label
data, _ = next(data_iter)

root = "/data/wlt/train_7/"
sym_file = root + "quickdraw_wlt_augmentation_epoch-4-0.8164531394275162.json"
prm_file = root + "quickdraw_wlt_augmentation_epoch-4-0.8164531394275162.params"

net1 = utils.load_model(sym_file, prm_file, inputs, ctx=ctx)
acc = mx.metric.Accuracy()
def quickdraw(data, label):
    res = net1.forward(data.as_in_context(ctx))
    acc.update(label, res)
    return "accuracy={:6.2%}".format(acc.get()[1])

if True:
    sym, params = mx.sym.load(sym_file), nd.load(prm_file)
    sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
    mrt = _mrt.MRT(sym, params, inputs_ext)
    mrt.set_data('data', data)
    mrt.calibrate(ctx=ctx)
    mrt.set_output_prec(8)
    qsym, qparams, inputs_ext = mrt.quantize()

    net2 = mx.gluon.nn.SymbolBlock(qsym, inputs)
    utils.load_parameters(net2, qparams, ctx=ctx)
    qacc = mx.metric.Accuracy()
    def cvm_quantize(data, label):
        data = sim.load_real_data(data, 'data', inputs_ext)
        res = net2.forward(data.as_in_context(ctx))
        qacc.update(label, res)
        return "accuracy={:6.2%}".format(qacc.get()[1])

if False:
    _mrt.std_dump(qsym, qparams, inputs_ext, data, "wlt_animal10")
    exit()

utils.multi_validate(quickdraw, data_iter_func,
        cvm_quantize,
        iter_num=100000)
