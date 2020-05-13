from __future__ import print_function  # only relevant for Python 2
import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import nnvm
import tvm
from tvm.contrib import graph_runtime

# from quant_utils import *
import utils
import mrt as _mrt
import sym_annotate as anno
import sym_utils as sutils
import sym_pass as spass
import sym_calib as calib
import sim_quant_helper as sim
import gluon_zoo as zoo

import numpy as np

def load_fname(version, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    if with_ext:
        return "./data/mnist%s%s.json"%(version, suffix), \
            "./data/mnist%s%s.params"%(version, suffix), \
            "./data/mnist%s%s.ext"%(version, suffix)
    else:
        return "./data/mnist%s%s.json"%(version, suffix), \
            "./data/mnist%s%s.params"%(version, suffix)

def data_xform(data):
    """Move channel axis to the beginning, cast to float32, and normalize to [0, 1]."""
    return nd.moveaxis(data, 2, 0).astype('float32') / 255

train_data = mx.gluon.data.vision.MNIST(train=True).transform_first(data_xform)
val_data = mx.gluon.data.vision.MNIST(train=False).transform_first(data_xform)

batch_size = 1
train_loader = mx.gluon.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = mx.gluon.data.DataLoader(val_data, shuffle=False, batch_size=batch_size)

version = 'lenet'
ctx = mx.gpu(2)
def train_mnist():
    # Select a fixed random seed for reproducibility
    mx.random.seed(42)

    if version == '':
        net = nn.HybridSequential(prefix='DApp_')
        with net.name_scope():
            net.add(
                nn.Conv2D(channels=16, kernel_size=(3, 3), activation='relu'),
                nn.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
                nn.Conv2D(channels=32, kernel_size=(3, 3), activation='relu'),
                nn.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
                nn.Conv2D(channels=64, kernel_size=(3, 3), activation='relu'),
                nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                nn.Conv2D(channels=128, kernel_size=(1, 1), activation='relu'),
                nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                nn.Flatten(),
                nn.Dense(10, activation=None),
            )
    elif version == 'lenet':
        net = nn.HybridSequential(prefix='LeNet_')
        with net.name_scope():
            net.add(
                nn.Conv2D(channels=20, kernel_size=(5, 5), activation='relu'),
                nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                nn.Conv2D(channels=50, kernel_size=(5, 5), activation='relu'),
                nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                nn.Flatten(),
                nn.Dense(500, activation='relu'),
                nn.Dense(10, activation=None),
            )
    elif version == 'mlp':
        net = nn.HybridSequential(prefix='MLP_')
        with net.name_scope():
            net.add(
                nn.Flatten(),
                nn.Dense(128, activation='relu'),
                nn.Dense(64, activation='relu'),
                nn.Dense(10, activation=None)  # loss function includes softmax already, see below
            )

    net.initialize(mx.init.Xavier(), ctx=ctx)
    net.summary(nd.zeros((1, 1, 28, 28), ctx=ctx))

    trainer = gluon.Trainer(
	params=net.collect_params(),
	optimizer='adam',
	optimizer_params={'learning_rate': 1e-3},
    )
    metric = mx.metric.Accuracy()
    loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
    num_epochs = 10

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.as_in_context(ctx)
            labels = labels.as_in_context(ctx)

            with autograd.record():
                outputs = net(inputs)
                loss = loss_function(outputs, labels)

            loss.backward()
            metric.update(labels, outputs)

            trainer.step(batch_size=inputs.shape[0])

        name, acc = metric.get()
        print('After epoch {}: {} = {:5.2%}'.format(epoch + 1, name, acc))
        metric.reset()

    for inputs, labels in val_loader:
        inputs = inputs.as_in_context(ctx)
        labels = labels.as_in_context(ctx)
        metric.update(labels, net(inputs))
    print('Validaton: {} = {}'.format(*metric.get()))
    assert metric.get()[1] > 0.96

    sym = net(mx.sym.var('data'))
    sym_file, param_file = load_fname(version)
    open(sym_file, "w").write(sym.tojson())
    net.collect_params().save(param_file)

def test_sym_pass(iter_num=10):
    inputs_ext = { 'data': {
            'shape': (batch_size, 1, 28, 28),
    } }
    inputs = [mx.sym.var(n) for n in inputs_ext]

    data_iter = iter(val_loader)
    def data_iter_func():
        return next(data_iter)
    data, _ = data_iter_func()

    net1 = utils.load_model(*load_fname(version), inputs, ctx=ctx)
    def graph_func(data):
        return net1.forward(data.as_in_context(ctx))

    sym_file, param_file = load_fname(version)
    sym, params = mx.sym.load(sym_file), nd.load(param_file)
    sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
    if True:
        mrt = _mrt.MRT(sym, params, inputs_ext)
        mrt.set_data('data', data)
        mrt.calibrate(ctx=ctx)
        mrt.set_output_prec(8)
        qsym, qparams, inputs_ext = mrt.quantize()
    else:
        inputs_ext['data']['data'] = data
        th_dict = calib.sym_calibrate(sym, params, inputs_ext, ctx=ctx)
        qsym, qparams, precs, _ = calib.sym_simulate(sym, params, inputs_ext, th_dict)
        qsym, qparams = calib.sym_realize(qsym, qparams, inputs_ext, precs, "cvm")
    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    sim.save_ext(dump_ext, inputs_ext)
    nd.save(dump_params, qparams)
    open(dump_sym, "w").write(qsym.tojson())

    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    (inputs_ext,) = sim.load_ext(dump_ext)
    inputs = [mx.sym.var(n) for n in inputs_ext]
    net2 = utils.load_model(dump_sym, dump_params, inputs, ctx=ctx)
    def cvm_quantize(data):
        data = sim.load_real_data(data, 'data', inputs_ext)
        return net2.forward(data.as_in_context(ctx))

    utils.multi_eval_accuracy(graph_func, data_iter_func,
            cvm_quantize,
            iter_num=iter_num)

def test_nnvm_pass(iter_num=10):
    logger = logging.getLogger("log.test.nnvm")
    logger.info("=== Log Test NNVM ===")

    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    sym, params = mx.sym.load(dump_sym), nd.load(dump_params)
    (inputs_ext,) = sim.load_ext(dump_ext)
    data_iter = iter(val_loader)
    data, _ = next(data_iter)
    _mrt.std_dump(sym, params, inputs_ext, data, "cvm_mnist")

print ("Test mnist", version)
train_mnist()
exit()

utils.log_init()
test_sym_pass(1000)
# test_nnvm_pass(10)
