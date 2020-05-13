import tvm
from tvm import relay
from tvm.relay.build_module import BuildConfig
from tvm.relay.backend import graph_runtime_codegen as _graph_gen
import mxnet as mx
from mxnet.gluon import nn
from mxnet import ndarray as nd

import nnvm
from tvm.contrib import graph_runtime

from functools import reduce
import numpy as np
import math
import struct
import inspect
import os

import utils
import sym_pass as spass
import sym_utils as sutils

shift_bits_dict = {
    # 'cv0_sb': relay.const(8, dtype='int32')
}

const_var_dict = {}
def mx_const(number):
    name = 'const_var' + str(number)
    if name not in const_var_dict:
        const_var_dict[name] = {
            'symbol': mx.sym.var(name, shape=(1,)),
            'number': number
        }

    return const_var_dict[name]['symbol']

def divide(data, deno):
    out = mx.sym.broadcast_div(data, deno)
    return mx.sym.fix(out)

def quantize(data, shift_bits):
    """Quantize output of layer, to be consistent with source code @yx

    Question: should the shift_bits participating to network control flow?
            At mxnet quantization with truman's code, the bits number of max_v
            is converted to normal interger using function `asscalar()`. However,
            I cannot find the related function in relay.
            I am confused with the control flow logic in model network, whether
            the condition `shift_bits == -1` should join in model network or just
            left it in python code flow. By Longtao.Wang

    Parameters
    ----------
    shift_bits: tvm.relay.Expr
        The shift_bits parameter is never used according to @yx's source code,
        which always be constant Expr(-1).
    """
    abs_data = mx.sym.abs(data)
    max_v = mx.sym.max(abs_data)

    total_bits = mx.sym.log2(max_v)
    total_bits = mx.sym.ceil(total_bits)

    shift_bits = mx.sym.broadcast_sub(total_bits, mx_const(7))
    shift_bits = mx.sym.maximum(shift_bits, mx_const(0))
    denominator = mx.sym.pow(2, shift_bits)
    out = divide(data, denominator)
    out = mx.sym.clip(out, a_min=-128, a_max=127)
    return out, max_v, None, shift_bits

def make_conv_relu(data, kernel_size, padding, strides, channels,
        prefix="conv", skip_relu=False, skip_quantize=False):
    prefix = "_conv_" + prefix
    weight = mx.sym.var(prefix+'_weight')
    bias = mx.sym.var(prefix+'_bias')

    out = mx.sym.Convolution(data, weight, bias, kernel=kernel_size,
                          pad=padding, stride=strides,
                          num_filter=channels, no_bias=False)

    if not skip_quantize:
        out, max_v, min_v, shift_bits = quantize(out, None)
    if not skip_relu:
        out = mx.sym.relu(out)
    return out, None, None, None

def make_max_pool(data):
    out = mx.sym.Pooling(data, kernel=(2, 2), stride=(2, 2), pool_type='max')
    return out

def make_dense(data, units, prefix="dense", skip_quantize=False):
    prefix = "_dense_" + prefix
    weight = mx.sym.var(prefix+'_weight')
    bias = mx.sym.var(prefix+"_bias")
    out = mx.sym.FullyConnected(data, weight, bias, num_hidden=units)
    if not skip_quantize:
        out, _,_,_ = quantize(out, None)
    return out, None, None, None

def make_mnist_graph():
    data = mx.sym.var('data')
    out, _, _, sb0 = make_conv_relu(data, (3, 3), (1, 1), (1, 1), 32, "cv0")
    out, max_v, min_v, sb1 = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 32, "cv1")

    mp = make_max_pool(out)
    out, _,_,_ = make_conv_relu(mp, (1, 1), (0, 0), (1, 1), 32, "cv2")
    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 32, "cv3")

    out = divide(out, mx_const(2))
    mp = divide(mp, mx_const(2))
    out = out + mp
    out = make_max_pool(out)

    out = mx.sym.flatten(out)
    out, _, _, _ = make_dense(out, 256, "dense0")
    out = mx.sym.relu(out)
    out, max_v, min_v, sb = make_dense(out, 10, "dense1", skip_quantize=True)

    return out

def make_dog_cat_graph():
    data = relay.var("data", relay.TensorType((1, 3, 224, 224), "int8"))
    out, _,_,_ = make_conv_relu(data, (3, 3), (1, 1), (1, 1), 64, "cv0")
    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 64, "cv1")
    out = make_max_pool(out)

    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 128, "cv2")
    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 128, "cv3")
    out = make_max_pool(out)

    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 256, "cv4")
    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 256, "cv5")
    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 256, "cv6")
    out = make_max_pool(out)

    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 512, "cv7")
    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 512, "cv8")
    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 512, "cv9")
    out = make_max_pool(out)

    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 512, "cv10")
    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 512, "cv11")
    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 512, "cv12")
    out = make_max_pool(out)

    out = relay.nn.batch_flatten(out).astype('int8')
    out, _,_,_ = make_dense(out, 1, "dense0")

    out = relay.Function(relay.ir_pass.free_vars(out), out)
    return out

def load_parameters(graph, infer_shapes, params_name):
    args = graph.list_inputs()
    print ('args = ', args)

    # Load int8 parameters from params file
    data = np.fromfile(params_name, dtype=np.int8)
    count = 0
    params_list = {}
    for idx, arg in enumerate(args):
        if arg == 'data':
            pass
        elif arg in const_var_dict:
            params_list[arg] = nd.array([const_var_dict[arg]['number']])
        else:
            # Weight size is batch*channel*size*size
            # Bias size is batch
            shape = infer_shapes[arg]
            # print ('shape = ', shape, arg, idx)
            assert len(shape) > 0, "parameter size should be 1 at least: " + str(arg)
            size = int(reduce((lambda x, y: x * y), shape))
            print (arg, ": ", size, data[count:count+2], shape)

            # Please refer to source code @yx for params shape details.
            # Current data format is (batch, channel, size, size)'s tensor.
            params = np.array(data[count:count+size], dtype='int32') \
                    .reshape([int(x) for x in shape])

            # Within @yx's code, weight params at dense layer is convert(weight),
            # where convert() is:
            #   weight = weight.reshape(reversed(weight.shape)).transpose()
            #
            # According to code at file `infernet/src/int_connected_layer.c`,
            # the dense layer was implemented falsely with expression as below:
            #   Y = X * W + B, instead of correctly format Y = W * X + B
            # which Y is output, X is input, W is weight and B is bias.
            # For more details of matrix computation in c source code, please refer to
            # file `infernet/src/trivial_mul_kernels.cu`, which leads to weight params 
            # in python code should be transformed with convert() function.
            if arg.startswith("_dense_"):
                params = params.reshape(list(reversed(list(params.shape)))).transpose()
                # print ('dense transpose = ', params.flatten()[0:2])

            if arg.endswith("_bias"):
                params = params.astype("int32")

            if arg.startswith("_dense_") and arg.endswith("_weight"):
                params = params.astype("int32")

            params_list[arg] = nd.array(params, dtype='float32')
            count += size

    print ("Parameters length: ", len(data), count)

    return graph, params_list

def test_yxnet_mnist():
    mnist_sym = make_mnist_graph()

    inputs_ext = { 'data': {
        'shape': (1, 1, 28, 28),
        'precision': 8,
    } }
    in_shape = (1, 1, 28, 28)
    arg_shapes, _, aux_shapes = mnist_sym.infer_shape(data=in_shape)
    args, auxs = mnist_sym.list_arguments(), mnist_sym.list_auxiliary_states()
    infer_shapes = {args[i]:arg_shapes[i] for i in range(len(args))}
    infer_shapes.update({auxs[i]:aux_shapes[i] for i in range(len(auxs))})

    root = "/home/serving/warehouse"
    _, bd = load_parameters(mnist_sym, infer_shapes,
            root + "/ca3d0286d5758697cdef653c1375960a868ac08a/data/params")
    mnist_sym, bd = spass.mx_set_precs(mnist_sym, bd, inputs_ext)

    dump_sym, dump_par = '/tmp/mnist_yxnet.symbol', '/tmp/mnist_yxnet.params'
    with open(dump_sym, 'w') as fout:
        fout.write(mnist_sym.tojson())
    nd.save(dump_par, bd)

    inputs = [mx.sym.var('data')]
    data = np.load(root + '/ba9fedfc87ccb6064fcd437fd2287f5edef1bd84/data')
    data = nd.array([data.astype(np.int8)])

    if False:
        graph =  nn.SymbolBlock(mnist_sym, inputs)
        utils.load_parameters(graph, bd)
        res = graph.forward(data).astype('int32')
    else:
        prefix = "/tmp/yxnet/mnist"
        dump_sym, dump_params = prefix+".json", prefix+".params"
        print (sutils.sym_collect_attr(mnist_sym))
        spass.mxnet_to_nnvm(mnist_sym, bd, {
            'data': {
                'shape': (1, 1, 28, 28)
        }}, dump_sym, dump_params)
        exit()
    print (res.asnumpy().flatten()[:100])

def test_naive():
    data = relay.var("data", relay.TensorType((1, 3, 224, 224), "int8"))
    out = data
    prefix = "_conv_" + 'cv0'
    weight = relay.var(prefix+"_weight", dtype="int8")
    out = relay.nn.conv2d(data, weight, kernel_size=(3, 3),
                          padding=(1, 1), strides=(1, 1),
                          channels=64, out_dtype="int32")
    out = relay.nn.leaky_relu(out, alpha = 0.1)
    out = relay.Function(relay.ir_pass.free_vars(out), out)
    graph = out
    with relay.build_config(opt_level=0):
        func = graph
        func = relay.ir_pass.infer_type(func)
        func = relay.ir_pass.fuse_ops(func, 0)
        func = relay.ir_pass.infer_type(func)
        graph_gen = _graph_gen.GraphRuntimeCodegen(mod=None, target='llvm')
        graph_json, lowered_funcs, params = graph_gen.codegen(func)
        print (graph_json)

if __name__ == "__main__":
    utils.log_init()
    #  test_naive()
    test_yxnet_mnist()

    #  test_yxnet_dog_cat()

