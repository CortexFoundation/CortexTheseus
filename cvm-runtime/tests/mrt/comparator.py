import tensorflow as tf
from tensorflow.core.framework import graph_pb2
# from tensorflow_parser import TFParser

import mxnet as mx
from mxnet import ndarray as nd

import sym_utils as sutils
import dataset as ds
# import from_tensorflow as ftf
import tfm_pass as tpass
import sim_quant_helper as sim
import cvm_op

import sys
import numpy as np
import logging
import os
from os import path

from tfm_pass import convert_params_dtype

def get_mxnet_outs(symbol, params, data, check_point, ctx=mx.cpu()):
    data = nd.array(data)
    _, deps = sutils.topo_sort(symbol, with_deps=True)
    out_cache, ans = {}, {}

    def _impl(op, params, graph, **kwargs):
        deps = kwargs['deps']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sutils.sym_iter(op.get_children()), op.list_attr()

        if op_name == 'null':
            out = data if sutils.is_inputs(op, params) else params[name]
        elif childs is None:
            out = sutils.get_nd_op(op_name)(**attr)
        else:
            cinfos = [(c.attr('name'), sutils.get_entry_id(c)) for c in childs]
            nd_inputs = [out_cache[n[0]][n[1]] for n in cinfos]
            out = sutils.get_nd_op(op_name)(*nd_inputs, **attr)
            for n, _ in cinfos:
                assert n in deps
                if name not in deps[n]:
                    # for op like: op = broadcast_mul(X, X)
                    # `cinfos` will have duplicate entries
                    # avoid removing more than once
                    continue
                deps[n].remove(name)
                if len(deps[n]) == 0:
                    del out_cache[n]
        if name == check_point:
            ans[check_point] = out
        out = [out] if len(op) == 1 else out
        out_cache[name] = [o.as_in_context(ctx) for o in out]

    print("itermediate result calculated from start_point: `data` to check_point: `%s`"%check_point)
    sutils.topo_visit_transformer(symbol, params, _impl, deps=deps, data=data)
    out_cache.clear()
    # res = ans[check_point].asnumpy()
    res = ans[check_point]
    return res

def get_tensorflow_outs(model, data, check_point, start_point):
    with tf.Session() as sess:
        graph_def = graph_pb2.GraphDef()
        with open(model, "rb") as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def)
        name_check = "import/"+check_point+":0"
        name_start = "import/"+start_point+":0"
        output_tensor = sess.graph.get_tensor_by_name(name_check)
        print("start calculating itermediate from start_point: `%s` to check_point: `%s`" \
            %(start_point, check_point))
        res = sess.run(output_tensor, {name_start: data})
    return res

def calculate_res(data1, data2, ctx=mx.cpu()):
    assert data1.shape == data2.shape
    ndims = np.product(data1.shape)
    data1 = np.reshape(data1, (ndims,))
    data2 = np.reshape(data2, (ndims,))
    res = data1-data2
    norm1 = np.linalg.norm(data1)
    norm2 = np.linalg.norm(data2)
    norm_res = np.linalg.norm(res)
    print(norm1, norm2, norm_res)

def calculate_norm(x, y):
    assert x.shape == y.shape
    ndims = np.product(x.shape)
    x = nd.reshape(x, shape=(ndims,))
    y = nd.reshape(y, shape=(ndims,))
    res = x-y
    nx = nd.norm(x)
    ny = nd.norm(y)
    nr = nd.norm(res)
    print("saving...")
    f = "/home/ryt/data/cmp_"
    names = ["nx", "ny", "nr"]
    objs = [nx, ny, nr]
    for obj in objs:
        print(type(obj), obj.shape)
    for i in range(3):
        nd.save(f+names[i], objs[i])
    print('success')

def get_metric(outs, label):
    outs = nd.array(outs)
    label = nd.array(label)
    metric = mx.metric.Accuracy()
    metric.update(label, outs)
    _, top1 = metric.get()
    print("model precision: ", top1)

def load_data_1(input_size=224, batch_size=1, layout='NHWC'):
    ds_name = 'imagenet'
    data_iter_func = ds.data_iter(ds_name, batch_size, input_size=input_size)
    data, label = data_iter_func()
    data = data.asnumpy()
    if layout == 'NHWC':
        data = np.transpose(data, axes=[0,2,3,1])
    print('data loaded with shape: ', data.shape)
    return data, label

def load_data_2(input_size=224, batch_size=1, redownload=False, layout='NHWC'):
    if redownload:
        _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
        path_to_zip = tf.keras.utils.get_file(
            'cats_and_dogs.zip', origin=_URL, extract=True)
        PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
    inps = []
    for i in range(batch_size):
        img = os.path.expanduser(os.path.join(
            '~/.keras/datasets/cats_and_dogs_filtered/validation/cats',
            'cat.'+str(2000+i)+'.jpg'))
        image = tf.keras.preprocessing.image.load_img(img, target_size=(input_size, input_size))
        inp = tf.keras.preprocessing.image.img_to_array(image)/255.
        inp = np.expand_dims(inp, axis=0)
        inps.append(inp)
    data = np.concatenate(inps, axis=0) if batch_size > 1 else inps[0]
    if layout == 'NCHW':
        data = np.transpose(data, axes=[0,3,1,2])
    print('data loaded with shape:', data.shape)
    return data

def load_data_3(modelname, input_size=224, batch_size=1, layout='NHWC', quantized=False):
    if quantized:
        data = nd.load(path.expanduser('~/data/' + modelname + '_qdata'))[0].asnumpy()
    else:
        data = nd.load(path.expanduser('~/data/' + modelname + '_data'))[0].asnumpy()
    label = nd.load(path.expanduser('~/data/' + modelname + '_label'))[0].asnumpy()
    if layout == 'NHWC':
        data = np.transpose(data, axes=[0,2,3,1])
    return data, label

mx_dir = path.expanduser("~/data")

def cmp_model(modelname, batch_size=160, revise_outs=False,
              tf_check_point=None, mx_check_point=None,
              tf_start_point=None):

    input_size = model_input_size[modelname]

    # mxnet model
    symbol_file = path.join(mx_dir, "tf_"+modelname+".json")
    params_file = path.join(mx_dir, "tf_"+modelname+".params")
    ftf.tf_dump_model(modelname, revise_outs=revise_outs)
    symbol = mx.sym.load(symbol_file)
    params = mx.nd.load(params_file)
    if mx_check_point is None:
        mx_check_point = sutils.topo_sort(symbol)[-1].attr('name')
    mx_data, mx_label = load_data_1(
        batch_size=batch_size, input_size=input_size, layout='NCHW')
    mx_outs = get_mxnet_outs(symbol, params, mx_data, mx_check_point)
    print('mxnet check with shape: ', mx_outs.shape)

    # evaluate
    if revise_outs:
        get_metric(mx_outs, mx_label)
        return

    # tensorflow model
    model = ftf.modelfile[modelname]
    tfparser = TFParser(model)
    tfgraph = tfparser.parse()
    if tf_start_point is None:
        for tfnode in ftf.topo_sort(tfgraph):
            name, op_name = tfnode.name, tfnode.op
            if op_name in ftf.default_tf_start_types:
                tf_start_point = name
    if tf_check_point is None:
        tf_check_point = ftf.topo_sort(tfgraph)[-1].name
    tf_data, tf_label = load_data_1(
        batch_size=batch_size, input_size=input_size, layout='NHWC')
    tf_outs = get_tensorflow_outs(
        model, tf_data, tf_check_point, tf_start_point)
    print("tensorflow check with shape: ", tf_outs.shape)

    # compare the result between tf and mx
    calculate_res(mx_outs, tf_outs)

def run_mx(modelname, batch_size=160, quantized=False,
           mx_check_point=None, float_dtype="float32",
           evaluate=False):
    input_size = model_input_size[modelname]
    suffix = ".mrt.quantize" if quantized else ""
    symbol_file = path.join(mx_dir, modelname+suffix+".json")
    params_file = path.join(mx_dir, modelname+suffix+".params")
    symbol = mx.sym.load(symbol_file)
    params = mx.nd.load(params_file)
    params = tpass.convert_params_dtype(params, dest_dtype="float32")
    if mx_check_point is None:
        mx_check_point = sutils.topo_sort(symbol)[-1].attr('name')
    mx_data, mx_label = load_data_3(
        modelname, batch_size=batch_size, input_size=input_size,
        layout='NCHW', quantized=quantized)
    if quantized:
        ext_file = path.join(mx_dir, modelname+suffix+".ext")
        _, _, _, scales = sim.load_ext(ext_file)
    else:
        op_names = tpass.collect_op_names(symbol, params)
        print(op_names)
        scales = None
    mx_outs = get_mxnet_outs(symbol, params, mx_data, mx_check_point)
    if scales and mx_check_point in scales:
        scale_factor = scales[mx_check_point]
        print('multiply scale factor: ', scale_factor)
        mx_outs /= scale_factor
    print('mxnet check with shape: ', mx_outs.shape)
    if evaluate:
        get_metric(mx_outs, mx_label)
    return mx_outs

def cmp_quantize(modelname, evaluate=False):
    mx_check_point = mx_check_points[modelname]
    mxq_check_point = mxq_check_points[modelname]

    # mxnet org model
    mx_outs = run_mx(modelname, mx_check_point=mx_check_point, evaluate=evaluate)
    print('\n\n\n\n\n\n')

    # mxnet quantized model
    mxq_outs = run_mx(
        modelname, quantized=True, mx_check_point=mxq_check_point, evaluate=evaluate)
    print('\n\n\n\n\n\n')

    # compare the result between mx and mxq
    # calculate_res(mx_outs, mxq_outs)
    calculate_norm(mx_outs, mxq_outs)
    exit()

model_input_size = {
                "resnet50_v1_new": 224,
                "inception_v3": 299,
                "mobilenet": 224,
                "densenet_lite": 224,
                "inception_v3_lite": 299,
                "mobilenet_v1_0.25_128_lite": 128,
                "mobilenet_v1_0.25_224_lite": 224,
                "mobilenet_v1_0.50_128_lite": 128,
                "mobilenet_v1_0.50_192_lite": 192,
                "mobilenet_v1_1.0_224_lite": 224,
                "mobilenet_v2_1.0_224_lite": 224,
                "resnet_v2_101_lite": 224,
                "ssd_512_vgg16_atrous_voc": 512,
            }

mx_check_points = {
                    "resnet50_v1_new": "softmax0",
                    "inception_v3": None,
                    "mobilenet": None,
                    "densenet_lite": None,
                    "inception_v3_lite": "broadcast_add0",
                    "mobilenet_v1_0.25_128_lite": None,
                    "mobilenet_v1_0.25_224_lite": None,
                    "mobilenet_v1_0.50_128_lite": None,
                    "mobilenet_v1_0.50_192_lite": None,
                    "mobilenet_v1_1.0_224_lite": "transpose5",
                    "mobilenet_v2_1.0_224_lite": "slice_axis0",
                    "resnet_v2_101_lite": None,
                    "ssd_512_vgg16_atrous_voc": "ssd0_vggatrousextractor0_broadcast_mul0",
                }

mxq_check_points = {
                    "resnet50_v1_new": None,
                    "inception_v3": None,
                    "mobilenet": None,
                    "densenet_lite": None,
                    "inception_v3_lite": "broadcast_add0",
                    "mobilenet_v1_0.25_128_lite": None,
                    "mobilenet_v1_0.25_224_lite": None,
                    "mobilenet_v1_0.50_128_lite": None,
                    "mobilenet_v1_0.50_192_lite": None,
                    "mobilenet_v1_1.0_224_lite": "clip0",
                    "mobilenet_v2_1.0_224_lite": "slice_axis0",
                    "resnet_v2_101_lite": None,
                    "ssd_512_vgg16_atrous_voc": "ssd0_vggatrousextractor0_broadcast_mul0",
                }

tf_check_points = {
                    "resnet50_v1_new": "fc1000_1/Softmax",
                    "inception_v3": "predictions_1/Softmax",
                    "mobilenet": 'act_softmax_1/Softmax',
                    "densenet_lite": "softmax_tensor",
                    "inception_v3_lite": None,
                    "mobilenet_v1_0.25_128_lite": None,
                    "mobilenet_v1_0.25_224_lite": None,
                    "mobilenet_v1_0.50_128_lite": None,
                    "mobilenet_v1_0.50_192_lite": None,
                    "mobilenet_v1_1.0_224_lite": None,
                    "mobilenet_v2_1.0_224_lite": None,
                    "resnet_v2_101_lite": None,
                }

tf_start_points = {
                    "resnet50_v1_new": "input_1_1",
                    "inception_v3": "input_1_1",
                    "mobilenet": "input_1_1",
                    "densenet_lite": "Placeholder",
                    "inception_v3_lite": None,
                    "mobilenet_v1_0.25_128_lite": None,
                    "mobilenet_v1_0.25_224_lite": None,
                    "mobilenet_v1_0.50_128_lite": None,
                    "mobilenet_v1_0.50_192_lite": None,
                    "mobilenet_v1_1.0_224_lite": None,
                    "mobilenet_v2_1.0_224_lite": None,
                    "resnet_v2_101_lite": "input",
                }


'''
if __name__ == "__main__":
    assert len(sys.argv) >= 2, "Please enter at least 2 python arguments."
    modelname = sys.argv[1]
    revise_outs= False if len(sys.argv)<3 or sys.argv[2] == 'False' else True
    tf_check_point = tf_check_points[modelname]
    mx_check_point = mx_check_points[modelname]
    tf_start_point = tf_start_points[modelname]
    cmp_model(modelname, revise_outs=revise_outs,
              tf_check_point=tf_check_point,
              mx_check_point=mx_check_point,
              tf_start_point=tf_start_point)
'''
if __name__ == "__main__":
    assert len(sys.argv) >= 2, "Please enter at least 2 python arguments."
    modelname = sys.argv[1]
    evaluate = True if len(sys.argv) > 2 and sys.argv[2] == 'True' else False
    cmp_quantize(modelname, evaluate=evaluate)
