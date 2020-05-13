import tensorflow as tf
import mxnet as mx
from mxnet import gluon, ndarray as nd
import numpy as np

import from_tensorflow as ftf
import transformer as tfm
import tfm_pass as tpass
import sym_utils as sutils
import utils
import dataset as ds
import sim_quant_helper as sim

import logging
import os
from os import path

# root = "/data/tfmodels/lite/Mobilenet_V1_1.0_224/"
root = "/data/tfmodels/mobilenet/"
# root = "/data/tfmodels/inception_v3"
# root = "/data/tfmodels/resnet50_v1_new"

# model_name = "mobilenet_v1_1.0_224_frozen.pb"
model_name = "model.pb"

pb_path = path.join(root, model_name)

print (pb_path)

utils.log_init()
sym, params = ftf.convert_model(pb_path)
sym, params = ftf._fuse_pad(sym, params)

print(tpass.collect_op_names(sym, params))

input_size=224
dataset = ds.DS_REG["imagenet"]((16, 3, input_size, input_size))
iter_func = dataset.iter_func()
iter_num=10
# ctx = [mx.gpu(i) for i in (3,4,5,6)]
ctx = mx.gpu(3)

params = {k:v for k, v in params.items() if v.shape != ()}
model = tfm.Model(sym, params)

# op_maps = {c.attr('name'): c for c in sutils.topo_sort(sym, params)}
# name = 'softmax0'
# sym = op_maps[name]
# _, oshp, _ = sym.infer_shape()
# print ("INFER SHAPE: ", oshp)

# model = tfm.Model(sym, params)

# graph = model.to_graph(ctx=ctx)

# cat_img = os.path.expanduser(
    # "~/.keras/datasets/cats_and_dogs_filtered/validation/cats/cat.2000.jpg")
# image = tf.keras.preprocessing.image.load_img(cat_img, target_size=(224, 224))
# data = tf.keras.preprocessing.image.img_to_array(
        # image, data_format="channels_first")
# data = np.expand_dims(data, axis=0) / 255.

# label_path = "/data/tfmodels/lite/DenseNet/labels.txt"
# with open(label_path, "r") as f:
    # lines = f.readlines()
    # labels = {i:l for i, l in enumerate(lines)}

# print (data.shape)
# res = graph(nd.array(data).as_in_context(ctx)).asnumpy()
# print (res.shape)
# print (res.reshape((-1,))[:10])
# argmax = res.flatten().argmax()
# print (argmax, labels[argmax] if argmax < 1000 else None)
# print (res.flatten()[argmax])
# # np.save("/tmp/mxnet.batchnorm.npy", res)
# exit()



model.prepare()
model.save("/tmp/tf_model.json", "/tmp/tf_model.params")

graph = model.to_graph(ctx=ctx)
metric = dataset.metrics()
def eval_func(data, label):
    # data = gluon.utils.split_and_load(
        # data, ctx_list=ctx, batch_axis=0, even_split=False)
    # outs = [graph(d) for d in data]
    # outs = nd.concatenate(outs)
    outs = graph(data.as_in_context(ctx))
    acc = dataset.validate(metric, outs, label)
    return acc

print("Quantization")
print(tpass.collect_op_names(model.symbol, model.params))
mrt = model.get_mrt()
data, _ = iter_func()
mrt.set_data(data)
mrt.calibrate(ctx=mx.gpu(2), lambd=20)
mrt.set_threshold("mrt_rewrite_transpose26_0", 20)
mrt.set_threshold("mrt_rewrite_transpose6_0", 20)
#  mrt.set_threshold("mrt_rewrite_bias_2", 20)
#  mrt.set_threshold("mrt_rewrite_transpose16_0", 15)
#  mrt.set_threshold("mrt_rewrite_bias_5", 13)
#  mrt.set_threshold("mrt_rewrite_conv1_bn_1/FusedBatchNorm_0", 6)
#  mrt.set_threshold("mrt_rewrite_conv_dw_1_bn_1/FusedBatchNorm_0", 6)
#  mrt.set_threshold("mrt_rewrite_conv_pw_1_bn_1/FusedBatchNorm_0", 6)
#  mrt.set_threshold("mrt_rewrite_conv_dw_2_bn_1/FusedBatchNorm_0", 6)
#  mrt.set_threshold("mrt_rewrite_conv_pw_2_bn_1/FusedBatchNorm_0", 6)
#  mrt.set_threshold("mrt_rewrite_conv_dw_3_bn_1/FusedBatchNorm_0", 6)
#  mrt.set_threshold("mrt_rewrite_conv_pw_3_bn_1/FusedBatchNorm_0", 6)
#  mrt.set_threshold("mrt_rewrite_conv_dw_4_bn_1/FusedBatchNorm_0", 6)
#  mrt.set_threshold("mrt_rewrite_conv_pw_4_bn_1/FusedBatchNorm_0", 6)
mrt.set_input_prec(8)
mrt.quantize()
mrt.save("tf_mobilenet", datadir="/tmp")
qmodel = mrt.current_model
qgraph = qmodel.to_graph(ctx=ctx)

qmetric = dataset.metrics()
def quantize(data, label):
    data = sim.load_real_data(data, 'data', mrt.get_inputs_ext())
    outs = qgraph(data.as_in_context(ctx))
    acc = dataset.validate(qmetric, outs, label)
    return acc


utils.multi_validate(eval_func, iter_func,
        quantize,
        iter_num=iter_num)

