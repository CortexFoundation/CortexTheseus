import cvm
from cvm.runtime import CVMAPILoadModel, CVMAPIInference
from cvm.runtime import CVMAPIGetOutputLength, CVMAPIFreeModel
from cvm import utils

import os
import time

ctx = cvm.cpu()

# model_root = "/home/serving/tvm-cvm/data/jetson/"
# model_root = "/data/std_out/ssd_512_mobilenet1.0_coco_tfm/"
#  model_root = "/tmp/resnet18_v1_tfm/"
# model_root = "/data/mrt/ssd_512_mobilenet1.0_voc_tfm"
#  model_root = "/data/std_out/cvm_mnist/"
#  model_root = "/data/std_out/resnet50_v2"
# model_root = "/data/std_out/ssd"
# model_root = "/data/std_out/resnet50_mxg/"
# model_root = "/data/ryt/alexnet_tfm"
# model_root = "/data/ryt/ssd_512_vgg16_atrous_voc_tfm"
# model_root = "/data/ryt/ssd_300_vgg16_atrous_voc_tfm"
# model_root = "/data/ryt/cifar_resnet20_v1_tfm"
# model_root = "/data/ryt/ssd_512_resnet50_v1_voc_tfm"
# model_root = "/data/ryt/ssd_512_mobilenet1.0_voc_tfm"
# model_root = "/data/ryt/yolo3_darknet53_voc_tfm"
# model_root = "/data/ryt/tf_mobilenet_v1_1.0_224_lite_tfm"
# model_root = "/data/ryt/mobilenetv2_1.0_tfm"
# model_root = "/data/ryt/tf_mobilenet_v2_1.0_224_lite_tfm"
# model_root = "/data/ryt/tf_mobilenet_v1_0.25_128_lite_tfm"
# model_root = "/data/ryt/tf_mobilenet_v1_0.25_224_lite_tfm"
# model_root = "/data/ryt/tf_mobilenet_v1_0.50_128_lite_tfm"
# model_root = "/data/ryt/tf_mobilenet_v1_0.50_192_lite_tfm"
# model_root = "/data/ryt/tf_mobilenet_v1_0.50_192_lite_tfm"
model_root = "/data/ryt/tf_mobilenet_tfm"

json, params = utils.load_model(
        os.path.join(model_root, "symbol"),
        os.path.join(model_root, "params"))

net = CVMAPILoadModel(json, params, ctx=ctx)
print(CVMAPIGetOutputLength(net),
    cvm.runtime.CVMAPIGetOutputTypeSize(net))

data_path = os.path.join(model_root, "data.npy")
data = utils.load_np_data(data_path)

iter_num = 1
start = time.time()
for i in range(iter_num):
    out = CVMAPIInference(net, data)
    utils.classification_output(out)
    # utils.detection_output(out)
end = time.time()
print ("Infer Time: ", (end - start) * 1e3 / iter_num, " ms")

CVMAPIFreeModel(net)

# import numpy as np

# import mxnet as mx
# from mxnet import ndarray as nd
# import gluoncv as gluon

# from mrt import sim_quant_helper as sim
# from mrt.transformer import reduce_graph, Model

# data = nd.array(np.load(data_path))
# input_shape = data.shape
# # model_name = "ssd_512_mobilenet1.0_voc.all.quantize"
# # model_name = "tf_mobilenet_v1_1.0_224_lite.mrt.quantize"
# model_name = "mobilenetv2_1.0.mrt.quantize"
# sym_file = "/home/ryt/data/" + model_name + ".json"
# prm_file = "/home/ryt/data/" + model_name + ".params"
# ext_file = "/home/ryt/data/" + model_name + ".ext"
# _, _, _, inputs_ext, _, _ = sim.load_ext(ext_file)
# model = Model.load(sym_file, prm_file)
# model = reduce_graph(model, {'data': input_shape})
# ctx = [mx.gpu(0)]
# graph = model.to_graph(ctx=ctx)
# data = sim.load_real_data(data, 'data', inputs_ext)
# data = gluon.utils.split_and_load(
    # data, ctx_list=ctx, batch_axis=0, even_split=False)
# outs = [graph(d) for d in data]
# olen = len(model.symbol)
# if olen == 1:
    # outs = nd.concatenate(outs)
# else:
    # outs = [nd.concatenate([outs[i][j] \
        # for i in range(len(outs))]) for j in range(olen)]
# arr = [o.asnumpy().astype('int32').tolist()[0] for o in outs]
# outs = []
# for i in range(100):
    # nums = [arr[0][i][0]]+[arr[1][i][0]]+arr[2][i][:]
    # for num in nums:
        # outs.append(num)
# assert len(out) == len(outs)
# print(out)
# print(outs)
# res = [out[i]-outs[i] for i in range(len(out))]
# # print(res)




