from mxnet.gluon.model_zoo import vision
import mxnet as mx

import gluoncv as cv

def load_inception_v3(ctx):
    return vision.inception_v3(pretrained=True, ctx=ctx, prefix="")
def save_inception_v3():
    graph = load_inception_v3(mx.cpu())
    sym = graph(mx.symbol.Variable('data'))
    with open('./data/inception_v3.json', 'w') as fout:
        fout.write(sym.tojson())
    graph.save_params('./data/inception_v3.params')

def load_mobilenet1_0(ctx):
    return vision.mobilenet1_0(pretrained=True, ctx=ctx, prefix="")
def save_mobilenet1_0():
    graph = load_mobilenet1_0(mx.cpu())
    sym = graph(mx.symbol.Variable('data'))
    with open('./data/mobilenet1_0.json', 'w') as fout:
        fout.write(sym.tojson())
    graph.save_params('./data/mobilenet1_0.params')

def load_mobilenet_v2_1_0(ctx):
    return vision.mobilenet_v2_1_0(pretrained=True, ctx=ctx, prefix="")
def save_mobilenet_v2_1_0():
    graph = load_mobilenet_v2_1_0(mx.cpu())
    sym = graph(mx.sym.var('data'))
    with open('./data/mobilenet_v2_1_0.json', 'w') as fout:
        fout.write(sym.tojson())
    graph.save_parameters('./data/mobilenet_v2_1_0.params')

def load_resnet18_v1_yolo():
    return cv.model_zoo.get_model('yolo3_resnet18_v1_voc',
            pretrained=False, pretrained_base=True,
            ctx=mx.gpu())

def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    classes : int
        Number of classes for the output layer.

    Returns
    -------
    HybridBlock
        The model.
    """
    return cv.model_zoo.get_model(name, pretrained=True,
            ctx=mx.gpu(), **kwargs)

def save_model(name, sym_path=None, prm_path=None, **kwargs):
    net = get_model(name, **kwargs)
    sym = net(mx.sym.var('data'))
    if isinstance(sym, tuple):
        sym = mx.sym.Group([*sym])
    sym_path = sym_path if sym_path else "./data/%s.json"%name
    prm_path = prm_path if prm_path else "./data/%s.params"%name
    with open(sym_path, "w") as fout:
        fout.write(sym.tojson())
    net.collect_params().save(prm_path)

""" Model List
resnet18_v1, resnet34_v1, resnet50_v1, resnet101_v1, resnet152_v1, resnet18_v2, resnet34_v2, resnet50_v2, resnet101_v2, resnet152_v2,
se_resnet18_v1, se_resnet34_v1, se_resnet50_v1, se_resnet101_v1, se_resnet152_v1, se_resnet18_v2, se_resnet34_v2, se_resnet50_v2, se_resnet101_v2, se_resnet152_v2,
vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,
alexnet,
densenet121, densenet161, densenet169, densenet201,
squeezenet1.0, squeezenet1.1,
inceptionv3,
mobilenet1.0, mobilenet0.75, mobilenet0.5, mobilenet0.25, mobilenetv2_1.0, mobilenetv2_0.75, mobilenetv2_0.5, mobilenetv2_0.25,
ssd_300_vgg16_atrous_voc, ssd_300_vgg16_atrous_coco, ssd_300_vgg16_atrous_custom, ssd_512_vgg16_atrous_voc, ssd_512_vgg16_atrous_coco, ssd_512_vgg16_atrous_custom, ssd_512_resnet18_v1_voc, ssd_512_resnet18_v1_coco, ssd_512_resnet50_v1_voc, ssd_512_resnet50_v1_coco, ssd_512_resnet50_v1_custom, ssd_512_resnet101_v2_voc, ssd_512_resnet152_v2_voc, ssd_512_mobilenet1.0_voc, ssd_512_mobilenet1.0_coco, ssd_512_mobilenet1.0_custom,
faster_rcnn_resnet50_v1b_voc, faster_rcnn_resnet50_v1b_coco, faster_rcnn_fpn_resnet50_v1b_coco, faster_rcnn_fpn_bn_resnet50_v1b_coco, faster_rcnn_resnet50_v1b_custom, faster_rcnn_resnet101_v1d_voc, faster_rcnn_resnet101_v1d_coco, faster_rcnn_fpn_resnet101_v1d_coco, faster_rcnn_resnet101_v1d_custom,
mask_rcnn_resnet50_v1b_coco, mask_rcnn_fpn_resnet50_v1b_coco, mask_rcnn_resnet101_v1d_coco, mask_rcnn_fpn_resnet101_v1d_coco,
cifar_resnet20_v1, cifar_resnet56_v1, cifar_resnet110_v1, cifar_resnet20_v2, cifar_resnet56_v2, cifar_resnet110_v2,
cifar_wideresnet16_10, cifar_wideresnet28_10, cifar_wideresnet40_8,
cifar_resnext29_32x4d, cifar_resnext29_16x64d,
fcn_resnet50_voc, fcn_resnet101_coco, fcn_resnet101_voc, fcn_resnet50_ade, fcn_resnet101_ade,
psp_resnet101_coco, psp_resnet101_voc, psp_resnet50_ade, psp_resnet101_ade, psp_resnet101_citys,
deeplab_resnet101_coco, deeplab_resnet101_voc, deeplab_resnet152_coco, deeplab_resnet152_voc, deeplab_resnet50_ade, deeplab_resnet101_ade,
resnet18_v1b, resnet34_v1b, resnet50_v1b, resnet50_v1b_gn, resnet101_v1b_gn, resnet101_v1b, resnet152_v1b, resnet50_v1c, resnet101_v1c, resnet152_v1c, resnet50_v1d, resnet101_v1d, resnet152_v1d, resnet50_v1e, resnet101_v1e, resnet152_v1e, resnet50_v1s, resnet101_v1s, resnet152_v1s, resnext50_32x4d, resnext101_32x4d, resnext101_64x4d,
se_resnext50_32x4d, se_resnext101_32x4d, se_resnext101_64x4d,
senet_154,
darknet53,
yolo3_darknet53_coco, yolo3_darknet53_voc, yolo3_darknet53_custom,
yolo3_mobilenet1.0_coco, yolo3_mobilenet1.0_voc, yolo3_mobilenet1.0_custom,
nasnet_4_1056, nasnet_5_1538, nasnet_7_1920, nasnet_6_4032,
simple_pose_resnet18_v1b, simple_pose_resnet50_v1b, simple_pose_resnet101_v1b, simple_pose_resnet152_v1b, simple_pose_resnet50_v1d, simple_pose_resnet101_v1d, simple_pose_resnet152_v1d,
residualattentionnet56, residualattentionnet92, residualattentionnet128, residualattentionnet164, residualattentionnet200, residualattentionnet236, residualattentionnet452,
cifar_residualattentionnet56, cifar_residualattentionnet92, cifar_residualattentionnet452,
resnet18_v1b_0.89, resnet50_v1d_0.86, resnet50_v1d_0.48, resnet50_v1d_0.37, resnet50_v1d_0.11, resnet101_v1d_0.76, resnet101_v1d_0.73,
mobilenet1.0_int8,
resnet50_v1_int8,
ssd_300_vgg16_atrous_voc_int8,
ssd_512_mobilenet1.0_voc_int8,
SSD_512_RESNET50_V1_VOC_INT8,
SSD_512_VGG16_ATROUS_VOC_INT8
"""
