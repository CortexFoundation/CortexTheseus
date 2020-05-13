import mxnet as mx

import utils
from model_utils import validate_model
from gluon_zoo import save_mobilenet1_0
from from_tensorflow import tf_dump_model

from os import path

def test_tf_resnet50_v1():
    sym_path = "./data/tf_resnet50_v1.json"
    prm_path = "./data/tf_resnet50_v1.params"
    # if not path.exists(sym_path) or not path.exists(prm_path):
    if True:
        tf_dump_model("resnet50_v1")
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx)

def test_tf_mobilenet():
    sym_path = "./data/tf_mobilenet.json"
    prm_path = "./data/tf_mobilenet.params"
    # if not path.exists(sym_path) or not path.exists(prm_path):
    if True:
        tf_dump_model("mobilenet")
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx)

def test_mobilenet1_0():
    sym_path = "./data/mobilenet1_0.json"
    prm_path = "./data/mobilenet1_0.params"
    if not path.exists(sym_path) or not path.exists(prm_path):
        save_mobilenet1_0()
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx, iter_num=999999, dump_model=True)

def test_mobilenet_v2_1_0():
    sym_path = "./data/mobilenetv2_1.0.json"
    prm_path = "./data/mobilenetv2_1.0.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx)

def test_tf_inceptionv3():
    sym_path = "./data/tf_inception_v3.json"
    prm_path = "./data/tf_inception_v3.params"
    if not path.exists(sym_path) or not path.exists(prm_path):
        tf_dump_model("inception_v3")
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    # validate_model(sym_path, prm_path, ctx, input_size=299, dump_model=True)
    validate_model(sym_path, prm_path, ctx, input_size=299, iter_num=99999999)

def test_alexnet():
    sym_path = "./data/alexnet.json"
    prm_path = "./data/alexnet.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    # validate_model(sym_path, prm_path, batch_size=700, ctx=ctx, dump_model=True)
    validate_model(sym_path, prm_path, batch_size=700, ctx=ctx, iter_num=9999999)

def test_cifar10_resnet20_v1():
    sym_path = "./data/cifar_resnet20_v1.json"
    prm_path = "./data/cifar_resnet20_v1.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    # validate_model(sym_path, prm_path, ctx, input_size=32,
    #                ds_name='cifar10', dump_model=True)
    validate_model(sym_path, prm_path, ctx, input_size=32,
                   ds_name='cifar10', iter_num=9999999)

def test_resnet(suffix):
    sym_path = "./data/resnet" + suffix + ".json"
    prm_path = "./data/resnet" + suffix + ".params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    # validate_model(sym_path, prm_path, ctx, lambd=16, dump_model=True)
    validate_model(sym_path, prm_path, ctx, lambd=16, iter_num=999999)

def test_densenet161():
    sym_path = "./data/densenet161.json"
    prm_path = "./data/densenet161.params"
    ctx = [mx.gpu(int(i)) for i in "1,2,3,4,5".split(',') if i.strip()]
    # validate_model(sym_path, prm_path, ctx, batch_size=16, dump_model=True)
    validate_model(sym_path, prm_path, ctx, batch_size=16, iter_num=9999999)

def test_qd10_resnetv1_20():
    sym_path = "./data/quick_raw_qd_animal10_2_cifar_resnet20_v2.json"
    prm_path = "./data/quick_raw_qd_animal10_2_cifar_resnet20_v2.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    # validate_model(sym_path, prm_path, ctx, num_channel=1,
    #         input_size=28, ds_name='quickdraw', dump_model=True)
    validate_model(sym_path, prm_path, ctx, num_channel=1,
            input_size=28, ds_name='quickdraw', iter_num=999999)

def test_shufflenet_v1():
    sym_path = "./data/shufflenet_v1.json"
    prm_path = "./data/shufflenet_v1.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    # validate_model(sym_path, prm_path, ctx, dump_model=True)
    validate_model(sym_path, prm_path, ctx, iter_num=9999999)

def test_squeezenet():
    sym_path = "./data/squeezenet1.0.json"
    prm_path = "./data/squeezenet1.0.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    # validate_model(sym_path, prm_path, ctx, batch_size=60, dump_model=True)
    validate_model(sym_path, prm_path, ctx, batch_size=60, iter_num=9999999)

def test_vgg19():
    sym_path = "./data/vgg19.json"
    prm_path = "./data/vgg19.params"
    ctx = [mx.gpu(int(i)) for i in "3".split(',') if i.strip()]
    # validate_model(sym_path, prm_path, ctx, dump_model=True)
    validate_model(sym_path, prm_path, ctx, iter_num=999999)

def test_quickdraw():
    sym_path = "./data/quickdraw_wlt_augmentation_epoch-4-0.8164531394275162.json"
    prm_path = "./data/quickdraw_wlt_augmentation_epoch-4-0.8164531394275162.params"
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    # validate_model(sym_path, prm_path, ctx, input_size=28, num_channel=1,
    #                ds_name="quickdraw", dump_model=True)
    validate_model(sym_path, prm_path, ctx, input_size=28, num_channel=1,
                   ds_name="quickdraw", iter_num=9999999)

def test_trec():
    sym_path = "./data/trec.json"
    prm_path = "./data/trec.params"
    ctx = [mx.gpu(int(i)) for i in "3".split(',') if i.strip()]
    validate_model(sym_path, prm_path, ctx, ds_name="trec",
                   input_shape=(38, 16), input_prec=16,
                   dump_model=True, dump_shape=(38, 1))
    # validate_model(sym_path, prm_path, ctx, ds_name="trec",
    #                input_shape=(38, 16), iter_num=999999)

def test_tf_densenet_lite():
    sym_path = "./data/tf_densenet_lite.json"
    prm_path = "./data/tf_densenet_lite.params"
    ctx = [mx.gpu(int(i)) for i in "1,2,3,4,5".split(',') if i.strip()]
    # validate_model(sym_path, prm_path, ctx, batch_size=16, dump_model=True)
    validate_model(sym_path, prm_path, ctx, batch_size=16, iter_num=10)

def test_tf_inceptionv3_lite():
    sym_path = "./data/tf_inception_v3_lite.json"
    prm_path = "./data/tf_inception_v3_lite.params"
    if not path.exists(sym_path) or not path.exists(prm_path):
        tf_dump_model("inception_v3")
    ctx = [mx.gpu(int(i)) for i in "4".split(',') if i.strip()]
    # validate_model(sym_path, prm_path, ctx, input_size=299, dump_model=True)
    validate_model(sym_path, prm_path, ctx, input_size=299, iter_num=10)

if __name__ == '__main__':
    utils.log_init()

    # TODO: test tfmodels
    # test_tf_mobilenet()           # 68% --> 8%, maybe due to pad
    # test_tf_resnet50_v1()         # 0% --> 0%
    # test_tf_densenet_lite()
    test_tf_inceptionv3_lite()

    # test_mobilenet1_0()
    '''
    2020-01-10 15:34:15
    top1: 70.76% --> 63.08%
    top5: 89.97% --> 85.02%
    Iteration: 3123
    Total Sample: 49984
    '''

    # test_mobilenet_v2_1_0()       # 73% --> 0%

    # test_tf_inceptionv3()
    '''
    2020-01-10 16:08:03
    top1: 55.57% --> 53.74%
    top5: 77.56% --> 76.01%
    Iteration: 3123
    Total Sample: 49984
    '''

    # test_alexnet()
    '''
    2020-01-10 16:23:24
    top1: 55.92% --> 55.15%
    top5: 78.74% --> 78.20%
    Iteration: 70
    Total Sample: 49700
    '''

    # test_cifar10_resnet20_v1()
    '''
    2020-01-10 16:37:35
    top1: 92.88% --> 92.83%
    top5: 99.78% --> 99.75%
    Iteration: 623
    Total Sample: 9984
    '''

    # test_resnet("50_v1")
    '''
    2020-01-10 17:04:50
    top1: 77.38% --> 75.81%
    top5: 93.58% --> 93.06%
    Iteration: 3123
    Total Sample: 49984
    '''

    # test_resnet("18_v1")
    '''
    2020-01-10 16:55:48
    top1: 70.94% --> 70.14%
    top5: 89.92% --> 89.54%
    Iteration: 3123
    Total Sample: 49984
    '''

    # test_resnet("50_v1d_0.86")    # not valid: Pooling count_include_pad:True

    # test_resnet("18_v1b_0.89")
    '''
    2020-01-10 17:00:43
    top1: 67.20% --> 63.82%
    top5: 87.45% --> 85.60%
    Iteration: 3123
    Total Sample: 49984
    '''

    # test_resnet("50_v2")
    '''
    2020-01-10 17:29:01
    top1: 77.15% --> 74.13%
    top5: 93.44% --> 91.76%
    Iteration: 3123
    Total Sample: 49984
    '''

    # test_densenet161()
    '''
    2020-01-10 20:33:58
    top1: 77.61% --> 77.32%
    top5: 93.82% --> 93.62%
    Iteration: 3127
    Total Sample: 49984
    '''

    # test_qd10_resnetv1_20()
    '''
    2020-01-10 17:57:44
    top1: 85.72% --> 85.73%
    top5: 98.71% --> 98.70%
    Iteration: 17330
    Total Sample: 277296
    '''

    # test_shufflenet_v1()
    '''
    2020-01-10 17:34:01
    top1: 63.48% --> 60.38%
    top5: 85.11% --> 82.88%
    Iteration: 3123
    Total Sample: 49984
    '''

    # test_squeezenet()
    '''
    2020-01-10 17:26:18
    top1: 57.20% --> 54.49%
    top5: 80.03% --> 77.86%
    Iteration: 832
    Total Sample: 49980
    '''

    # test_vgg19()
    '''
    2020-01-10 17:40:53
    top1: 74.12% --> 73.68%
    top5: 91.77% --> 91.66%
    Iteration: 3123
    Total Sample: 49984
    '''

    # test_quickdraw()
    '''
    2020-01-10 16:39:51
    top1: 81.66% --> 81.57%
    top5: 98.22% --> 98.20%
    Iteration: 17330
    Total Sample: 277296
    '''

    # test_trec()
    '''
    2020-01-10 
    top1: --> 
    top5: --> 
    Iteration: 
    Total Sample: 
    '''




