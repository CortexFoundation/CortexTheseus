import mxnet as mx
from mxnet import ndarray as nd
import gluoncv as cv

import logging

import utils
import cvm_op
import sim_quant_helper as sim
import dataset as ds
import sym_pass as spass
import mrt as _mrt
import gluon_zoo as zoo

def load_fname(version, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    fname = "./data/resnet%s%s"%(version, suffix)
    return utils.extend_fname(fname, with_ext)

version = "18_v1"
def save_model(): # load original model from network
    sym_fname, param_fname = load_fname(version)
    name = "resnet" + version
    net = cv.model_zoo.get_model(name, pretrained=True,
            ctx=mx.gpu())
    sym = net(mx.sym.var('data'))
    with open(sym_fname, "w") as fout:
        fout.write(sym.tojson())
    net.collect_params().save(param_fname)

def test_sym_pass(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.sym.pass")
    calib_ctx = mx.gpu(2)
    ctx = mx.gpu(2)
    inputs_ext = { 'data': {
            'shape': (batch_size, 3, 224, 224),
    } }
    inputs = [mx.sym.var(name) for name in inputs_ext]

    logger.info("load dataset, symbol and parameters")
    # load dataset and iter function
    data_iter = ds.load_imagenet_rec(batch_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    data, _ = data_iter_func()

    # load original model for accuracy
    net1 = utils.load_model(*load_fname(version), inputs, ctx=ctx)
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    def resnet(data, label):
        res = net1.forward(data.as_in_context(ctx))
        acc_top1.update(label, res)
        _, top1 = acc_top1.get()
        acc_top5.update(label, res)
        _, top5 = acc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

    # load original model
    sym_fname, param_fname = load_fname(version)
    sym, params = mx.sym.load(sym_fname), nd.load(param_fname)
    sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)

    # quantize process
    mrt = _mrt.MRT(sym, params, inputs_ext)     # initialize
    mrt.set_data('data', data)                  # set input data
    mrt.calibrate(ctx=calib_ctx)                # calibration
    mrt.set_output_prec(8)                      # set output prec, do nothing by default
    qsym, qparams, inputs_ext = mrt.quantize()  # quantization

    if True:
        # dump quantized model
        dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
        sim.save_ext(dump_ext, inputs_ext)
        nd.save(dump_params, qparams)
        open(dump_sym, "w").write(qsym.tojson())

        # convert to cvm executor model
        inputs_ext['data']['shape'] = (1, 3, 224, 224)
        nnvm_sym, nnvm_params = spass.mxnet_to_nnvm(qsym, qparams, inputs_ext)
        spass.cvm_build(nnvm_sym, nnvm_params, inputs_ext, *load_fname(version, "nnvm"))

    # load quantized model for accuracy
    net3 = mx.gluon.nn.SymbolBlock(qsym, inputs)
    utils.load_parameters(net3, qparams, ctx=ctx)
    qacc_top1 = mx.metric.Accuracy()
    qacc_top5 = mx.metric.TopKAccuracy(5)
    qacc_top1.reset()
    qacc_top5.reset()
    def cvm_quantize(data, label):
        data = sim.load_real_data(data, 'data', inputs_ext)
        res = net3.forward(data.as_in_context(ctx))
        qacc_top1.update(label, res)
        _, top1 = qacc_top1.get()
        qacc_top5.update(label, res)
        _, top5 = qacc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

    # compare accuracy between models
    utils.multi_validate(resnet, data_iter_func,
            cvm_quantize,
            iter_num=iter_num, logger=logger)

if __name__ == "__main__":
    utils.log_init()

    save_model()
    test_sym_pass(batch_size=16, iter_num=100)



