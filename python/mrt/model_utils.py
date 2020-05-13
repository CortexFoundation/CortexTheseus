import sys
import logging
from os import path
import numpy as np

import mxnet as mx
from mxnet import gluon, ndarray as nd

import tfm_pass as tpass
import dataset as ds
from transformer import Model, MRT # , init, compile_to_cvm
import sim_quant_helper as sim
import utils

def load_model(model, ctx, inputs_qext=None):
    net = model.to_graph(ctx=ctx)

    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    def model_func(data, label):
        data = sim.load_real_data(data, 'data', inputs_qext) \
               if inputs_qext else data
        data = gluon.utils.split_and_load(data, ctx_list=ctx,
                                          batch_axis=0, even_split=False)
        res = [net.forward(d) for d in data]
        res = nd.concatenate(res)
        acc_top1.update(label, res)
        _, top1 = acc_top1.get()
        acc_top5.update(label, res)
        _, top5 = acc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)
    return model_func

def validate_model(sym_path, prm_path, ctx, num_channel=3,
                   input_size=224, batch_size=16, iter_num=10,
                   ds_name='imagenet', from_scratch=0, lambd=None,
                   dump_model=False, input_shape=None):
    from gluon_zoo import save_model

    flag = [False]*from_scratch + [True]*(2-from_scratch)
    model_name, _ = path.splitext(path.basename(sym_path))
    model_dir = path.dirname(sym_path)
    input_shape = input_shape if input_shape else \
                  (batch_size, num_channel, input_size, input_size)
    logger = logging.getLogger("log.validate.%s"%model_name)

    if not path.exists(sym_path) or not path.exists(prm_path):
        save_model(model_name)
    model = Model.load(sym_path, prm_path)
    model.prepare(input_shape)
    # model = init(model, input_shape)

    print(tpass.collect_op_names(model.symbol, model.params))

    data_iter_func = ds.data_iter(ds_name, batch_size, input_size=input_size)
    data, _ = data_iter_func()

    # prepare
    mrt = model.get_mrt()
    # mrt = MRT(model)

    # calibrate
    mrt.set_data(data)
    prefix = path.join(model_dir, model_name+'.mrt.dict')
    _, _, dump_ext = utils.extend_fname(prefix, True)
    if flag[0]:
        th_dict = mrt.calibrate(lambd=lambd)
        sim.save_ext(dump_ext, th_dict)
    else:
        (th_dict,) = sim.load_ext(dump_ext)
        mrt.set_th_dict(th_dict)

    mrt.set_input_prec(8)
    mrt.set_output_prec(8)

    if flag[1]:
        mrt.quantize()
        mrt.save(model_name+".mrt.quantize", datadir=model_dir)
    else:
        mrt = MRT.load(model_name+".mrt.quantize", datadir=model_dir)

    # dump model
    if dump_model:
        datadir = "/data/ryt"
        model_name = model_name + "_tfm"
        dump_shape = (1, num_channel, input_size, input_size)
        mrt.current_model.to_cvm(
            model_name, datadir=datadir, input_shape=input_shape)
        data = data[0].reshape(dump_shape)
        data = sim.load_real_data(
            data.astype("float64"), 'data', mrt.get_inputs_ext())
        np.save(datadir+"/"+model_name+"/data.npy", data.astype('int8').asnumpy())
        sys.exit(0)

    # validate
    org_model = load_model(Model.load(sym_path, prm_path), ctx)
    cvm_quantize = load_model(
        mrt.current_model, ctx,
        inputs_qext=mrt.get_inputs_ext())

    utils.multi_validate(org_model, data_iter_func, cvm_quantize,
                         iter_num=iter_num,
                         logger=logging.getLogger('mrt.validate'))
    logger.info("test %s finished.", model_name)
