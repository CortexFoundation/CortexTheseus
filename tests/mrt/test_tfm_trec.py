import gluon_zoo as gz
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
import tvm
from tvm.contrib import graph_runtime
import nnvm
import pickle

import sym_pass as spass
import dataset as ds
import sim_quant_helper as sim
import ops_generator as opg
import utils
from transformer import *

from os import path

def load_fname(suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    prefix = "./data/trec%s" % (suffix)
    return utils.extend_fname(prefix, with_ext=with_ext)

def test_mrt_quant(batch_size=1, iter_num=10):
    ctx = mx.gpu(3)
    input_shape = (38, batch_size)
    inputs = [mx.sym.var('data')]

    utils.log_init()

    data_iter = ds.load_trec(batch_size)
    def data_iter_func():
        return next(data_iter)
    data, label = data_iter_func()

    sym_path, prm_path = load_fname()
    model_name, _ = path.splitext(path.basename(sym_path))
    model_dir = path.dirname(sym_path)

    model = Model.load(sym_path, prm_path)
    model = init(model, input_shape)

    net1 = model.to_graph(ctx=ctx)
    def trec(data):
        res = net1(data.as_in_context(ctx))
        return res

    qsym, qparams, inputs_qext = None, None, None
    if True:
        mrt = MRT(model)
        mrt.set_data(data)
        mrt.calibrate(ctx=ctx)
        mrt.set_input_prec(16)
        # mrt.set_fixed('data')
        mrt.set_output_prec(8)
        mrt.quantize()
        mrt.save(model_name+".mrt.quantize", datadir=model_dir)
        # mrt.compile("trec_tfm", datadir="/data/ryt")
        # data = sim.load_real_data(data, 'data', inputs_qext)
        # np.save("/data/ryt/trec_tfm/data.npy",
        #         sim.load_real_data(data, 'data', inputs_qext).asnumpy().astype('int32'))
        # exit()
    else:
        inputs_qext['data']['data'] = data
        th_dict = calib.sym_calibrate(sym, params, inputs_qext, ctx=ctx)
        qsym, qparams, _ = calib.pure_int8_quantize(sym, params, inputs_qext, th_dict)

    net2 = mrt.current_model.to_graph(ctx=ctx)
    # net2 = gluon.nn.SymbolBlock(qsym, inputs)
    # utils.load_parameters(net2, qparams, ctx=ctx)
    inputs_qext = mrt.get_inputs_ext()
    def quantize(data):
        data = sim.load_real_data(data, 'data', inputs_qext)
        res = net2(data.as_in_context(ctx))
        return res

    if False:
        inputs_qext['data']['shape'] = (38, 1)
        data = data[:, 0].reshape(38, 1)
        _mrt.std_dump(qsym, qparams, qinputs_ext, data, "trec",
                batch=True, data_dtype="int32", max_num=1000,
                dump_ops=["sentimentnet0_embedding0_fwd"])
        opg.dump_file("take",
                ["/data/std_out/trec/sentimentnet0_embedding0_fwd_0.mrt.dump.in.npy",
                 "/data/std_out/trec/sentimentnet0_embedding0_fwd_1.mrt.dump.in.npy"],
                ["/data/std_out/trec/sentimentnet0_embedding0_fwd_0.mrt.dump.out.npy"],
                "/data/std_out/trec/sentimentnet0_embedding0_fwd.attr")

    if True:
        while True:
            data, _ = next(data_iter)
            inputs_qext = mrt.get_inputs_ext()
            data = sim.load_real_data(data, 'data', inputs_qext)
            inputs_qext['data']['data'] = data
            spass.sym_dump_ops(mrt.current_model.symbol, mrt.current_model.params, inputs_qext,
                    ctx=mx.gpu(3))
        exit()

    utils.multi_eval_accuracy(trec, data_iter_func,
            quantize,
            iter_num=iter_num)

if __name__ == '__main__':
    test_mrt_quant(16, 100) # 97% --> 97% StopIteration

