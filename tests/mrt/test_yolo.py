import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
from mxnet import gluon

import tvm
from tvm.contrib import graph_runtime
import nnvm
from tvm import relay

import sym_calib as calib
import utils
import mrt as _mrt
import gluon_zoo as zoo
import sym_pass as spass
import sym_utils as sutils
import sym_annotate as anno
import sim_quant_helper as sim
import dataset

import logging
import numpy as np
import os

def load_fname(version, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    prefix = "./data/yolo3%s%s"%(version, suffix)
    return utils.extend_fname(prefix, with_ext)

def validate(net, val_data, eval_metric, iter_num, logger=logging):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    # net.set_nms(nms_thresh=0.45, nms_topk=400)
    mx.nd.waitall()
    for idx, batch in enumerate(val_data):
        if idx >= iter_num:
            break
        data, label = batch[0], batch[1]
        acc = validate_data(net, data, label, eval_metric)
        logger.info('Validation: {:5.2%}'.format(acc))

def validate_data(net, data, label, eval_metric):
    det_ids, det_scores, det_bboxes = [], [], []
    gt_ids, gt_bboxes, gt_difficults = [], [], []

    # get prediction results
    x, y = data, label
    ids, scores, bboxes = net(x)
    det_ids.append(ids)
    det_scores.append(scores)
    # clip to image size
    det_bboxes.append(bboxes.clip(0, x.shape[2]))
    # split ground truths
    gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
    gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
    gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

    # update metric
    eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    map_name, mean_ap = eval_metric.get()
    acc = {k:v for k,v in zip(map_name, mean_ap)}['mAP']
    return acc

def split_model(symbol, params, inputs_ext, keys, logger=logging):
    infer_shapes = spass.sym_infer_shape(symbol, params, inputs_ext)
    bases = [s for s in sutils.topo_sort(symbol) if s.attr('name') in keys]
    base = mx.sym.Group(bases)
    base_params = {k:params[k] for k in base.list_inputs() if k in params}
    base_inputs_ext = inputs_ext

    graph = {}
    inputs = {k:v for k,v in inputs_ext.items()}
    for sym in sutils.topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sutils.sym_iter(sym.get_children()), sym.list_attr()
        node = sym
        if childs is not None:
            childs = [graph[c.attr('name')] for c in childs]
            node = sutils.get_mxnet_op(op_name)(*childs, **attr, name=name)
        if name in keys:
            node = mx.sym.var(name)
            inputs[name] = {'shape': infer_shapes[name]}
        graph[name] = node
    nodes = [graph[sym.attr('name')] for sym in symbol]
    top = nodes[0] if len(nodes) == 1 else mx.sym.Group(nodes)
    top_params = {k:params[k] for k in top.list_inputs() if k in params}
    top_inputs_ext = {k:v for k,v in inputs.items() if k not in inputs_ext}

    return base, base_params, base_inputs_ext, top, top_params, top_inputs_ext

def merge_model(base, base_params, base_inputs_ext, top, top_params, maps):
    graph = {maps[c.attr('name')]:c for c in base}
    for sym in sutils.topo_sort(top):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sutils.sym_iter(sym.get_children()), sym.list_attr()
        node = sym
        if childs is not None:
            childs = [graph[c.attr('name')] for c in childs]
            node = sutils.get_mxnet_op(op_name)(*childs, **attr, name=name)
        if name in graph:
            node = graph[name]
        graph[name] = node
    symbols = [graph[s.attr('name')] for s in top]
    symbol = symbols[0] if len(symbols) == 1 else mx.sym.Group(symbols)
    params = base_params
    params.update(top_params)
    params = {k:params[k] for k in symbol.list_inputs() if k in params}
    return symbol, params

def test_mrt_quant(batch_size=1, iter_num=10):
    logger = logging.getLogger("log.test.mrt.quantize")

    base_ctx = mx.gpu(1)
    ctx = mx.gpu(2)
    qctx = mx.gpu(3)
    input_size = 416
    h, w = input_size, input_size
    inputs_ext = { 'data': {
        'shape': (batch_size, 3, h, w),
    } }

    val_data = dataset.load_voc(batch_size, input_size)
    val_data_iter = iter(val_data)
    def data_iter_func():
        data, label = next(val_data_iter)
        return data, label

    if False:
        sym_file, param_file = load_fname("_darknet53_voc")
        sym, params = mx.sym.load(sym_file), nd.load(param_file)
        sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
        keys = [
          'yolov30_yolooutputv30_expand_dims0',
          'yolov30_yolooutputv31_expand_dims0',
          'yolov30_yolooutputv32_expand_dims0',
          'yolov30_yolooutputv30_tile0',
          'yolov30_yolooutputv31_tile0',
          'yolov30_yolooutputv32_tile0',
          'yolov30_yolooutputv30_broadcast_add1',
          'yolov30_yolooutputv31_broadcast_add1',
          'yolov30_yolooutputv32_broadcast_add1',
        ]
        base, base_params, base_inputs_ext, top, top_params, top_inputs_ext \
                = split_model(sym, params, inputs_ext, keys, logger)
        dump_sym, dump_params = load_fname("_darknet53_voc", "mrt.base")
        open(dump_sym, "w").write(base.tojson())
        nd.save(dump_params, base_params)
        dump_sym, dump_params, dump_ext = load_fname("_darknet53_voc", "mrt.top", True)
        open(dump_sym, "w").write(top.tojson())
        nd.save(dump_params, top_params)
        sim.save_ext(dump_ext, top_inputs_ext)

    dump_sym, dump_params = load_fname("_darknet53_voc", "mrt.base")
    base, base_params = mx.sym.load(dump_sym), nd.load(dump_params)
    dump_sym, dump_params, dump_ext = load_fname("_darknet53_voc", "mrt.top", True)
    top, top_params = mx.sym.load(dump_sym), nd.load(dump_params)
    (top_inputs_ext,) = sim.load_ext(dump_ext)

    base_inputs = [mx.sym.var(n) for n in inputs_ext]
    base_graph = mx.gluon.nn.SymbolBlock(base, base_inputs)
    utils.load_parameters(base_graph, base_params, ctx=ctx)

    top_inputs = [mx.sym.var(n) for n in top_inputs_ext]
    top_graph = mx.gluon.nn.SymbolBlock(top, top_inputs)
    utils.load_parameters(top_graph, top_params, ctx=ctx)

    metric = dataset.load_voc_metric()
    metric.reset()
    def yolov3(data, label):
       def net(data):
           tmp = base_graph(data.as_in_context(ctx))
           outs = top_graph(*tmp)
           # print ([o[0][0][:] for o in outs])
           return outs
       acc = validate_data(net, data, label, metric)
       return "{:6.2%}".format(acc)

    if False:
        mrt = _mrt.MRT(base, base_params, inputs_ext)
        for i in range(16):
            data, _ = data_iter_func()
            mrt.set_data('data', data)
            th_dict = mrt.calibrate(ctx=ctx)
        _, _, dump_ext = load_fname("_darknet53_voc", "mrt.dict", True)
        sim.save_ext(dump_ext, th_dict)

    _, _, dump_ext = load_fname("_darknet53_voc", "mrt.dict", True)
    (th_dict,) = sim.load_ext(dump_ext)
    if True:
        mrt = _mrt.MRT(base, base_params, base_inputs_ext)
        mrt.set_th_dict(th_dict)
        mrt.set_threshold('data', 2.64)
        mrt.set_threshold('yolov30_yolooutputv30_expand_dims0', 1)
        mrt.set_threshold('yolov30_yolooutputv31_expand_dims0', 1)
        mrt.set_threshold('yolov30_yolooutputv32_expand_dims0', 1)
        mrt.set_threshold('yolov30_yolooutputv30_tile0', 416)
        mrt.set_threshold('yolov30_yolooutputv31_tile0', 416)
        mrt.set_threshold('yolov30_yolooutputv32_tile0', 416)
        # mrt.set_fixed('yolov30_yolooutputv30_broadcast_add1')
        # mrt.set_fixed('yolov30_yolooutputv31_broadcast_add1')
        # mrt.set_fixed('yolov30_yolooutputv32_broadcast_add1')
        mrt.set_output_prec(30)
        qbase, qbase_params, qbase_inputs_ext = mrt.quantize()
        oscales = mrt.get_output_scales()
        dump_sym, dump_params, dump_ext = load_fname("_darknet53_voc", "mrt.quantize", True)
        open(dump_sym, "w").write(qbase.tojson())
        nd.save(dump_params, qbase_params)
        sim.save_ext(dump_ext, qbase_inputs_ext, oscales)

    if True:
        dump_sym, dump_params, dump_ext = load_fname("_darknet53_voc", "mrt.quantize", True)
        net2_inputs_ext, oscales = sim.load_ext(dump_ext)
        inputs = [mx.sym.var(n) for n in net2_inputs_ext]
        net2 = utils.load_model(dump_sym, dump_params, inputs, ctx=qctx)
        net2_metric = dataset.load_voc_metric()
        net2_metric.reset()
        def mrt_quantize(data, label):
            def net(data):
                data = sim.load_real_data(data, 'data', net2_inputs_ext)
                outs = net2(data.as_in_context(qctx))

                outs = [o.as_in_context(ctx) / oscales[i] for i, o in enumerate(outs)]
                # outs = b2_graph(*data)
                outs = top_graph(*outs)
                return outs
            acc = validate_data(net, data, label, net2_metric)
            return "{:6.2%}".format(acc)

    utils.multi_validate(yolov3, data_iter_func,
            mrt_quantize,
            iter_num=iter_num, logger=logger)

def test_sym_pass(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.sym.pass")

    base_ctx = mx.gpu(1)
    ctx = mx.gpu(2)
    input_size = 416
    h, w = input_size, input_size
    inputs_ext = { 'data': {
        'shape': (batch_size, 3, h, w),
    } }

    val_data = dataset.load_voc(batch_size, input_size)
    val_data_iter = iter(val_data)
    def data_iter_func():
        data, label = next(val_data_iter)
        return data, label

    sym_file, param_file = load_fname("_darknet53_voc")
    sym, params = mx.sym.load(sym_file), nd.load(param_file)
    sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
    if False:
        th_dict = {}
        for i in range(16):
          data, _ = data_iter_func()
          for k, v in inputs_ext.items():
              v['data'] = data
          th_dict = calib.sym_calibrate(sym, params, inputs_ext,
                  old_ths=th_dict, ctx=ctx)
        _, _, dump_ext = load_fname("_darknet53_voc", "dict", True)
        sim.save_ext(dump_ext, th_dict)

    _, _, dump_ext = load_fname("_darknet53_voc", "dict", True)
    (th_dict,) = sim.load_ext(dump_ext)
    inputs = [mx.sym.var(name) for name in inputs_ext]
    net1 = mx.gluon.nn.SymbolBlock(sym, inputs)
    utils.load_parameters(net1, params, ctx=ctx)
    metric = dataset.load_voc_metric()
    metric.reset()
    def yolov3(data, label):
       def net(data):
           out = net1(data.as_in_context(ctx))
           print ([o[0][0][:] for o in out])
           return out
       acc = validate_data(net, data, label, metric)
       return "{:6.2%}".format(acc)

    keys = [
        'yolov30_yolooutputv30_conv0_fwd',
        'yolov30_yolooutputv31_conv0_fwd',
        'yolov30_yolooutputv32_conv0_fwd',
    ]
    base, base_params, base_inputs_ext, top, top_params, top_inputs_ext \
            = split_model(sym, params, inputs_ext, keys, logger)
    dump_sym, dump_params = load_fname("_darknet53_voc", "base")
    open(dump_sym, "w").write(base.tojson())
    dump_sym, dump_params, dump_ext = load_fname("_darknet53_voc", "top", True)
    open(dump_sym, "w").write(top.tojson())
    nd.save(dump_params, top_params)
    sim.save_ext(dump_ext, top_inputs_ext)

    base_inputs = [mx.sym.var(n) for n in base_inputs_ext]
    base_graph = mx.gluon.nn.SymbolBlock(base, base_inputs)
    utils.load_parameters(base_graph, base_params, ctx=base_ctx)

    top_inputs = [mx.sym.var(n) for n in top_inputs_ext]
    top_graph = mx.gluon.nn.SymbolBlock(top, top_inputs)
    utils.load_parameters(top_graph, top_params, ctx=ctx)

    # quantize base graph
    if False:
        qbase, qbase_params, qbase_prec, base_oscales = calib.sym_simulate(
                base, base_params, base_inputs_ext, th_dict)
        qbase, qbase_params = calib.sym_realize(qbase, qbase_params, base_inputs_ext, qbase_prec)
        dump_sym, dump_params, dump_ext = load_fname("_darknet53_voc", "base.quantize", True)
        open(dump_sym, "w").write(qbase.tojson())
        sim.save_ext(dump_ext, base_inputs_ext, base_oscales)
        nd.save(dump_params, qbase_params)

    if False:
        qb_sym, qb_params, qb_ext = load_fname("_darknet53_voc", "base.quantize", True)
        net2_inputs_ext, base_oscales = sim.load_ext(qb_ext)
        net2_inputs = [mx.sym.var(n) for n in net2_inputs_ext]
        net2 = utils.load_model(qb_sym, qb_params, net2_inputs, ctx=ctx)
        base_metric = dataset.load_voc_metric()
        base_metric.reset()
        def base_quantize(data, label):
           def net(data):
               data = sim.load_real_data(data, 'data', net2_inputs_ext)
               tmp = list(net2(data.as_in_context(ctx)))
               tmp = [t / base_oscales[i] for i,t in enumerate(tmp)]
               return top_graph(*tmp)
           acc = validate_data(net, data, label, base_metric)
           return "{:6.2%}".format(acc)

    # quantize top graph
    if False:
        in_bit, out_bit = 8, 30
        outputs_ext = {
           'yolov30_yolooutputv30_expand_dims0': { 'threshold': 1, 'type': 'score' },
           'yolov30_yolooutputv31_expand_dims0': { 'threshold': 1, 'type': 'score' },
           'yolov30_yolooutputv32_expand_dims0': { 'threshold': 1, 'type': 'score' },
           'yolov30_yolooutputv30_tile0': { 'threshold': 416, 'type': 'bbox' },
           'yolov30_yolooutputv31_tile0': { 'threshold': 416, 'type': 'bbox' },
           'yolov30_yolooutputv32_tile0': { 'threshold': 416, 'type': 'bbox' },
           'yolov30_yolooutputv30_broadcast_add1': { 'fixed': True, 'type': 'ids' },
           'yolov30_yolooutputv31_broadcast_add1': { 'fixed': True, 'type': 'ids' },
           'yolov30_yolooutputv32_broadcast_add1': { 'fixed': True, 'type': 'ids' },
        }
        qsym, qparams, type_ext = anno.mixed_precision(top, top_params,
               top_inputs_ext, th_dict, in_bit=in_bit, out_bit=out_bit,
               out_ext=outputs_ext, runtime="cvm")
        out_scales = [type_ext['ids'], type_ext['score'], type_ext['bbox']]

        dump_sym, dump_params, dump_ext = load_fname("_darknet53_voc", "top.quantize", True)
        open(dump_sym, "w").write(qsym.tojson())
        sim.save_ext(dump_ext, top_inputs_ext, out_scales)
        nd.save(dump_params, qparams)

    if True:
        sym_file, param_file, ext_file = load_fname("_darknet53_voc", "top.quantize", True)
        net3_inputs_ext, net3_scales = sim.load_ext(ext_file)
        top_sym = base_graph(mx.sym.Group(base_inputs))
        top_names = [c.attr('name') for c in top_sym]
        net3_inputs = [mx.sym.var(n) for n in net3_inputs_ext]
        net3 = utils.load_model(sym_file, param_file, net3_inputs, ctx=ctx)
        top_qmetric = dataset.load_voc_metric()
        top_qmetric.reset()
        def top_quantize(data, label):
            def net(data):
                tmp = base_graph(data.as_in_context(base_ctx))
                tmp = [t.as_in_context(ctx) for t in tmp]
                tmp = [sim.load_real_data(tmp[i], n, net3_inputs_ext) for i,n in enumerate(top_names)]
                out = net3(*tmp)
                out = [(t / net3_scales[i]) for i,t in enumerate(out)]
                print ([o[0][0][:] for o in out])
                return out
            acc = validate_data(net, data, label, top_qmetric)
            return "{:6.2%}".format(acc)

    # merge quantize model
    if False:
        qb_sym, qb_params, qb_ext = load_fname("_darknet53_voc", "base.quantize", True)
        qbase, qbase_params = mx.sym.load(qb_sym), nd.load(qb_params)
        qbase_inputs_ext, _ = sim.load_ext(qb_ext)
        qt_sym, qt_params, qt_ext = load_fname("_darknet53_voc", "top.quantize", True)
        qtop, qtop_params = mx.sym.load(qt_sym), nd.load(qt_params)
        _, out_scales = sim.load_ext(qt_ext)
        maps = dict(zip([c.attr('name') for c in qbase], [c.attr('name') for c in base]))
        qsym, qparams = merge_model(qbase, qbase_params, qbase_inputs_ext,
                qtop, qtop_params, maps)
        sym_file, param_file, ext_file = load_fname("_darknet53_voc", "all.quantize", True)
        open(sym_file, "w").write(qsym.tojson())
        nd.save(param_file, qparams)
        sim.save_ext(ext_file, qbase_inputs_ext, out_scales)

    if False:
        sym_file, param_file, ext_file = load_fname("_darknet53_voc", "all.quantize", True)
        net4_inputs_ext, net4_scales = sim.load_ext(ext_file)
        net4_inputs = [mx.sym.var(n) for n in net4_inputs_ext]
        net4 = utils.load_model(sym_file, param_file, net4_inputs, ctx=ctx)
        all_qmetric = dataset.load_voc_metric()
        all_qmetric.reset()
        def all_quantize(data, label):
            def net(data):
                data = sim.load_real_data(data, 'data', net4_inputs_ext)
                out = net4(data.as_in_context(ctx))
                out = [(t / net4_scales[i]) for i,t in enumerate(out)]
                return out
            acc = validate_data(net, data, label, all_qmetric)
            return "{:6.2%}".format(acc)

    if False:
        sym_file, param_file, ext_file = load_fname("_darknet53_voc", "all.quantize", True)
        net4_inputs_ext, net4_scales = sim.load_ext(ext_file)
        datadir = "/data/voc/data/"
        for i in range(50):
            countdir = datadir + "/" + str(i)
            os.makedirs(countdir, exist_ok=True)
            data, label = data_iter_func()
            data = sim.load_real_data(data, 'data', net4_inputs_ext)
            np.save(countdir+"/data.npy", data.asnumpy().astype('int8'))
            np.save(countdir+"/label.npy", label.asnumpy())

        # data = sim.load_real_data(data, 'data', net4_inputs_ext)
        # np.save("/tmp/yolo/data", data.asnumpy().astype('int8'))
        # out = net4(data.as_in_context(ctx))
        # for i, o in enumerate(out):
        #    np.save("/tmp/yolo/result"+str(i), o.asnumpy().astype('int32'))
        exit()

    utils.multi_validate(yolov3, data_iter_func,
            top_quantize,
            # base_quantize, # top_quantize, all_quantize,
            iter_num=iter_num, logger=logger)

def test_sym_nnvm(batch_size, iter_num):
    logger = logging.getLogger("log.test.nnvm")
    logger.info("=== Log Test NNVM ===")

    sym_file, param_file, ext_file = load_fname("_darknet53_voc", "all.quantize", True)
    dump_sym, dump_params = load_fname("_darknet53_voc", "all.nnvm.compile")
    sym, params = mx.sym.load(sym_file), nd.load(param_file)
    inputs_ext, _ = sim.load_ext(ext_file)
    spass.mxnet_to_nnvm(sym, params, inputs_ext, dump_sym, dump_params)

if __name__ == '__main__':
    utils.log_init()

    # zoo.save_model('yolo3_darknet53_voc')
    # name = "yolo3_resnet18_v1_voc"
    # net = zoo.load_resnet18_v1_yolo()
    # sym = net(mx.sym.var('data'))
    # if isinstance(sym, tuple):
    #     sym = mx.sym.Group([*sym])
    # open("./data/%s.json"%name, "w").write(sym.tojson())
    # exit()

    if False:
        val_data = dataset.load_voc(1, 416)
        sym_file, param_file, ext_file = load_fname("_darknet53_voc", "all.quantize", True)
        sym, params = mx.sym.load(sym_file), nd.load(param_file)
        inputs_ext, _ = sim.load_ext(ext_file)
        if False:
            for data, _ in val_data:
                data = sim.load_real_data(data, 'data', inputs_ext)
                inputs_ext['data']['data'] = data
                spass.sym_dump_ops(sym, params, inputs_ext,
                        datadir="/data/wlt", ctx=mx.gpu(2))
        else:
            val_data_iter = iter(val_data)
            data, _ = next(val_data_iter)
            data = sim.load_real_data(data, 'data', inputs_ext)
            inputs_ext['data']['data'] = data
            spass.sym_dump_layer_outputs(sym, params, inputs_ext,
                    datadir="/tmp/yolo/out",
                    dump_ops=[""])
        exit()
    # zoo.save_model("yolo3_mobilenet1.0_voc")

    # test_sym_pass(1, 1)
    test_mrt_quant(1, 100)
    # test_sym_nnvm(16, 0)
