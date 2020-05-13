import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
from mxnet import gluon

import utils
import mrt as _mrt
import gluon_zoo as zoo
import sym_pass as spass
import sym_utils as sutils
import sim_quant_helper as sim
import dataset

import logging

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
            = _mrt.split_model(sym, params, inputs_ext, keys)
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
           return outs
       acc = validate_data(net, data, label, metric)
       return "{:6.2%}".format(acc)

    if True:
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
        mrt.set_fixed('yolov30_yolooutputv30_broadcast_add1')
        mrt.set_fixed('yolov30_yolooutputv31_broadcast_add1')
        mrt.set_fixed('yolov30_yolooutputv32_broadcast_add1')
        mrt.set_output_prec(30)
        qbase, qbase_params, qbase_inputs_ext = mrt.quantize()
        oscales = mrt.get_output_scales()
        maps = mrt.get_maps()
        dump_sym, dump_params, dump_ext = load_fname("_darknet53_voc", "mrt.quantize", True)
        open(dump_sym, "w").write(qbase.tojson())
        nd.save(dump_params, qbase_params)
        sim.save_ext(dump_ext, qbase_inputs_ext, oscales, maps)

    # merge quantize model
    if True:
        qb_sym, qb_params, qb_ext = load_fname("_darknet53_voc", "mrt.quantize", True)
        qbase, qbase_params = mx.sym.load(qb_sym), nd.load(qb_params)
        qbase_inputs_ext, oscales, maps = sim.load_ext(qb_ext)

        def box_nms(node, params, graph):
            name, op_name = node.attr('name'), node.attr('op_name')
            childs, attr = sutils.sym_iter(node.get_children()), node.list_attr()
            if op_name == '_contrib_box_nms':
                valid_thresh = sutils.get_attr(attr, 'valid_thresh', 0)
                attr['valid_thresh'] = int(valid_thresh * oscales[3])
                node = sutils.get_mxnet_op(op_name)(*childs, **attr, name=name)
            return node
        qsym, qparams = _mrt.merge_model(qbase, qbase_params,
                top, top_params, maps, box_nms)
        sym_file, param_file, ext_file = load_fname("_darknet53_voc", "mrt.all.quantize", True)
        open(sym_file, "w").write(qsym.tojson())
        nd.save(param_file, qparams)
        sim.save_ext(ext_file, qbase_inputs_ext,
                [oscales[0], oscales[3], oscales[4]])

    if True:
        dump_sym, dump_params, dump_ext = load_fname("_darknet53_voc", "mrt.all.quantize", True)
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
                return outs
            acc = validate_data(net, data, label, net2_metric)
            return "{:6.2%}".format(acc)

    utils.multi_validate(yolov3, data_iter_func,
            mrt_quantize,
            iter_num=iter_num, logger=logger)

def test_sym_nnvm(batch_size, iter_num):
    logger = logging.getLogger("log.test.nnvm")
    logger.info("=== Log Test NNVM ===")

    sym_file, param_file, ext_file = load_fname("_darknet53_voc", "mrt.all.quantize", True)
    sym, params = mx.sym.load(sym_file), nd.load(param_file)
    inputs_ext, _ = sim.load_ext(ext_file)
    nnvm_sym, nnvm_params = spass.mxnet_to_nnvm(sym, params, inputs_ext)
    spass.cvm_build(nnvm_sym, nnvm_params, inputs_ext, *load_fname("_darknet53_voc", "nnvm"))

if __name__ == '__main__':
    utils.log_init()

    # zoo.save_model('yolo3_darknet53_voc')

    test_mrt_quant(1, 100)
    # test_sym_nnvm(16, 0)
