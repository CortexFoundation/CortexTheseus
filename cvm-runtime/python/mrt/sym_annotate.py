import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn, SymbolBlock
from mxnet import gluon
import numpy as np
import math

from sym_utils import *
from utils import *
import sym_pass as spass
import sim_quant_helper as sim

PLACE_HOLDER = 32 # INT32
out_key = 'out_key'
target_key = 'target_key'
disable_requant_ops = [
    'Activation', 'relu',
    'Pooling',
    'slice', 'slice_like', 'slice_axis',
    'clip', 'negative',
    'repeat', 'tile', 'expand_dims',
    'Reshape', 'transpose', 'Flatten', 'UpSampling'
]
class ANNO_TYPE():
    REQUANT = '_requant'
    IN_PREC_SQUEEZE = '_in_prec_squeeze'
    IN_PREC_SCALE = '_in_prec_scale'

def _infer_fixed_precs(sym, params, graph, inputs_ext, precs):
    logger = logging.getLogger("log.infer.fixed.precision")
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    cprecs = [precs[c.attr('name')] for c in childs] if childs else []
    precs[name] = precs[name] if name in precs else {}
    if op_name == 'null':
        return sym, params
    elif op_name in disable_requant_ops:
        pass
    elif op_name in ['sigmoid', 'exp']:
        cprecs[0][name] = 16
    elif op_name in ['Convolution', 'FullyConnected']:
        cprecs[0][name], cprecs[1][name] = 8, 8
        if eval(attr['no_bias']) == False:
            cprecs[2][name] = PLACE_HOLDER-1
    elif op_name in ['broadcast_add', 'broadcast_sub', 'elemwise_add',
            'elemwise_sub']:
        cprecs[0][name], cprecs[1][name] = PLACE_HOLDER-1, PLACE_HOLDER-1
        # cprecs[0][name], cprecs[1][name] = PLACE_HOLDER//2, PLACE_HOLDER//2
        # cprecs[0][name], cprecs[1][name] = 8, 8
    elif op_name in ['broadcast_mul']:
        cprecs[0][name], cprecs[1][name] = PLACE_HOLDER//2, PLACE_HOLDER//2
    elif op_name in ['Concat']:
        for prec in cprecs:
            prec[name] = PLACE_HOLDER
            # prec[name] = 8
    elif op_name in ['sum']:
        cprecs[0][name] = 8
    else:
        logger.critical("%s name=%-40s has not been considered.",
                op_name, name)
    return sym, params

def _update_input_precs(precs, in_bit, inputs_ext):
    for k in inputs_ext:
        inputs_ext[k]['target_bit'] = in_bit
        precs[k][out_key] = in_bit
        for n, v in precs[k].items():
            assert v >= in_bit, "input %s out of bit %s vs. %s" \
                    % (k, v, in_bit)
            precs[k][n] = in_bit

def _infer_dynamic_precs(sym, params, graph, inputs_ext, infer_shapes, precs,
        fix_param=False):
    logger = logging.getLogger("log.infer.dynamic.precision")
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    cprecs = [precs[c.attr('name')] for c in childs] if childs else []
    if op_name == 'null':
        return sym, params
    elif op_name in disable_requant_ops:
        cprecs[0][name] = cprecs[0][out_key]
        precs[name][out_key] = cprecs[0][name]
        return sym, params

    # update childs precision
    for i, c in enumerate(childs):
        c_tb = cprecs[i][name]
        c_bit = cprecs[i][out_key] if out_key in cprecs[i] else cprecs[i][name]
        if c_tb >= c_bit:
            cprecs[i][name] = c_bit

    cbits = [prec[name] for prec in cprecs]
    if op_name in ['sigmoid', 'exp']:
        precs[name][out_key] = 16
    elif op_name in ['Convolution', 'FullyConnected']:
        W_shape = infer_shapes[childs[1].attr('name')]
        sum_len = np.product(W_shape[1:])
        sum_bit = math.ceil(math.log2(sum_len))
        out_bit = cbits[0] + cbits[1] + sum_bit
        if get_attr(attr, 'no_bias', False) == False:
            if fix_param:
                out_bit = max(out_bit, cprecs[2][name])
            else:
                cprecs[2][name] = out_bit
            out_bit += 1
        precs[name][out_key] = out_bit
    elif op_name in ['broadcast_add', 'broadcast_sub', 'elemwise_add',
            'elemwise_sub', 'Concat']:
        is_params = lambda s : s.attr('op_name')=='null' and \
                s.attr('name') not in inputs_ext
        params_idx = [idx for idx,s in enumerate(childs) if is_params(s)]
        inputs_idx = [idx for idx,s in enumerate(childs) if not is_params(s)]
        assert len(inputs_idx) > 0, "Forget apply fuse constant pass first"
        out_bit = max([cbits[i] for i in inputs_idx])
        for idx in params_idx:
            if cbits[idx] > out_bit:
                if fix_param:
                    out_bit = max(out_bit, cprecs[idx][name])
                else:
                    cprecs[idx][name] = out_bit
        precs[name][out_key] = out_bit if op_name in ['Concat'] else out_bit+1
    elif op_name in ['broadcast_mul']:
        precs[name][out_key] = cbits[0] + cbits[1]
    elif op_name in ['sum']:
        axis = eval(attr['axis'])
        dshape = infer_shapes[childs[0].attr('name')]
        sum_len = np.product([dshape[i] for i in axis])
        sum_bit = math.ceil(math.log2(sum_len))
        precs[name][out_key] = cbits[0] + sum_bit

    assert precs[name][out_key] <= PLACE_HOLDER, \
        "%s name=%-40s out of PLACE_HOLDER %s" % (op_name, name, PLACE_HOLDER)
    return sym, params

def _infer_parameter_precs(sym, params, graph, inputs_ext, precs, outputs_ext):
    logger = logging.getLogger('log.infer.parameters.precision')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    if op_name != 'null' or name in inputs_ext:
        return sym, params
    param_prec = precs[name]
    if name in outputs_ext:
        ext = outputs_ext[name]
        if 'fixed' in ext and ext['fixed']:
            alpha = params[name].abs().max().asscalar()
            precs[name][out_key] = math.ceil(math.log2(alpha)) + 1
    else:
        min_prec = min(list(param_prec.values()))
        assert min_prec >= 8, "%s precision=%s from %s" % (name, min_prec, param_prec)
        assert out_key not in param_prec
        param_prec[out_key] = min_prec
    logger.debug("Fixed parameter %-40s out precision: %2d from %s",
        name, precs[name][out_key], param_prec)
    return sym, params

def _annotate(sym, graph, precs, out_bit, out_tb, anno_type, logger):
    name, op_name = sym.attr('name'), sym.attr('op_name')
    logger.info("Requantize layer %-20s name=%-40s out of bit %s vs. %s",
        op_name, name, out_tb, out_bit)
    tmp_name = name + '_requant_' + str(out_tb)
    if tmp_name not in graph:
        graph[tmp_name] = mx.sym.Custom(sym, in_prec=out_bit,
                out_prec=out_tb, anno_type=anno_type,
                name=tmp_name, op_type='cvm_annotate')
        if tmp_name not in precs:
            precs[tmp_name] = { out_key: out_tb }
        precs[name][tmp_name] = out_bit
    return graph[tmp_name]
def _is_annotate_op(sym):
    op_name, attr = sym.attr('op_name'), sym.list_attr()
    if op_name == 'Custom' and attr['op_type'] == 'cvm_annotate':
        return True
    return False
def _sym_annotate(sym, params, graph, inputs_ext, precs, th_dict):
    logger = logging.getLogger("log.sym.annotate")
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    if childs is None:
        return sym, params
    cprecs = [precs[c.attr('name')] for c in childs]
    new_childs = []
    for i, c in enumerate(childs):
        c_tb = cprecs[i][name]
        c_bit = cprecs[i][out_key] if out_key in cprecs[i] else cprecs[i][name]
        tmp = c
        if c_tb < c_bit:
            tmp = _annotate(c, graph, precs, c_bit, c_tb,
                    ANNO_TYPE.REQUANT, logger)
            precs[tmp.attr('name')][name] = c_tb
            precs[tmp.attr('name')][out_key] = c_tb
            th_dict[tmp.attr('name')] = th_dict[c.attr('name')]
        new_childs.append(tmp)
    node = get_mxnet_op(op_name)(*new_childs, **attr, name=name)

    if op_name in ['sigmoid', 'exp']:
        c_tb = cprecs[0][name]
        c_bit = cprecs[0][out_key] if out_key in cprecs[0] else cprecs[0][name]
        tmp = _annotate(childs[0], graph, precs, c_bit, c_tb,
        ANNO_TYPE.IN_PREC_SCALE, logger)
        precs[tmp.attr('name')][name] = c_tb
        precs[tmp.attr('name')][out_key] = c_tb
        th_dict[tmp.attr('name')] = th_dict[childs[0].attr('name')]
        node = get_mxnet_op(op_name)(tmp, **attr, name=name)

    if target_key in precs[name]:
        out_tb, out_bit = precs[name][target_key], precs[name][out_key]
        if out_tb < out_bit:
            node = _annotate(node, graph, precs, out_bit, out_tb,
                    ANNO_TYPE.IN_PREC_SCALE, logger)
            precs[node.attr('name')][name] = out_tb
            precs[tmp.attr('name')][out_key] = out_tb
            th_dict[node.attr('name')] = th_dict[name]
    return node, params

def _update_scale_and_precs(symbol, params, inputs_ext, th_dict, precs, scales):
    logger = logging.getLogger('log.simulate.update.scale')
    def _get_scale(alpha, prec):
        tb_max = 2 ** (prec - 1) - 1
        return tb_max / alpha
    for sym in topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        cscales = [scales[c.attr('name')] for c in childs] if childs else []
        if name in scales:
            logger.debug("Fixed scale of %s with value: %s",
                    name, scales[name])
        elif op_name == 'null':
            prec = precs[name][out_key]
            scales[name] = _get_scale(th_dict[name], prec)
        elif op_name in ['Convolution', 'FullyConnected']:
            if get_attr(attr, 'no_bias', False) == False:
                B_name = childs[2].attr('name')
                scales[B_name] = cscales[0] * cscales[1]
            scales[name] = cscales[0] * cscales[1]
        elif op_name in ['broadcast_mul']:
            scales[name] = cscales[0] * cscales[1]
        elif op_name in ['elemwise_add', 'elemwise_sub',
            'broadcast_add', 'broadcast_sub', 'Concat']:
            is_params = lambda s : s.attr('op_name')=='null' and \
                    s.attr('name') not in inputs_ext
            params_idx = [idx for idx,s in enumerate(childs) if is_params(s)]
            inputs_idx = [idx for idx,s in enumerate(childs) if not is_params(s)]
            assert len(inputs_idx) > 0, "Forget apply fuse constant pass first"
            out_scale = min([cscales[i] for i in inputs_idx])
            for i in params_idx:
                cname = childs[i].attr('name')
                if scales[cname] > out_scale:
                    scales[cname] = cscales[i] = out_scale
            scales[name] = min(cscales)
            for c in childs:
               cname = c.attr('name')
               int_alpha = th_dict[cname] * scales[cname]
               prec = math.ceil(math.log2(int_alpha)) + 1
               assert prec <= precs[cname][out_key], \
                        "Update %s for %s precision %s vs. %s" \
                        % (cname, name, prec, precs[cname][out_key])
               precs[cname][out_key] = prec
        elif op_name in ['sum']:
            scales[name] = cscales[0]
        elif op_name in disable_requant_ops:
            scales[name] = cscales[0]
        elif _is_annotate_op(sym):
            cname = childs[0].attr('name')
            int_alpha = th_dict[cname] * scales[cname]
            prec = math.ceil(math.log2(int_alpha)) + 1
            assert prec <= precs[cname][out_key], \
                    "Update %s for %s precision %s vs. %s" \
                    % (cname, name, prec, precs[cname][out_key])
            precs[cname][out_key] = prec
            scales[name] = _get_scale(th_dict[cname], precs[name][out_key])
            if attr['anno_type'] == ANNO_TYPE.REQUANT:
                if prec <= precs[name][out_key]:
                    scales[name] = scales[cname]
        elif op_name in ['sigmoid', 'exp']:
            cname = childs[0].attr('name')
            in_prec = precs[cname][name]
            alpha = (2 ** (in_prec - 1)) - 1
            data = nd.array([-alpha, alpha])
            out = get_nd_op(op_name)(data / cscales[0])
            alpha = out.abs().max().asscalar()
            scales[name] = _get_scale(alpha, precs[name][out_key])
        else:
            logger.critical('Unrecognized op:%s(%s) . attrs(%s)', op_name, name, attr)
        logger.debug("collect layer %-20s name=%-40s infos: out_scale=%-15.5f " +
                "out_prec=%-2s in_scales=%s in_prec=%s",
                op_name, name, scales[name], precs[name][out_key],
                cscales,
                [precs[c.attr('name')][out_key] for c in childs] if childs else [])

def _simulate(sym, scale, in_prec, out_prec, name):
    node = mx.sym.Custom(sym, in_prec=in_prec, out_prec=out_prec,
            scale=scale, name=name, op_type='cvm_sim_quant')
    return node
def _is_simulate_op(sym):
    op_name, attr = sym.attr('op_name'), sym.list_attr()
    if op_name == 'Custom' and attr['op_type'] == 'cvm_sim_quant':
        return True
    return False
def _simulate_layer(sym, params, graph, inputs_ext, scales, precs):
    logger = logging.getLogger('log.calib.sym.sim.requant')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()

    node = sym
    cscales = [scales[c.attr('name')] for c in childs] if childs else []
    def _restore():
        new_childs = []
        for idx, c in enumerate(childs):
            tmp = c / cscales[idx]
            new_childs.append(tmp)
        out = get_mxnet_op(op_name)(*new_childs, **attr, name=name)
        out = out * scales[name]
        return out
    if _is_annotate_op(sym):
        X_name = childs[0].attr('name')
        requant_scale = scales[name] / scales[childs[0].attr('name')]
        in_prec, out_prec = precs[X_name][out_key], precs[name][out_key]
        node = _simulate(childs[0], requant_scale, in_prec, out_prec, name)
        logger.debug("layer %-40s requant scale=%-16.8f  out=%-16.8f in=%s",
                name, requant_scale, scales[name],
                [scales[c.attr('name')] for c in childs] if childs else [])
    elif op_name in ['broadcast_add', 'broadcast_sub',
            'elemwise_add', 'elemwise_sub', 'Concat']:
        cscales = [scales[c.attr('name')] for c in childs]
        new_childs = []
        out_scale = min(cscales)
        for idx, c in enumerate(childs):
            relative_scale = out_scale / cscales[idx]
            if relative_scale != 1:
                cname = c.attr('name')
                in_prec, out_prec = precs[cname][out_key], precs[cname][out_key]
                c = _simulate(c, relative_scale, in_prec, out_prec,
                        "%s_in%d_squeeze"%(name, idx))
                logger.debug("layer %-40s  adjust scale=%-16.8f orig=%-16.8f" + \
                        " for requant %-40s input scale %-16.8f",
                        c.attr('name'), relative_scale,
                        cscales[idx], name, out_scale)
            new_childs.append(c)
        node = get_mxnet_op(op_name)(*new_childs, **attr, name=name)
    elif op_name in ['sigmoid', 'exp']:
        cname = childs[0].attr('name')
        in_prec = precs[cname][name]
        alpha = (2 ** (in_prec - 1)) - 1
        data = nd.arange(-alpha, alpha+1)
        out = get_nd_op(op_name)(data / cscales[0])
        weight = (out * scales[name]).round().reshape(2*alpha, 1)
        W_name = name + '_weight'
        assert W_name not in graph
        W = graph[W_name] = mx.sym.var(W_name, shape=weight.shape)
        params[W_name] = weight
        precs[W_name] = { out_key: precs[name][out_key] }
        alpha_sym, alpha_name = op_const(alpha, graph, var=mx.sym.var)
        precs[alpha_name] = { out_key: in_prec }
        params[alpha_name] = nd.array([alpha])
        X = mx.sym.broadcast_add(childs[0], alpha_sym)
        node = mx.sym.Custom(X, W, in_dim=2*alpha,
                name=name, op_type='cvm_lut')

    scales[node.attr('name')] = scales[name]
    precs[node.attr('name')] = precs[name]
    return node, params
def _simulate_parameters(sym, params, graph, inputs_ext, scales):
    logger = logging.getLogger('log.annotate.parameters')
    if sym.attr('op_name') != 'null':
        return sym, params
    name = sym.attr('name')
    if name in inputs_ext:
        inputs_ext[name]['scale'] = float(scales[name])
    elif name in scales:
        params[name] = params[name] * scales[name]
    return sym, params

def _realize_tvm(sym, sb, prec, params, graph):
    """Requantize Op:
        out = round(sym >> sb)  if sb >  0
        out = round(sym)        if sb == 0
        out = round(sym << -sb) if sb <  0

        round(sym >> sb) = int((int(sym >> (sb - 1)) + 1) >> 1)

        out = clip_int(out)
    """
    out = mx.sym.round(sym) # avoid precision loss represented in float32
    if sb < 0:
        out = out * (2 ** -sb)
        out = mx.sym.round(sym)
    elif sb > 0:
        if sb > 1:
            out = out / (2 ** (sb - 1))
            out = mx.sym.floor(out)
        out = out + 1
        out = out / 2
        out = mx.sym.floor(out)
    clip_range = 2 ** (prec - 1) - 1
    out = mx.sym.clip(out, a_min=-clip_range, a_max=clip_range)
    return out
def _realize_cvm(sym, sb, prec, params, graph):
    name = sym.attr('name')
    requant_op = name + '_cvm_shift'
    assert requant_op not in graph
    if sb == 0:
        return mx.sym.Custom(sym, precision=prec,
                cvm_name=requant_op,
                name=requant_op, op_type='cvm_clip')
    elif sb < 0:
        return mx.sym.Custom(sym, shift_bit=-sb, precision=prec,
                name=requant_op, op_type='cvm_left_shift')
    else:
        return mx.sym.Custom(sym, shift_bit=sb, precision=prec,
                cvm_name=requant_op,
                name=requant_op, op_type='cvm_right_shift')

def _realize_layer(sym, params, graph, inputs_ext, runtime):
    logger = logging.getLogger('log.realize.layer')
    name = sym.attr('name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    if not _is_simulate_op(sym):
        return sym, params

    X, scale = childs[0], eval(attr['scale'])
    in_prec, out_prec = eval(attr['in_prec']), eval(attr['out_prec'])
    frac, sb = sim.extract_float(scale)

    assert runtime in ['cvm', 'tvm']
    _realize_func = _realize_cvm if runtime == 'cvm' else _realize_tvm

    def cal_bit(A_bit, B_bit, sb):
       max_bit = 32
       total_bit = A_bit + B_bit
       excess_bit = (total_bit - max_bit) // 2 if total_bit > max_bit else 0
       A_target_bit = A_bit - excess_bit
       B_target_bit = min(B_bit - excess_bit, 32 - A_target_bit)
       A_sb, B_sb = A_bit - A_target_bit, B_bit - B_target_bit
       Y_sb = (-sb) - A_sb - B_sb
       return A_sb, A_target_bit, B_sb, B_target_bit, Y_sb

    if scale == 1:
        node = _realize_func(X, 0, out_prec, params, graph)
        logger.debug("layer %-40s skip prec=%s", name, out_prec)
    elif frac == 1:
        node =_realize_func(X, -sb, out_prec, params, graph)
        logger.debug("layer %-40s X(%s >> %s) prec=%s",
                name, in_prec, -sb, out_prec)
    else:
        B_bit = math.ceil(math.log2(frac)) + 1
        A_sb, A_tb, B_sb, B_tb, Y_sb = cal_bit(in_prec, B_bit, sb)

        X = _realize_func(X, A_sb, A_tb, params, graph)
        B_name = name + '_scale'
        params[B_name] = nd.array([round(frac / (2 ** B_sb))])
        B_range = 2 ** (B_tb - 1) - 1
        params[B_name] = nd.clip(params[B_name],
                a_min=-B_range, a_max=B_range)
        attr = { 'precision': str(B_tb) }
        B = graph[B_name] = mx.sym.var(B_name, shape=(1,), attr=attr)
        node = mx.sym.broadcast_mul(X, B)
        node = _realize_func(node, Y_sb, out_prec, params, graph)
        logger.debug("layer %-40s Y(INT%s >> %s) X(%s >> %s) B(%s vs. %s %s >> %s)",
                name, out_prec, Y_sb, in_prec, A_sb, scale, frac, sb, B_sb)

    #  if childs[0].attr('name') in [
        #  'yolov30_yolooutputv30_tile0',
        #  'yolov30_yolooutputv31_tile0',
        #  'yolov30_yolooutputv32_tile0',
        #  'yolov30_yolooutputv30_expand_dims0',
        #  'yolov30_yolooutputv31_expand_dims0',
        #  'yolov30_yolooutputv32_expand_dims0',
    #  ]:
        #  sb = out_prec - 16
        #  node = _realize_func(node, sb, 16, params, graph)
    return node, params

def _realize_parameters(sym, params, graph, inputs_ext, precs):
    logger = logging.getLogger('log.realize.parameters')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    attr = sym.list_attr()
    if op_name != 'null':
        return sym, params
    if name in inputs_ext:
        attr['precision'] = str(precs[name][out_key])
        return mx.sym.var(name, attr=attr), params
    prec = precs[name][out_key]
    data = params[name]
    params[name] = sim.int_realize(data, prec, logger=logger)
    # calculate error
    error = params[name].astype('float32') - data
    if nd.sum(error).asscalar() == 0:
        rate = 0
    else:
        rate = nd.norm(error / data).asscalar() / np.product(data.shape)
    if rate > 0.001:
        logger.warn("realize parameter %-60s avg error=%10.9f shape=%s",
                name, rate, data.shape)
    else:
        logger.debug("realize parameter %-60s avg error=%10.9f shape=%s",
                name, rate, data.shape)
    attr['precision'] = str(prec)
    node = mx.sym.var(name, attr=attr)
    return node, params

def _extract_symbol(symbol, params, outputs_ext):
    bases = []
    graph = {}
    for sym in topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        node = sym
        if childs is not None:
            childs = [graph[c.attr('name')] for c in childs]
            node = get_mxnet_op(op_name)(*childs, **attr, name=name)
        if name in outputs_ext:
            bases.append(sym)
            node = mx.sym.var(name)
        graph[name] = node
    base = bases[0] if len(bases) == 1 else mx.sym.Group(bases)
    base_params = {k:params[k] for k in base.list_inputs() if k in params}
    tops = [graph[sym.attr('name')] for sym in symbol]
    top = tops[0] if len(tops) == 1 else mx.sym.Group(tops)
    top_params = {k:params[k] for k in top.list_inputs() if k in params}
    return base, base_params, top, top_params
def _merge_symbol(base, base_params, top, top_params, maps):
    graph = {maps[c.attr('name')]:c for c in base}
    for sym in topo_sort(top):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        node = sym
        if childs is not None:
            childs = [graph[c.attr('name')] for c in childs]
            node = get_mxnet_op(op_name)(*childs, **attr, name=name)
        if name in graph:
            node = graph[name]
        graph[name] = node
    symbols = [graph[s.attr('name')] for s in top]
    symbol = symbols[0] if len(symbols) == 1 else mx.sym.Group(symbols)
    params = base_params
    params.update(top_params)
    params = {k:params[k] for k in symbol.list_inputs() if k in params}
    return symbol, params

def sym_annotate(symbol, params, inputs_ext, outputs_ext, th_dict,
        in_bit=8, out_bit=8):
    logger = logging.getLogger('log.infer.precision')
    precs = {}
    topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op, logger=logger,
            callback=_infer_fixed_precs, precs=precs)
    _update_input_precs(precs, in_bit, inputs_ext)
    infer_shapes = spass.sym_infer_shape(symbol, params, inputs_ext)
    topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op, logger=logger,
            callback=_infer_dynamic_precs,
            infer_shapes=infer_shapes, precs=precs, fix_param=False)
    topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op, logger=logger,
            callback=_infer_parameter_precs,
            precs=precs, outputs_ext=outputs_ext)
    topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op, logger=logger,
            callback=_infer_dynamic_precs,
            infer_shapes=infer_shapes, precs=precs, fix_param=True)

    for sym in symbol:
        precs[sym.attr('name')][target_key] = out_bit
    symbol, params = topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op, logger=logger,
            callback=_sym_annotate, precs=precs,
            th_dict=th_dict)

    for sym in topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs = sym_iter(sym.get_children())
        childs = childs if childs else []
        logger.debug("%-20s name=%-40s out_prec=%s in_precs=%s",
                op_name, name, precs[name][out_key],
                [precs[c.attr('name')][name] for c in childs])
    return symbol, params, precs

def sym_simulate(symbol, params, inputs_ext, outputs_ext, precs, th_dict):
    logger = logging.getLogger('log.simulate')

    infer_shapes = spass.sym_infer_shape(symbol, params, inputs_ext)
    scales = {}
    for k, v in outputs_ext.items():
        if 'threshold' in v:
            logger.debug("Update thresholds of output %s", k)
            th_dict[k] = v['threshold']
        if 'fixed' in v and v['fixed']:
            scales[k] = 1
    _update_scale_and_precs(symbol, params, inputs_ext,
            th_dict, precs, scales)

    ssym, sparams = topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op, logger=logger,
            callback=_simulate_layer,
            scales=scales, precs=precs)
    _, sparams = topo_visit(ssym, sparams, inputs_ext,
            get_op=get_mxnet_op, logger=logger,
            callback=_simulate_parameters, scales=scales)
    sparams = examine_parameters(ssym, sparams, inputs_ext)
    return ssym, sparams, scales

def sym_realize(symbol, params, inputs_ext, precs, runtime="cvm"):
    logger = logging.getLogger('log.realize')
    qsym, qparams = topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op,
            callback=_realize_parameters, precs=precs)
    qsym, qparams = topo_visit(qsym, qparams, inputs_ext,
            get_op=get_mxnet_op,
            callback=_realize_layer, runtime=runtime)

    def _check_int_params(params, arg):
       param = params[arg]
       msg = "key:%s max_val:%s, min_val:%s %s"%(arg, param.max().asscalar(),
               param.min().asscalar(), param)
       flat = param.asnumpy().flatten()
       assert all(flat >= INT32_MIN) and all(flat <= INT32_MAX), msg
       assert all(flat.astype('int32').astype(flat.dtype) == flat), msg

    qparams = examine_parameters(qsym, qparams, inputs_ext,
          callback=_check_int_params)
    return qsym, qparams

def post_quantize(symbol, params, inputs_ext, extra_ext):
    quantize_identity = ['_contrib_box_nms']
    def _post_quantize(sym, params, graph, inputs_ext):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        node = sym
        if op_name == '_contrib_box_nms':
            score_scale = extra_ext['score']
            bbox_scale = extra_ext['bbox']
            valid_thresh = get_attr(attr, 'valid_thresh', 0)
            attr['valid_thresh'] = int(valid_thresh * score_scale) # / (2 **8))
            node = get_mxnet_op(op_name)(*childs, **attr, name=name)
        return node, params
    qsym, qparams = topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op,
            callback=_post_quantize)
    return qsym, qparams

def mixed_precision(symbol, params, inputs_ext, th_dict,
        in_bit=8, out_bit=8, out_ext=None, runtime="cvm"):
    if out_ext is None:
        out_ext = {s.attr('name'):{ 'type': s.attr('name') } for s in symbol}
    base, base_params, top, top_params = _extract_symbol(symbol, params,
            out_ext)

    sbase, sbase_params, precs = sym_annotate(base, base_params, inputs_ext,
            out_ext, in_bit=in_bit, out_bit=out_bit, th_dict=th_dict)
    qbase, qbase_params, scales = sym_simulate(sbase, sbase_params,
            inputs_ext, out_ext, precs, th_dict)

    # update type_ext
    maps = dict(zip([c.attr('name') for c in qbase], [c.attr('name') for c in base]))
    ext = {maps[c.attr('name')]:scales[c.attr('name')] for c in qbase}
    type_ext = {out_ext[k]['type']:v for k,v in ext.items() \
        if 'type' in out_ext[k]}

    qsym, qparams = _merge_symbol(qbase, qbase_params, top, top_params, maps)
    qsym, qparams = sym_realize(qsym, qparams, inputs_ext, precs, runtime)
    qsym, qparams = post_quantize(qsym, qparams, inputs_ext, type_ext)
    return qsym, qparams, type_ext









