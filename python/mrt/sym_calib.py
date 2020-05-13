import logging
import os
import numpy as np

import mxnet as mx
from mxnet.gluon import nn, SymbolBlock
from mxnet import ndarray as nd
import nnvm as nnvm
import tvm

from sym_utils import *
from sym_pass import *
from utils import *
import sim_quant_helper as sim
import cvm_op

max_bit = 32 # INT32
default_target_bit = 8 # INT8
bias_target_bit = default_target_bit * 4 - 1
disable_requant_ops = [
    'Activation', 'relu',
    'Pooling',
    'slice', 'slice_like', 'slice_axis',
    'clip', 'negative',
    'repeat', 'tile', 'expand_dims', 'squeeze',
    'Reshape', 'transpose', 'Flatten',
    'max',
    'UpSampling',
]

def sym_calibrate(symbol, params, inputs_ext, old_ths={}, ctx=mx.cpu()):
    logger = logging.getLogger('log.calibration')
    check_ext_deps(inputs_ext, 'data', logger)

    out_as_threholds = set()
    for sym in topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs = sym_iter(sym.get_children())
        if op_name in disable_requant_ops:
            continue
        elif op_name in ['Embedding']:
            continue
        elif op_name in ['sigmoid', 'exp']:
            out_as_threholds.add(childs[0].attr('name'))
        out_as_threholds.add(name)

    order, deps = topo_sort(symbol, logger=logger, with_deps=True)
    th_dict, out_cache = {}, {}
    for sym in order:
        name, op_name = sym.attr('name'), sym.attr('op_name')
        attr, childs = sym.list_attr(), sym_iter(sym.get_children())
        if op_name == 'null':
            out = inputs_ext[name]['data'] if name in inputs_ext \
                  else params[name]
        elif childs is None:
            out = get_nd_op(op_name)(**attr)
        else:
            cinfos = [(c.attr('name'), get_entry_id(c)) for c in childs]
            nd_inputs = [out_cache[n[0]][n[1]] for n in cinfos]
            out = get_nd_op(op_name)(*nd_inputs, **attr)
            for n, _ in cinfos:
                assert n in deps
                deps[n].remove(name)
                if len(deps[n]) == 0:
                    del out_cache[n]
        out = [out] if len(sym) == 1 else out
        out_cache[name] = [o.as_in_context(ctx) for o in out]
        if name in out_as_threholds:
            # TODO: set multiple output
            opts = [float(o.abs().max().asscalar()) for o in out][0]
        elif op_name in disable_requant_ops:
            opts = th_dict[childs[0].attr('name')]
        elif op_name in ['Embedding']:
           opts = th_dict[childs[1].attr('name')]
        else:
           print (name, op_name)
           assert False
        if name in old_ths:
            #  th_dict[name] = [max(old_ths[name][i], o) for i,o in enumerate(opts)]
            th_dict[name] = max(old_ths[name], opts)
            logger.debug("update symbol %-40s out_shape=%-20s th_dict: (%s)",
                   name, [o.shape for o in out], th_dict[name])
        else:
            th_dict[name] = opts
            logger.debug("collect symbol %-40s out_shape=%-20s th_dict: (%s)",
                   name, [o.shape for o in out], th_dict[name])

    out_cache.clear()
    for k, v in inputs_ext.items():
        del v['data']
    return th_dict

def _sim_requantize_op(sym, scale, params, graph, prefix=None):
    name = sym.attr('name') if prefix is None else prefix
    scale_name = name + '_requant_scale'
    assert scale_name not in graph, "scale name %s has existed in graph" \
            % (scale_name)
    scale_sym = graph[scale_name] = mx.sym.var(scale_name, shape=(1,))
    params[scale_name] = nd.array([scale])

    requant_op_name = name + '_requant_op'
    assert requant_op_name not in graph
    node = mx.sym.broadcast_mul(sym, scale_sym, name=requant_op_name)
    graph[requant_op_name] = node
    return node
def _is_sim_requantize_op(sym):
    name = sym.attr('name')
    return True if name.endswith('_requant_op') else False
def _realize_tvm_requant_op(sym, sb, params, graph, target_bit):
    """Requantize Op:
        out = round(sym >> sb)  if sb >  0
        out = round(sym)        if sb == 0
        out = round(sym << -sb) if sb <  0

        round(sym >> sb) = int((int(sym >> (sb - 1)) + 1) >> 1)

        out = clip_int(out)
    """
    out = mx.sym.round(sym) # avoid precision loss represented in float32
    sb, tb = sb.asscalar(), target_bit.asscalar()
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
    clip_range = 2 ** (tb - 1) - 1
    out = mx.sym.clip(out, a_min=-clip_range, a_max=clip_range)
    return out
def _realize_cvm_requant_op(sym, sb, params, graph, target_bit):
    name = sym.attr('name')
    requant_op = name + '_cvm_shift'
    assert requant_op not in graph
    sb, tb = int(sb.asscalar()), int(target_bit.asscalar())
    if sb == 0:
        return mx.sym.Custom(sym, precision=tb,
                cvm_name=requant_op,
                name=requant_op, op_type='cvm_clip')
    elif sb < 0:
        return mx.sym.Custom(sym, shift_bit=-sb, precision=tb,
                name=requant_op, op_type='cvm_left_shift')
    else:
        return mx.sym.Custom(sym, shift_bit=sb, precision=tb,
                cvm_name=requant_op,
                name=requant_op, op_type='cvm_right_shift')

def _collect_scale_helper(sym, params, graph, inputs_ext,
        th_dict, get_scale, scale_helper, target_bits):
    logger = logging.getLogger('log.calib.sim.scale')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()

    scale_helper[name] = get_scale(th_dict[name], default_target_bit)
    target_bits[name] = default_target_bit
    if op_name in ['Convolution', 'FullyConnected']:
        X_name, W_name = childs[0].attr('name'), childs[1].attr('name')
        if not get_attr(attr, 'no_bias', False):
            B_name = childs[2].attr('name')
            scale_helper[B_name] = scale_helper[X_name] * scale_helper[W_name]
            target_bits[B_name] = bias_target_bit
    elif op_name in ['Embedding']:
        X_name, W_name = childs[0].attr('name'), childs[1].attr('name')
        X_range = params[W_name].shape[0]
        target_bits[X_name] = math.ceil(math.log2(X_range)) + 1
        scale_helper[X_name] = 1
    return sym, params
def _annotate_layer(sym, params, graph, inputs_ext,
        scale_helper, target_bits, infer_shapes):
    logger = logging.getLogger('log.calib.sym.sim.requant')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()

    node = sym
    cscales = [scale_helper[c.attr('name')] for c in childs] if childs else []
    cbits = [target_bits[c.attr('name')] for c in childs] if childs else []
    if op_name == 'null':
        return node, params
    elif op_name in disable_requant_ops:
        return node, params
    elif op_name in ['Convolution', 'FullyConnected', 'broadcast_mul']:
        requant_scale = scale_helper[name] / (cscales[0] * cscales[1])
    elif op_name in ['elemwise_add', 'elemwise_sub',
            'broadcast_add', 'broadcast_sub', 'Concat'] or (op_name == 'add_n' and len(childs) == 2):
        new_childs = []
        in_scale = min(cscales)
        for idx, c in enumerate(childs):
            relative_scale = in_scale / cscales[idx]
            if relative_scale != 1:
                c = _sim_requantize_op(c, relative_scale, params, graph,
                        "%s_in%d"%(name, idx))
                target_bits[c.attr('name')] = cbits[idx]
                logger.debug("layer %-40s  adjust scale=%-16.8f orig=%-16.8f" + \
                        " for requant %-40s input scale %-16.8f",
                        c.attr('name'), relative_scale, cscales[idx],
                        name, in_scale)
            new_childs.append(c)
        requant_scale = scale_helper[name] / in_scale
        node = get_mxnet_op(op_name)(*new_childs, **attr, name=name)
    elif op_name in ['Embedding']:
        requant_scale = scale_helper[name] / cscales[1]
    elif op_name in ['sum']:
        requant_scale = scale_helper[name] / cscales[0]
    else:
        logger.critical('Unrecognized op:%s(%s) attrs(%s)', op_name, name, attr)

    # if requant_scale != 1:
    r = (2**(default_target_bit-1)-1) / requant_scale
    target_bits[node.attr('name')] = math.ceil(math.log2(r)) + 1
    node = _sim_requantize_op(node, requant_scale, params, graph)
    logger.debug("layer %-40s requant scale=%-16.8f  out=%-16.8f in=%s",
            name, requant_scale, scale_helper[name],
            [scale_helper[c.attr('name')] for c in childs] \
            if childs else [])
    scale_helper[node.attr('name')] = scale_helper[name]
    target_bits[node.attr('name')] = default_target_bit
    infer_shapes[node.attr('name')] = infer_shapes[name]
    return node, params
def _annotate_parameters(sym, params, graph, inputs_ext,
        scale_helper, target_bits):
    logger = logging.getLogger('log.annotate.parameters')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    if op_name != 'null' or name in inputs_ext:
        return sym, params
    if name in scale_helper:
        params[name] = params[name] * scale_helper[name]
    return sym, params
def _realize_symbol(sym, params, graph, inputs_ext,
        target_bits, runtime="cvm"):
    logger = logging.getLogger('log.calib.realize')
    if not _is_sim_requantize_op(sym):
        return sym, params

    assert runtime in ["cvm", "tvm"]
    if runtime == "cvm":
        _realize_func = _realize_cvm_requant_op
        # _realize_func = _realize_broadcast_op
    else:
        _realize_func = _realize_tvm_requant_op

    childs = sym_iter(sym.get_children())
    X, B = childs[0], childs[1]
    X_name, B_name = X.attr('name'), B.attr('name')
    name = sym.attr('name')
    assert X_name in target_bits and name in target_bits, \
        "%s(%s, %s) not in precs %s" \
        % (name, X_name, B_name, target_bits.keys())

    def cal_bit(A_bit, B_bit, sb):
        # A_target_bit, B_target_bit = 16, 16
        # A_target_bit = min(A_bit, A_target_bit)
        # B_target_bit = min(B_bit, B_target_bit)
        # A_target_bit = 32 - B_target_bit if B_target_bit < 16 else A_target_bit
        # B_target_bit = 32 - A_target_bit if A_target_bit < 16 else B_target_bit
        # A_target_bit = min(A_bit, A_target_bit)
        # B_target_bit = min(B_bit, B_target_bit)
        max_bit = 32
        total_bit = A_bit + B_bit
        excess_bit = (total_bit - max_bit) // 2 if total_bit > max_bit else 0
        A_target_bit = A_bit - excess_bit
        B_target_bit = min(B_bit - excess_bit, 32 - A_target_bit)
        A_sb, B_sb = A_bit - A_target_bit, B_bit - B_target_bit
        Y_sb = (-sb) - A_sb - B_sb
        return A_sb, A_target_bit, B_sb, B_target_bit, Y_sb

    scale = params[B_name].asscalar()
    if scale == 1:
        node = _realize_func(X, nd.array([0]), params, graph,
                nd.array([target_bits[name]]))
    else:
        frac, sb = sim.extract_float(params[B_name].asscalar())
        shape = params[B_name].shape

        B_range = frac
        Y_tb = target_bits[name]
        Y_range = 2 ** (Y_tb - 1) - 1
        A_range = Y_range / params[B_name].asscalar()
        A_bit = target_bits[X_name]
        B_bit = math.ceil(math.log2(B_range)) + 1
        A_sb, A_tb, B_sb, B_tb, Y_sb = cal_bit(A_bit, B_bit, sb)

        X = _realize_func(X, nd.array(A_sb).reshape(shape), params, graph,
                nd.array(A_tb).reshape(shape))
        params[B_name] = nd.array([round(frac / (2 ** B_sb))])
        B_range = 2 ** (B_tb - 1) - 1
        params[B_name] = nd.clip(params[B_name],
                a_min=-B_range, a_max=B_range)
        attr = { 'precision': str(B_tb) }
        graph[B_name] = B = mx.sym.var(B_name, shape=shape, attr=attr)
        node = mx.sym.broadcast_mul(X, B)
        node = _realize_func(node, nd.array(Y_sb).reshape(shape), params, graph,
                nd.array(Y_tb).reshape(shape))
        logger.debug("layer %s Y(INT%s >> %s) X(%s|%s >> %s) B(%s|%s vs. %s %s >> %s)",
               name, Y_tb, Y_sb, A_range, A_bit, A_sb, B_range,
               B_bit, frac, sb, B_sb)
    target_bits[node.attr('name')] = target_bits[name]
    return node, params
def _realize_parameters(sym, params, graph, inputs_ext,
        target_bits={}, params_sim={}):
    logger = logging.getLogger('log.calib.realize.parameters')
    name = sym.attr('name')
    attr = sym.list_attr()
    if 'precision' not in attr or name in inputs_ext:
        return sym, params
    target_bit = int(attr['precision'])
    data = params[name]
    params[name] = sim.int_realize(data, target_bit, logger=logger)
    # calculate error
    error = params[name].astype('float32') - data
    error_rate = error / data
    if nd.sum(error).asscalar() == 0:
        rate = 0
    else:
        rate = nd.norm(error_rate).asscalar() / np.product(data.shape)
    if rate > 0.001:
        logger.warn("realize parameter %-60s avg error=%10.9f shape=%s",
                name, rate, data.shape)
    else:
        logger.debug("realize parameter %-60s avg error=%10.9f shape=%s",
                name, rate, data.shape)
    return sym, params


# interface API
def sym_simulate(symbol, params, inputs_ext, th_dict):
    logger = logging.getLogger('log.simulate')

    scale_helper, target_bits = {}, {}
    _, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_collect_scale_helper, th_dict=th_dict,
            get_scale=sim.get_sim_scale,
            scale_helper=scale_helper, target_bits=target_bits)

    # update inputs_ext
    for k, v in inputs_ext.items():
        v['scale'] = scale_helper[k]
        v['target_bit'] = target_bits[k]

    infer_shapes = sym_infer_shape(symbol, params, inputs_ext)
    symbol, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_annotate_layer, scale_helper=scale_helper,
            target_bits=target_bits, infer_shapes=infer_shapes)
    _, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_annotate_parameters,
            scale_helper=scale_helper, target_bits=target_bits)

    out_scales = [scale_helper[s.attr('name')] for s in symbol]

    params = examine_parameters(symbol, params, inputs_ext)
    symbol, params = sym_attach_attrs(symbol, params, inputs_ext,
            precision=target_bits)

    return symbol, params, target_bits, out_scales

def sym_realize(symbol, params, inputs_ext, precs, runtime="cvm"):
    logger = logging.getLogger('log.realize')
    _, params = topo_visit(symbol, params, get_op=get_mxnet_op,
           logger=logger, inputs_ext=inputs_ext,
           callback=_realize_parameters)
    symbol, params = topo_visit(symbol, params, get_op=get_mxnet_op,
           logger=logger, inputs_ext=inputs_ext,
           callback=_realize_symbol,
           target_bits=precs, runtime=runtime)

    def _check_int_params(params, arg):
       param = params[arg]
       amin, amax = param.min().asscalar(), param.max().asscalar()
       msg = "key:%s max_val:%s, min_val:%s"%(arg, amax, amin)
       assert amin >= INT32_MIN and amax <= INT32_MAX, msg
       flat = param.asnumpy().flatten()
       assert all(flat.astype('int32').astype(flat.dtype) == flat), msg

    params = examine_parameters(symbol, params, inputs_ext,
          callback=_check_int_params)
    return symbol, params

def pure_int8_quantize(symbol, params, inputs_ext, th_dict,
        runtime="cvm"):
    """Layer-Wise Quantization Method
    Quantize graph into INT8 for each layer including inputs and outputs

    Parameters
    ----------
    symbol: mx.sym.Symbol
        Mxnet symbol object.
    params: dict of NDArray
        Parameters of model.
    inputs_ext: dict of inputs information
        Extension info for inputs, such as inputs shape, calibration data,
        or precision after quantize.
    runtime: string
        Belongs to  "cvm" or "tvm", indicate the quantize target for runtime.
    ctx: mx.Context

    Returns
    -------
    qsym: mx.sym.Symbol
        Quantized symbol.
    qparams: dict of NDArray
        Quantized params.
    oscales: array of float
        Output of quantized graph / Output of original graph

    """
    ssym, sparams, precs, oscales = sym_simulate(symbol, params, inputs_ext, th_dict)
    qsym, qparams = sym_realize(ssym, sparams, inputs_ext, precs, runtime=runtime)

    return qsym, qparams, oscales
















