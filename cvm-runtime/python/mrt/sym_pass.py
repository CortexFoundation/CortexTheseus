import math
import numpy as np
import os

import mxnet as mx
from mxnet.gluon import nn, SymbolBlock
from mxnet import ndarray as nd
# import nnvm as nnvm
# import tvm
# from tvm import relay
from sym_utils import *
from utils import *

def mx_set_precs(symbol, params, inputs_ext):
    logger = logging.getLogger("log.pass.set.precisions")
    def _set_prec(sym, params, graph, inputs_ext):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        if name in inputs_ext:
            prec = inputs_ext[name]['precision']
            sym = mx.sym.var(name, attr={ 'precision': str(prec) })
        elif name in params:
            alpha = params[name].abs().max().asscalar()
            if alpha == 0:
                prec = "1"
            else:
                prec = str(math.ceil(math.log2(alpha)) + 1)
            sym = mx.sym.var(name, attr={ 'precision': str(prec) }, **attr)
        return sym, params

    ret_sym, ret_params = topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op, logger=logger,
            callback=_set_prec)
    return ret_sym, ret_params

def prepare_for_cvm(symbol, params, inputs_ext):
    infer_shapes = sym_infer_shape(symbol, params, inputs_ext)
    def _mx_prepare(sym, params, graph, inputs_ext):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        if op_name == 'null':
            return sym, params

        if 'scalar' in attr:
            scalar = float(attr['scalar'])
            msg = "name:%s, op_name:%s, scalar:%s"%(name, op_name, attr)
            assert scalar >= INT32_MIN and scalar <= INT32_MAX, msg
            assert float(int(scalar)) == scalar, msg
            attr['scalar'] = int(scalar)
        if 'overlap_thresh' in attr:
            thresh = float(attr['overlap_thresh']) * 100
            attr['overlap_thresh'] = int(thresh)
        node = get_mxnet_op(op_name)(*childs, **attr)

        if op_name in ['slice_axis']:
            X = childs[0]
            cshape = infer_shapes[X.attr('name')]
            axis = get_attr(attr, 'axis')
            axis_begin = get_attr(attr, 'begin')
            axis_end = get_attr(attr, 'end')
            if axis_end is None:
                axis_end = cshape[axis]
            begin = [0 for s in cshape]
            end = [s for s in cshape]
            begin[axis], end[axis] = axis_begin, axis_end
            node = get_mxnet_op('slice')(X, begin=begin, end=end, name=name)
        elif op_name in ['slice']:
            X = childs[0]
            cshape = infer_shapes[X.attr('name')]
            begin = get_attr(attr, 'begin')
            end = get_attr(attr, 'end')
            begin = [0 if s is None else s for s in begin]
            end = [cshape[i] if s is None else s for i,s in enumerate(end)]
            node = get_mxnet_op('slice')(X, begin=begin, end=end, name=name)
        elif op_name in ['floor', 'ceil', 'round', 'fix', 'Cast']:
            node = childs[0]
        elif op_name == '_greater_scalar':
            X = childs[0]
            scalar = int(attr['scalar'])
            assert int(scalar) == scalar
            var = mx_const(scalar, graph, params)
            node = mx.sym.broadcast_greater(X, var, name=name)
        infer_shapes[node.attr('name')] = infer_shapes[name]
        return node, params
    psym, pparams = topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op,
            callback=_mx_prepare)
    return psym, pparams

def mxnet_build(sym, params, inputs_ext, dump_sym, dump_params,
        runtime="cvm", target="cuda", logger=logging):
    nnvm_sym, nnvm_params = mxnet_to_nnvm(sym, params, logger=logging)
    return cvm_build(nnvm_sym, nnvm_params, inputs_ext, dump_sym, dump_params,
            runtime=runtime, target=target, logger=logger)

def mxnet_to_nnvm(sym, params, inputs_ext, logger=logging):
    sym, params = prepare_for_cvm(sym, params, inputs_ext)
    nnvm_sym, _ = nnvm.frontend.from_mxnet(sym)


    # nnvm_sym, params = nnvm_realize(nnvm_sym, params, inputs_ext)

    args = nnvm_sym.list_input_names()
    real_params = {}
    use_dtype = "int32"
    tvm_ctx = tvm.context("llvm", 0)
    for key, value in params.items():
        if key not in args:
            logger.warn("key:%s not exists in graph", key)
        else:
            msg = "key:%s value:%s"%(key, value)
            flat = value.asnumpy().flatten()
            assert all(flat >= INT32_MIN) and all(flat <= INT32_MAX), msg
            assert all(flat.astype('int32').astype('float32') == flat), msg
            real_params[key] = tvm.nd.array(value.astype(use_dtype).asnumpy(), tvm_ctx)
    return nnvm_sym, real_params

def cvm_build(nnvm_sym, nnvm_params, inputs_ext, dump_sym, dump_params,
        runtime="cvm", target="cuda", logger=logging, dtype="int32"):
    logger.debug("Compile nnvm graph to %s", runtime)
    tvm_ctx = tvm.context(target, 0)
    inputs_shape = {k:v['shape'] for k,v in inputs_ext.items()}

    with nnvm.compiler.build_config(opt_level=0, runtime=runtime):
        deploy_graph, lib, real_params = nnvm.compiler.build(
            nnvm_sym, target=target, shape=inputs_shape,
            params=nnvm_params, dtype=dtype)
    if runtime == "cvm":
        real_params = tvm_params_reduce(nnvm_sym, real_params, inputs_ext, tvm_ctx)
    open(dump_sym, "w").write(deploy_graph.json())
    param_bytes = nnvm.compiler.save_param_dict(real_params)
    open(dump_params, "wb").write(param_bytes)
    if runtime == "cvm":
        return deploy_graph, real_params
    else:
        return deploy_graph, real_params, lib


def mxnet_to_tvm(sym, params, inputs_ext, dump_sym, dump_params,
        logger=logging):
    inputs_shape = {k:v['shape'] for k,v in inputs_ext.items()}
    sym, params = prepare_for_cvm(sym, params, inputs_ext)
    relay_sym, relay_params = relay.frontend.from_mxnet(sym, inputs_shape,
            arg_params=params)
    real_params = tvm_params_reduce(sym, relay_params, inputs_ext, tvm.context("cpu"))
    # TODO: conv and dense layer dump without precision
    logger.debug("Compiling into CVM Executor Graph")
    with relay.build_config(opt_level=0):
        graph_json, lib, graph_params = relay.build(relay_sym, "cvm")
    assert len(graph_params.keys()) == 0, graph_params.keys()
    logger.debug("Dump json & params to file")
    open(dump_sym, "w").write(graph_json)
    param_bytes = nnvm.compiler.save_param_dict(real_params)
    open(dump_params, "wb").write(param_bytes)

def nnvm_realize(symbol, params, inputs_ext):
    logger = logging.getLogger("log.quant.nnvm.realize")
    def nnvm_const(number, graph, params):
        name = 'const_var_' + str(number)
        prec = math.ceil(math.log2(number)) + 1
        if name not in graph:
            graph[name] = nnvm.sym.Variable(name, shape=(1,), precision=str(prec))
            params[name] = nd.array([number])
        return graph[name]

    def _realize(sym, params, graph, inputs_ext):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        attr, childs = sym.list_attr(), sym_iter(sym.get_children())
        node = sym
        if op_name == 'null':
            return node, params
        elif op_name in ['__rpow_scalar__']:
            base = int(attr['scalar'])
            assert base == 2
            var = nnvm_const(1, graph, params)
            node = nnvm.sym.broadcast_left_shift(var, childs[0])
        elif op_name not in nnvm_identity_ext:
            logger.critical(
                "Unsupported op:%s(name=%s, attr=%s) in INT8 Inference network",
                op_name, name, attr)
        return node, params
    print (sym_collect_attr(symbol))
    ret_sym, params = topo_visit(symbol, params, get_op=get_nnvm_op,
            logger=logger, inputs_ext=inputs_ext, callback=_realize)
    return ret_sym, params

def tvm_params_reduce(symbol, params, inputs_ext, ctx):
    for sym in topo_sort(symbol):
        name, attr = sym.attr('name'), sym.list_attr()
        if sym.attr('op_name') == 'null' and name not in inputs_ext:
            precision = get_attr(attr, "precision")
            val = params[name]
            if precision > 8:
                params[name] = tvm.nd.array(val.asnumpy().astype('int32'), ctx)
            else:
                params[name] = tvm.nd.array(val.asnumpy().astype('int8'), ctx)
    return params

MATRIX_MAXIMUM_SIZE = 65536 # 2 ** 16
def _matrix_decomposition(sym, params, graph, inputs_ext, infer_shapes):
    logger = logging.getLogger('log.sym.pass.matrix_decomposition')
    name = sym.attr('name')
    op_name = sym.attr('op_name')
    childs = sym_iter(sym.get_children())
    attr = sym.list_attr()

    node = sym
    if op_name == 'Convolution':
        # TODO: do matrix decomposition for conv op
        # childs_name = [c.attr('name') for c in childs]
        # childs_shape = [infer_shapes[n] for n in childs_name]

        # for idx, cshape in enumerate(childs_shape):
        #     cname = childs_name[idx]
        #     if cname in params and cshape != params[cname].shape:
        #         logger.critical(
        #             "parameter(%s): infer shape(%s) in graph isn't consistent \
        #             with params dict(%s)",
        #             cshape, params[cname].shape)

        # assert 'layout' not in attr or attr['layout'] == 'NCHW'
        # data_shape = childs_shape[0] # (batch, channel, height, weight)
        # weight_shape = childs_shape[1] # (filter, channel, kernel, kernel)

        # channel = data_shape[1] # channel
        # kernel = [weight_shape[2], weight_shape[3]] # kernel size
        # matrix_len = channel * kernel[0] * kernel[1]
        # print (data_shape, weight_shape, matrix_len)
        pass

    elif op_name == 'FullyConnected':
        childs_name = [c.attr('name') for c in childs]
        childs_shape = [infer_shapes[n] for n in childs_name]

        for idx, cshape in enumerate(childs_shape):
            cname = childs_name[idx]
            if cname in params and cshape != params[cname].shape:
                logger.critical(
                    "parameter(%s): infer shape(%s) in graph isn't consistent \
                    with params dict(%s)",
                    cshape, params[cname].shape)

        # X * W + B
        # (N, C) * (C, M) -> (N, M)
        # C multiply and (C - 1) add
        # (N, 0...K) (N, K...2K) ... (N, pK...C) K = 65526
        # (0...K, M) (K...2K, M) ... (pK...C, M)
        # (N, iK...(i+1)K) * (iK...(i+1)K, M) -> (p, N, M)
        # add
        batch, matrix_len = childs_shape[1]
        if matrix_len > MATRIX_MAXIMUM_SIZE:
            weight_name_prefix = childs[1].attr('name')
            bias = childs[2] if attr['no_bias']=='False' else None

            X, W = childs[0], childs[1]
            if X.attr('op_name') != 'Flatten':
                X = mx.sym.flatten(X)
            weight_params = params[weight_name_prefix]

            nodes = []
            start, step, idx = 0, MATRIX_MAXIMUM_SIZE, 0
            while start < matrix_len:
                stop = min(start + step, matrix_len)

                weight_name = weight_name_prefix + '_split' + str(idx)
                assert weight_name not in graph
                weight = mx.sym.var(weight_name)
                graph[weight_name] = weight

                # TODO: use slice_axis instead of slice
                tmp = mx.sym.slice(X, begin=(0, start), end=(batch, stop))
                tmp = mx.sym.FullyConnected(tmp, weight, bias, **attr)
                nodes.append(tmp)

                params[weight_name] = weight_params.slice(
                        begin=(0, start), end=(batch, stop))
                start, idx = stop, idx+1

            # N1, N2, ..., Np
            #  N1.2, N3.4, ..., N(p-1)p
            #   -> reduce
            while len(nodes) > 1:
                a, b = nodes.pop(0), nodes.pop(0)
                tmp = a + b
                nodes.append(tmp)
            node = nodes[0]

            logger.info("split %s(%s) with shape (%s, %s -> %s(%s)) array",
                    op_name, name, batch, matrix_len, idx, step)

    return node, params

def sym_infer_shape(symbol, params, inputs_ext):
    logger = logging.getLogger('log.symbol.infer_shape')
    check_ext_deps(inputs_ext, 'shape')

    def _infer_shape(sym, params, graph, inputs_ext, infer_shapes):
        logger = logging.getLogger('log.symbol.infer_shape')
        name, op_name = sym.attr('name'), sym.attr('op_name')

        if op_name == 'null':
            if name in params:
                assert params[name].shape == infer_shapes[name], \
                        "parameter %s shape %s is inconsistent with \
                        params dict %s"%(name, infer_shapes[name], params[name].shape)
            return sym, params

        args = sym.list_inputs()
        inputs_shape = {k:tuple(v['shape']) for k,v in inputs_ext.items() if k in args}
        _, out_shapes, _ = sym.infer_shape(**inputs_shape)

        assert len(out_shapes) == 1, 'Infer shape %s'%(name)
        if name in infer_shapes:
            logger.warn("Symbol:%s has been infered shape in graph", out_shapes)
            assert infer_shapes[name] == out_shapes[0], "%s shape %s vs. %s" \
                    % (name, infer_shapes[name], out_shapes)

        infer_shapes[name] = out_shapes[0]

        return sym, params

    inputs = symbol.list_inputs()
    args, auxs = symbol.list_arguments(), symbol.list_auxiliary_states()
    inputs_shape = {k:tuple(v['shape']) for k, v in inputs_ext.items() if k in inputs}
    arg_shapes, _, aux_shapes = symbol.infer_shape(**inputs_shape)
    infer_shapes = {args[i]:arg_shapes[i] for i in range(len(args))}
    infer_shapes.update({auxs[i]:aux_shapes[i] for i in range(len(auxs))})

    _, _ = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_infer_shape, infer_shapes=infer_shapes)

    return infer_shapes

def convert_input_format(symbol, params, logger=logging,
        src_format="NHWC", des_format="NCHW"):
    assert sorted(src_format) == sorted(des_format)
    assert len(set(src_format)) == len(src_format)

    if src_format == des_format:
        return symbol, params

    axes = [des_format.find(c) for c in src_format]
    axes = tuple(axes)
    def _data_convert(sym, params, graph, inputs_ext):
        name, attr = sym.attr("name"), sym.list_attr()
        if name == "data":
            shp = None
            if "__shape__" in attr:
                shp_axes = [src_format.find(c) for c in des_format]
                src_shp = eval(attr["__shape__"])
                shp = [src_shp[ax] for ax in shp_axes]
            sym = mx.sym.var("data", shape=shp)
            sym = mx.sym.transpose(sym, axes=axes, name="data_transpose")
        return sym, params

    return topo_visit(symbol, params, {},
            callback=_data_convert, logger=logger)

    # assert src_format in ["NCHW", "NHWC"]
    # assert des_format in ["NCHW", "NHWC"]
    # axes = tuple()
    # if src_format == "NCHW" and des_format == "NHWC":
    #     axes = (0, 2, 3, 1)
    # elif src_format == "NHWC" and des_format == "NCHW":
    #     axes = (0, 3, 1, 2)
    # graph, inp = {}, {}
    # for sym in topo_sort(symbol, logger=logger):
    #     name, op_name = sym.attr('name'), sym.attr('op_name')
    #     childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    #     if childs is not None:
    #         nchilds = []
    #         for c in childs:
    #             if c.attr('name') == 'data':
    #                 tname = 'data_transpose'
    #                 csym = mx.sym.transpose(data=graph['data'], axes=axes, name=tname)
    #                 nchilds.append(csym)
    #                 graph[tname] = csym
    #             else:
    #                 nchilds.append(graph[c.attr('name')])
    #         sym = get_mxnet_op(op_name)(*nchilds, **attr, name=name)
    #     graph[name] = sym

    # nodes = [get_node(sym, graph) for sym in symbol]
    # ret = mx.sym.Group(nodes) if len(nodes) > 1 else nodes[0]
    # return ret, params

def sym_robust_infer_shape(symbol, params, inputs_ext):
    logger = logging.getLogger('log.symbol.infer_shape')
    check_ext_deps(inputs_ext, 'shape')

    def _infer_shape(sym, params, graph, inputs_ext, infer_shapes):
        logger = logging.getLogger('log.symbol.infer_shape')
        name, op_name = sym.attr('name'), sym.attr('op_name')

        if op_name == 'null':
            if name in params:
                assert params[name].shape == infer_shapes[name], \
                        "parameter %s shape %s is inconsistent with \
                        params dict %s"%(name, infer_shapes[name], params[name].shape)
            return sym, params

        args = sym.list_inputs()
        inputs_shape = {k:tuple(v['shape']) for k,v in inputs_ext.items() if k in args}
        _, out_shapes, _ = sym.infer_shape(**inputs_shape)

        # assert len(out_shapes) == 1, 'Infer shape %s'%(name)
        if name in infer_shapes:
            logger.warn("Symbol:%s has been infered shape in graph", out_shapes)
            assert infer_shapes[name] == out_shapes[0], "%s shape %s vs. %s" \
                    % (name, infer_shapes[name], out_shapes)

        infer_shapes[name] = out_shapes

        return sym, params

    inputs = symbol.list_inputs()
    args, auxs = symbol.list_arguments(), symbol.list_auxiliary_states()
    inputs_shape = {k:tuple(v['shape']) for k, v in inputs_ext.items() if k in inputs}
    print (symbol.attr('name'), inputs_shape)
    arg_shapes, _, aux_shapes = symbol.infer_shape(**inputs_shape)
    infer_shapes = {args[i]:arg_shapes[i] for i in range(len(args))}
    infer_shapes.update({auxs[i]:aux_shapes[i] for i in range(len(auxs))})

    _, _ = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_infer_shape, infer_shapes=infer_shapes)

    return infer_shapes

def _sym_check(sym, params, graph, inputs_ext):
    logger = logging.getLogger('log.prepare.symbol.check')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    if op_name not in mx_identity_ext:
        logger.error("%s(%s) has not been considered in quantization",
                name, op_name)
        return sym, params
    attr = sym.list_attr()
    std_attr = mx_identity_ext[op_name]
    for k,v in std_attr.items():
        if k in attr:
            assert attr[k] in v, \
                "%s(%s attr=%s) not match attribute %s (%s vs. %s)" \
                % (name, op_name, attr, k, attr[k], v)
        else:
            assert v[0], "%s(%s attr=%s) not contains attribute %s" \
                % (name, op_name, attr, k)

    if op_name == 'Pooling' and attr['pool_type'] == 'avg':
        msg = "%s(%s attr=%s) not match attribute %s (%s vs. %s)"
        if 'pooling_convention' in attr:
            pooling_convention = attr['pooling_convention']
            if pooling_convention == 'full':
                assert 'global_pool' in attr and \
                    attr['global_pool'] == 'True', msg \
                    % (name, op_name, attr, 'pooling_convention&global_pool',
                    [attr['pooling_convention'], attr['global_pool']],
                    ['full', 'True'])
            else:
                assert pooling_convention == 'valid', msg \
                    % (name, op_name, attr, 'pooling_convention',
                    attr['pooling_convention'], 'valid')
    return sym, params

def _sym_rewrite(sym, params, graph, inputs_ext, infer_shapes):
    logger = logging.getLogger('log.prepare.symbol.rewrite')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    node = sym
    if op_name == 'Convolution':
        X, W = childs[0], childs[1]
        X_name, W_name = X.attr('name'), W.attr('name')
        # X_name, W_name = childs[0].attr('name'), childs[1].attr('name')
        layout = get_attr(attr, 'layout', "NCHW")
        if layout == "NCW":
            no_bias = get_attr(attr, 'no_bias', False)
            dilate, kernel = get_attr(attr, 'dilate'), get_attr(attr, 'kernel')
            pad, stride = get_attr(attr, 'pad'), get_attr(attr, 'stride')
            num_filter = get_attr(attr, 'num_filter')
            num_group = get_attr(attr, 'num_group', 1)
            attr = {
                'layout': "NCHW", 'no_bias': no_bias,
                'dilate': (*dilate, 1), 'kernel': (*kernel, 1),
                'pad': (*pad, 0), 'stride': (*stride, 1),
                'num_filter': num_filter, 'num_group': num_group,
            }
            X = mx.sym.expand_dims(X, axis=3)
            params[W_name] = params[W_name].expand_dims(axis=3)
            W = graph[W_name] = mx.sym.var(W_name, shape=params[W_name].shape)
            B = None if no_bias else childs[2]
            node = get_mxnet_op(op_name)(X, W, B, **attr, name=name)
            node = mx.sym.squeeze(node, axis=3)
        else:
            assert layout == "NCHW", "%s(name=%-40s attr=%s)" \
                           % (op_name, name, attr)
    elif op_name == 'Pooling':
        pool_type = attr['pool_type']
        is_global = get_attr(attr, "global_pool", False)
        node = get_mxnet_op(op_name)(*childs, **attr, name=name)
        if pool_type == 'avg' and is_global:
            input_name = childs[0].attr('name')
            input_shape = infer_shapes[input_name]
            assert len(input_shape) == 4

            scale_name = input_name + '_avg_scale'
            assert scale_name not in graph
            scale_sym = mx.sym.var(scale_name, shape=(1,))
            graph[scale_name] = scale_sym

            params[scale_name] = nd.array([1. /
                    (input_shape[2] * input_shape[3])])

            node = mx.sym.sum(childs[0], axis=(2, 3))
            node = mx.sym.broadcast_mul(node, scale_sym)
        elif pool_type == 'avg':
            X = childs[0]
            X_shape = infer_shapes[X.attr('name')]
            in_channel = X_shape[1]
            kernel = get_attr(attr, 'kernel')
            if isinstance(kernel, int):
                kernel = (kernel, kernel)
            conv_attr = {
                'no_bias': 'True',
                'dilate': '(1, 1)',
                'kernel': kernel,
                'stride': attr['stride'],
                'pad': attr['pad'],
                'layout': 'NCHW',
                'num_filter': in_channel,
                'num_group': in_channel,
            }
            conv_name = name.replace('pool', 'pool_conv')
            W_name = conv_name + '_weight'
            assert W_name not in graph
            W_shape = (in_channel, 1, *kernel)
            graph[W_name] = W = mx.sym.var(W_name, shape=W_shape)
            params[W_name] = nd.full(shape=W_shape, val=(1/np.product(kernel)))
            node = mx.sym.Convolution(X, W, **conv_attr, name=conv_name)
        else:
            assert pool_type == 'max', "Unsupported Pooling \
                    %s(%s, pool_type=%s)"%(op_name, name, pool_type)
    elif op_name == 'LeakyReLU':
        act = get_attr(attr, 'act_type', 'leaky')
        slope = get_attr(attr, 'slope', 0.25)
        assert act == 'leaky', "Unsupported LeakyReLU %s for act_type: %s" \
                % (name, act)
        X = childs[0]
        posi_X = mx.sym.relu(X)
        nega_X = mx.sym.negative(X)
        nega_X = mx.sym.relu(nega_X)
        slope_name = name + "_slope"
        params[slope_name] = nd.array([slope])
        graph[slope_name] = slope_sym = mx.sym.var(slope_name, shape=(1,))
        scale_X = mx.sym.broadcast_mul(nega_X, slope_sym)
        node = posi_X - scale_X
    elif op_name == 'BatchNorm':
        # data, gamma, beta, data_mean, data_var
        assert len(childs) == 5
        X = childs[0]
        X_shape = infer_shapes[X.attr('name')]
        in_channel = X_shape[1]
        gamma = params[childs[1].attr('name')]
        beta = params[childs[2].attr('name')]
        data_mean = params[childs[3].attr('name')]
        data_var = params[childs[4].attr('name')]

        fix_gamma = get_attr(attr, 'fix_gamma', True)
        gamma = 1 if fix_gamma else gamma
        axis = get_attr(attr, 'axis', 1)
        assert axis == 1

        epsilon = float(attr['eps']) if 'eps' in attr else 1e-5
        scale = gamma / nd.sqrt(data_var + epsilon)
        bias = beta - scale * data_mean

        if X.attr('op_name') == 'Convolution':
            conv_attr = X.list_attr()
            conv_childs = sym_iter(X.get_children())

            conv_name = combile_name(X.attr('name'), name)
            W_name = conv_name + '_weight'
            weight = params[conv_childs[1].attr('name')]
            params[W_name] = weight * scale.reshape(*scale.shape, 1, 1, 1)
            W = graph[W_name] = mx.sym.var(W_name, shape=params[W_name].shape)

            B_name = conv_name + '_bias'
            assert B_name not in graph, "bias name %s has existed in graph %s" \
               % (B_name, graph.keys())
            if not get_attr(conv_attr, 'no_bias', False):
               bias += params[conv_childs[2].attr('name')]
            params[B_name] = bias
            B = graph[B_name] = mx.sym.var(B_name, shape=bias.shape)

            conv_attr['no_bias'] = 'False'
            node = mx.sym.Convolution(conv_childs[0], W,
                   B, **conv_attr, name=conv_name)
            logger.info("fuse Convolution=%-40s and batchnorm=%-40s",
                   X.attr('name'), name)
        else:
            w_name = name + "_weight"
            params[w_name] = scale.reshape((1, in_channel, 1, 1))
            graph[w_name] = W = mx.sym.var(w_name, shape=(1, in_channel, 1, 1))
            node = mx.sym.broadcast_mul(X, W, name=name+"_mul")
            bias_name = name + "_bias"
            params[bias_name] = bias.reshape((1, in_channel, 1, 1))
            graph[bias_name] = B = mx.sym.var(bias_name,
                    shape=(1, in_channel, 1, 1))
            node = mx.sym.broadcast_add(node, B, name=name+"_add")
            logger.info("rewrite BatchNorm=%-40s into alpha * X + beta", name)
    elif op_name == 'Dropout':
        # dropout is identity during testing
        node = childs[0]
    elif op_name == '_mul_scalar':
        X = childs[0]
        scalar = get_attr(attr, 'scalar')
        if scalar == 0:
            params[name] = nd.zeros(infer_shapes[name])
            node = mx.sym.var(name, shape=infer_shapes[name])
        else:
            sname = name + '_scalar'
            params[sname] = nd.array([scalar])
            graph[sname] = scale = mx.sym.var(sname, shape=(1,))
            node = mx.sym.broadcast_mul(X, scale, name=name)
    elif op_name == '_div_scalar':
        X = childs[0]
        scalar = get_attr(attr, 'scalar')
        sname = name + '_scalar'
        params[sname] = nd.array([1 / scalar])
        graph[sname] = scale = mx.sym.var(sname, shape=(1,))
        node = mx.sym.broadcast_mul(X, scale, name=name)
    elif op_name == '_plus_scalar':
        X = childs[0]
        scalar = get_attr(attr, 'scalar')
        if scalar == 0:
            node = X
        else:
            sname = name + '_scalar'
            params[sname] = nd.array([scalar])
            graph[sname] = offset = mx.sym.var(sname, shape=(1,))
            node = mx.sym.broadcast_add(X, offset, name=name)
    elif op_name == 'zeros_like':
        X = childs[0]
        params[name] = nd.zeros(infer_shapes[name])
        node = mx.sym.var(name, shape=infer_shapes[name])
    elif op_name == 'ones_like':
        X = childs[0]
        params[name] = nd.zeros(infer_shapes[name])
        node = mx.sym.var(name, shape=infer_shapes[name])
    elif op_name == 'SwapAxis':
        dim1 = get_attr(attr, 'dim1', 0)
        dim2 = get_attr(attr, 'dim2', 0)
        ndim = len(infer_shapes[name])
        new_axes = [None] * ndim
        for i in range(ndim):
            new_axes[i] = dim2 if i==dim1 else i
            new_axes[i] = dim1 if i==dim2 else new_axes[i]
        node = mx.sym.transpose(childs[0], tuple(new_axes), name=name)
    infer_shapes[node.attr('name')] = infer_shapes[name]
    return node, params

def _fuse_bias(sym, params, graph, inputs_ext, infer_shapes):
    name = sym.attr('name')
    op_name = sym.attr('op_name')
    childs = sym_iter(sym.get_children())
    attr = sym.list_attr()

    node = sym
    if op_name in ['FullyConnected', 'Convolution']:
        if attr['no_bias'] == 'False':
            attr['no_bias'] = 'True'

            bias_name = childs[2].attr('name')
            bias = params[bias_name]

            shape = list(infer_shapes[name])
            assert len(bias.shape) == 1
            assert shape [1] == bias.shape[0]
            shape = [1 if i!=1 else s for i,s in enumerate(shape)]

            params[bias_name] = bias.reshape(shape)
            bias_sym = mx.sym.var(bias_name, shape=shape)
            graph[bias_name] = bias_name

            node = get_mxnet_op(op_name)(childs[0], childs[1],
                    **attr, name=name)
            node = mx.sym.broadcast_add(node, bias_sym, name=name+'_add')
    return node, params

def _fuse_constant(sym, params, graph, inputs_ext):
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    node = sym
    if op_name == 'null':
        return node, params
    elif childs is None:
        out = get_nd_op(op_name)(**attr)
        params[name] = out
        node = mx.sym.var(name, shape=out.shape)
    else:
        is_param = lambda c: (c.attr('op_name')=='null') and \
                        (c.attr('name') not in inputs_ext)
        flag = all([is_param(c) for c in childs])
        if flag:
            in_params = [params[c.attr('name')] for c in childs]
            out = get_nd_op(op_name)(*in_params, **attr)
            params[name] = out
            node = mx.sym.var(name, shape=out.shape)
    return node, params

def _reduce_graph(sym, params, graph, inputs_ext):
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    is_param = lambda c: (c.attr('op_name')=='null') and \
                    (c.attr('name') not in inputs_ext)
    # node param1
    #   \   /            reduce      node op(param1, param2)
    #  operator param1   =======>      \   /
    #        \   /                    operator
    #       operator
    node = sym
    if op_name in ['broadcast_mul']:
        A, B = childs[0], childs[1]
        if A.attr('op_name') not in ['broadcast_mul']:
            return node, params
        if not is_param(B):
            return node, params
        A_A, A_B = sym_iter(A.get_children())
        if not is_param(A_B):
            return node, params
        B_name, A_B_name = B.attr('name'), A_B.attr('name')
        if params[B_name].shape != (1,) and params[A_B_name].shape != (1,):
            return node, params
        fuse_name = B_name.split("_")
        fuse_name = "%s_%s"%("_".join(fuse_name),
                "_".join([n for n in A_B_name.split("_") if n not in fuse_name]))
        params[fuse_name] = get_nd_op(op_name)(params[B_name], params[A_B_name])
        fuse_sym = mx.sym.var(fuse_name, shape=params[fuse_name].shape)
        node = get_mxnet_op(op_name)(A_A, fuse_sym, **attr, name=name)
    return node, params

def fuse_transpose(symbol, params, logger=logging):
    def _fuse_transpose(sym, params, graph, inputs_ext):
        name, op = sym.attr('name'), sym.attr('op_name')
        childs = sym_iter(sym.get_children())

        #   node                                 node              node
        #     |                reduce              |
        # transpose1         =========>        transpose3     or
        #     |
        # transpose2
        if op == 'transpose':
            axes = get_attr(sym.list_attr(), 'axes')
            if childs[0].attr('op_name') == 'transpose':
                caxes = get_attr(childs[0].list_attr(), 'axes')
                axes = [caxes[ii] for ii in axes]
                sym = sym_iter(childs[0].get_children())[0]
                if axes != sorted(axes):
                    sym = mx.sym.transpose(sym, axes=axes, name=name)

        #   node                                 node
        #     |                switch              |
        # transpose1         =========>          relu
        #     |                                    |
        #   relu                               transpose1
        elif op == 'relu':
            if childs[0].attr('op_name') == 'transpose':
                name, attr = childs[0].attr('name'), childs[0].list_attr()
                axes = get_attr(attr, 'axes')
                sym = mx.sym.relu(sym_iter(childs[0].get_children())[0])
                sym = mx.sym.transpose(sym, axes=axes, name=name)

        #     node     ...    node
        #       |               |
        #   transpose1 ...  transpose1                     node    node
        #       \               /              reduce        \       /
        #             concat                 =========>        concat
        #                                                        |
        #                                                    transpose1
        elif op == 'Concat':
            same, axeses = True, set()
            for child in childs:
                if child.attr('op_name') != 'transpose':
                    same = False
                    break
                attr = child.list_attr()
                axeses.add(get_attr(attr, 'axes'))
            if same and len(axeses) == 1:
                clist, attr = [], sym.list_attr()
                dim = get_attr(attr, 'dim')
                axes = list(axeses)[0]
                for child in childs:
                    cchild = sym_iter(child.get_children())[0]
                    clist.append(cchild)
                sym = mx.sym.concat(*clist, dim=axes[dim])
                name = 'fusetranspose_' + name
                sym = mx.sym.transpose(sym, axes=axes, name=name)

        #    node                           node             
        #      |              reduce          |
        #  transpose1       =========>       sum
        #      |
        #     sum
        # 
        # only when 'keepdims' == False
        # if not specified, defalt: 'exclude' == False, 'keepdims' == False
        elif op == 'sum':
            attr = sym.list_attr()
            axis, keepdims = get_attr(attr, 'axis', None), get_attr(attr, 'keepdims', False)
            if childs[0].attr('op_name') == 'transpose' and not keepdims:
                attr = childs[0].list_attr()
                axes = get_attr(attr, 'axes')
                sym = sym_iter(childs[0].get_children())[0]
                axis = [axes[i] for i in axis]
                sym = mx.sym.sum(sym, axis=axis, keepdims=keepdims)
            #  TODO(ryt): if an op 'sum' has attr 'exclude=True'
            #  change it to 'False'

        return sym, params

    logger.info("redundant transposes fused.")

    return topo_visit(symbol, params, {},
            callback=_fuse_transpose, logger=logger)

def fuse_multiple_outputs(symbol, params, inputs_ext, logger):
    infer_shapes = sym_robust_infer_shape(symbol, params, inputs_ext)
    print ("Robust infer shape")
    channel, graph = {}, {}
    for sym in topo_sort(symbol, logger=logger):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        if childs is not None:
            childs = [get_node(c, graph) for c in childs]
            sym = get_mxnet_op(op_name)(*childs, **attr, name=name)
        if op_name == 'SliceChannel':
            # Only designed for special usei, thus
            # check of "SliceChannel" has not been added to _sym_check
            assert childs is not None and len(childs) == 1, \
                "Invalid Layer: %s, the 'SliceChannel' \
                operator must have exactly one input" % name
            axis = get_attr(attr, 'axis', 1)
            num_outputs = get_attr(attr, 'num_outputs')
            chchild_shape = infer_shapes[childs[0].attr('name')]
            eid = get_entry_id(childs[0])
            dim = chchild_shape[eid][axis]
            assert num_outputs > 0 and dim % num_outputs == 0, \
                "Invalid Layer: %s, the 'SliceChannel' operator \
                has a wrong attribute, 'num_outputs': %d" \
                % (name, num_outputs)
            stride = int(dim / num_outputs)
            interval = [(i * stride, (i + 1) * stride) \
                       for i in range(num_outputs)]
            channel[name] = [childs, axis, interval]
        elif childs is not None:
            is_split = False
            for i in range(len(childs)):
                cname = childs[i].attr('name')
                if cname in channel:
                    is_split = True
                    eid = get_entry_id(childs[i])
                    chchilds, axis, interval = channel[cname]
                    begin, end = interval[eid]
                    chattr = {'axis': axis, 'begin': begin, 'end': end}
                    slp_name = "%s_slice_axis%d" % (cname, eid)
                    if slp_name not in graph:
                        graph[slp_name] = mx.sym.slice_axis(*chchilds,
                                **chattr, name=slp_name)
                    childs[i] = graph[slp_name]
            if is_split:
                sym = get_mxnet_op(op_name)(*childs, **attr, name=name)
        graph[name] = sym

    nodes = [get_node(sym, graph) for sym in symbol]
    ret = mx.sym.Group(nodes) if len(nodes) > 1 else nodes[0]
    return ret, params

def sym_quant_prepare(symbol, params, inputs_ext, graph_ext={}):
    logger = logging.getLogger('log.sym.pass.prepare')

    topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_sym_check)

    symbol, params = fuse_multiple_outputs(symbol, params, inputs_ext, logger)

    infer_shapes = sym_infer_shape(symbol, params, inputs_ext)
    sym, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_sym_rewrite, infer_shapes=infer_shapes)

    # infer_shapes = sym_infer_shape(sym, params, inputs_ext)
    # sym, params = topo_visit(sym, params, get_op=get_mxnet_op,
    #         logger=logger, inputs_ext=inputs_ext,
    #         callback=_fuse_bias, infer_shapes=infer_shapes)

    infer_shapes = sym_infer_shape(sym, params, inputs_ext)
    sym, params = topo_visit(sym, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_matrix_decomposition, infer_shapes=infer_shapes)

    sym, params = topo_visit(sym, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_fuse_constant)

    sym, params = topo_visit(sym, params, get_op=get_mxnet_op,
           logger=logger, inputs_ext=inputs_ext,
           callback=_reduce_graph)

    sym, params = check_graph(sym, params)
    return sym, params

def sym_attach_attrs(symbol, params, inputs_ext, **kwargs):
    logger = logging.getLogger('log.sym.attach.attrs')
    def _attach_attr(sym, params, graph, inputs_ext, **kwargs):
        name = sym.attr('name')
        op_name = sym.attr('op_name')
        attr = sym.list_attr()
        childs = sym_iter(sym.get_children())
        for k,v in kwargs.items():
            if name not in v:
                continue
            attr[k] = str(v[name])

        if op_name == 'null':
            sym = mx.sym.var(name, attr=attr)
        return sym, params

    return topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_attach_attr, **kwargs)

import hashlib
import shutil
NAME_MAPS = {
    'Convolution': 'conv2d',
    'stride': 'strides',
    'kernel': 'kernel_size',
    'pad': 'padding',
    'num_filter': 'channels',
    'num_group': 'groups',
    'dilate': 'dilation',
    'no_bias': 'use_bias',
}
def sym_dump_ops(symbol, params, inputs_ext, datadir="/data/op_std_out",
        dtype="float64", out_dtype="int32",
        ctx=mx.gpu(), cleanDir=False, ops=None):
    logger = logging.getLogger('log.sym.dump.ops')
    check_ext_deps(inputs_ext, 'data')

    if cleanDir:
        logger.info("Clean directory: %s", datadir)
        shutil.rmtree(datadir, ignore_errors=True)

    npdir = "%s/%s" % (datadir, ".hidden.out")
    os.makedirs(npdir, exist_ok=True)

    def npy_txt(data):
        data = data.astype("int32")
        shp = data.shape
        txt = "{}\n{}\n{}\n".format(
            len(shp),
            " ".join([str(s) for s in shp]),
            " ".join([str(s) for s in data.flatten()]),
        )
        return txt
    def txt_sha256(data):
        return hashlib.sha256(data.encode()).hexdigest()
    def dump_txt(hsh_file, ln_file, data):
        logger = logging.getLogger('log.ops.txt.dump')

        if os.path.exists(hsh_file):
            loaded = open(hsh_file, "r").read()
            if data != loaded:
                logger.error(
                    "Dump op failed: hash file=%s, link file=%s",
                        hsh_file, ln_file)
                return False
        open(hsh_file, "w").write(data)
        os.symlink(hsh_file, ln_file)
        return True
    def dump_op(op_name, attr, ins, outs):
        op_name = NAME_MAPS.get(op_name, op_name)
        ins = [_in.asnumpy().astype(out_dtype) for _in in ins]
        ins = [npy_txt(_in) for _in in ins]
        hshes = [txt_sha256(_in) for _in in ins]
        hsh = hashlib.sha1("{}{}".format(hshes, attr)
                    .encode()).hexdigest()
        hsh_dir = "%s/%s/%s" % (datadir, op_name, hsh)
        if os.path.exists(hsh_dir):
            logger.info("Skip op:%-20s hashdir=%s",
                op_name, hsh)
            return

        os.makedirs(hsh_dir, exist_ok=True)
        attr_file = "%s/%s" % (hsh_dir, "attr.txt")
        for k, v in NAME_MAPS.items():
            if k in attr:
                attr[v] = eval(attr[k])
                del attr[k]
        if 'use_bias' in attr:
            attr['use_bias'] = not attr['use_bias']
        attr_str = str(attr).replace("'", "\"")
        with open(attr_file, "w") as fout:
            fout.write(attr_str + "\n")
        for i, _in in enumerate(ins):
            in_file = "%s/in_%d.txt" % (hsh_dir, i)
            hsh_file = "%s/%s.txt" % (npdir, hshes[i])
            if not dump_txt(hsh_file, in_file, _in):
                shutil.rmtree(hsh_dir)
                return
        for i, _out in enumerate(outs):
            out_file = "%s/out_%d.txt" % (hsh_dir, i)
            _out = _out.asnumpy().astype(out_dtype)
            _out = npy_txt(_out)
            out_hsh = txt_sha256(_out)
            hsh_file = "%s/%s.txt" % (npdir, out_hsh)
            if not dump_txt(hsh_file, out_file, _out):
                shutil.rmtree(hsh_dir)
                return

        idx_file = "%s/%s/%s" % (datadir, op_name, "index")
        open(idx_file, "a").write(hsh + "\n")

    order, deps = topo_sort(symbol, logger=logger, with_deps=True)
    out_cache = {}
    for sym in order:
        name, op_name = sym.attr('name'), sym.attr('op_name')
        attr, childs = sym.list_attr(), sym_iter(sym.get_children())
        if op_name == 'null':
            out = inputs_ext[name]['data'] if name in inputs_ext \
                  else params[name]
            out_cache[name] = [out.round().astype(dtype).as_in_context(ctx)]
            continue

        assert childs is not None
        cinfos = [(c.attr('name'), get_entry_id(c)) for c in childs]
        nd_inputs = [out_cache[n[0]][n[1]] for n in cinfos]
        if get_attr(attr, 'op_type', 'null')=='cvm_clip' and \
                dtype=="float64":
            np_inputs = [o.asnumpy() for o in nd_inputs]
            precision = get_attr(attr, 'precision')
            amax = (2 ** (precision - 1)) - 1
            amin = -amax
            np_out = np.clip(np_inputs[0], amin, amax)
            out = nd.array(np_out, dtype=dtype)
        else:
            out = get_nd_op(op_name)(*nd_inputs, **attr)
            if op_name == 'broadcast_div':
                out = out.astype('int32').astype(dtype)
        out = [out] if len(sym) == 1 else out
        out_cache[name] = [o.round().astype(dtype).as_in_context(ctx) for o in out]
        out = out_cache[name]

        op_name = attr['op_type'] if op_name=='Custom' else op_name

        if ops is None or name in ops:
            logger.debug("Dump op:%s attr=%s", op_name, attr)
            dump_op(op_name, attr, nd_inputs, out)
        for n, _ in cinfos:
            assert n in deps
            deps[n].remove(name)
            if len(deps[n]) == 0:
                del out_cache[n]

def sym_dump_layer_outputs(symbol, params, inputs_ext,
        datadir, max_num=20,
        dtype="float64", out_dtype='int32', data_dtype="int8",
        ctx=mx.gpu(),
        dump_ops=[]):
    logger = logging.getLogger('log.sym.dump.internals')
    check_ext_deps(inputs_ext, 'data')
    def _str_output(out, start=None, end=None):
        out = out.asnumpy().flatten().astype(out_dtype)
        out = out[start:end] if end else out
        dump = ' '.join(str(d) for d in out)
        return dump
    def _str_feature(out):
        maxes = [o.max().astype(out_dtype).asscalar() for o in out]
        mines = [o.min().astype(out_dtype).asscalar() for o in out]
        return "min=%s, max=%s" % (min(mines), max(maxes))

    DUMP_SUFFIX = "mrt.dump"
    NPY_SUFFIX = "npy"
    ATTR_SUFFIX = "attr"
    logger.info("Clean datadir: %s", datadir)
    os.makedirs(datadir, exist_ok=True)
    for fname in os.listdir(datadir):
        if fname.endswith(DUMP_SUFFIX) or fname.endswith(NPY_SUFFIX) or \
        fname.endswith(ATTR_SUFFIX):
            os.remove(datadir + '/' + fname)

    for k, v in inputs_ext.items():
        np.save("%s/%s"%(datadir, k),
                v['data'].asnumpy().astype(data_dtype))

    order, deps = topo_sort(symbol, logger=logger, with_deps=True)
    out_cache = {}
    for sym in order:
        name, op_name = sym.attr('name'), sym.attr('op_name')
        attr, childs = sym.list_attr(), sym_iter(sym.get_children())
        if name in dump_ops or op_name in dump_ops:
            cs = [] if childs is None else childs
            for i, c in enumerate(cs):
                dump_in = "%s/%s_%d.%s.in" % (datadir, name, i, DUMP_SUFFIX)
                out = out_cache[c.attr('name')][get_entry_id(c)].asnumpy().round().astype(out_dtype)
                np.save(dump_in, out)
            dump_attr = "%s/%s.%s" % (datadir, name, ATTR_SUFFIX)
            open(dump_attr, "w").write(str(attr))

        if op_name == 'null':
            out = inputs_ext[name]['data'] if name in inputs_ext \
                  else params[name]
        elif childs is None:
            out = get_nd_op(op_name)(**attr)
        elif get_attr(attr, 'op_type', 'null')=='cvm_clip' and \
                dtype=="float64":
            cinfos = [(c.attr('name'), get_entry_id(c)) for c in childs]
            nd_inputs = [out_cache[n[0]][n[1]] for n in cinfos]
            np_inputs = [o.asnumpy() for o in nd_inputs]
            precision = get_attr(attr, 'precision')
            amax = (2 ** (precision - 1)) - 1
            amin = -amax
            np_out = np.clip(np_inputs[0], amin, amax)
            out = nd.array(np_out, dtype=dtype)
        else:
            cinfos = [(c.attr('name'), get_entry_id(c)) for c in childs]
            nd_inputs = [out_cache[n[0]][n[1]] for n in cinfos]
            out = get_nd_op(op_name)(*nd_inputs, **attr)
            if op_name == 'broadcast_div':
                out = out.astype('int32').astype(dtype)
            for n, _ in cinfos:
                assert n in deps
                deps[n].remove(name)
                if len(deps[n]) == 0:
                    del out_cache[n]
        out = [out] if len(sym) == 1 else out
        out_cache[name] = [o.round().astype(dtype).as_in_context(ctx) for o in out]
        out = out_cache[name]

        if name in dump_ops or op_name in dump_ops:
            cs = [] if childs is None else childs
            for i, o in enumerate(out):
                dump_out = "%s/%s_%d.%s.out" % (datadir, name, i, DUMP_SUFFIX)
                np.save(dump_out, o.asnumpy().astype(out_dtype))

        prefix = "parameters" if op_name=='null' else op_name
        prefix = attr['op_type'] if op_name=='Custom' else prefix
        dump_file = "%s/%s.%s" % (datadir, prefix, DUMP_SUFFIX)
        with open(dump_file, "a+") as fout:
            fout.write(name + ": ")
            fout.write(_str_feature(out))
            fout.write("\n")
            for i, o in enumerate(out):
                fout.write(_str_output(o, end=max_num))
                fout.write("\n")
        logger.debug("Dump %-20s name=%-40s", op_name, name)

    for i, sym in enumerate(symbol):
        name, eid = sym.attr('name'), get_entry_id(sym)
        np.save("%s/result_%d"%(datadir, i),
                out_cache[name][eid].asnumpy().astype(out_dtype))

def sym_calculate_ops(symbol, params, inputs_ext):
    logger = logging.getLogger("log.calculate.ops")
    ops = {}
    infer_shapes = sym_infer_shape(symbol, params, inputs_ext)
    for sym in topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        msg = "%-20s name=%-40s ops=%-15s oshape=%-20s ishape=%-50s attr=%s"
        cshapes = [infer_shapes[c.attr('name')] for c in childs] if childs else []
        if op_name == 'null':
            continue
        base_ops, ext = 1, "{}"
        if op_name in ['Convolution', 'FullyConnected']:
            W_shape = cshapes[1]
            base_ops = np.product(W_shape[1:]) * 2
            if not get_attr(attr, 'no_bias', False):
                base_ops += 1
        elif op_name in ['BatchNorm']:
            base_ops = 4
        elif op_name in ['Activation']:
            if attr['act_type'] != "relu":
                assert False
        elif op_name in ['Pooling']:
            pool_type = attr['pool_type']
            is_global = get_attr(attr, 'global_pool', False)
            if is_global:
                _, _, K1, K2 = cshapes[0]
            else:
                K1, K2 = eval(attr['kernel'])
            assert pool_type in ['avg', 'max']
            base_ops = K1 * K2
            if pool_type == 'avg':
                base_ops += 1
            ext = "{'kernel': %s}"%attr['kernel']
        elif op_name in ['Custom']:
            op_type = attr['op_type']
            assert op_type in ['cvm_clip', 'cvm_left_shift', 'cvm_right_shift']
        elif op_name in ['broadcast_mul', 'broadcast_add', 'broadcast_sub', 'Flatten',
            'elemwise_add', 'elemwise_sub', 'relu', 'slice', 'clip', 'negative',
            'slice_like', 'slice_axis', 'repeat', 'tile', 'expand_dims',
            'Reshape', 'transpose', 'Flatten', 'Concat', 'UpSampling']:
            # base op is 1, do nothing
            pass
        elif op_name in ['Dropout']:
            base_ops = 0
        elif op_name in ['sum']:
            axis = eval(attr['axis'])
            base_ops = np.product([cshapes[0][i] for i in axis])
            ext = "{'axis': %s}"%attr['axis']
        else:
            logger.critical("%s(%s) has not been considered", op_name, name)
        count = np.product(infer_shapes[name][1:]) * base_ops
        ops[name] = count
        logger.debug(msg, op_name, name, count,
                infer_shapes[name], cshapes, ext)

    total_ops = 0
    for k,v in ops.items():
        total_ops += v
    LEVELS = ['', 'K', 'M', 'G', 'T']
    idx, red_ops = 0, total_ops
    while red_ops > 1000:
        red_ops /= 1000
        idx += 1
    logger.info("Graph Total OPs: {} eqs. {:5.2f}{}".format(total_ops,
            red_ops, LEVELS[idx]))
    top_k = 5
    logger.info("========== Top %d OPs ==========", top_k)
    sorted_ops = sorted(ops.items(), key=lambda item: item[1], reverse=True)
    for i in range(top_k):
        k, v = sorted_ops[i]
        logger.info("{:3d} | name={:40s} ops={:<15d} percent={:6.2%}".format(
                i, k, v, v / total_ops))
    return total_ops

def sym_infer_precision(symbol, params, inputs_ext):
    logger = logging.getLogger('log.infer.precision')
    infer_shapes = sym_infer_shape(symbol, params, inputs_ext)
    MAX_BIT = 32
    precs = {}
    for sym in topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        cns = [c.attr('name') for c in childs] if childs else []
        cprecs = [precs[n] for n in cns]
        if op_name == 'null':
            oprec = get_attr(attr, 'precision')
        elif op_name in ['Convolution', 'FullyConnected']:
            wshp = infer_shapes[cns[1]]
            sum_len = np.product(wshp[1:])
            sum_bit = math.ceil(math.log2(sum_len))
            oprec = cprecs[0] + cprecs[1] + sum_bit
            if not get_attr(attr, 'no_bias', False):
                oprec = max(oprec, cprecs[2])
                oprec += 1
        elif op_name in [
                'Activation', 'relu',
                'Pooling',
                'slice', 'slice_like', 'slice_axis',
                'clip', 'negative',
                'repeat', 'tile', 'expand_dims',
                'Reshape', 'transpose', 'Flatten',
        ]:
            oprec = cprecs[0]
        elif op_name in ['broadcast_add', 'broadcast_sub',
                'elemwise_add', 'elemwise_sub', 'Concat']:
            oprec = max(cprecs)
        elif op_name in ['broadcast_mul']:
            oprec = cprecs[0] + cprecs[1]
        elif op_name in ['sum']:
            axis = get_attr(attr, 'axis', None)
            shp = infer_shapes[cns[0]]
            sum_axis = [shp[i] for i in axis] if axis else shp
            sum_len = np.product(sum_axis)
            sum_bit = math.ceil(math.log2(sum_len))
            oprec = cprecs[0] + sum_bit
        elif op_name in ['sigmiod', 'exp']:
            oprec = cprecs[1]
        elif op_name == 'Custom':
            op_type = get_attr(attr, 'op_type', 'null')
            assert op_type in ['cvm_clip', 'cvm_left_shift', 'cvm_right_shift',
                    'cvm_lut']
            if op_type in ['cvm_clip', 'cvm_left_shift', 'cvm_right_shift']:
                oprec = get_attr(attr, 'precision')
            else:
                oprec = cprecs[1]
        else:
            print (name, op_name, attr)
            assert False

        logger.debug("%-20s name=%-40s precision=%s from %s",
                op_name, name, oprec, cprecs)
        assert oprec <= MAX_BIT
        precs[name] = oprec
    return precs






