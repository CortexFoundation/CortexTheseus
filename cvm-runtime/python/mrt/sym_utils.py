from mxnet.symbol import _internal
from mxnet import symbol as _sym
from mxnet import ndarray as nd
import mxnet as mx
import cvm
import logging
import json
import math

INT32_MIN, INT32_MAX = -2147483647, 2147483647
INT8_MIN, INT8_MAX = -127, 127

INT8_TYPE, INT32_TYPE= ('int8', 'int32')

def is_op(sym, params):
    return (sym.attr('op_name') != 'null')
def is_var(sym, params):
    return (sym.attr('op_name') == 'null')
def is_params(sym, params):
    return is_var(sym, params) and \
        (sym.attr('name') in params)
def is_inputs(sym, params):
    return is_var(sym, params) and \
        (sym.attr('name') not in params)

def nd_array(source_array, ctx=None, dtype="float64"):
    return nd.array(source_array, ctx=ctx, dtype=dtype)
def nd_arange(*args, **kwargs):
    return nd.arange(*args, dtype="float64", **kwargs)
def nd_full(*args, **kwargs):
    return nd.full(*args, dtype="float64", **kwargs)
def nd_zeros(*args, **kwargs):
    return nd.zeros(*args, dtype="float64", **kwargs)
def nd_ones(*args, **kwargs):
    return nd.ones(*args, dtype="float64", **kwargs)

DATA_NAME = "data"
_name_dict = {}
def gen_name(name):
    if name not in _name_dict:
        _name_dict[name] = 0
    _name_dict[name] += 1
    return name + '_' + str(_name_dict[name] - 1)

def check_graph(symbol, params, logger=logging):
    # check duplicate name
    graph_str = json.loads(symbol.tojson())
    nodes = graph_str['nodes']
    name_set = []
    for node in nodes:
        assert node['name'] not in name_set, \
            "NameError, duplicate name '%s'" % node['name']
        name_set.append(node['name'])

    # check input name and params name, remove unused params name
    new_params = {}
    for sym in topo_sort(symbol, logger==logger):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        if is_params(sym, params):
            assert name != DATA_NAME, \
                "NameError, param should not be named by 'data'"
            new_params[name] = params[name]
        elif is_inputs(sym, params):
            assert name == DATA_NAME, \
                "NameError, input '%s' should be named by 'data'" % name
    logger.info("Model Checked Passed.")

    return symbol, new_params

def combile_name(n1, n2):
    t1, t2 = n1.split("_"), n2.split("_")
    len1, len2 = len(t1), len(t2)
    min_len = min(len1, len2)
    begin, end = 0, 0
    for i in range(min_len):
        if t1[i] != t2[i]:
            break
        begin = i + 1
    for i in range(min_len):
        if t1[len1-1-i] != t2[len2-1-i]:
            break
        end = i + 1
    res = t1[:begin]
    res.extend([n for n in t1[begin:len1-end]])
    res.extend([n for n in t2[begin:len2-end]])
    res.extend(t1[len1-end:])
    return "_".join(res)

def check_ext_deps(ext, deps=[], logger=logging):
    if isinstance(deps, str):
        deps = [deps]
    for k, v in ext.items():
        for dep in deps:
            if dep not in v:
                logger.critical("ext must have attribute %s vs. %s",
                        dep, ext)
                assert False

NoneAttr = object()
def get_attr(attr, name, default=NoneAttr):
    if name in attr:
        if isinstance(default, str):
            return attr[name]
        return eval(attr[name])
    if default == NoneAttr:
        assert False, "attr %s is not exists in %s" % (name, attr)
    return default

_MX_OP_CONTRIB_PREFIX = '_contrib_'
def get_nd_op(op_name):
    op = getattr(nd, op_name, None)
    if op is None:
        op = getattr(nd._internal, op_name, None)
    if op_name.startswith(_MX_OP_CONTRIB_PREFIX):
        op = getattr(nd.contrib, op_name[len(_MX_OP_CONTRIB_PREFIX):], None)

    if op is None:
        raise RuntimeError("Unable to map op_name {} to mxnet.ndarray".format(op_name))
    return op

def get_mxnet_op(op_name):
    op = getattr(_internal, op_name, None)
    if op is None:
        op = getattr(_sym, op_name, None)
    if op_name.startswith(_MX_OP_CONTRIB_PREFIX):
        op = getattr(_sym.contrib, op_name[len(_MX_OP_CONTRIB_PREFIX):], None)

    if op is None:
        raise RuntimeError("Unable to map op_name {} to mxnet.sym".format(op_name))
    return op

def get_nnvm_op(op_name):
    op = getattr(cvm.symbol, op_name)

    if not op:
        raise RuntimeError("Unable to map op_name {} to nnvm.sym".format(op_name))
    return op

def sym_iter(sym):
    if sym is None:
        return None

    if isinstance(sym, mx.sym.Symbol):
        sym = [sym[i] for i in range(len(sym))]
    else:
        assert isinstance(sym, cvm.sym.Symbol)
        size = len(sym.list_output_names())
        sym = [sym[i] for i in range(size)]
    return sym


def examine_parameters(symbol, params, inputs_ext, allows=[], callback=None):
    args, new_params = symbol.list_inputs(), {}
    for arg in args:
        if arg not in inputs_ext:
            assert arg in params, 'arg(%s) not exists in params dict(%s)' \
                % (arg, params.keys())

            if callback is not None:
                callback(params, arg)

            new_params[arg] = params[arg]

    for name in allows:
        if name in params:
            new_params[name] = params[name]
    return new_params

def nd_const(number, graph, params):
    name = 'const_var_' + str(number)
    prec = math.ceil(math.log2(math.fabs(number)+1)) + 1
    if name not in params and name not in graph:
        attr = { 'precision': str(prec) }
        graph[name] = mx.sym.var(name, shape=(1,), attr=attr)
        params[name] = nd_array([number])
    return graph[name]

def mx_const(number, graph, params):
    name = 'const_var_' + str(number)
    prec = math.ceil(math.log2(number+1)) + 1
    if name not in graph:
        attr = { 'precision': str(prec) }
        graph[name] = mx.sym.var(name, shape=(1,), attr=attr)
        params[name] = nd.array([number])
    return graph[name]

def op_const(number, graph, var=mx.sym.var):
    name = 'const_var_' + str(number)
    if name not in graph:
        graph[name] = var(name, shape=(1,))
    return graph[name], name

def topo_sort(symbol, logger=logging, with_deps=False):
    """Sort all symbols in the mxnet graph in topological order.
    """
    queue = []
    symbol_map = {}
    deps = {}
    dep_cnts = {}
    for s in symbol:
        symbol_map[s.attr('name')] = s
        queue.append(s)

    while queue:
        sym = queue.pop(0)
        name = sym.attr('name')
        childs = sym.get_children()
        if childs is None:
            dep_cnts[name] = 0
        else:
            childs = sym_iter(childs)
            # remove duplication dependency
            dep_cnts[name] = len({c.attr('name') for c in childs})
            for child in childs:
                child_name = child.attr('name')
                if child_name not in deps:
                    deps[child_name] = set()
                deps[child_name].add(name)
                if child_name not in symbol_map:
                    symbol_map[child_name] = child
                    queue.append(child)

    order = []
    reduce_flag = True
    while dep_cnts:
        if not reduce_flag:
            logger.critical("deps cannot reduce -> %s", dep_cnts)
            assert False

        remove = []
        reduce_flag = False
        for name in dep_cnts:
            if dep_cnts[name] == 0:
                order.append(symbol_map[name])
                remove.append(name)
                if name in deps:
                    for other in deps[name]:
                        dep_cnts[other] -= 1

                reduce_flag = True
        for name in remove:
            del dep_cnts[name]
    if with_deps:
        return order, deps
    else:
        return order

def sym_collect_attr(symbol, attr_name='op_name'):
    return {sym.attr(attr_name) for sym in topo_sort(symbol)}

MULTIPYE_OUTS_NODE = [
    'get_valid_counts', 'SliceChannel',
    # group's op_name is None
    'None',
]
def get_entry_id(sym):
    oindex = 0
    if sym.attr('op_name') in MULTIPYE_OUTS_NODE:
        if isinstance(sym, _sym.Symbol):
            oindex = json.loads(sym.tojson())['heads'][0][1]
        elif isinstance(sym, cvm.sym.Symbol):
            graph = cvm.graph.create(sym)
            oindex = json.loads(graph.json())['heads'][0][1]
    return oindex

def get_node(sym, graph):
    """ Assume all graph node have single output.
        Multiple output node will be fused
        by `fuse_multiple_outputs` sym_pass.
    """
    name = sym.attr('name')
    if name not in graph:
        assert False, "Unrecognized layer:%s in graph keys:%s" \
            % (name, graph.keys())
    return graph[name][get_entry_id(sym)]

def topo_visit(symbol, params, inputs_ext, callback,
        get_op=get_mxnet_op, logger=logging,
        with_maps=False, **kwargs):
    graph, maps = {}, {}
    params = {k:v[:] for k,v in params.items()} # copy params
    for sym in topo_sort(symbol, logger=logger):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        node = sym
        if childs is not None:
            childs = [get_node(c, graph) for c in childs]
            node = get_op(op_name)(*childs, **attr, name=name)
        if callback is not None:
            node, params = callback(node, params, graph, inputs_ext, **kwargs)
        maps[node.attr('name')] = name
        graph[name] = node
    nodes = [get_node(sym, graph) for sym in symbol]
    ret = get_op("Group")(nodes) if len(nodes) > 1 else nodes[0]
    if with_maps:
        return ret, params, maps
    else:
        return ret, params

def topo_visit_transformer(symbol, params, callback,
        get_op=get_mxnet_op, logger=logging, **kwargs):
    graph = {}
    params = {k:v[:] for k, v in params.items()}
    for op in topo_sort(symbol, logger=logger):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        if childs is not None:
            childs = [get_node(c, graph) for c in childs]
            op = get_op(op_name)(*childs, **attr, name=name)

        graph[name] = callback(op, params=params, graph=graph, **kwargs)
        if graph[name] is None:
            graph[name] = op
    nodes = [get_node(op, graph) for op in symbol]
    ret = get_op("Group")(nodes) if len(nodes) > 1 else nodes[0]
    return ret, params


"""Deterministic Op Description
The specific op for quantization with Int8 or Int32, more details
described as belows:

In: inputs variable, maybe followed with int counter.
Out: output variable, maybe followed with int counter.
P_X: params variable, load from params file.
C_X: constant variable, fixed in graph.

Activation: specific indicate relu.
    In[Int8] -> Out[Int8]
Pooling: sepcific indicate max pool.
    In[Int8] -> Out[Int8]
Convolution:
    In[Int8] * P_weight[Int8] + P_bias[Int32] -> Out[Int64]
FullyConnected|Dense:
    In[Int8] * P_weight[Int8] + P_bias[Int32] -> Out[Int64]
elemwise_add:
    In1[Int8] + In2[Int8] -> Out[Int32]
sum: reduce op over specific axis, sum(data, axis=[1, 2])
    In[Int8] -> Out[Int32]

Reshape:
    In[Int32] -> Out[Int32]
Flatten:
    In[Int32] -> Out[Int32]

broadcast_add:
    In1[Int32] + In2[Int32] -> Out[Int64]
    In1[Int8]  + In2[Int8]  -> Out[Int32]
broadcast_sub:
    In1[Int32] + In2[Int32] -> Out[Int64]
    In1[Int8]  - In2[Int8]  -> Out[Int32]
broadcast_mul:
    In1[Int32] * In2[Int32] -> Out[Int64]
    In1[Int8]  * In2[Int8]  -> Out[Int32]
broadcast_div:
    In1[Int32] / In2[Int32] -> Out[Int32]
    In1[Int8]  / In2[Int8]  -> Out[Int8]

_plus_scalar:
    In[Int32] + C_scale[Int32] -> Out[Int64]
_sub_scalar:
    In[Int32] - C_scale[Int32] -> Out[Int64]
_mul_scalar:
    In[Int32] * C_scale[Int32] -> Out[Int64]
_div_scalar:
    In[Int32] / C_scale[Int32] -> Out[Int32]

# Requant Op
cvm_right_shift:
    assert P_shift_bits > 0
    In[Int8|Int32|Int64] >> P_shift_bits[Int8] -> Out[Int8]
cvm_left_shift:
    assert 0 <= P_shift_bits < 24
    In[Int8|Int32|Int64] << P_shift_bits[Int8] -> Out[Int8]

"""
nnvm_identity_ext = [
    'null',
    'relu', 'upsampling', 'max_pool2d',
    'conv2d', 'dense', 'sum', 'elemwise_add', 'elemwise_sub',
    'reshape', 'flatten', 'strided_slice', 'slice_like',

    'broadcast_left_shift', 'broadcast_right_shift',
    'broadcast_div', 'broadcast_mul', 'broadcast_add', 'broadcast_sub',
    'broadcast_max',

    '__add_scalar__',

    'max', 'abs', 'log2',
    'clip', 'concatenate', 'negative',
    'cvm_clip', 'cvm_left_shift', 'cvm_right_shift',
    'cvm_lut',

    'take', 'repeat', 'tile', 'transpose',
    'expand_dims', 'squeeze',
    'get_valid_counts', 'non_max_supression',
]

"""Mxnet Symbol Operator Extension
Attribute Options:
    0   : whether flag by default is support
    1...: optional type
"""
mx_identity_ext = {
    'null': {},
    'Convolution': {},
    'BatchNorm': {},
    'Pooling': {
        'pool_type': [False, 'max', 'avg'],
        'count_include_pad': [True, 'True'],
    },
    'Flatten': {},
    'FullyConnected': {},
    'Activation': {
        'act_type': [False, 'relu'], # Only supported relu
    },
    'relu': {},
    'sum': {},
    'Dropout': {
        'mode': [True, 'training'],
    },
    'Concat': {},
    'elemwise_add': {},
    'elemwise_sub': {},
    'LeakyReLU': {
        'act_type': [True, 'leaky']
    },
    'slice_like': {},
    'slice_axis': {},
    'repeat': {},
    'Reshape': {},
    'UpSampling': {},
    'transpose': {},
    'tile': {},
    'expand_dims': {},

    '_arange': {},

    # Not supported broadcast_div
    'broadcast_mul': {},
    'broadcast_add': {},
    'broadcast_sub': {},

    '_mul_scalar': {},
    '_div_scalar': {},
    '_plus_scalar': {},

    'max': {},
    'Embedding': {},
    'squeeze': {},
    'SwapAxis': {},

    'sigmoid': {},
    'exp': {},

    'SliceChannel': {},
    'zeros_like': {},
    '_greater_scalar': {},
    'where': {},
    'ones_like': {},
    '_contrib_box_nms': {},
    'softmax': {},
}
