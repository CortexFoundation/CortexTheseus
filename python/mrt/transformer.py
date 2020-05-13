""" MRT Interface API

    Refractor of source code, using the registry pattern.
    Rules of coding with pylint.
    Collection of hyper-parameters controller.
    Simplification of public API.
"""

import logging
import os
from os import path
import numpy as np

import mxnet as mx
from mxnet import gluon, ndarray as nd
import cvm

# import as registry pattern
from . import tfm_ops  # pylint: disable=unused-import
from . import cvm_op   # pylint: disable=unused-import

from . import tfm_pass as tpass
from .tfm_pass import OUT_KEY, convert_params_dtype
from .tfm_pass import sym_calibrate, quantize, to_cvm
from .tfm_pass import prepare_for_compile, fuse_constant
from .tfm_pass import calculate_ops, collect_op_names

from . import sym_utils as sutils
from .sym_utils import topo_sort, sym_iter, get_mxnet_op
from . import utils
from . import sim_quant_helper as sim

# TODO: collect hyper-parameters

__all__ = ["Model", "MRT", "ModelSpliter", "ModelMerger",
           # "init", "MRT", "compile_to_cvm",
           # "split_model", "merge_model",
           # transformer helper pass
           # "convert_params_dtype",
]


class Model:
    """ Wrapper of Mxnet symbol and params, design
            with user-friendly model API.
    """
    def __init__(self, symbol, params, dtype="float64"):
        self.symbol = symbol
        self.params = convert_params_dtype(params, dest_dtype=dtype)

    def __iter__(self):
        return iter(self.symbol)

    def input_names(self):
        """ List model input names.  """
        return [s.attr("name") for s in topo_sort(self.symbol) \
                if sutils.is_inputs(s, self.params)]

    def output_names(self):
        """ List model output names. """
        return [s.attr('name') for s in self.symbol]

    def names(self):
        return self.output_names()

    def to_graph(self, dtype="float32", ctx=mx.cpu()):
        """ Convenient helper function to create model runtime,
                returns gluon.nn.SymbolBlock.
        """
        graph = gluon.nn.SymbolBlock(self.symbol, \
            [mx.sym.var(n) for n in self.input_names()])
        utils.load_parameters(graph, convert_params_dtype(
            self.params,
            dest_dtype=dtype), ctx=ctx)
        return graph

    def save(self, symbol_file, params_file):
        """ Model dump to disk. """
        with open(symbol_file, 'w') as fout:
            fout.write(self.symbol.tojson())
        nd.save(params_file, self.params)

    @staticmethod
    def load(symbol_file, params_file):
        """ Model load from disk. """
        symbol = mx.sym.load(symbol_file)
        params = nd.load(params_file)
        return Model(symbol, params)

    def split(self, keys):
        return split_model(self, keys)

    @staticmethod
    def merger(base, top, base_name_map=None):
        return ModelMerger(base, top, base_name_map)

    def prepare(self, input_shape=None):
        model = init(self, input_shape)
        self.symbol, self.params = model.symbol, model.params

    def get_mrt(self):
        return MRT(self)

    def to_cvm(self, model_name, datadir="/data/stdout",
                       input_shape=None, target="gpu"):
        return compile_to_cvm(self, model_name, datadir,
                              input_shape, target)

def init(model, input_shape=None):
    logger = logging.getLogger("mrt.prepare")
    logger.info("Model initializing...")

    _sym, _prm = model.symbol, model.params
    tpass.name_duplicate_check(_sym, _prm)

    if isinstance(input_shape, dict):
        _sym, _prm = tpass.attach_input_shape(_sym, _prm, input_shape)
        _sym, _prm = tpass.fuse_multiple_inputs(_sym, _prm)
    elif input_shape is not None:
        model_inputs = tpass.model_inputs(_sym, _prm)
        assert model_inputs == 1, "Multiple inputs non-known shape"
        _sym, _prm = tpass.input_name_replace(_sym, _prm)
        _sym, _prm = tpass.attach_input_shape(_sym, _prm,
                                              {"data": input_shape})
    tpass.infer_shape(_sym, _prm) # check infer_shape is correct

    _sym, _prm = tpass.fuse_multiple_outputs(_sym, _prm)
    _sym, _prm = tpass.fuse_constant(_sym, _prm)
    _sym, _prm = tpass.fuse_transpose(_sym, _prm)
    _sym, _prm = tpass.rewrite(_sym, _prm)
    _sym, _prm = tpass.fuse_constant(_sym, _prm)
    _sym, _prm = tpass.params_unique(_sym, _prm)

    return Model(_sym, _prm)

class MRT:
    """ An MRT quantization class contained many helper functions.

    Quantization Procedures:
    ========================
        1. prepare: initial of model graph, such as fuse_constant,
            rewrite, validate, ...etc;
        2. calibration: caculate the internal thresholds of layers;
        3. quantization: quantize the floating parameters into INT(p)
            precision with scales, using the floading data simulate
            the realized environment of interger dataflow;
    """
    def __init__(self, model, input_prec=8):
        self.old_names = model.output_names()
        self.current_model = model

        self._data = None
        self.th_dict = {}

        self.restore_names = set()
        self._op_default_input_precs()
        self.precs = {s.attr('name'):{} \
            for s in topo_sort(self.current_model)}
        if 'data' not in self.precs:
            raise RuntimeError("please invoke `init` function first")
        self.precs['data'][OUT_KEY] = input_prec
        self.scales = {}
        self.softmax_lambd = 10
        self.shift_bits = 5

    def set_data(self, data):
        self._data = data

    def calibrate(self, ctx=mx.cpu(), lambd=None, old_ths=None):
        self.th_dict = sym_calibrate(
            self.current_model.symbol, self.current_model.params,
            self._data, ctx=ctx, lambd=lambd, old_ths=old_ths)
        return self.th_dict

    def set_restore(self, name):
        self.restore_names.add(name)

    def set_threshold(self, name, threshold):
        self.th_dict[name] = threshold

    def set_th_dict(self, th_dict):
        self.th_dict = th_dict

    def _op_default_input_precs(self):
        op_precs = self.op_input_precs = {}
        for name in ['Convolution', 'FullyConnected',
                     'sigmoid', 'exp', 'softmax']:
            op_precs[name] = 8
        op_precs['sum'] = 8
        for name in ['broadcast_add', 'broadcast_sub',
                     'elemwise_add', 'elemwise_sub', 'slice_like']:
            op_precs[name] = 16
        op_precs['broadcast_mul'] = 16
        op_precs['L2Normalization'] = 8
        op_precs['Concat'] = 16
        op_precs['Embedding'] = 16
        op_precs['slice_like'] = 30

    def set_input_prec(self, prec):
        self.precs['data'][OUT_KEY] = prec

    def set_output_prec(self, prec):
        for sym in self.current_model:
            name = sym.attr('name')
            self.precs[name][name] = prec

    def set_softmax_lambd(self, val):
        self.softmax_lambd = val

    def set_shift_bits(self, val):
        self.shift_bits = val

    def quantize(self):
        _sym, _prm = quantize(
            self.current_model.symbol, self.current_model.params,
            self.th_dict, self.precs, self.scales, self.op_input_precs,
            self.restore_names, self.shift_bits, self.softmax_lambd)
        self.current_model = Model(_sym, _prm)
        return self.current_model

    def get_output_scales(self):
        return [self.scales[s.attr("name")] for s in self.current_model]

    def get_maps(self):
        return dict(zip([c.attr('name') for c in self.current_model],
                        self.old_names))

    def get_inputs_ext(self):
        inputs_ext = {'data': {
            'scale': self.scales['data'],
            'target_bit': self.precs['data'][OUT_KEY]}}
        return inputs_ext

    def save(self, model_name, datadir="./data"):
        # pylint: disable=unbalanced-tuple-unpacking
        sym_file, params_file, ext_file = \
            utils.extend_fname(path.join(datadir, model_name), True)
        sim.save_ext(ext_file, self.old_names, self.th_dict,
                     self.precs, self.scales)
        self.current_model.save(sym_file, params_file)

    @staticmethod
    def load(model_name, datadir="./data"):
        # pylint: disable=unbalanced-tuple-unpacking
        sym_file, params_file, ext_file = \
            utils.extend_fname(path.join(datadir, model_name), True)
        mrt = MRT(Model.load(sym_file, params_file))
        mrt.old_names, mrt.th_dict, mrt.precs, mrt.scales = \
            sim.load_ext(ext_file)
        return mrt

def split_model(model, keys):
    symbol, params = model.symbol, model.params
    nodes = [s for s in topo_sort(symbol) if s.attr('name') in keys]
    base = nodes[0] if len(nodes) == 1 else mx.sym.Group(nodes)
    base_params = {k:params[k] for k in base.list_inputs() if k in params}

    graph = {}
    infer_shapes = tpass.infer_shape(symbol, params)
    for sym in topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        node = sym
        if childs is not None:
            childs = [sutils.get_node(c, graph) for c in childs]
            node = get_mxnet_op(op_name)(*childs, **attr, name=name)
        if name in keys:
            node = mx.sym.var(name, \
                shape=infer_shapes[name][sutils.get_entry_id(sym)])
        graph[name] = node
    nodes = [sutils.get_node(c, graph) for c in symbol]
    top = nodes[0] if len(nodes) == 1 else mx.sym.Group(nodes)
    top_params = {k:params[k] for k in top.list_inputs() if k in params}

    return Model(base, base_params), Model(top, top_params)

def merge_model(base_model, top_model, base_name_maps=None, callback=None):
    base_name_maps = {} if base_name_maps is None else base_name_maps
    # topo sort base model for duplicated name symbol
    # graph = {base_name_maps.get(c.attr('name'), c.attr('name')): c \
        # for c in base_model.symbol}
    graph = {base_name_maps.get(c.attr('name'), c.attr('name')): c \
        for c in topo_sort(base_model.symbol)}
    for sym in topo_sort(top_model.symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        node = sym
        if childs is not None:
            childs = [sutils.get_node(c, graph) for c in childs]
            node = get_mxnet_op(op_name)(*childs, **attr, name=name)
        if name in graph:
            node = graph[name]
        if callback is not None:
            node = callback(node, top_model.params, graph)
        graph[name] = node
    nodes = [sutils.get_node(s, graph) for s in top_model.symbol]
    symbol = nodes[0] if len(nodes) == 1 else mx.sym.Group(nodes)
    params = base_model.params
    params.update(top_model.params)
    return Model(symbol, params)


class ModelSpliter:
    def __init__(self, model, keys):
        self.model = model
        self.keys = keys

    def split(self):
        return split_model(self.model, self.keys)


class ModelMerger:
    def __init__(self, base_model, top_model, base_name_maps=None):
        self.base, self.top = base_model, top_model
        base_name_maps = {} if base_name_maps is None else base_name_maps
        self.base_name_maps = base_name_maps

    def merge(self, callback=None):
        return merge_model(
            self.base, self.top, self.base_name_maps, callback)

    def get_output_scales(self, base_oscales, maps):
        name_idx = {self.base_name_maps.get(
            s.attr("name"), s.attr("name")): i \
            for i, s in enumerate(self.base)}
        return [1 if v=="None" else base_oscales[name_idx[v]] \
            for k, v in maps.items()]

def reduce_graph(model, input_shapes):
    _sym, _prm = model.symbol, model.params
    _sym, _prm = tpass.attach_input_shape(
        _sym, _prm, input_shapes)

    _sym, _prm = prepare_for_compile(_sym, _prm)
    _sym, _prm = fuse_constant(_sym, _prm)
    return Model(_sym, _prm)

def compile_to_cvm(model, model_name, datadir="/data/std_out",
                   input_shape=None, target="gpu"):
    """ Compile Mxnet model into CVM Accept-JSON&BIN-Format
    """
    logger = logging.getLogger("mrt.compile")
    datadir = path.join(datadir, model_name)
    os.makedirs(datadir, exist_ok=True)

    if input_shape is None:
        for sym in topo_sort(symbol):
            if sutils.is_inputs(sym, params):
                _, oshp, _ = sym.infer_shape()
                input_shape = oshp[0]
                break
    input_shapes = {'data': input_shape}

    # transform from mxnet symbol to cvm
    logger.info("Transform Mxnet symbol into CVM")
    model = reduce_graph(model, input_shapes)
    symbol, params = model.symbol, model.params
    cvm_sym, params = to_cvm(symbol, params)
    logger.info("Transform Mxnet symbol into CVM finished")

    dtype, cvm_params = "int32", {}
    cvm_ctx = cvm.context(target, 0)
    for sym in topo_sort(cvm_sym):
        if sutils.is_params(sym, params):
            key, value = sym.attr('name'), params[sym.attr('name')]
            flat = value.asnumpy()
            assert np.abs(flat).max() <= sutils.INT32_MAX, \
                "key: {}\nvalue: {}".format(key, value)
            assert (flat.astype(dtype).astype("float64") == flat).all(), \
                "key: {}\nvalue: {}".format(key, value)
            cvm_params[key] = cvm.nd.array(flat.astype(dtype), ctx=cvm_ctx)
        elif sutils.is_inputs(sym, params):
            assert sym.attr('name') == 'data'

    # compile to JSON&Bytes format
    logger.info("Compile into CVM graph")
    deploy_graph, cvm_params = cvm.graph.build(
        cvm_sym, cvm_params, shape=input_shapes)

    # cvm parameters reduce
    logger.info("Parameters precision reduce")
    for sym in topo_sort(cvm_sym):
        if sutils.is_params(sym, cvm_params):
            name, attr = sym.attr('name'), sym.list_attr()
            precision = sutils.get_attr(attr, "precision")
            dtype = "int32" if precision > 8 else "int8"
            cvm_params[name] = cvm.nd.array(
                params[name].asnumpy().astype(dtype), ctx=cvm_ctx)

    # dump
    logger.info("CVM Json&Params dump")
    with open(path.join(datadir, "symbol"), "w") as fout:
        fout.write(deploy_graph.json())
    param_bytes = cvm.nd.save_param_dict(cvm_params)
    with open(path.join(datadir, "params"), "wb") as fout:
        fout.write(param_bytes)
    return deploy_graph, cvm_params
