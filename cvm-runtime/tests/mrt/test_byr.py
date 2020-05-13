import transformer as tfm

import sym_pass as spass
import sym_utils as sutils
from sym_utils import topo_visit_transformer
import mxnet as mx
import nnvm
import numpy as np
from ut_base import *

class TestFuseMultiplyInputs(TfmTest):
    def test_fmi(self):
        d1 = mx.sym.var('d1', shape=(2, 3))
        d2 = mx.sym.var('d2', shape=(2, 4))
        d3 = mx.sym.var('d3', shape=(2, 3))
        op = mx.sym.concat(d1, d2, d3)
        sym = transfer_multiple_inputs(op, {})

        data = mx.sym.var('data', shape=(20,))
        s1 = mx.sym.slice(data, begin=(0,), end=(6,))
        r1 = mx.sym.reshape(s1, shape=(2, 3))
        s2 = mx.sym.slice(data, begin=(6,), end=(14,))
        r2 = mx.sym.reshape(s2, shape=(2, 4))
        s3 = mx.sym.slice(data, begin=(14,), end=(20,))
        r3 = mx.sym.reshape(s3, shape=(2, 3))
        des = mx.sym.concat(r1, r2, r3)

        self._assert_equal(sym, des)

@tfm.N.register_nm("fmi")
def transfer_multiple_inputs(sym, params):
    infer_shapes = tfm.infer_shape(sym, params)
    dim_sum, dim_per, dims = 0, {}, {}
    def _sum_input(node, params, **kwargs):
        name = node.attr('name')
        nonlocal dim_sum, dim_per, dims
        if sutils.is_inputs(node, params):
            dims[name] = infer_shapes[name][0]
            dot = np.product(dims[name])
            dim_per[name] = dot
            dim_sum += dot
    topo_visit_transformer(sym, params, _sum_input)
    data_sum = mx.sym.var('data', shape=(dim_sum,))
    first, last = 0, 0
    def _change_node(op, params, graph, **kwargs):
        name = op.attr('name')
        if sutils.is_inputs(op, params):
            nonlocal first, last
            last = first + dim_per[name]
            op = mx.sym.slice(data_sum, name=tfm.N.n('slice'),
                    begin=(first,), end=(last,))
            op = mx.sym.reshape(op, name=tfm.N.n('reshape'),
                    shape=dims[name])
            first = last
        return op
    sym, params = topo_visit_transformer(sym, params, _change_node)
    return sym

