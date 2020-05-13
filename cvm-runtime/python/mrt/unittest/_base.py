import unittest

import mxnet as mx

import transformer as tfm
import sym_utils as sutils

def graph_equal(src, des):
    if isinstance(src, mx.sym.Symbol) and len(src) > 1:
        for i, op in enumerate(src):
            if i >= len(des):
                return op, des[-1]
            r1, r2 = graph_equal(op, des[i])
            if r1 is not None:
                return r1, r2
        return None, None
    if src.attr('op_name') != des.attr('op_name'):
        return src, des
    if sutils.get_entry_id(src) != sutils.get_entry_id(des):
        return src, des
    if src.list_attr() != des.list_attr():
        return src, des
    src_childs = sutils.sym_iter(src.get_children())
    des_childs = sutils.sym_iter(des.get_children())
    if src_childs is None:
        if des_childs is None:
            return None, None
        else:
            return src, des
    if len(src_childs) != len(des_childs):
        return src, des
    for i, op in enumerate(src_childs):
        r1, r2 = graph_equal(op, des_childs[i])
        if r1 is not None:
            return r1, r2
    return None, None

def summary(sym, err_op=None):
    _s = ""
    for op in sutils.topo_sort(sym):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sutils.sym_iter(op.get_children()), op.list_attr()
        prefix = "Op " if op_name == "null" else "Var"
        if (err_op is not None) and (name == err_op.attr('name')):
            prefix = "> " + prefix
        _s += "%5s:%-10s, Name=%-15s, Attr=%-40s" \
               % (prefix, op_name, name, attr)
        if childs is not None:
            cinfos = ["%s(%d)" % (c.attr('name'), sutils.get_entry_id(c)) \
                         for c in childs]
            _s += ", Inputs=%s" % ", ".join(cinfos)
        _s += "\n"
    return _s


class TfmTest(unittest.TestCase):
    def _collect_params(self, symbol):
        params = {}
        for op in sutils.topo_sort(symbol):
            if sutils.is_var(op, params):
                _, shp, _ = op.infer_shape()
                params[op.attr('name')] = mx.nd.uniform(-1, 1, shp[0])
        return params

    def _assert_equal(self, op, des, passes=[]):
        if isinstance(passes, str):
            passes = [passes]
        params = self._collect_params(op)
        for p in passes:
            op, params = getattr(tfm, p)(op, params)

        r1, r2 = graph_equal(op, des)
        _s ="Graph Not Equal\n" + "-" * 20 +"\n"
        _s += summary(op, r1) + "-" * 20 + "\n"
        _s += summary(des, r2)
        self.assertIsNone(r1, _s)

    def _assert_error(self, op, passes):
        if isinstance(passes, str):
            passes = [passes]
        params = self._collect_params(op)
        with self.assertRaises(AssertionError):
            for p in passes:
                op, params = getattr(tfm, p)(op, params)
