import numpy as np

from .sym_utils import *

class Transformer(object):
    """ Base transformer object

        All subclass inherited from this should be registered maually
            using helper function `register_transformer`, and then
            all class function should be well-considered to override
            or use helper function `register_pass` to annotate using
            function defined in base class (that is this object),
            if there's no point to redefine duplicate function.

        Subclass should only implement function defined in base object,
            and we advise any helper function to be named with underline
            prefix.

        Please refer to file `tfm_ops.py` for more examples about
            operator transformers.

    Attributes:
    ==========
    op_name: Transformer is associated with operator which is defined
            in mxnet, and the variable indicates the type name of mxnet
            symbol.
            Attention please, the base transformer should not be instantiated
            since it's just an abstarct aggregation of graph pass, and it's
            named `none` by default.
    """
    op_name = "none"

    def __init__(self):
        if self.op_name == "none":
            raise RuntimeError("Base transformer should not be instantiated")

    def validate(self, op, **kwargs):
        """ All operators should be validated before another pass,
                neither correcting the invalid format nor asserting
                error to announce unsupported graph.

            Do nothing by default.
        """
        return op

    def rewrite(self, op, **kwargs):
        """ Operators may need to rewrite to equivalent graph which is
                easier to quantize for later procedure.

            Do nothing by default.
        """
        return op

    def quantize(self, op, **kwargs):
        """ Main procedure for quantization.

            Do nothing by default.
        """
        precs, scales = kwargs['precs'], kwargs['scales']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs = sym_iter(op.get_children())

        cname = childs[0].attr('name')
        precs[name][OUT_KEY] = precs[cname][OUT_KEY]
        scales[name] = scales[cname]

        logger = logging.getLogger('log.mrt.realize')
        logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
               op_name, name, scales[name], scales[cname])
        return op

    def prepare_for_compile(self, op, **kwargs):
        """ Equivalent graph transition may be needed before `compile`
                dynamic shape fixxation for `MulScalar`, `DivScalar`, `Zeroslike`
                and 'OnesLike' that is only needed in quantization:
            Do nothing by default.
        """
        return op

    def compile(self, op, **kwargs):
        """ Compile mxnet symbol into nnvm symbol.

            Throw exception by default.
        """
        childs = kwargs['childs']
        attrs = kwargs['attr']
        sym = get_nnvm_op(self.op_name)(*childs, name=N.n(),
                                        **attrs)
        return sym

    def fuse_transpose(self, op, **kwargs):
        return op

    def calculate_ops(self, op, **kwargs):
        """ Calculate the amount of computations for operator.

            Returns the output size by default.
        """
        base_ops = kwargs.get('base_ops', 1)
        infer_shapes = kwargs['infer_shapes']
        count = sum(np.product(shp) for shp in infer_shapes[op.attr('name')])
        return count * base_ops

_tfm_manager = {}
def register_transformer(op_name):
    def wrapper(tfm):
        tfm.op_name = op_name
        if op_name in _tfm_manager:
            raise NameError("Transformer %s has been registered" % op_name)
        _tfm_manager[op_name] = tfm()

        rpass = [k for k, v in tfm.__dict__.items() \
                if not k.startswith("_") and callable(v)]
        for p in rpass:
            tfm = register_pass(p)(tfm)
        return tfm
    return wrapper

def get_transformer(op):
    op_name = op.attr('op_name')
    if op_name not in _tfm_manager:
        raise NotImplementedError( \
                "Transformer %s has not been registered" % op_name)
    return _tfm_manager[op_name]

_op_manager = {}
_pass_manager = {k:[] for k, v in Transformer.__dict__.items() \
        if not k.startswith("_") and callable(v)}
def register_pass(pass_t):
    def wrapper(tfm):
        if tfm.op_name not in _op_manager:
            _op_manager[tfm.op_name] = []
        if pass_t in _op_manager[tfm.op_name]:
            raise NameError( \
                    "Transformer %s pass:%s has been registered" \
                    % (tfm.op_name, pass_t))
            return tfm
        _op_manager[tfm.op_name].append(pass_t)
        if pass_t in _pass_manager:
            _pass_manager[pass_t].append(tfm.op_name)
        return tfm
    return wrapper

def pass_info(arg=None):
    if arg is None:
        return _pass_manager
    if isinstance(arg, mx.sym.Symbol):
        return _op_manager.get(arg.attr('op_name'), [])
    return _pass_manager.get(arg, [])

def apply_pass(pass_t, **updates):
    def wrapper(op, **kwargs):
        tfm = get_transformer(op)
        assert pass_t in pass_info(op), \
                "Transformer %s has not been registered pass:%s" \
                % (op.attr('op_name'), pass_t)
        kwargs.update(updates)
        ret = getattr(tfm, pass_t)(op, **kwargs)
        for n in updates:
            assert op.attr('name') in kwargs[n], "%s %s %s"%(n, op.attr('name'), ret.attr('name'))
            kwargs[n][ret.attr('name')] = kwargs[n][op.attr('name')]
        return ret
    return wrapper

OUT_KEY = "out_key"
TARGET_KEY = "target_key"
MAX_BIT = 32

# === name manager 

_NoneName = object()
class N(object):
    _global_name = _NoneName
    _count = {}
    _name_manager = {}

    @staticmethod
    def n(name=""):
        assert N._global_name != _NoneName, \
                "register name manager first please"
        if name not in N._name_manager:
            N._name_manager[name] = 0
        _n = "mrt"
        if N._global_name:
            _n += "_" + N._global_name
        if name != "":
            _n += "_" + name
        _n += "_%d" % N._name_manager[name]
        N._name_manager[name] += 1
        return _n

    @staticmethod
    def _set_global(name):
        if name not in N._count:
            N._count[name] = 0
        else:
            name += str(N._count[name])
        N._global_name = name

    @staticmethod
    def register_nm(name):
        def wrapper(pass_f):
            def run(symbol, params, *args, **kwargs):
                old_name = N._global_name
                N._set_global(name)
                N._count[name] += 1
                ret = pass_f(symbol, params, *args, **kwargs)
                N._global_name = old_name
                return ret
            return run
        return wrapper
