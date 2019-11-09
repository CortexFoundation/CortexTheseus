import z3

_ops_manager = {}
def register_op(op_name):
    def wrapper(op):
        op.op_name = op_name
        _ops_manager[op_name] = op
        return op
    return wrapper


_INT32_MAX = (2 ** (32 - 1)) - 1
class _Base(object):
    op_name = "NONE" # override

    def __init__(self, *args, **attr):
        for v in args:
            if not isinstance(v, _Base):
                raise TypeError(
                    'Operator:%s only accept input _Base' % self.op_name)

        self._attr = attr
        self._childs = args
        self.v, self.p = None, None
        self._forward(*args, **self._attr)
        if self.v is None:
            raise TypeError(
                'Operator:%s seems to forget to set v(value)'
                'in func:_forward' % self.op_name)
        if self.p is None:
            raise TypeError(
                'Operator:%s seems to forget to set p(precision)'
                'in func:_forward' % self.op_name)

    def _forward(self, *args, **kwargs):
        raise NotImplementedError("current operator's logic mathmatic")

    def _cstr(self):
        raise NotImplementedError("current operator's constraint condition")

    def cstr(self):
        cstr = [c.cstr() for c in self._childs]
        return z3.And(*cstr, self._cstr())

    def asrt(self):
        """ Assert the data overflow problem.
        """
        cstr = [c.asrt() for c in self._childs]
        asrt = z3.And(- _INT32_MAX <= self.v,
                self.v <= _INT32_MAX)
        return z3.And(*cstr, asrt)

    def __add__(self, b):
        return Add(self, b)

    def __mul__(self, b):
        return Mul(self, b)

    def __sub__(self, b):
        return Sub(self, b)

def InClosedInterval(data, start, end):
    return z3.And(start <= data, data <= end)


@register_op("var")
class Var(_Base):
    def _forward(self, name=None, prec=None):
        assert name is not None
        self.v = z3.Int(name)
        if prec is None:
            self.p = z3.Int("p_%s" % name)
        else:
            self.p = z3.IntVal(prec)

    def _cstr(self):
        r = (2 ** (self.p - 1)) - 1
        return z3.And(
                InClosedInterval(self.p, 1, 32),
                InClosedInterval(self.v, -r, r),
        )

def var(name, prec=None):
    return Var(name=name, prec=prec)


@register_op("scalar_add")
class Add(_Base):
    def _forward(self, a, b):
        # The interger addition is deterministic
        self.v = a.v + b.v
        self.p = z3.If(a.p > b.p, a.p, b.p) + 1

    def _cstr(self):
        a, b = self._childs
        return z3.And(
                InClosedInterval(a.p, 1, 16),
                InClosedInterval(b.p, 1, 16),
            )


@register_op("scalar_sub")
class Sub(_Base):
    def _forward(self, a, b):
        # The interger subtraction is deterministic
        self.v = a.v - b.v
        self.p = z3.If(a.p > b.p, a.p, b.p) + 1

    def _cstr(self):
        a, b = self._childs
        return z3.And(
                InClosedInterval(a.p, 1, 16),
                InClosedInterval(b.p, 1, 16),
            )


@register_op("scalar_mul")
class Mul(_Base):
    def _forward(self, a, b):
        # The interger multiply is deterministic
        self.v = a.v * b.v
        self.p = a.p + b.p

    def _cstr(self):
        # TODO(wlt): no need to consider larger than 0?
        # refer to source code.
        a, b = self._childs
        return z3.And(
                InClosedInterval(a.p, 1, 16),
                InClosedInterval(b.p, 1, 16),
            )


def prove_model(model, show_prop=False):
    cstr = model.cstr()
    s = z3.Solver()
    s.add(cstr)
    if show_prop:
        print ("Assumption: \n", cstr, "\n")
    if s.check() == z3.unsat:
        print ("Model cannot be satisfied, "
               "so it's proved to be deterministic")
        return

    asrt = model.asrt()
    if show_prop:
        print ("Assertion: \n", asrt, "\n")
    statement = z3.Implies(cstr, asrt)
    s = z3.Solver()
    s.add(z3.Not(statement))
    status = s.check()
    if status == z3.unsat:
        print ("Success: The model is deterministic")
    elif status == z3.sat:
        print ("Error: The model is undeterministic")
        m = s.model()
        for d in m.decls():
            print ("%s = %s" % (d.name(), m[d]))
    elif status == z3.unknown:
        print ("Error: The model cannot be proved to deterministic")


a, b = var('a'), var('b')
c = a * b
prove_model(c, True)
