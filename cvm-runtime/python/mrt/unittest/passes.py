from _base import *
from transformer import transfer_multiple_inputs

class TestFuseMultiplyInputs(TfmTest):
    def test_fmi(self):
        d1 = mx.sym.var('d1', shape=(2, 3))
        d2 = mx.sym.var('d2', shape=(2, 4))
        d3 = mx.sym.var('d3', shape=(2, 3))
        op = mx.sym.concat(d1, d2, d3)
        sym, _ = transfer_multiple_inputs(op, {})

        data = mx.sym.var('data', shape=(20,))
        s1 = mx.sym.slice(data, begin=(0,), end=(6,))
        r1 = mx.sym.reshape(s1, shape=(2, 3))
        s2 = mx.sym.slice(data, begin=(6,), end=(14,))
        r2 = mx.sym.reshape(s2, shape=(2, 4))
        s3 = mx.sym.slice(data, begin=(14,), end=(20,))
        r3 = mx.sym.reshape(s3, shape=(2, 3))
        des = mx.sym.concat(r1, r2, r3)

        self._assert_equal(sym, des)

if __name__ == "__main__":
    import sys
    unittest.main(argv=sys.argv, verbosity=5)
