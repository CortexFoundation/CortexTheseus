import mxnet as mx
from os import path
import sym_pass as spass

sym = mx.sym.Variable(name='input')
sym = mx.sym.transpose(sym, axes=(0, 2, 3, 1))
sym1 = mx.sym.transpose(sym, axes=(0, 1, 3, 2))
sym2 = mx.sym.transpose(sym, axes=(1, 0, 2, 3))
sym3 = mx.sym.transpose(sym, axes=(1, 0, 3, 2))
sym = mx.sym.Group([sym1, sym2, sym3])
params = {}
sym, _ = spass.fuse_transpose(sym, params)
dump = path.expanduser('~/tvm-cvm/data/dump_ryt.json')
with open(dump, 'w') as f:
    f.write(sym.tojson())
