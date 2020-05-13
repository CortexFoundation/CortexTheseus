import mxnet as mx
from mxnet import ndarray as nd
import sym_utils as sutils

sym_file = "/home/test/tvm-cvm/data/ssd_512_mobilenet1.0_coco.prepare.json"
params_file = "/home/test/tvm-cvm/data/ssd_512_mobilenet1.0_coco.prepare.params"
symbol = mx.sym.load(sym_file)
params = nd.load(params_file)
for sym in sutils.topo_sort(symbol):
    name = sym.attr('name')
    if sym.attr('op_name') == 'Convolution':
        s = "\""+name+"\","
        print(s)

