import cvm_op
import mxnet as mx

qsym_file = '/home/test/tvm-cvm/data/ssd_512_mobilenet1.0_coco.all.quantize.json'
qsym = mx.sym.load(qsym_file)
