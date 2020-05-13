import tvm
from tvm.contrib import graph_runtime
import nnvm

import utils
import sim_quant_helper as sim
import sym_pass as spass
import cvm_op

def load_fname(prefix, suffix=None):
    suffix = "."+suffix if suffix is not None else ""
    load_prefix = prefix + suffix
    names = list(utils.extend_fname(load_prefix, True))
    names, ext_file = names[:-1], names[-1]
    (inputs_ext,) = sim.load_ext(ext_file)
    dump_prefix = prefix + ".nnvm.compile"
    names.extend(utils.extend_fname(dump_prefix, False))
    return names, inputs_ext

names, inputs_ext = load_fname("./data/quick_raw_qd_animal10_2_cifar_resnet20_v2", "sym.quantize")
print (names, inputs_ext)
spass.to_nnvm(*names, inputs_ext)


