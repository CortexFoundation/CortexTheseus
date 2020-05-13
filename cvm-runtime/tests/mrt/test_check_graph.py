import mxnet as mx
import sym_utils as sutils
import sym_pass as spass
import logging
import utils
import json
from mxnet import ndarray as nd

version = "v3"

def load_fname(version, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    fname = "./data/tf_inception%s%s"%(version, suffix)
    return utils.extend_fname(fname, with_ext)


utils.log_init()
logger = logging.getLogger("log.test.check_graph")
sym, prm = load_fname(version)
symbol, params = mx.sym.load(sym), nd.load(prm)


symbol, params = sutils.check_graph(symbol, params, logger)
symbol, params = spass.fuse_transpose(symbol, params, logger)
fsym, _ = load_fname("vt2")
with open(fsym, "w") as f:
    f.write(symbol.tojson())
