import sys
from os import path
import logging

import mxnet as mx
from mxnet import ndarray as nd

import utils
import dataset as ds
import sym_pass as spass
import sym_utils as sutils
import sim_quant_helper as sim
import mrt as _mrt

NoneCfg, NoneValue = object(), object()
class Pack(object):
    def __init__(self, name="", parent=None, dscr=""):
        self._name, self._parent = name, parent
        self._dscr = dscr
        self._cfg = {}

    def path(self):
        if self._parent:
            return self._parent.path() + " > " + self._name
        return self._name

    def descripe(self, dscr):
        self._dscr = dscr
        return self

    def item(self):
        return self

    def usage(self, count=0, ident="    "):
        helper = "\n%s%s: %s\n" % (ident * count, self._name, self._dscr)
        for k, v in self._cfg.items():
            helper += v.usage(count + 1, ident)
        return helper

    def parse(self, cfg):
        cfg = {} if cfg == NoneValue else cfg
        for k, v in self._cfg.items():
            val = cfg[k] if k in cfg else NoneValue
            v.parse(val)

        for k in cfg:
            assert k in self._cfg, "%s > %s is not supported" \
                % (self.path(), k)


class Type(Pack):
    def item(self):
        return self._cfg["parsed"]

    def type(self, dtype):
        self._cfg["type"] = dtype
        self._cfg["value"] = NoneValue
        return self

    def value(self, val):
        self._cfg["type"] = type(val)
        self._cfg["value"] = val
        return self

    def usage(self, count, ident):
        helper = "%s[%-15s]: type -> %-7s" % (ident * count,
                self._name, self._cfg["type"].__name__)
        if self._cfg["value"] != NoneValue:
            helper += " default -> %s" % self._cfg["value"]
        if self._dscr:
            helper += ", %s" % self._dscr
        return helper + "\n"

    def parse(self, cfg):
        val = self._cfg["value"] if cfg == NoneValue else cfg
        assert val != NoneValue, "%s is not set" % self.path()
        assert isinstance(val, self._cfg["type"]), \
            "%s error: invalid type, expected %s vs. %s" \
            % (self.path(), self._cfg["type"].__name__, type(val).__name)
        self._cfg["parsed"] = val


class Config(Pack):
    def __getitem__(self, key):
        return self._cfg[key].item()

    def declare(self, key, dtype):
        assert isinstance(dtype, type)
        return self._cfg.setdefault(key, Type(key, self).type(dtype))

    def default(self, key, default):
        return self._cfg.setdefault(key, Type(key, self).value(default))

    def config(self, key):
        return self._cfg.setdefault(key, Config(key, self))

cfg = Config("Configuration").descripe("MRT Json Format")
cfg.declare("symbol", str).descripe("symbol file path")
cfg.declare("params", str).descripe("params file path")
cfg.declare("input_shape", tuple)
cfg.declare("dataset", str).descripe( \
        "optional ['imagenet', 'voc', 'mnist', 'quickdraw']")

_cfg = cfg.config("quantization").descripe("Config For Quantization")
_cfg.declare("batch_size", int)
_cfg.default("pure_int8", False)
_cfg.default("calibrate_num", 1)
_cfg.default("device", "cpu:0").descripe( \
        "(format) device:number optional ['cpu', 'gpu']")
_cfg.declare("output_precision", int)
_cfg.default("fixed", [])
_cfg.default("thresholds", {})
_cfg.default("split_names", [])
_cfg.default("name_maps", {})
_cfg.default("attr_scales", {})
_cfg.default("log", False)

_cfg = cfg.config("cvm").descripe("Config For Compiling to CVM")
_cfg.default("batch_size", -1).descripe( \
        "-1 indicates batch in section `quantization`")
_cfg.default("save_ext", False)
_cfg.default("dir", "./")

_cfg = cfg.config("accuracy").descripe("Config For Accuracy Evalulate")
_cfg.default("iter_num", 0)

if __name__ == "__main__":
    utils.log_init()
    logger = logging.getLogger("log.main")

    assert len(sys.argv) == 2, cfg.usage()
    cfgPath = sys.argv[1]
    baseDir = path.abspath(path.dirname(cfgPath))
    logger.info("Load config file: %s", cfgPath)
    with open(cfgPath, "r") as fin:
        lines = [l.strip() for l in fin.readlines()]
        lines = [l for l in lines if not l.startswith("#")]
        lines = [l for l in lines if not l == ""]
        config = eval(" ".join(lines))

    cfg.parse(config)

    sym_file, prm_file = cfg["symbol"], cfg["params"]
    if not path.isabs(sym_file):
        sym_file = path.abspath(path.join(baseDir, sym_file))
    if not path.isabs(prm_file):
        prm_file = path.abspath(path.join(baseDir, prm_file))

    qconf = cfg["quantization"]
    batch_size = qconf["batch_size"]
    pure_int8 = qconf["pure_int8"]
    calibrate_num = qconf["calibrate_num"]

    device = qconf["device"].split(":")
    ctx = mx.gpu(int(device[1])) if device[0] == "gpu" else mx.cpu()

    input_shape = cfg["input_shape"]
    shp = tuple(batch_size if s == -1 else s for s in input_shape)
    inputs_ext = { "data": {
        "shape": shp,
    } }

    dataset = cfg["dataset"]
    if dataset == "imagenet":
        data_iter = ds.load_imagenet_rec(batch_size, shp[2])
        def data_iter_func():
            data = data_iter.next()
            return data.data[0], data.label[0]
    elif dataset == "voc":
        val_data = ds.load_voc(batch_size, shp[2])
        data_iter = iter(val_data)
        def data_iter_func():
            return next(data_iter)
    elif dataset == "trec":
        data_iter = ds.load_trec(batch_size)
        def data_iter_func():
            return next(data_iter)
    elif dataset == "mnist":
        val_loader = ds.load_mnist(batch_size)
        data_iter = iter(val_loader)
        def data_iter_func():
            return next(data_iter)
    elif dataset == "quickdraw":
        val_data = ds.load_quickdraw10(batch_size)
        data_iter = iter(val_data)
        def data_iter_func():
            return next(data_iter)
    else:
        assert False, "dataset:%s is not supported" % (dataset)

    inputs = [mx.sym.var("data")]
    sym, params = mx.sym.load(sym_file), nd.load(prm_file)
    sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)

    debug = qconf["log"]
    if debug:
        with open(baseDir + "/mrt.prepare.json", "w") as fout:
            fout.write(sym.tojson())

    keys = qconf["split_names"]
    if len(keys) > 0:
        sym, params, inputs_ext, sym2, prm2, ins_ext2 \
            = _mrt.split_model(sym, params, inputs_ext, keys)
        name_maps = qconf["name_maps"]

    if debug:
        with open(baseDir + "/mxnet.split.json", "w") as fout:
            fout.write(sym.tojson())

    thresholds = qconf["thresholds"]
    fixed = qconf["fixed"]
    oprec = qconf["output_precision"]

    mrt = _mrt.MRT(sym, params, inputs_ext)     # initialize
    for i in range(calibrate_num):
        data, _ = data_iter_func()
        mrt.set_data('data', data)              # set input data
        mrt.calibrate(ctx=ctx)                  # calibration
    for k, v in thresholds.items():
        mrt.set_threshold(k, v)
    for k in fixed:
        mrt.set_fixed(k)
    if oprec > 0:
        mrt.set_output_prec(oprec)
    if pure_int8:
        mrt.set_pure_int8()
    qsym, qparams, inputs_ext = mrt.quantize()  # quantization

    oscales = mrt.get_output_scales()

    if debug:
        sim.save_ext(baseDir + "/mrt.quantize.ext", inputs_ext, oscales)
        with open(baseDir + "/mrt.quantize.json", "w") as fout:
            fout.write(qsym.tojson())
        nd.save(baseDir + "/mrt.quantize.params", qparams)

    if len(keys) > 0:
        oscales_dict = dict(zip([c.attr('name') for c in sym], oscales))
        oscales = [oscales_dict[name_maps[c.attr('name')]] for c in sym2]

        attr_scales = qconf["attr_scales"]
        def op_scales(node, params, graph):
            name, op_name = node.attr('name'), node.attr('op_name')
            childs, attr = sutils.sym_iter(node.get_children()), node.list_attr()
            if name in attr_scales:
                scales = attr_scales[name]
            elif op_name in attr_scales:
                scales = attr_scales[op_name]
            else:
                return node

            for k, v in scales.items():
                assert k in attr, "attribute %s not in %s(%s) with %s" \
                    % (k, op_name, name, attr.keys())
                attr[k] = int(float(attr[k]) * oscales_dict[v])
                node = sutils.get_mxnet_op(op_name)(*childs, **attr, name=name)
            return node
        maps = mrt.get_maps()
        qsym, qparams = _mrt.merge_model(qsym, qparams, sym2, prm2, maps, op_scales)

    cvm_flag = cfg["cvm"]
    cvm_batch_size = cvm_flag["batch_size"]
    cvm_batch_size = batch_size if cvm_batch_size == -1 else cvm_batch_size

    shp = tuple(cvm_batch_size if s == -1 else s for s in input_shape)
    inputs_ext["data"]["shape"] = shp
    nnvm_sym, nnvm_params = spass.mxnet_to_nnvm(qsym, qparams, inputs_ext)

    cvm_dir = cfg["cvm"]["dir"]
    spass.cvm_build(nnvm_sym, nnvm_params, inputs_ext,
            path.join(cvm_dir, "cvm.symbol"),
            path.join(cvm_dir, "cvm.params"))

    if cfg["cvm"]["save_ext"]:
        sim.save_ext(path.join(cvm_dir, "cvm.ext"), inputs_ext, oscales)

