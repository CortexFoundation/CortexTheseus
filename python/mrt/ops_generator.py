import hashlib
import shutil
import numpy as np
import random
import logging
import os
import copy
import re
import itertools
from mxnet import nd

logger = logging.getLogger('log.ops.unittest')
random.seed(42)

class AttrName():
    def __init__(self, name="", cvm_name=None):
        self.nd_name = name
        self.cvm_name = cvm_name if cvm_name else name

def iter_constraint(number):
    return lambda : [i for i in range(number)]
def std_int_constraint(number):
    return lambda : [-abs(number), 0, abs(number)]
def shape_constraint(number):
    return lambda : list(range(1, number+1))
def range_constraint(start, end, step=1):
    return lambda : [i for i in range(start, end, step)]
def gen_non_constraint():
    return lambda : [None]
def gen_num_constraint(number):
    return lambda : [number]
def list_constraint(arr):
    return lambda : arr
def rand_constraint(start, end, size=1):
    return lambda : [random.randint(start, end) for _ in range(size)]

class NoneIter():
    def __len__(self):
        return 1
    def __getitem__(self, key):
        assert key == 0
        return None

class BoolIter(AttrName):
    def __init__(self, const=None, **kwargs):
        super(BoolIter, self).__init__(**kwargs)
        if const is not None:
            self._size = 1
            self._const = const
        else:
            self._size = 2
    def __len__(self):
        return self._size
    def __getitem__(self, key):
        assert key >= 0 and key < self._size
        if self._size == 1:
            return self._const
        return (key == 0)

class RandomIter(AttrName):
    def __init__(self, start, end, divisor=False, **kwargs):
        super(RandomIter, self).__init__(**kwargs)
        self._diter = rand_constraint(start, end, 1)
        self.divisor = divisor
    def __len__(self):
        return 1
    def __getitem__(self, key):
        assert key == 0
        val = self._diter()[0]
        if self.divisor:
            while val == 0:
                val = self._diter()[0]
        return val



class IntIter(AttrName):
    def __init__(self, *cstrs, **kwargs):
        super(IntIter, self).__init__(**kwargs)
        self._iter = []
        for cstr in cstrs:
            data = cstr()
            for d in data:
                if d not in self._iter:
                    self._iter.append(d)
        self._size = len(self._iter)
    def __len__(self):
        return self._size
    def __getitem__(self, key):
        assert key >= 0 and key < self._size, \
            "index %d out of range [0, %d)" % (key, self._size)
        return self._iter[key]

class ConstantIter(AttrName):
    def __init__(self, cstr, shape=None):
        flatten = cstr()
        if shape == None:
            shape = (len(flatten),)
        size = np.product(shape)
        assert size == len(flatten)
        for shp in reversed(shape):
            size = size // shp
            flatten = [flatten[i*shp:(i+1)*shp] for i in range(size)]
        self._iter = flatten
    def __len__(self):
        return 1
    def __getitem__(self, key):
        assert key == 0
        return self._iter[key]

class PermutationIter(AttrName):
    def __init__(self, *cstrs, **kwargs):
        super(PermutationIter, self).__init__(**kwargs)
        self._iter = []
        for cstr in cstrs:
            ava = cstr()
            for possible in itertools.permutations(ava):
                self._iter.append(possible)
        self._size = len(self._iter)
    def __len__(self):
        return self._size
    def __getitem__(self, key):
        assert key >= 0 and key < self._size, \
            "index %d out of range [0, %d)" % (key, self._size)
        return self._iter[key]

class RandomBoolIter(AttrName):
    def __init__(self, **kwargs):
        super(RandomBoolIter, self).__init__(**kwargs)
        self._diter = rand_constraint(0, 2, 1)
    def __len__(self):
        return 1
    def __getitem__(self, key):
        assert key == 0
        return (self._diter()[0] == 0)

class RandomVectorIter(AttrName):
    def __init__(self, start, end, length, size=5, **kwargs):
        super(RandomVectorIter, self).__init__(**kwargs)
        self._diter = rand_constraint(start, end, length)
        self._vec_len = length
        self._size = size
    def __len__(self):
        return self._size
    def __getitem__(self, key):
        assert key >= 0 and key < self._size, \
            "index %d out of range [0, %d)" % (key, self._size)
        return self._diter()
        vec = [None] * self._vec_len
        for i in range(self._vec_len):
            vec[i] = self._diter()
        return vec

class VectorIter(AttrName):
    def __init__(self, dattr, size, **kwargs):
        super(VectorIter, self).__init__(**kwargs)
        self._dattr = dattr
        self._dim = size
        self._size = len(dattr) ** size if size > 0 else 1
    def __len__(self):
        return self._size
    def __getitem__(self, key):
        assert key >= 0 and key < self._size, \
            "index %d out of range [0, %d)" % (key, self._size)
        if self._dim == 0:
            return []
        step = len(self._dattr)
        vec = [None] * self._dim
        for i in range(self._dim):
            vec[self._dim-1-i] = self._dattr[key % step]
            key = key // step
        return vec

class RepeatIter(AttrName):
    def __init__(self, dattr, size, **kwargs):
        super(RepeatIter, self).__init__(**kwargs)
        self._dattr = dattr
        self._dim = size
        self._size = len(dattr)
    def __len__(self):
        return self._size
    def __getitem__(self, key):
        assert key >= 0 and key < self._size, \
            "index %d out of range [0, %d)" % (key, self._size)
        vec = [None] * self._dim
        for i in range(self._dim):
            vec[i] = self._dattr[key]
        return vec

class ShapeIter(AttrName):
    def __init__(self, iter, shape, attrs=None, **kwargs):
        super(ShapeIter, self).__init__(**kwargs)
        self._arr = iter
        if attrs is None:
            attrs = [VectorIter for _ in range(len(shape))]
        assert len(shape) == len(attrs)
        ndim = len(shape)
        for i in range(ndim):
            idx = ndim - 1 - i
            self._arr = attrs[idx](self._arr, shape[idx])
        self._size = len(self._arr)
    def __len__(self):
        return self._size
    def __getitem__(self, key):
        assert key >= 0 and key < self._size, \
            "index %d out of range [0, %d)" % (key, self._size)
        return self._arr[key]

class ExtendIter(AttrName):
    def __init__(self, *iters, **kwargs):
        super(ExtendIter, self).__init__(**kwargs)
        self.iters = iters
        self._size = np.product([len(it) for it in iters])
    def __len__(self):
        return self._size
    def __getitem__(self, key):
        assert key >= 0 and key < self._size, \
            "index %d out of range [0, %d)" % (key, self._size)
        vec = [None] * len(self.iters)
        for i, it in enumerate(self.iters):
            vec[i] = it[key % len(it)]
        return vec

class ConcatIter(AttrName):
    def __init__(self, *iters, **kwargs):
        super(ConcatIter, self).__init__(**kwargs)
        self._iters = iters
        self._size = np.sum([len(c) for c in iters])
    def __len__(self):
        return self._size
    def __getitem__(self, key):
        assert key >= 0 and key < self._size, \
            "index %d out of range [0, %d)" % (key, self._size)
        for it in self._iters:
            if key >= len(it):
                key = key - len(it)
            else:
                return it[key]

class AllOverIter(AttrName):
    def __init__(self, dattr, shp_cstr, **kwargs):
        super(AllOverIter, self).__init__(**kwargs)
        self._dattr = dattr
        self._shp_iter = shp_cstr()

        step = len(dattr)
        self._size = 0
        self._shp_size = [0]
        for ndim in self._shp_iter:
            self._size += step ** ndim
            self._shp_size.append(self._size)
    def __len__(self):
        return self._size
    def __getitem__(self, key):
        assert key >= 0 and key < self._size, \
            "index %d out of range [0, %d)" % (key, self._size)
        step = len(self._dattr)
        for shp, size in enumerate(self._shp_size):
            if key < size:
                key = key - self._shp_size[shp-1]
                vec = [None] * shp
                for i in range(shp):
                    vec[shp-1-i] = self._dattr[key % step]
                    key = key // step
                return vec

class OpUnitIter():
    def __init__(self, inputs, attr_index=None, constraints=[]):
        self.inputs = inputs
        self.attr_idx = attr_index if attr_index else len(inputs)
        self.cstr_funcs = constraints
        size = np.product([len(c) for c in self.inputs])

        self._iter = []
        print ("OpUnitIter size: ", size)
        for i in range(size):
            idxes, ins = [], []
            for c in self.inputs:
                idx = i % len(c)
                ins.append(c[idx])
                idxes.append(idx)
                i = i // len(c)
            flag = True
            for func in self.cstr_funcs:
                if not func(*ins):
                    flag = False
                    break
            if flag:
                self._iter.append(ins)
        self._size = len(self._iter)
    def __len__(self):
        return self._size
    def __getitem__(self, key):
        assert key >= 0 and key < self._size, \
            "index %d out of range [0, %d)" % (key, self._size)
        return self._iter[key]
    def attrs(self, inputs):
        attrs = {}
        for i in range(self.attr_idx, len(self.inputs)):
            if inputs[i] is not None:
                attrs[self.inputs[i].cvm_name] = inputs[i]
        return attrs
    def eval_data(self, op_name, op_func, is_dump=False):
        logger.info("Eval operator %s size=%s", op_name, len(self))
        for i in range(len(self)):
            out_npys, err = None, None
            inputs = self[i]
            attrs = self.attrs(inputs)
            try:
                out_npys = op_func(*[copy.deepcopy(d) for d in inputs])
            except Exception as e:
                err = "Error:\n" + str(e)
            logger.debug("Inputs: %s, Attr: %-30s, Outputs: %s, Error: %s",
                    inputs[:self.attr_idx], attrs,
                    str(out_npys).replace("\n", "").replace(" ", ""),
                    str(err).split("\n"))
            if is_dump:
                dump(op_name, attrs, inputs[:self.attr_idx], out_npys, err)

def npy_sha256(data):
    hsh = hashlib.sha256(data.data.tobytes()).hexdigest()
    return hashlib.sha256("{}{}{}{}{}{}{}{}{}".format(
            hsh, data.shape, data.dtype,
            data.min(), data.max(),
            data.argmin(), data.argmax(),
            data.mean(), data.var(),
    ).encode()).hexdigest()

def txt_sha256(data):
    return hashlib.sha256(data.encode()).hexdigest()

def _dump_np(hsh_file, ln_file, data):
    logger = logging.getLogger('log.ops.numpy.dump')

    if os.path.exists(hsh_file):
        loaded = np.load(hsh_file)
        if not np.equal(loaded, data).all():
            logger.error(
                "Dump op failed:%-20s hash file=%s, link file=%s",
                    op_name, hsh_file, ln_file)
            return False
    np.save(hsh_file, data)
    os.symlink(hsh_file, ln_file)
    return True

def _dump_txt(hsh_file, ln_file, data):
    logger = logging.getLogger('log.ops.txt.dump')

    if os.path.exists(hsh_file):
        loaded = open(hsh_file, "r").read()
        if data != loaded:
            logger.error(
                "Dump op failed: hash file=%s, link file=%s",
                    hsh_file, ln_file)
            return False
    open(hsh_file, "w").write(data)
    os.symlink(hsh_file, ln_file)
    return True

def npy_txt(data):
    data = data.astype("int32")
    shp = data.shape
    txt = "{}\n{}\n{}\n".format(
        len(shp),
        " ".join([str(s) for s in shp]),
        " ".join([str(s) for s in data.flatten()]),
    )
    return txt

def txt_npy(data):
    ndim, shp, nums = data.split("\n")
    shp = [int(s) for s in shp.strip().split(' ')]
    assert len(shp) == eval(ndim)
    arr = nums.strip().split(' ')
    npy = np.array([int(a) for a in arr]).reshape(shp)
    return npy

def txt_npy(data):
    dim, shp, data_str = data.strip().split("\n")
    dim = int(dim)
    shp = tuple([int(s) for s in shp.strip().split(" ")])
    arr = [int(s) for s in data_str.strip().split(" ")]
    arr = np.array(arr).reshape(shp)
    return arr

def _dump_dict(attr, attr_file):
    fout = open(attr_file, "w+")
    fout.write("{")
    arr = ["\"%s\":\"%s\""%(k, v) for k, v in attr.items()]
    fout.write(", ".join(arr))
    fout.write("}\n")

def clean_dir(datadir="/data/ops_generator"):
    shutil.rmtree(datadir, ignore_errors=True)
    logger.info("Clean directory: %s", datadir)

def dump(op_name, attr, ins, outs, err=None,
        datadir="/data/ops_generator"):
    logger = logging.getLogger('log.ops.unittest')

    npdir = "%s/%s" % (datadir, ".hidden.out")
    os.makedirs(npdir, exist_ok=True)

    for i, _in in enumerate(ins):
        if isinstance(_in, nd.NDArray):
            ins[i] = _in.asnumpy()
    for i, _in in enumerate(ins):
        if not isinstance(_in, np.ndarray):
            ins[i] = np.array(_in, dtype="int32")

    ins = [npy_txt(_in) for _in in ins]
    hshes = [txt_sha256(_in) for _in in ins]
    hsh = hashlib.sha1("{}{}".format(hshes, attr)
                .encode()).hexdigest()
    hsh_dir = "%s/%s/%s" % (datadir, op_name, hsh)
    if os.path.exists(hsh_dir):
        logger.info("Skip op:%-20s hashdir=%s",
            op_name, hsh)
        return

    os.makedirs(hsh_dir, exist_ok=True)
    attr_file = "%s/%s" % (hsh_dir, "attr.txt")
    _dump_dict(attr, attr_file)
    for i, _in in enumerate(ins):
        in_file = "%s/in_%d.txt" % (hsh_dir, i)
        hsh_file = "%s/%s.tx" % (npdir, hshes[i])
        if not _dump_txt(hsh_file, in_file, _in):
            shutil.rmtree(hsh_dir)
            return

    if err:
        err_file = "%s/err.txt" % (hsh_dir)
        open(err_file, "w").write(err + "\n")
        return

    for i, _out in enumerate(outs):
        if isinstance(_out, nd.NDArray):
            outs[i] = _out.asnumpy()
    for i, _out in enumerate(outs):
        if not isinstance(_out, np.ndarray):
            outs[i] = np.array(_out, dtype="int32")
    for i, _out in enumerate(outs):
        out_file = "%s/out_%d.txt" % (hsh_dir, i)
        _out = npy_txt(_out)
        out_hsh = txt_sha256(_out)
        #  out_hsh = npy_sha256(_out)
        hsh_file = "%s/%s.txt" % (npdir, out_hsh)
        if not _dump_txt(hsh_file, out_file, _out):
            shutil.rmtree(hsh_dir)
            return

def dump_file(op_name, in_file, out_file, attr_file, root="./"):
    attr = open(root+"/"+attr_file, "r").read()
    print (attr)
    attr = eval(attr.strip())
    ins = [np.load(root+"/"+f) for f in in_file]
    outs = [np.load(root+"/"+f) for f in out_file]
    dump(op_name, attr, ins, outs, None)


