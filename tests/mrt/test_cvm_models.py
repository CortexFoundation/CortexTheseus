import mxnet as mx
import nnvm
import random
import itertools
import numpy as np
from nnvm.compiler import graph_attr, graph_util
from nnvm import graph as _graph
import math
import tvm
from mxnet import nd

import sym_utils as sutils
import utils
import mrt as _mrt

def random_shape(shp, min_dim=1, max_dim=64):
    for i in range(len(shp)):
        if shp[i] is None:
            shp[i] = random.randint(min_dim, max_dim)
    while np.product(shp) > (1 << 20):
        rand_idx = random.randint(0, len(shp)-1)
        shp[rand_idx] = max(shp[rand_idx] // 2, 1)
    return shp
def random_bool():
    return random.randint(0, 1) == 0
def random_select(arr):
    return arr[random.randint(0, len(arr)-1)]
def random_int():
    int_max = (2 ** 31) - 1
    return random.randint(-int_max, int_max)

def get_cvm_op(op_name):
    if op_name == 'null':
        return nnvm.sym.Variable
    op = getattr(nnvm.sym, op_name)
    if not op:
        raise OpNotImplemented(
            'Operator {} is not supported.'.format(op_name))
    return op

def infer_shape(symbol):
    nnvm_graph = _graph.create(symbol)
    return graph_util.infer_shape(nnvm_graph)
def to_json(symbol):
    graph = _graph.create(symbol)
    return graph.json()

def adjust_shape(sym, ishp, oshp):
    isize, osize = np.product(ishp), np.product(oshp)
    if ishp != (isize,):
        print ("\tflatten", sym.attr('name'), ishp, "into (", isize, ",)")
        sym = get_cvm_op("reshape")(sym, shape=(isize,))
    if isize < osize:
        repeats = int(math.ceil(osize / isize))
        print ("\trepeat", sym.attr('name'), "times=", repeats)
        sym = get_cvm_op("repeat")(sym, repeats=repeats)
    _, shape = infer_shape(sym)
    print ("\tstrided_slice", sym.attr('name'), "with (0, ", osize, ")", shape)
    sym = get_cvm_op("strided_slice")(sym, begin=(0,), end=(osize,))
    if oshp != (osize,):
        print ("\treshape", sym.attr('name'), "into ", oshp)
        sym = get_cvm_op("reshape")(sym, shape=oshp)
    return sym

class Attr():
    bool_t = '__bool_type'
    int_t = '__int_type'
    list_10_t = '__list_10_type'
    list_4_t = '__list_4_type'
    list_2_t = '__list_2_type'
    def __init__(self, in_len=1, out_len=1, param_candidate_rate=0, **kwargs):
        assert out_len > 0
        self.in_len = in_len
        self.out_len = out_len
        self.param_candidate_rate = param_candidate_rate
        self.attrs = kwargs
    def input_size(self):
        return self.in_len
    def output_size(self):
        return self.out_len
    def attr(self):
        attrs = {}
        for k, v in self.attrs.items():
            if v == Attr.bool_t:
                attrs[k] = random_bool()
            elif v == Attr.int_t:
                attrs[k] = random_int()
            elif v == Attr.list_2_t:
                attrs[k] = [random_int(), random_int()]
            elif v == Attr.list_4_t:
                attrs[k] = [random_int() for _ in range(4)]
            elif v == Attr.list_10_t:
                attrs[k] = [random_int() for _ in range(10)]
            else:
                attrs[k] = v
        return attrs
NullAttr = Attr(0, 1)

class EntryId():
    def __init__(self, node, entry=0):
        assert entry < len(node)
        self._node = node
        self._entry = entry
    def __call__(self, node_id=True):
        if node_id:
            return self._node.id()
        return self._entry
    def __repr__(self):
        return "<%d, %d>" % (self._node.id(), self._entry)

class Node():
    _eid = 0
    def __init__(self, op_name="null", attr=NullAttr):
        self._op_name = op_name
        self._attr = attr
        self._input_size = attr.input_size()
        if self._input_size is None:
            self._input_size = random.randint(1, 4)
        elif isinstance(self._input_size, list):
            length = len(self._input_size) - 1
            self._input_size = self._input_size[random.randint(0, length)]
        self._output_size = attr.output_size()
        self._id = Node._eid
        Node._eid += self._output_size
        self._inputs = None
    def attr(self, name=None):
        if name is None:
            return self._attr.attr()
        return self._attr.__dict__[name]
    def name(self):
        return self._op_name
    def set_inputs(self, inputs):
        self._inputs = inputs
    def get_children(self):
        return self._inputs
    def input_size(self):
        return self._input_size
    def id(self):
        return self._id
    def entry(self, entry=None):
        if entry is None:
            if self._op_name == 'get_valid_counts':
                return EntryId(self, 1)
            return EntryId(self, 0)
        return EntryId(self, entry)
    def __len__(self):
        return self._output_size
    def op(self):
        return get_cvm_op(self._op_name)
    def __repr__(self):
        return "id=%3d op=%-20s inputs=%s" % (self._id, self._op_name, self._inputs)

class IndexGraph():
    def __init__(self):
        self.idx_graph = {}
        self.nid = 0
        self.last_node = None
    def add_node(self, node):
        print ("Index Graph add node nid", "%3d"%self.nid, "node", node)
        self.idx_graph[self.nid] = node
        self.last_node = node
        self.nid = self.nid + 1
    def random_entry(self, param_rate=0):
        nid = random.randint(0, self.nid - 1)
        if random.randint(0, 99) < param_rate:
            self.add_node(Node())
            nid = self.nid - 1
        node = self.idx_graph[nid]
        return node.entry()
    def __iter__(self):
        for i in sorted(self.idx_graph.keys()):
            yield self.idx_graph[i]
    def __getitem__(self, key):
        assert key >= 0 and key < self.nid
        return self.idx_graph[key]

CVM_OPS = {
    # nn
    'conv2d': Attr(in_len=[2, 3], param_candidate_rate=100,
            channels=Attr.int_t, kernel_size=Attr.list_2_t,
            strides=Attr.list_2_t, padding=Attr.list_2_t,
            dilation=Attr.list_2_t, groups=Attr.int_t,
            use_bias=Attr.bool_t),
    'dense': Attr(in_len=[2, 3], param_candidate_rate=100,
            units=Attr.int_t, use_bias=Attr.bool_t),
    'relu': Attr(),
    'upsampling':Attr(scale=Attr.int_t),
    'max_pool2d': Attr(pool_size=Attr.list_2_t, strides=Attr.list_2_t,
            padding=Attr.list_2_t),

    # reduce
    'max': Attr(axis=Attr.list_10_t, keepdims=Attr.bool_t, exclude=Attr.bool_t),
    'sum': Attr(axis=Attr.list_10_t, keepdims=Attr.bool_t, exclude=Attr.bool_t),

    # elemwise
    'elemwise_add': Attr(in_len=2, param_candidate_rate=5),
    'elemwise_sub': Attr(in_len=2, param_candidate_rate=5),
    'abs': Attr(),
    'log2': Attr(),
    'negative': Attr(),
    'clip': Attr(),
    'cvm_clip': Attr(precision=Attr.int_t),
    'cvm_left_shift': Attr(shift_bit=Attr.int_t, precision=Attr.int_t),
    'cvm_right_shift': Attr(shift_bit=Attr.int_t, precision=Attr.int_t),

    # broadcast
    'broadcast_add': Attr(in_len=2, param_candidate_rate=10),
    'broadcast_sub': Attr(in_len=2, param_candidate_rate=10),
    'broadcast_mul': Attr(in_len=2, param_candidate_rate=70),
    'broadcast_max': Attr(in_len=2, param_candidate_rate=50),

    # vision
    'get_valid_counts': Attr(out_len=2, score_threshold=Attr.int_t),
    'non_max_suppression': Attr(in_len=2, param_candidate_rate=100,
            iou_threshold=Attr.int_t, force_suppress=Attr.bool_t,
            top_k=Attr.int_t, max_output_size=Attr.int_t),

    # transform
    'expand_dims': Attr(axis=Attr.int_t, num_newaxis=Attr.int_t),
    'transpose': Attr(axes=Attr.list_10_t),
    'reshape': Attr(shape=Attr.list_10_t),
    'squeeze': Attr(axis=Attr.int_t),
    'concatenate': Attr(in_len=None, param_candidate_rate=5,
            axis=Attr.int_t),
    'take': Attr(in_len=2, axis=Attr.int_t),
    'strided_slice': Attr(begin=Attr.list_10_t, end=Attr.list_10_t, stride=Attr.list_10_t),
    'repeat': Attr(repeats=Attr.int_t, axis=Attr.int_t),
    'tile': Attr(reps=Attr.list_10_t),
    'slice_like': Attr(in_len=2, param_candidate_rate=10, axis=Attr.list_10_t),
    'cvm_lut': Attr(in_len=2),
    'flatten': Attr(),
}

op_names = [
    'conv2d', 'dense', 'expand_dims', 'transpose', 'reshape', 'squeeze',
    'concatenate', 'take', 'strided_slice', 'repeat', 'tile',
    'broadcast_add', 'broadcast_sub', 'broadcast_mul', 'broadcast_mul',
    'elemwise_add', 'elemwise_sub',
]
name_count = {}
def uniq_name(name):
    if name not in name_count:
        name_count[name] = 0
    uniq = "%s_%d" % (name, name_count[name])
    name_count[name] += 1
    return uniq

def sequence2symbol(idx_graph):
    def constraint_attr(attr, name, a_min, a_max):
        num = attr[name]
        if isinstance(num, list):
            for i, a in enumerate(num):
                alpha = a_max[i] - a_min[i] + 1
                attr[name][i] = a % alpha + a_min[i]
            return
        alpha = a_max - a_min + 1
        attr[name] = num % alpha + a_min

    graph, shapes = {}, {}
    params = {}
    inputs_ext = { 'data': {} }
    for node in idx_graph:
        input_eids = node.get_children()
        if input_eids is not None:
            childs = [graph[c()][c(False)] for c in input_eids]
        else:
            childs = None

        op, op_name = node.op(), node.name()
        nid, attr = node.id(), node.attr()
        shpes = []
        requantize = False
        shift_bit = 0
        if op_name == 'conv2d':
            ishp = random_shape([None, None, None, None])
            ishp[0] = random.randint(1, 16)
            ishp[1] = random_select([3, 4, 16])
            is_depthwise = random.randint(0, 100) > 80
            attr['groups'] = ishp[1] if is_depthwise else 1
            matrix_len = 1
            if is_depthwise:
                attr['channels'] = ishp[1]
            else:
                constraint_attr(attr, 'channels', a_min=1, a_max=256)
                matrix_len = attr['channels']
            h, w = ishp[2:]
            dh, dw = random.randint(1, min(3, h)), random.randint(1, min(3, w))
            kh, kw = random.randint(1, max(1, h // dh)), random.randint(1, max(1, w // dw))
            while (kh * kw) > (32768 / matrix_len):
                kh = kh // 2
                kw = kw // 2
            bit = math.ceil(math.log2(matrix_len * kh * kw))
            attr['kernel_size'] = (kh, kw)
            attr['dilation'] = (dh, dw)
            constraint_attr(attr, 'strides', a_min=(1, 1), a_max=(5, 5))
            constraint_attr(attr, 'padding', a_min=(1, 1), a_max=(7, 7))
            attr['use_bias'] = False if len(childs) == 2 else True
            if is_depthwise:
                wshp = (ishp[1], 1, kh, kw)
            else:
                wshp = (attr['channels'], ishp[1], kh, kw)
            shpes = [ishp, wshp]
            if attr['use_bias']:
                shpes.append((wshp[0],))
                bit += 1
            requantize = True
            shift_bit = bit + 16 - 14
        elif op_name == 'dense':
            ishp = random_shape([None, None], 10, 64)
            constraint_attr(attr, 'units', a_min=1, a_max=1000)
            bit = math.ceil(math.log2(attr['units']))
            attr['use_bias'] = False if len(childs) ==2 else True
            wshp = (attr['units'], ishp[1])
            shpes = [ishp, wshp]
            if attr['use_bias']:
                shpes.append((wshp[0],))
                bit += 1
            requantize = True
            shift_bit = bit + 16 - 14
        elif op_name == 'expand_dims':
            ndim = random.randint(1, 3)
            ishp = random_shape([None] * ndim)
            constraint_attr(attr, 'axis', a_min=-ndim-1, a_max=ndim)
            constraint_attr(attr, 'num_newaxis', a_min=1, a_max=6-ndim)
            shpes = [ishp]
        elif op_name == 'transpose':
            ndim = random.randint(1, 6)
            ishp = random_shape([None] * ndim, max_dim=32)
            axes = list(range(0, ndim))
            random.shuffle(axes)
            attr['axes'] = axes
            shpes = [ishp]
        elif op_name == 'reshape':
            ndim = random.randint(1, 4)
            shape = random_shape([None] * ndim)
            attr['shape'] = shape
            size = np.product(shape)
            shpes = [(size,)]
        elif op_name == 'squeeze':
            ndim = random.randint(2, 5)
            ishp = random_shape([None] * ndim)
            constraint_attr(attr, 'axis', a_min=-ndim, a_max=ndim-1)
            ishp[attr['axis']] = 1
            shpes = [ishp]
        elif op_name == 'concatenate':
            ndim = random.randint(1, 4)
            ishp = random_shape([None] * ndim)
            constraint_attr(attr, 'axis', a_min=-ndim, a_max=ndim-1)
            axis = attr['axis'] if attr['axis'] >= 0 else attr['axis']+ndim
            for _ in range(len(childs)):
                shp = [random.randint(1, 64) if i==axis else s for i,s in enumerate(ishp)]
                shpes.append(shp)
        elif op_name == 'take':
            ndim = random.randint(1, 3)
            ishp = random_shape([None] * ndim)
            constraint_attr(attr, 'axis', a_min=-ndim, a_max=ndim)
            attr['axis'] = None if attr['axis'] == ndim else attr['axis']

            ndim = random.randint(1, 2)
            wshp = random_shape([None] * ndim)
            shpes = [ishp, wshp]
        elif op_name == 'strided_slice':
            ndim = random.randint(1, 4)
            ishp = random_shape([None] * ndim)
            begin, end, stride = [], [], []
            for s in ishp:
                st = random_select([-3, -2, -1, 1, 2, 3])
                if s == 1:
                    b = 0
                    e = 1 if st > 0 else -s-1
                else:
                    b = random.randint(0, s-1)
                    e = random.randint(0, s-2)
                    e = e if e < b else e+1
                    if st > 0:
                        b, e = (b, e) if b < e else (e, b)
                    else:
                        b, e = (e, b) if b < e else (b, e)
                    b = b-s if random_bool() else b
                    e = e-s if random_bool() else e
                begin.append(b)
                end.append(e)
                stride.append(st)
            attr['begin'], attr['end'], attr['stride'] = begin, end, stride
            shpes = [ishp]
        elif op_name == 'repeat':
            ndim = random.randint(1, 4)
            ishp = random_shape([None] * ndim)
            constraint_attr(attr, 'repeats', 1, 10)
            constraint_attr(attr, 'axis', -ndim, ndim-1)
            shpes = [ishp]
        elif op_name == 'tile':
            ndim = random.randint(1, 4)
            ishp = random_shape([None] * ndim)
            rdim = random.randint(1, 5)
            attr['reps'] = [random.randint(1, 4) for _ in range(rdim)]
            shpes = [ishp]
        elif op_name == 'flatten':
            ndim = random.randint(1, 4)
            ishp = random_shape([None] * ndim)
            shpes = [ishp]
        elif op_name in ['broadcast_add', 'broadcast_sub', 'broadcast_mul', 'broadcast_max']:
            adim = random.randint(1, 4)
            bdim = random.randint(1, 4)
            max_dim = max(adim, bdim)
            shp = random_shape([None] * max_dim)
            ashp = [1 if random_bool() else shp[max_dim-adim+i] for i in range(adim)]
            bshp = [1 if random_bool() else shp[max_dim-bdim+i] for i in range(bdim)]
            shpes = [ashp, bshp]
        elif op_name in ['elemwise_add', 'elemwise_sub']:
            ndim = random.randint(1, 4)
            ishp = random_shape([None] * ndim)
            shpes = [ishp, ishp]
        print (op_name, attr, "childs shape:", shpes)

        if nid==0:
            ndim = random.randint(1, 4)
            oshape = [random_shape([None] * ndim)]
            node = op("data", shape=oshape[0], precision=8)
            inputs_ext["data"]["shape"] = oshape[0]
            print ("data", oshape)
        elif op_name == 'null':
            node = op(uniq_name("param"))
            oshape = None
        elif childs is not None:
            new_childs = []
            for i, c in enumerate(childs):
                cname, cop_name = c.attr('name'), c.attr('op_name')
                if shapes[cname] is None:
                    new_name = uniq_name("parameter")
                    new_c = get_cvm_op("null")(new_name, shape=shpes[i],
                            precision=8)
                    shapes[new_name] = shpes[i]
                    param = np.random.randint(-127, 127, shpes[i], "int32")
                    params[new_name] = tvm.nd.array(param)
                else:
                    new_c = adjust_shape(c, shapes[cname][input_eids[i](False)], shpes[i])
                new_childs.append(new_c)
            node = op(*new_childs, **attr)
            if requantize:
                #  node = get_cvm_op("cvm_right_shift")(node, shift_bit=shift_bit, precision=8)
                node = get_cvm_op("cvm_clip")(node, precision=8)
            ishape, oshape = infer_shape(node)
            print (op_name, "name:", node.attr('name'), "output shape:", oshape)
            if len(oshape) == 1:
                bias = get_cvm_op("null")(uniq_name("parameter"), shape=oshape[0],
                        precision=8)
                params[bias.attr('name')] = tvm.nd.array(np.random.randint(-127, 127, oshape[0], "int32"))
                node = get_cvm_op("elemwise_add")(node, bias)
                node = get_cvm_op("cvm_clip")(node, precision=8)
        else:
            assert False
        shapes[node.attr('name')] = oshape
        graph[nid] = node
    return graph[idx_graph.last_node.id()], params, inputs_ext

def gen_sequence():
    graph_size = random.randint(1, 100)
    print ("Graph Length", graph_size)
    #  op_names.extend(['conv2d' for _ in range(5)])
    #  op_names.extend(['dense' for _ in range(2)])
    ops_count = len(op_names)
    ops = [random.randint(0, ops_count-1) for _ in range(graph_size)]
    ops = [op_names[idx] for idx in ops]
    print ("Graph Ops", " -> ".join(ops))

    idx_graph = IndexGraph()
    Node._eid = 0
    out = Node()
    idx_graph.add_node(out)
    for op_name in ops:
        op = Node(op_name, CVM_OPS[op_name])
        input_size = op.input_size()
        param_rate = op.attr('param_candidate_rate')
        inputs = [out.entry() if i==0 else idx_graph.random_entry(param_rate) \
                for i in range(input_size)]
        op.set_inputs(inputs)
        idx_graph.add_node(op)
        out = op
    return idx_graph

def to_nnvm(sym, params, inputs_ext, model_name):
    dshp = inputs_ext['data']['shape']
    data = tvm.nd.array(np.random.randint(-127, 127, dshp, "int32"))
    _mrt.std_dump(sym, params, inputs_ext, data, model_name,
            is_mxnet=False, batch=True)

def cvm_model():
    graph = gen_sequence()
    symbol, params, inputs_ext = sequence2symbol(graph)
    open("/tmp/cvm_model.json", "w").write(to_json(symbol))
    param_bytes = nnvm.compiler.save_param_dict(params)
    open("/tmp/cvm_model.params", "wb").write(param_bytes)

    model_name = uniq_name("random_4")
    to_nnvm(symbol, params, inputs_ext, model_name)

def load_model():
    sym_json = open("/tmp/cvm_model.json", "r").read()
    param_bytes = open("/tmp/cvm_model.params", "rb").read()
    params = nnvm.compiler.load_param_dict(param_bytes)
    nnvm_graph = _graph.load_json(sym_json)

    for sym in sutils.topo_sort(nnvm_graph.symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        if op_name == 'repeat':
            _, oshape = infer_shape(sym)
            print (op_name, name, oshape)
    #  ishape, oshape = graph_util.infer_shape(nnvm_graph)
    #  print (ishape, oshape)

    model_name = "random_test"
    inputs_ext = { 'data': { 'shape': (25, 32, 28, 28) }}
    to_nnvm(nnvm_graph.symbol, params, inputs_ext, model_name)
    exit()

for i in range(10):
    cvm_model()
