import time
import numpy as np
import logging

from mxnet import ndarray as nd
import mxnet as mx

from sym_utils import topo_sort, sym_iter, is_inputs, get_nd_op
from sym_utils import get_entry_id, topo_visit_transformer
import dataset as ds
import utils
import cvm_op
from tfm_pass import convert_params_dtype

def load_data(input_size=224, batch_size=1, layout='NHWC'):
    ds_name = 'imagenet'
    data_iter_func = ds.data_iter(ds_name, batch_size, input_size=input_size)
    data, label = data_iter_func()
    data = data.asnumpy()
    if layout == 'NHWC':
        data = np.transpose(data, axes=[0,2,3,1])
    return data, label

def get_mxnet_outs(symbol, params, input_shape, ctx, gpu_flag, logger, iter_num):
    batch_size, _, input_size, _ = input_shape
    data, _ = load_data(input_size=input_size, batch_size=batch_size, layout='NCHW')
    data = nd.array(data)
    _, deps = topo_sort(symbol, with_deps=True)
    out_cache, ans = {}, {}
    times = {}

    def _impl(op, params, graph, **kwargs):
        deps = kwargs['deps']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()

        if op_name == 'null':
            start_time = None
            out = data if is_inputs(op, params) else params[name]
        elif childs is None:
            start_time= time.time()
            out = get_nd_op(op_name)(**attr)
            if gpu_flag:
                nd.waitall()
            end_time = time.time()
        else:
            cinfos = [(c.attr('name'), get_entry_id(c)) for c in childs]
            nd_inputs = [out_cache[n[0]][n[1]] for n in cinfos]
            start_time = time.time()
            out = get_nd_op(op_name)(*nd_inputs, **attr)
            if gpu_flag:
                nd.waitall()
            end_time = time.time()
            for n, _ in cinfos:
                assert n in deps
                deps[n].remove(name)
                if len(deps[n]) == 0:
                    del out_cache[n]
        if start_time is not None:
            if op_name not in times:
                times[op_name] = {}
            times[op_name][name] = end_time - start_time
        out = [out] if len(op) == 1 else out
        out_cache[name] = [o.as_in_context(ctx) for o in out]

    topo_visit_transformer(symbol, params, _impl, deps=deps, data=data)
    out_cache.clear()
    stime = {}
    for opn, dct in times.items():
        stime[opn] = {}
        tl = list(dct.values())
        # stime[opn]['num'] = len(tl)
        # stime[opn]['max'] = max(tl)
        # stime[opn]['min'] = min(tl)
        # stime[opn]['mean'] = np.mean(tl)
        # stime[opn]['std'] = np.std(tl)
        stime[opn]['total'] = sum(tl)
    logger.info('successfully run: #%s'%iter_num)
    return stime

def data_process(quantize_flag, input_shape, gpu_flag, num_test):
    if quantize_flag:
        symbol = mx.sym.load('/home/test/tvm-cvm/data/ssd_512_mobilenet1.0_coco.all.quantize.json')
        params = nd.load('/home/test/tvm-cvm/data/ssd_512_mobilenet1.0_coco.all.quantize.params')
        pfx = 'quant_'
    else:
        symbol = mx.sym.load('/home/test/tvm-cvm/data/ssd_512_mobilenet1.0_coco.json')
        params = nd.load('/home/test/tvm-cvm/data/ssd_512_mobilenet1.0_coco.params')
        pfx = 'org_'
    if gpu_flag:
        ctx = mx.gpu(3)
        pfx += 'gpu_'
    else:
        ctx = mx.cpu()
        pfx += 'cpu_'
    params = convert_params_dtype(params, dest_dtype="float32")
    utils.log_init()
    logger = logging.getLogger('main')
    stimes = {}
    for iter_num in range(num_test):
        for opn, dct in get_mxnet_outs(symbol, params, input_shape, ctx, gpu_flag, logger, iter_num).items():
            if opn not in stimes:
                stimes[opn] = {'sample_total': []}
            stimes[opn]['sample_total'].append(dct['total'])
    for opn, dct in stimes.items():
        stimes[opn]['mean_total'] = sum(dct['sample_total'][1:]) / len(dct['sample_total'][1:])
    arr = sorted([(stimes[opn]['mean_total'], opn) for opn in stimes], reverse=True)
    total = sum([dct['mean_total'] for opn, dct in stimes.items()])
    s = 'total forward time: %s second\n'%total
    s += '\n'
    for _, opn in arr:
        dct = stimes[opn]
        # s += 'op_name: %s\nmin: %s second\nmax: %s second\nmean: %s second\nstd: %s second\ntotal: %s second\n'%\
                # (opn, dct['max'], dct['min'], dct['mean'], dct['std'], dct['total'])
        s += 'op: %s\ntotal: %s second\n'%(opn, dct['mean_total'])
        s += '---------------------------\n'
        s += '---------------------------\n'
        s += '\n'
    filename = '/home/test/'+pfx+'test.txt'
    with open(filename, 'w') as f:
        f.write(s)


quantize_flag = True
input_shape = (1, 3, 512, 512)
gpu_flag = True
num_test = 17
data_process(quantize_flag, input_shape, gpu_flag, num_test)
