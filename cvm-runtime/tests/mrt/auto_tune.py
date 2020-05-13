import mxnet as mx
from mxnet import ndarray as nd
from os import path
import tfm_pass as tpass
import sym_utils as sutils

import os
from os import path
import sys
import math

lambd2 = 0.95
lambd = 0.55
mu = 0.3
stride = 1

def get_conv_names(modelname):
    conv_name_dct_old = {}

    weight_thresh = {}
    myfile = path.expanduser('~/tvm-cvm/cvm/models/'+modelname+'_mid.ini')
    for k, v in [v.split(':') for v in load_file(myfile)]:
        wname = k.strip()
        thresh = float(v.replace(',', '').strip())
        weight_thresh[wname] = thresh

    myfile = path.expanduser('~/tvm-cvm/cvm/models/'+modelname+'_normal.ini')
    weight_normal = [v.strip() for v in load_file(myfile)]

    sym_file = path.expanduser('~/tvm-cvm/data/'+modelname+'.prepare.json')
    params_file = path.expanduser('~/tvm-cvm/data/'+modelname+'.prepare.params')
    sym = mx.sym.load(sym_file)
    params = nd.load(params_file)
    weight_2_conv = []
    for sym in sutils.topo_sort(sym, params):
        if sym.attr('op_name') == 'Convolution':
            name = sym.attr('name')
            wname = sutils.sym_iter(sym.get_children())[1].attr('name')
            if wname in weight_thresh or wname in weight_normal:
                continue
            weight_2_conv.append((wname, name))
    return weight_2_conv

def run_model(modelname):
    combine_file(modelname)
    os.system('python cvm/quantization/main2.py cvm/models/'+modelname+'_auto.ini')
    myfile = '/home/test/tvm-cvm/data/th.txt'
    content = load_file(myfile)
    w, th = [v.strip() for v in content[-1].split(',')]
    th = float(th)
    myfile = '/home/test/tvm-cvm/data/acc.txt'
    p0, p = [float(v.split('%')[0].split('=')[1].strip())/100 for v in load_file(myfile)[0].split(',')]
    return w, th, p0, p

def load_file(filepath):
    with open(filepath, 'r') as f:
        s = f.readlines()
    return s

def write_file(filepath, content):
    with open(filepath, 'w') as f:
        for line in content:
            if line[-1] == '\n':
                f.write(line)
            else:
                f.write(line+'\n')

def combine_file(modelname):
    pfx = path.expanduser('~/tvm-cvm/cvm/models/'+modelname)
    top = load_file(pfx+'_top.ini')
    mid = load_file(pfx+'_mid.ini')
    base = load_file(pfx+'_base.ini')
    content = top+mid+base
    auto_file = pfx+'_auto.ini'
    write_file(auto_file, content)

def model_tuning(modelname):
    conv_restore_names = [name for _, name in get_conv_names(modelname)]
    filepath = path.expanduser("~/tvm-cvm/data/conv_restore_names.txt")
    write_file(filepath, conv_restore_names)
    print('run conv restore model')
    _, _, p0, p = run_model(modelname)
    assert p0*lambd <= p, "problem still exists besides conv"
    while True:
        w2c = get_conv_names(modelname)
        if not w2c:
            print('finished')
            break
        wname, conv_name = w2c[0]
        conv_restore_names = [name for _, name in w2c[1:]]
        filepath = path.expanduser("~/tvm-cvm/data/conv_restore_names.txt")
        write_file(filepath, conv_restore_names)
        w, th, p0, p = run_model(modelname)
        print('juding: ', w, th, p0, p)
        results = [(p, -1)]
        if p0*lambd > p:
            m_th = max(25, min(math.ceil(mu*th), 50))
            c_ths = [i for i in range(m_th, 0, -1)]
            max_p = p
            max_th = -1
            while c_ths:
                for cp, cth in results:
                    print(cp, cth)
                c_th = c_ths.pop()
                myfile = path.expanduser('~/tvm-cvm/cvm/models/'+modelname+'_mid.ini')
                content = load_file(myfile)
                pwname = content[-1].split(':')[0].strip() if content else None
                nline = '  ' + wname + ': ' + str(c_th)
                if not content:
                    pass
                elif pwname != wname:
                    content[-1] = '  '+content[-1].strip()+','
                else:
                    content.pop()
                content.append(nline)
                write_file(myfile, content)
                print('trial: ', wname, c_th)
                _, _, _, p = run_model(modelname)
                results.append((p, c_th))
                if p>max_p:
                    max_th = c_th
                    max_p = p
                if p>lambd2*p0:
                    break
            print('juding finished: ', w, th)
            for cp, cth in results:
                print(cp, cth)
            if max_p < p0*lambd:
                assert False, "lambd: " + str(lambd) + " is too large. " + \
                    str(max_p) + ',' + str(max_th) + ','+str(max_p/p0)
            myfile = path.expanduser('~/tvm-cvm/cvm/models/'+modelname+'_mid.ini')
            content = load_file(myfile)
            content.pop()
            nline = '  ' + wname + ': ' + str(max_th)
            content.append(nline)
            write_file(myfile, content)
            print(conv_name, wname, ': tuned', max_th)
        else:
            myfile = path.expanduser('~/tvm-cvm/cvm/models/'+modelname+'_normal.ini')
            wnames = load_file(myfile) +[wname]
            write_file(myfile, wnames)
            print(conv_name, wname, ': directly passed')

if __name__ == '__main__':
    assert len(sys.argv) == 2
    modelname = sys.argv[1]
    model_tuning(modelname)
