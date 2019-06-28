from __future__ import print_function
import math
import numpy as np
import tvm
import topi
import topi.testing

from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple
from topi.vision import ssd, non_max_suppression, get_valid_counts
import os

DIR = '/data/zkh/non_max_suppression'
def save_dict(params, fn, change=False):
    with open(fn, 'w+') as fout:
        fout.write("{")
        is_first = True
        for k, v in params.items():
            if is_first:
                is_first = False
            else:
                fout.write(", ")
            if k == 'step':
                k = 'stride'
            if k == 'axes' and change == True:
                k = 'axis'
            fout.write("\"%s\": \"%s\"" % (k, v))
        fout.write("}\n")

def save_txt(data, filename):
    with open(filename, mode='a') as f:
        shape = data.shape
        ndim = len(shape)
        f.write(str(ndim))
        f.write('\n')
        for i in shape:
            f.write(str(i))
            f.write(' ')
        f.write('\n')
        for i in data.asnumpy().flatten().astype('int32'):
            f.write(str(i))
            f.write(' ')
        f.write('\n')

def test_non_max_suppression():
    print("test non max suppression")
    for case_index in range(0, 100):
        case_dir = DIR + "/" + str(case_index)
        print(case_dir)
        os.makedirs(case_dir, exist_ok=True)

        batch = np.random.randint(low=1, high=10)
        n = np.random.randint(low=10, high=11)
        k = 6
        dshape = (batch, n, k)
        data = tvm.placeholder(dshape, name="data")
        valid_count = tvm.placeholder((dshape[0],), dtype="int32", name="valid_count")
        nms_threshold = np.random.randint(low=1, high=10)
        force_suppress = True if np.random.randint(low=0, high=1) == 1 else False
        nms_topk = np.random.randint(low=1, high=9)
        params = {'iou_threshold':nms_threshold*10, 'coord_start':2, 'score_index':1, 'id_index':0,
                'force_suppress':force_suppress, 'top_k': nms_topk, 'return_indices':False}
        save_dict(params, case_dir + '/attr.txt')

       # np_data = np.array([[[0, 8, 1, 20, 25, 45], [1, 7, 30, 60, 50, 80],
       #                      [0, 4, 4, 21, 19, 40], [2, 9, 35, 61, 52, 79],
       #                      [1, 5, 100, 60, 70, 110]]]).astype(data.dtype)
       # np_valid_count = np.array([4]).astype(valid_count.dtype)
        np_data = np.random.randint(low=-(2**31-1), high=(2**31-1), size=dshape).astype(data.dtype)
        np_valid_count = np.random.randint(low=1, high=10, size=(batch)).astype(valid_count.dtype)

        device = 'llvm'
        ctx = tvm.context(device, 0)
        with tvm.target.create(device):
            out = non_max_suppression(data, valid_count, -1, nms_threshold, force_suppress, nms_topk, return_indices=False)
            s = topi.generic.schedule_nms(out)

        tvm_data = tvm.nd.array(np_data, ctx)
        tvm_valid_count = tvm.nd.array(np_valid_count, ctx)
        save_txt(tvm_data, case_dir + "/in_0.txt")
        save_txt(tvm_valid_count, case_dir + "/in_1.txt")

        tvm_out = tvm.nd.array(np.zeros(dshape, dtype=data.dtype), ctx)
        f = tvm.build(s, [data, valid_count, out], device)
        f(tvm_data, tvm_valid_count, tvm_out)

        save_txt(tvm_out, case_dir + "/out_0.txt")



if __name__ == "__main__":
    test_non_max_suppression()
