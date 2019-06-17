import mxnet as mx
from mxnet import ndarray as nd
import numpy as np
import json
import os

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

DIR = "/data/"

def test_concatenate():
    print("test concatenate")
    tmp_dir = DIR + "concatenate/"

    os.makedirs(tmp_dir + "0/", exist_ok=True)
    shape = np.random.randint(low=1,high=100, size=(4)).astype("int32")
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "0/in_0.npy", a.astype("int32"));
    b = np.random.randint(low=-127, high=127, size=shape)#np.array([[1,2],[3,4]])
    np.save(tmp_dir + "0/in_1.npy", b.astype("int32"));
    params = {'axis': 0}
    save_dict(params, tmp_dir + "0/attr.txt")
    c = np.concatenate((a,b), **params)
    np.save(tmp_dir + "0/out_0.npy", c.astype("int32"));
    print(c.shape)

    os.makedirs(tmp_dir + "1/", exist_ok=True)
    shape = np.random.randint(low=1,high=100, size=(4)).astype("int32")
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "1/in_0.npy", a.astype("int32"));
    b = np.random.randint(low=-127, high=127, size=shape)#np.array([[1,2],[3,4]])
    np.save(tmp_dir + "1/in_1.npy", b.astype("int32"));
    params = {'axis': 1}
    c = np.concatenate((a,b), **params)
    save_dict(params, tmp_dir + "1/attr.txt")
    np.save(tmp_dir + "1/out_0.npy", c.astype("int32"));
    print(c.shape)

    os.makedirs(tmp_dir + "2/", exist_ok=True)
    shape = np.random.randint(low=1,high=100, size=(4)).astype("int32")
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "2/in_0.npy", a.astype("int32"));
    b = np.random.randint(low=-127, high=127, size=shape)#np.array([[1,2],[3,4]])
    np.save(tmp_dir +"2/in_1.npy", b.astype("int32"));
    params = {'axis': 3}
    c = np.concatenate((a,b), **params)
    save_dict(params, tmp_dir +"2/attr.txt")
    np.save(tmp_dir +"2/out_0.npy", c.astype("int32"));
    print(c.shape)

    os.makedirs(tmp_dir + "3/", exist_ok=True)
    shape = np.random.randint(low=1,high=100, size=(4)).astype("int32")
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "3/in_0.npy", a.astype("int32"));
    b = np.random.randint(low=-127, high=127, size=shape)#np.array([[1,2],[3,4]])
    np.save(tmp_dir + "3/in_1.npy", b.astype("int32"));
    params = {'axis': -1}
    c = np.concatenate((a,b), **params)
    save_dict(params, tmp_dir + "3/attr.txt")
    np.save(tmp_dir + "3/out_0.npy", c.astype("int32"));
    print(c.shape)

   # c = np.concatenate((a,b), axis=4)
   # np.save("/tmp/concatenate/out4.npy", c.astype("int32"));
   # print(c.shape)

def test_repeat():
    print("test repeat")
    tmp_dir = DIR + "repeat/"

    os.makedirs(tmp_dir + "0/", exist_ok=True)
    shape = np.random.randint(low=1,high=50, size=(4)).astype("int32")
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir +"0/in_0.npy", a.astype("int32"))
    params = {'axis':0, 'repeats':2}
    b = np.repeat(a,**params)
    save_dict(params, tmp_dir +"0/attr.txt")
    np.save(tmp_dir +"0/out_0.npy", b.astype("int32"))
    print(b.shape)

    os.makedirs(tmp_dir + "1/", exist_ok=True)
    shape = np.random.randint(low=1,high=50, size=(4)).astype("int32")
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir +"1/in_0.npy", a.astype("int32"))
    params = {'axis':1, 'repeats':3}
    b = np.repeat(a,**params)
    save_dict(params, tmp_dir +"1/attr.txt")
    np.save(tmp_dir +"1/out_0.npy", b.astype("int32"))
    print(b.shape)

    os.makedirs(tmp_dir + "2/", exist_ok=True)
    shape = np.random.randint(low=1,high=50, size=(4)).astype("int32")
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir +"2/in_0.npy", a.astype("int32"))
    params = {'axis':3, 'repeats':2}
    b = np.repeat(a,**params)
    save_dict(params,tmp_dir + "2/attr.txt")
    np.save(tmp_dir +"2/out_0.npy", b.astype("int32"))
    print(b.shape)

    os.makedirs(tmp_dir + "3/", exist_ok=True)
    shape = np.random.randint(low=2,high=5, size=(4)).astype("int32")
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir +"3/in_0.npy", a.astype("int32"))
    params = {'axis':-1, 'repeats':3}
    b = np.repeat(a,**params)
    save_dict(params, tmp_dir +"3/attr.txt")
    np.save(tmp_dir +"3/out_0.npy", b.astype("int32"))
    print(b.shape)

def test_tile():
    print("test tile")
    tmp_dir = DIR + "tile/"

    os.makedirs(tmp_dir + "0/", exist_ok=True)
    shape = np.random.randint(low=1, high=50, size=(4)).astype("int32")
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir +"0/in_0.npy", a.astype("int32"))
    params = {'reps':[2]}
    save_dict(params, tmp_dir +"0/attr.txt")
    b = np.tile(a, **params)
    np.save(tmp_dir +"0/out_0.npy", b.astype("int32"))
    print(b.shape)

    os.makedirs(tmp_dir + "1/", exist_ok=True)
    shape = np.random.randint(low=1, high=50, size=(4)).astype("int32")
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir +"1/in_0.npy", a.astype("int32"))
    params = {'reps':[2, 3]}
    save_dict(params, tmp_dir +"1/attr.txt")
    b = np.tile(a, **params)
    np.save(tmp_dir +"1/out_0.npy", b.astype("int32"))
    print(b.shape)

    os.makedirs(tmp_dir + "2/", exist_ok=True)
    shape = np.random.randint(low=1, high=50, size=(4)).astype("int32")
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir +"2/in_0.npy", a.astype("int32"))
    params = {'reps':[2, 3, 1]}
    save_dict(params, tmp_dir +"2/attr.txt")
    b = np.tile(a, **params)
    np.save(tmp_dir +"2/out_0.npy", b.astype("int32"))
    print(b.shape)

    os.makedirs(tmp_dir + "3/", exist_ok=True)
    shape = np.random.randint(low=1, high=50, size=(4)).astype("int32")
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir +"3/in_0.npy", a.astype("int32"))
    params = {'reps':[2, 3, 1, 4]}
    save_dict(params,tmp_dir + "3/attr.txt")
    b = np.tile(a, **params)
    np.save(tmp_dir +"3/out_0.npy", b.astype("int32"))
    print(b.shape)

    os.makedirs(tmp_dir + "4/", exist_ok=True)
    shape = np.random.randint(low=1, high=50, size=(4)).astype("int32")
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir +"4/in_0.npy", a.astype("int32"))
    params = {'reps':[2, 3, 1, 4, 5]}
    save_dict(params,tmp_dir + "4/attr.txt")
    b = np.tile(a, **params)
    np.save(tmp_dir +"4/out_0.npy", b.astype("int32"))
    print(b.shape)

def test_transpose():
    print("test transpose")
    tmp_dir = DIR + "transpose/"

    os.makedirs(tmp_dir + "0/", exist_ok=True)
    shape = np.random.randint(low=1, high=5, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "0/in_0.npy", a.astype("int32"))
    params = {'axes':[0,1,2,3]}
    save_dict(params, tmp_dir + "0/attr.txt");
    b = np.transpose(a, **params)
    print(b.shape)
    np.save(tmp_dir + "0/out_0.npy", b.astype("int32"))
    print(a.flatten())
    print(b.flatten())

    os.makedirs(tmp_dir + "1/", exist_ok=True)
    shape = np.random.randint(low=1, high=5, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "1/in_0.npy", a.astype("int32"))
    params = {'axes':[1,0,2,3]}
    save_dict(params, tmp_dir + "1/attr.txt");
    b = np.transpose(a, **params)
    print(b.shape)
    np.save(tmp_dir + "1/out_0.npy", b.astype("int32"))
    print(a.flatten())
    print(b.flatten())

    os.makedirs(tmp_dir + "2/", exist_ok=True)
    shape = np.random.randint(low=1, high=5, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "2/in_0.npy", a.astype("int32"))
    params = {'axes':[1,2,0,3]}
    save_dict(params, tmp_dir + "2/attr.txt");
    b = np.transpose(a, **params)
    print(b.shape)
    np.save(tmp_dir + "2/out_0.npy", b.astype("int32"))
    print(a.flatten())
    print(b.flatten())

    os.makedirs(tmp_dir + "3/", exist_ok=True)
    shape = np.random.randint(low=1, high=5, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "3/in_0.npy", a.astype("int32"))
    params = {'axes':[3,1,2,0]}
    save_dict(params, tmp_dir + "3/attr.txt");
    b = np.transpose(a, **params)
    print(b.shape)
    np.save(tmp_dir + "3/out_0.npy", b.astype("int32"))
    print(a.flatten())
    print(b.flatten())

#    os.makedirs(tmp_dir + "4/", exist_ok=True)
#    shape = np.random.randint(low=1, high=5, size=(4))
#    print(shape)
#    a = np.random.randint(low=-127, high=127, size=shape)
#    np.save(tmp_dir + "4/in_0.npy", a.astype("int32"))
#    params = {}
#    save_dict(params, tmp_dir + "4/attr.txt");
#    b = np.transpose(a, **params)
#    print(b.shape)
#    np.save(tmp_dir + "4/out_0.npy", b.astype("int32"))
#    print(b.flatten())

def test_strided_slice():
    print("test strided slice")
    tmp_dir = DIR + "strided_slice/"

    os.makedirs(tmp_dir + "0/", exist_ok=True)
    shape = np.random.randint(low=3, high=4, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    print(a)
    np.save(tmp_dir + "0/in_0.npy", a.astype("int32"))
    params = {"begin":[2,0], "end":[0,3], "step":[-1,2]}
    save_dict(params, tmp_dir + "0/attr.txt")
    b = nd.slice(nd.array(a), **params)
    np.save(tmp_dir + "0/out_0.npy", b.asnumpy().astype("int32"))
    print(b.shape)
    print(b)

    os.makedirs(tmp_dir + "1/", exist_ok=True)
    shape = np.random.randint(low=3, high=4, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    print(a)
    np.save(tmp_dir + "1/in_0.npy", a.astype("int32"))
    params = {"begin":[0,0], "end":[2,3]}
    save_dict(params, tmp_dir + "1/attr.txt")
    b = nd.slice(nd.array(a), **params)
    np.save(tmp_dir + "1/out_0.npy", b.asnumpy().astype("int32"))
    print(b.shape)

    os.makedirs(tmp_dir + "2/", exist_ok=True)
    shape = np.random.randint(low=3, high=4, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    print(a)
    np.save(tmp_dir + "2/in_0.npy", a.astype("int32"))
    params = {"begin":[0,0,1,1], "end":[1,2,3,3]}
    save_dict(params, tmp_dir + "2/attr.txt")
    b = nd.slice(nd.array(a), **params)
    np.save(tmp_dir + "2/out_0.npy", b.asnumpy().astype("int32"))
    print(b.shape)

def test_slice_like():
    print("test slice like")
    tmp_dir = DIR + "slice_like/"

    os.makedirs(tmp_dir + "0/", exist_ok=True)
    shape = np.random.randint(low=5, high=10, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "0/in_0.npy", a.astype("int32"))
    b = np.zeros((4,4,4))
    np.save(tmp_dir + "0/in_1.npy", b.astype("int32"))
    params = {'axes': [2,1]}
    save_dict(params, tmp_dir + "0/attr.txt", change=True)
    c = nd.slice_like(nd.array(a), nd.array(b), **params)
    np.save(tmp_dir + "0/out_0.npy", c.asnumpy().astype("int32"))
    print(c.shape)

    os.makedirs(tmp_dir + "1/", exist_ok=True)
    shape = np.random.randint(low=5, high=10, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "1/in_0.npy", a.astype("int32"))
    b = np.zeros((1,2))
    np.save(tmp_dir + "1/in_1.npy", b.astype("int32"))
    params = {'axes': [0,1]}
    save_dict(params, tmp_dir + "1/attr.txt", change=True)
    c = nd.slice_like(nd.array(a), nd.array(b), **params)
    np.save(tmp_dir + "1/out_0.npy", c.asnumpy().astype("int32"))
    print(c.shape)

    os.makedirs(tmp_dir + "2/", exist_ok=True)
    shape = np.random.randint(low=5, high=10, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "2/in_0.npy", a.astype("int32"))
    b = np.zeros((4,4,4))
    np.save(tmp_dir + "2/in_1.npy", b.astype("int32"))
    params = {'axes': [1,2]}
    save_dict(params, tmp_dir + "2/attr.txt", change=True)
    c = nd.slice_like(nd.array(a), nd.array(b), **params)
    np.save(tmp_dir + "2/out_0.npy", c.asnumpy().astype("int32"))
    print(c.shape)

def test_get_valid_counts():
    print("test get_valid_counts")

def test_non_max_suppression():
    print("test non_max_suppression")

def test_take():
    print("test take")
    tmp_dir = DIR + "take/"

    os.makedirs(tmp_dir + "0/", exist_ok=True)
    shape = np.random.randint(low=4, high=10, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "0/in_0.npy", a.astype("int32"))
    indices = [[0,1],[2,3]]
    np.save(tmp_dir + "0/in_1.npy", np.array(indices).astype("int32"))
    params = {}
    save_dict(params, tmp_dir + "0/attr.txt")
    c = np.take(a, indices, **params)
    np.save(tmp_dir + "0/out_0.npy", c.astype("int32"))
    print(c.shape)

    os.makedirs(tmp_dir + "1/", exist_ok=True)
    shape = np.random.randint(low=4, high=10, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "1/in_0.npy", a.astype("int32"))
    indices = [0]
    np.save(tmp_dir + "1/in_1.npy", np.array(indices).astype("int32"))
    params = {}
    save_dict(params, tmp_dir + "1/attr.txt")
    c = np.take(a, indices, **params)
    np.save(tmp_dir + "1/out_0.npy", c.astype("int32"))
    print(c.shape)

    os.makedirs(tmp_dir + "2/", exist_ok=True)
    shape = np.random.randint(low=2, high=4, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "2/in_0.npy", a.astype("int32"))
    indices = [[10,11],[13,15]]
    np.save(tmp_dir + "2/in_1.npy", np.array(indices).astype("int32"))
    params = {}
    save_dict(params, tmp_dir + "2/attr.txt")
    c = np.take(a, indices, **params)
    np.save(tmp_dir + "2/out_0.npy", c.astype("int32"))
    print(c.shape)

    os.makedirs(tmp_dir + "3/", exist_ok=True)
    shape = np.random.randint(low=10, high=11, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "3/in_0.npy", a.astype("int32"))
    indices = [[9,9],[9,9]]
    np.save(tmp_dir + "3/in_1.npy", np.array(indices).astype("int32"))
    params = {'axis':1}
    save_dict(params, tmp_dir + "3/attr.txt")
    c = np.take(a, indices, **params)
    np.save(tmp_dir + "3/out_0.npy", c.astype("int32"))
    print(c.shape)

    os.makedirs(tmp_dir + "4/", exist_ok=True)
    shape = np.random.randint(low=10, high=11, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "4/in_0.npy", a.astype("int32"))
    indices = [1,2,3,4]
    np.save(tmp_dir + "4/in_1.npy", np.array(indices).astype("int32"))
    params = {'axis':-1}
    save_dict(params, tmp_dir + "4/attr.txt")
    c = np.take(a, indices, **params)
    np.save(tmp_dir + "4/out_0.npy", c.astype("int32"))
    print(c.shape)

    os.makedirs(tmp_dir + "5/", exist_ok=True)
    shape = np.random.randint(low=10, high=11, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "5/in_0.npy", a.astype("int32"))
    indices = [1,2,3,4]
    np.save(tmp_dir + "5/in_1.npy", np.array(indices).astype("int32"))
    params = {'axis':-3}
    save_dict(params, tmp_dir + "5/attr.txt")
    c = np.take(a, indices, **params)
    np.save(tmp_dir + "5/out_0.npy", c.astype("int32"))
    print(c.shape)

def test_max_pool():
    print("test max pool")


def test_cvm_lut():
    print("test cvm_lut")

def test_upsampling():
    print("test upsampling")
    tmp_dir = DIR + "upsampling/"

    os.makedirs(tmp_dir + "0/", exist_ok=True)
    shape = np.random.randint(low=3, high=5, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "0/in_0.npy", a.astype("int32"))
    params = {'method':'NEAREST_NEIGHBOR', 'layout':'NCHW', 'scale': 1}
    save_dict(params, tmp_dir + "0/attr.txt")
    params = {'scale':1, 'sample_type':'nearest'}
    b = nd.UpSampling(nd.array(a), **params)
    np.save(tmp_dir + "0/out_0.npy", b.asnumpy().astype("int32"))
    print (b.shape)
    print(b)

    os.makedirs(tmp_dir + "1/", exist_ok=True)
    shape = np.random.randint(low=3, high=4, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "1/in_0.npy", a.astype("int32"))
    params = {'method':'NEAREST_NEIGHBOR', 'layout':'NCHW', 'scale': 2}
    save_dict(params, tmp_dir + "1/attr.txt")
    params = {'scale':2, 'sample_type':'nearest'}
    b = nd.UpSampling(nd.array(a), **params)
    np.save(tmp_dir + "1/out_0.npy", b.asnumpy().astype("int32"))
    print (b.shape)

    os.makedirs(tmp_dir + "2/", exist_ok=True)
    shape = np.random.randint(low=2, high=4, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "2/in_0.npy", a.astype("int32"))
    params = {'method':'NEAREST_NEIGHBOR', 'layout':'NCHW', 'scale': 3}
    save_dict(params, tmp_dir + "2/attr.txt")
    params = {'scale':3, 'sample_type':'nearest'}
    b = nd.UpSampling(nd.array(a), **params)
    np.save(tmp_dir + "2/out_0.npy", b.asnumpy().astype("int32"))
    print (b.shape)
    print(b)

def test_squeeze():
    print("test squeeze")

def test_expand_dims():
    print("test expand dims")

def test_negative():
    print("test negative")

def test_max():
    print("test max")
    tmp_dir = DIR + "max/"

    os.makedirs(tmp_dir + "0/", exist_ok=True)
    shape = np.random.randint(low=2, high=10, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "0/in_0.npy", a.astype("int32"))
    params = {'axis':[1,3]}
    save_dict(params, tmp_dir + "0/attr.txt")
    b = nd.max(nd.array(a), **params)
    np.save(tmp_dir + "0/out_0.npy", b.asnumpy().astype("int32"))
#    print(b.asnumpy().astype("int32").flatten())

    os.makedirs(tmp_dir + "1/", exist_ok=True)
    shape = np.random.randint(low=2, high=10, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "1/in_0.npy", a.astype("int32"))
    params = {}
    save_dict(params, tmp_dir + "1/attr.txt")
    b = nd.max(nd.array(a), **params)
    np.save(tmp_dir + "1/out_0.npy", b.asnumpy().astype("int32"))
#    print(b.asnumpy().astype("int32").flatten())

    os.makedirs(tmp_dir + "2/", exist_ok=True)
    shape = np.random.randint(low=2, high=10, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "2/in_0.npy", a.astype("int32"))
    params = {'axis':[0]}
    save_dict(params, tmp_dir + "2/attr.txt")
    b = nd.max(nd.array(a), **params)
    np.save(tmp_dir + "2/out_0.npy", b.asnumpy().astype("int32"))
#    print(b.asnumpy().astype("int32").flatten())

    os.makedirs(tmp_dir + "3/", exist_ok=True)
    shape = np.random.randint(low=2, high=10, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "3/in_0.npy", a.astype("int32"))
    params = {'axis':[2]}
    save_dict(params, tmp_dir + "3/attr.txt")
    b = nd.max(nd.array(a), **params)
    np.save(tmp_dir + "3/out_0.npy", b.asnumpy().astype("int32"))
#    print(b.asnumpy().astype("int32").flatten())

    os.makedirs(tmp_dir + "4/", exist_ok=True)
    shape = np.random.randint(low=2, high=10, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "4/in_0.npy", a.astype("int32"))
    params = {'axis':[3]}
    save_dict(params, tmp_dir + "4/attr.txt")
    b = nd.max(nd.array(a), **params)
    np.save(tmp_dir + "4/out_0.npy", b.asnumpy().astype("int32"))
#    print(b.asnumpy().astype("int32").flatten())

    os.makedirs(tmp_dir + "5/", exist_ok=True)
    shape = np.random.randint(low=2, high=10, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "5/in_0.npy", a.astype("int32"))
    params = {'axis':[1,2,3]}
    save_dict(params, tmp_dir + "5/attr.txt")
    b = nd.max(nd.array(a), **params)
    np.save(tmp_dir + "5/out_0.npy", b.asnumpy().astype("int32"))
#   print(b.asnumpy().astype("int32").flatten())

    os.makedirs(tmp_dir + "6/", exist_ok=True)
    shape = np.random.randint(low=2, high=10, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "6/in_0.npy", a.astype("int32"))
    params = {'axis':[0,1,2,3]}
    save_dict(params, tmp_dir + "6/attr.txt")
    b = nd.max(nd.array(a), **params)
    np.save(tmp_dir + "6/out_0.npy", b.asnumpy().astype("int32"))
#    print(b.asnumpy().astype("int32").flatten())

def test_sum():
    print("test sum")
    tmp_dir = DIR + "sum/"

    os.makedirs(tmp_dir + "0/", exist_ok=True)
    shape = np.random.randint(low=2, high=10, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "0/in_0.npy", a.astype("int32"))
    params = {'axis':[1,3]}
    save_dict(params, tmp_dir + "0/attr.txt")
    b = nd.sum(nd.array(a), **params)
    np.save(tmp_dir + "0/out_0.npy", b.asnumpy().astype("int32"))
#    print(b.asnumpy().astype("int32").flatten())

    os.makedirs(tmp_dir + "1/", exist_ok=True)
    shape = np.random.randint(low=2, high=10, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "1/in_0.npy", a.astype("int32"))
    params = {}
    save_dict(params, tmp_dir + "1/attr.txt")
    b = nd.sum(nd.array(a), **params)
    np.save(tmp_dir + "1/out_0.npy", b.asnumpy().astype("int32"))
#    print(b.asnumpy().astype("int32").flatten())

    os.makedirs(tmp_dir + "2/", exist_ok=True)
    shape = np.random.randint(low=2, high=10, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "2/in_0.npy", a.astype("int32"))
    params = {'axis':[0]}
    save_dict(params, tmp_dir + "2/attr.txt")
    b = nd.sum(nd.array(a), **params)
    np.save(tmp_dir + "2/out_0.npy", b.asnumpy().astype("int32"))
#    print(b.asnumpy().astype("int32").flatten())

    os.makedirs(tmp_dir + "3/", exist_ok=True)
    shape = np.random.randint(low=2, high=10, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "3/in_0.npy", a.astype("int32"))
    params = {'axis':[2]}
    save_dict(params, tmp_dir + "3/attr.txt")
    b = nd.sum(nd.array(a), **params)
    np.save(tmp_dir + "3/out_0.npy", b.asnumpy().astype("int32"))
#    print(b.asnumpy().astype("int32").flatten())

    os.makedirs(tmp_dir + "4/", exist_ok=True)
    shape = np.random.randint(low=2, high=10, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "4/in_0.npy", a.astype("int32"))
    params = {'axis':[3]}
    save_dict(params, tmp_dir + "4/attr.txt")
    b = nd.sum(nd.array(a), **params)
    np.save(tmp_dir + "4/out_0.npy", b.asnumpy().astype("int32"))
#    print(b.asnumpy().astype("int32").flatten())

    os.makedirs(tmp_dir + "5/", exist_ok=True)
    shape = np.random.randint(low=2, high=10, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "5/in_0.npy", a.astype("int32"))
    params = {'axis':[1,2,3]}
    save_dict(params, tmp_dir + "5/attr.txt")
    b = nd.sum(nd.array(a), **params)
    np.save(tmp_dir + "5/out_0.npy", b.asnumpy().astype("int32"))
#   print(b.asnumpy().astype("int32").flatten())

    os.makedirs(tmp_dir + "6/", exist_ok=True)
    shape = np.random.randint(low=2, high=10, size=(4))
    print(shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "6/in_0.npy", a.astype("int32"))
    params = {'axis':[0,1,2,3]}
    save_dict(params, tmp_dir + "6/attr.txt")
    b = nd.sum(nd.array(a), **params)
    np.save(tmp_dir + "6/out_0.npy", b.asnumpy().astype("int32"))
#    print(b.asnumpy().astype("int32").flatten())

def test_elemwise_add():
    print("test elemwise_add")
    tmp_dir = DIR + "elemwise_add/"
    os.makedirs(tmp_dir + "0/", exist_ok=True)
    shape = (1)
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "0/in_0.npy", a.astype("int32"))
    b = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "0/in_1.npy", b.astype("int32"))
    params = {}
    save_dict(params, tmp_dir + "0/attr.txt")
    c = nd.elemwise_add(nd.array(a), nd.array(b))
    np.save(tmp_dir + "0/out_0.npy", c.asnumpy().astype("int32"))

    os.makedirs(tmp_dir + "1/", exist_ok=True)
    shape = np.random.randint(low=10, high=100, size=(4))
    a = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "1/in_0.npy", a.astype("int32"))
    b = np.random.randint(low=-127, high=127, size=shape)
    np.save(tmp_dir + "1/in_1.npy", b.astype("int32"))
    params = {}
    save_dict(params, tmp_dir + "1/attr.txt")
    c = nd.elemwise_add(nd.array(a), nd.array(b))
    np.save(tmp_dir + "1/out_0.npy", c.asnumpy().astype("int32"))

def test_conv2d():
    print("test conv2d")
    batch = np.random.randint(low=1, high=32)
    i_c = np.random.randint(low=1, high=32)
    i_h = np.random.randint(low=7, high=256)
    i_w = np.random.randint(low=7, high=256)
    xshape = (batch, i_c, i_h, i_w)
    print(xshape)
    x = np.random.randint(low=-127, high=127, size=xshape)
    o_c = np.random.randint(low=1, high=1024)
    f_h = 3
    f_w = 3
    wshape = (o_c, i_c, f_h, f_w)
    print(wshape)
    w = np.random.randint(low=-127, high=127, size=wshape)

    stride=(1,1)
    padding=(0,0)
    dilation=(1,1)
    kernel_size=(f_h, f_w)
    o_h = (i_h + 2*padding[0] - f_h) / stride[0] + 1
    o_w = (i_w + 2*padding[1] - f_w) / stride[1] + 1

    oshape = (batch, o_c, o_h, o_w)
    params = {'stride':stride, 'pad':padding, 'dilate':dilation, 'kernel': kernel_size,'num_filter':o_c}
    y = nd.Convolution(nd.array(x), nd.array(w), None, **params)

def test_broadcast_add():
    print("test broadcast add")
    shape = np.random.randint(low=2, high=127, size=(4))
    print (shape)
    a = np.random.randint(low=-127, high=127, size=shape)
    b = np.random.randint(low=-127, high=127, size=shape)
   # c = np.braodcast_add(a,b)

def test_nms():
    batch = np.random.randint(low=1, high=10)
    n = np.random.randint(low=1, high=10)
    k = 6
    print(batch, n, k)
    data = np.random.randint(low=0, high=100, size=(batch, n, k))
    params = {'overlap_thresh':10, 'coord_start':2, 'score_index':1, 'id_index':0, 'force_suppress':True, 'in_format':'corner', 'out_type':'corner'}
    y = nd.contrib.box_nms(data, **params)
    print(y.shape)
#test_concatenate()
#test_repeat()
#test_tile()
#test_transpose()
#test_strided_slice()
#test_slice_like()
#test_take()
#test_max()
#test_sum()
test_upsampling()
#test_elemwise_add()
##test_conv2d()
##test_broadcast_add()
#test_nms()
