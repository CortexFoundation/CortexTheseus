import mxnet as mx
import numpy as np
import datetime
import traceback
import sys
# define a simple data batch
from collections import namedtuple
import json
import time
import os
import random
import threading
from multiprocessing import Pool, Pipe, Process

class Inference:
    def __init__(self,config):
        self.batchsize = config["batchsize"]
        self.model_names = config["models"]
        self.img_dir = config["img_dir"]
        self.result_dir = config["result_dir"]
        self.model_dir = config["model_dir"]
        self.models = config["models"]
        self.model_list = {}
        self.lock = threading.Lock()
        print ('Inference init')

    #load model
    def loadModel(self,name,type="cpu",gpu_id = 0):
        begin = time.time()
        sym, arg_params, aux_params = mx.model.load_checkpoint(name, 0)
        mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None) if type=="cpu" else mx.mod.Module(symbol=sym, context=mx.gpu(gpu_id), label_names=None)
        mod.bind(for_training=False, data_shapes=[('data', (self.batchsize, 3, 224, 224))])
        mod.set_params(arg_params, aux_params, allow_missing=True)
        end = time.time()
        return mod,end-begin

    #predict single image
    def predictSingle(self,name,mod):
        imgs = []
        img = np.load(os.path.join(self.img_dir,name))
        imgs.append(img)
        Batch = namedtuple('Batch', ['data'])
        begin = time.time()
        mod.forward(Batch([mx.nd.array(imgs)]))
        prob = mod.get_outputs()[0].asnumpy()
        end = time.time()
        return np.argmax(prob), end-begin

    #get file size
    def getFileSize(self,filePath):
        # filePath = unicode(filePath,'utf8')
        fsize = os.path.getsize(filePath)
        fsize = fsize/float(1024*1024)
        return round(fsize,2)

    #test single inference
    def testSingleInference(self,model,data,t="cpu"):
        self.lock.acquire()
        try:
            if model not in self.model_list.keys():
                mod = self.loadModel(os.path.join(self.model_dir,model),type=t)
                self.model_list[model] = mod[0]
            return {"status":"ok","info":self.predictSingle(data,self.model_list[model])[0]}
        except Exception as ex:
            return {"status":"error","info":str(ex)}
        finally:
            self.lock.release()

class InferHelper(object):
    def __init__(self, config, n_jobs = 10):
        self.workers = []
        self.msg_queues = []
        for i in range(n_jobs):
            parent_conn, child_conn = Pipe()
            self.msg_queues.append(parent_conn)
            self.workers.append(Process(target=self.__handler, args=(config, child_conn, i)))
        for p in self.workers:
            p.start()
        self.q_lock = threading.Lock()
        self.has_model = [set() for _ in range(n_jobs)]
        self.idle = [True] * n_jobs

    def getQ(self, model):
        worker = None
        while worker is None:
            self.q_lock.acquire()
            idx = None
            for i in range(len(self.msg_queues)):
                if self.idle[i] and (model in self.has_model[i]):
                    idx = i
                    break
            if idx is None:
                for i in range(len(self.msg_queues)):
                    if self.idle[i]:
                        idx = i
                        break
            if idx is not None:
                worker = self.msg_queues[idx]
                self.idle[idx] = False
                self.has_model[idx].add(model)
                self.q_lock.release()
                break
            self.q_lock.release()
        return worker

    def putQ(self, q):
        if q is None:
            return
        self.q_lock.acquire()
        for i in range(len(self.msg_queues)):
            if q == self.msg_queues[i]:
                self.idle[i] = True
                break
        self.q_lock.release()

    def makeInfer(self, model, data, t='cpu'):
        worker = None
        worker = self.getQ(model)
        worker.send((model, data, t))
        res = worker.recv()
        self.putQ(worker)
        return res

    def __handler(self, config, q, idx):
        inferObj = Inference(config)
        while True:
            model, data, t = q.recv()
            print ('Process %d: '% idx, model, data, t, file=sys.stderr)
            res = inferObj.testSingleInference(model, data, t)
            q.send(res)

    def __deallocate__(self):
        for p in self.workers:
            p.join()

if __name__ == "__main__":
    inferObj = Inference(json.load(open("config.json","r")))
    res = inferObj.testSingleInference(
            "5c4d1f84063be8e25e83da6452b1821926548b3c2a2a903a0724e14d5c917b00",
            "e23438003ce13f86ff46e41441d568dfc6b8a7279a52a2f0b01c3aa5ed85c2c6")
    print(res)
