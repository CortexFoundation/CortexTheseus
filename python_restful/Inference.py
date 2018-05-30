import cv2
import mxnet as mx
import numpy as np
import datetime
import traceback
# define a simple data batch
from collections import namedtuple
import json
import time
import os
import random
#parse json file
def parse_js(expr):
    import ast
    m = ast.parse(expr)
    a = m.body[0]

    def parse(node):
        if isinstance(node, ast.Expr):
            return parse(node.value)
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Dict):
            return dict(zip(map(parse, node.keys), map(parse, node.values)))
        elif isinstance(node, ast.List):
            return map(parse, node.elts)
        else:
            raise NotImplementedError(node.__class__)

    return parse(a)
class Inference:
    def __init__(self,config):
        #map define each class name
        #self.map = open("map_clsloc.txt",'r').read()
        #self.map = self.map.split('\n')
        #for i in range(len(self.map)-1):
        #    self.map[i] = self.map[i].split(' ')[2]
        #self.map = parse_js(open("gistfile1.txt",'r').read())
        tt = open("map.txt",'r').readlines()
        self.map = []
        for i in tt:
            a = i.split(' ')
            self.map.append(a[1].strip())
        self.batchsize = config["batchsize"]
        self.model_names = config["models"]
        self.img_dir = config["img_dir"]
        self.result_dir = config["result_dir"]
        self.model_dir = config["model_dir"]
        self.models = config["models"]
        self.model_list = {}

    def getImage(self,name):
        #fetch image
        img = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
        if img is None:
             return None
        #standard input is batchsize x 3 x 224 x 224
        img = cv2.resize(img, (224, 224))
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        return img
    #load model
    def loadModel(self,name,type="cpu",gpu_id = 0):
        begin = time.time()
        sym, arg_params, aux_params = mx.model.load_checkpoint(name, 0)
        mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None) if type=="cpu" else mx.mod.Module(symbol=sym, context=mx.gpu(gpu_id), label_names=None)
        mod.bind(for_training=False, data_shapes=[('data', (self.batchsize, 3, 224, 224))])
        mod.set_params(arg_params, aux_params, allow_missing=True)
        end = time.time()
        return mod,end-begin
    #predict
    def predict(self,names,mod):
        imgs = []
        for name in names:
            img = self.getImage(name)
            imgs.append(img)
        Batch = namedtuple('Batch', ['data'])
        begin = time.time()
        mod.forward(Batch([mx.nd.array(imgs)]))
        prob = mod.get_outputs()[0].asnumpy()
        end = time.time()
        return end-begin
    #predict single image
    def predictSingle(self,name,mod):
        imgs = []
        img = self.getImage(name)
        imgs.append(img)
        Batch = namedtuple('Batch', ['data'])
        begin = time.time()
        mod.forward(Batch([mx.nd.array(imgs)]))
        prob = mod.get_outputs()[0].asnumpy()
        end = time.time()
        return self.map[np.argmax(prob)], end-begin

    #get file size    
    def getFileSize(self,filePath):
        # filePath = unicode(filePath,'utf8')
        fsize = os.path.getsize(filePath)
        fsize = fsize/float(1024*1024)
        return round(fsize,2)
    #remove invalid data
    def rmInvalidData(self,rm_img = False):
        if rm_img:
            predict_files = []
            for filename in os.listdir(self.img_dir):
                if filename.endswith("jpg") or filename.endswith("jpeg") or filename.endswith(".png") or filename.endswith(".gif"):
                    predict_files.append(filename)
            for name in predict_files:
                try:
                    img = self.getImage(name)
                except Exception,ex:
                    os.remove(os.path.join(self.img_dir,name))
        self.predict_files = []
        for filename in os.listdir(self.img_dir):
            if filename.endswith("jpg") or filename.endswith("jpeg") or filename.endswith(".png") or filename.endswith(".gif"):
                self.predict_files.append(filename)
    #load model for cortex test
    def loadModelTest(self):
        self.cpu_model_list = []
        self.gpu_model_list = []
        self.cpu_model_time = []
        self.gpu_model_time = []
        for model_name in self.model_names:
            mod,load_model_time = self.loadModel(os.path.join(self.model_dir, model_name),type="cpu")
            self.cpu_model_list.append(mod)
            self.cpu_model_time.append(
                {
                    "model_name":model_name,
                    "model_size":self.getFileSize(os.path.join(self.model_dir,model_name+"-0000.params")),
                    "model_load_time_cpu":load_model_time
                }
            )
            mod,load_model_time = self.loadModel(os.path.join(self.model_dir, model_name),type="gpu")
            self.gpu_model_list.append(mod)
            self.gpu_model_time.append(
                {
                    "model_name":model_name,
                    "model_size":self.getFileSize(os.path.join(self.model_dir,model_name+"-0000.params")),
                    "model_load_time_gpu":load_model_time
                }
            )
        json.dump(self.cpu_model_time,open(os.path.join(self.result_dir,"gpu_model_time.json"),'w'),indent=2)
        json.dump(self.gpu_model_time,open(os.path.join(self.result_dir,"gpu_model_time.json"),'w'),indent=2)
    #test inference performance
    def testInference(self,times = 10000):
        self.cpu_infer_time = []
        self.gpu_infer_time = []
        for i in range(times):
            sampled = random.sample(self.predict_files,self.batchsize)
            for j in range(len(self.models)):
                t = self.predict(sampled,self.cpu_model_list[j])
                self.cpu_infer_time.append(
                    {
                        "model_name":self.models[j],
                        "data_set":i,
                        "batchsize":self.batchsize,
                        "cpu_time":t
                    }
                )
                print i,self.cpu_model_time[j]["model_name"],t
            for j in range(len(self.models)):
                t = self.predict(sampled,self.gpu_model_list[j])
                self.gpu_infer_time.append(
                    {
                        "model_name":self.models[j],
                        "data_set":i,
                        "batchsize":self.batchsize,
                        "gpu_time":t
                    }
                )
                print i,self.gpu_model_time[j]["model_name"],t
        json.dump(self.cpu_infer_time,open(os.path.join(self.result_dir,"cpu_infer_time_%d.json"%self.batchsize),'w'),indent=2)
        json.dump(self.gpu_infer_time,open(os.path.join(self.result_dir,"gpu_infer_time_%d.json"%self.batchsize),'w'),indent=2)
    #test single inference
    def testSingleInference(self,model,data,t="cpu"):
        try:
            if model not in self.model_list.keys():
                mod,load_model_time = self.loadModel(os.path.join(model),type=t)
                self.model_list[model] = mod
        except Exception as ex:
                return "error"+str(ex)
        return str(self.predictSingle(data,self.model_list[model])[0])
            
