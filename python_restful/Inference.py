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
        self.batchsize = config["batchsize"]
        self.model_names = config["models"]
        self.img_dir = config["img_dir"]
        self.result_dir = config["result_dir"]
        self.model_dir = config["model_dir"]
        self.models = config["models"]
        self.model_list = {}
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
        img = np.load(os.path.join(self.img_dir,name))
        print(img)
        imgs.append(img)
        Batch = namedtuple('Batch', ['data'])
        begin = time.time()
        print(mod)
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
                except Exception as ex:
                    os.remove(os.path.join(self.img_dir,name))
        self.predict_files = []
        for filename in os.listdir(self.img_dir):
            if filename.endswith("jpg") or filename.endswith("jpeg") or filename.endswith(".png") or filename.endswith(".gif"):
                self.predict_files.append(filename)
    #test single inference
    def testSingleInference(self,model,data,t="cpu"):
        try:
            if model not in self.model_list.keys():
                mod = self.loadModel(os.path.join(self.model_dir,model),type=t)
                self.model_list[model] = mod[0]
        except Exception as ex:
            return {"status":"error","info":str(ex)}
        return {"status":"ok","info":self.predictSingle(data,self.model_list[model])[0]}
if __name__ == "__main__":
    inferObj = Inference(json.load(open("config.json","r")))
    res = inferObj.testSingleInference("5c4d1f84063be8e25e83da6452b1821926548b3c2a2a903a0724e14d5c917b00","e23438003ce13f86ff46e41441d568dfc6b8a7279a52a2f0b01c3aa5ed85c2c6")
    print(res)