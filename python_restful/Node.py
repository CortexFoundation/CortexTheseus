import hashlib
import imghdr
from collections import defaultdict
from flask import Flask, request, render_template_string, render_template,url_for,redirect,jsonify
import sys
from Crypto.Hash import SHA256
import os
import json
import time
import random
import threading
import uuid
import Inference
class Node:
    def __init__(self,end = None):
        self.node=Flask(__name__)
        self.model_lock=defaultdict(lambda:threading.Lock())
        self.input_lock=defaultdict(lambda:threading.Lock())
        self.inferObj = Inference.Inference(json.load(open("config.json","r")))
        @self.node.route('/txion', methods = ['POST'])
        def transaction():
            try:
                if request.method == 'POST':
                    # On each new POST request,
                    # we extract the transaction data

                    new_txion=json.loads(request.files["json"].read())
                    # Then we add the transaction to our list
                    # Because the transaction was successfully
                    # submitted, we log it to our console
                    info={}
                    contractType=new_txion["type"]

                    if contractType == "model_data":
                        f=request.files["params_file"]
                        sha=SHA256.new()
                        sha.update(f.read())

                        f1=request.files["json_file"]
                        sha.update(f1.read())
                        addr = sha.hexdigest()
                        print(addr,flush=True)
                        p=os.path.join("model", addr+"-0000.params")
                        p1=os.path.join("model", addr+"-symbol.json")
                        f.stream.seek(0)
                        f1.stream.seek(0)
                        self.model_lock[addr].acquire()
                        try:
                            f.save(p)
                            f1.save(p1)
                        except Exception as e:
                            self.model_lock[addr].release()
                            return jsonify({"msg":"error","info":str(e)})

                        self.model_lock[addr].release()
                        info["model_addr"]=addr
                    #update param
                    elif contractType == "input_data":
                        f=request.files["file"]
                        addr=SHA256.new(
                            data = f.read()
                            ).hexdigest()
                        print(addr,flush=True)
                        p=os.path.join("input_data", addr)
                        f.stream.seek(0)
                        self.input_lock[addr].acquire()
                        try:
                            f.save(p)
                        except Exception as e:
                            self.input_lock[addr].release()
                            return jsonify({"msg":"error","info":str(e)})
                        self.input_lock[addr].release()
                        info["input_addr"]=addr
                    return jsonify({"msg": "ok", "info": info})
            except Exception as e:
                return jsonify({"msg":"error","info":str(e)})
        @self.node.route('/infer', methods = ['POST'])
        def infer():
            try:
                if request.method == 'POST':
                    new_infer=request.get_json()
                    print(new_infer)
                    if not os.path.exists(os.path.join("model",new_infer["model_addr"]+"-symbol.json")):
                        return jsonify({"msg":"error","info":"json file does not exist"})
                    if not os.path.exists(os.path.join("model",new_infer["model_addr"]+"-0000.params")):
                        return jsonify({"msg":"error","info":"params file does not exist"})
                    if not os.path.exists(os.path.join("input_data",new_infer["input_addr"])):
                        return jsonify({"msg":"error","info":"input file does not exist"})
                    self.input_lock[new_infer["input_addr"]].acquire()
                    self.model_lock[new_infer["model_addr"]].acquire()
                    try:
                        res = self.inferObj.testSingleInference(
                            new_infer["model_addr"],
                            new_infer["input_addr"]
                            )
                        if res["status"] == "error":
                            self.input_lock[new_infer["input_addr"]].release()
                            self.model_lock[new_infer["model_addr"]].release()
                            return jsonify({"msg": "error", "info": res})
                    except Exception as ex:
                        self.input_lock[new_infer["input_addr"]].release()
                        self.model_lock[new_infer["model_addr"]].release()
                        return jsonify({"msg":"error","info":str(ex)})
                    self.input_lock[new_infer["input_addr"]].release()
                    self.model_lock[new_infer["model_addr"]].release()
                    return jsonify({"msg": "ok", "info": str(res["info"])})
            except Exception as ex:
                return jsonify({"msg":"error","info":str(ex)})
if __name__ == "__main__":
    node = Node()
    node.node.run(host='0.0.0.0',port=5000)
