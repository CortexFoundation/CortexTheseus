import hashlib
import imghdr
from flask import Flask, request, render_template_string, render_template,url_for,redirect,jsonify
import sys
from Crypto.Hash import SHA256
import os
import json
import time
import random
import threading
import uuid


class Node:
    def __init__(self,end = None):
        self.node=Flask(__name__)
        self.lock=threading.Lock()
        @self.node.route('/txion', methods = ['POST'])
        def transaction():
            self.lock.acquire()
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
                        f=request.files["file"]
                        p=os.path.join("model", new_txion["filename"])
                        model_addr=SHA256.new(
                            data = f.read()
                            ).hexdigest()
                        print(model_addr,flush=True)
                        
                        
                        f.save(p)
                        new_txion["model_addr"]=model_addr
                        new_txion["model_path"]=p
                        info["model_addr"]=model_addr
                    #update param
                    elif contractType == "param_data":
                        assert("filename" in new_txion.keys())
                        f=request.files["file"]
                        p=os.path.join("param", new_txion["filename"])
                        param_addr=SHA256.new(
                            data = (str(long(time.time()*1000))).encode()).hexdigest()
                        f.save(p)
                        new_txion["param_addr"]=param_addr
                        new_txion["param_path"]=p
                        info["param_addr"]=param_addr
                    #update input_data
                    elif contractType == "input_data":
                        assert("filename" in new_txion.keys())
                        f=request.files["file"]
                        p=os.path.join("input_data", new_txion["filename"])
                        input_addr=SHA256.new(
                            data = (str(long(time.time()*1000))).encode()).hexdigest()
                        f.save(p)
                        new_txion["input_addr"]=input_addr
                        new_txion["input_path"]=p
                        info["input_addr"]=input_addr
                    self.lock.release()
                    return jsonify({"msg": "ok", "info": info})
            except Exception as e:
                self.lock.release()
                return jsonify({"msg":"error","info":str(e)})
            self.lock.release()
        #get state data
        @self.node.route('/getStates', methods = ['GET'])
        def getStates():
            self.lock.acquire()
            s=json.dumps(self.blockchain.CVM.state)
            self.lock.release()
            return s
        #get block data
        @self.node.route('/getBlock', methods = ['GET'])
        def getBlock():
            self.lock.acquire()
            addr = request.args.get("address")
            if addr:
                result = []
                start = -1
                for i in range(len(self.blockchain._chain_name[addr])-1,start,-1):
                    result.append(self.blockchain._chain_name[addr][i].getDict())
                s = jsonify({"block":result,"account":self.blockchain.CVM.state["account"][addr]})
                self.lock.release()
                return s
            else:
                result = [] 
                start = len(self.blockchain._chain)-11
                if start<-1:
                    start = -1
                for i in range(len(self.blockchain._chain)-1,start,-1):
                    result.append(self.blockchain._chain[i].getDict())
                s=jsonify({"block":result,"account":0})
                self.lock.release()
                return s
if __name__ == "__main__":
    node = Node()
    node.node.run(host='0.0.0.0',port=5000)
