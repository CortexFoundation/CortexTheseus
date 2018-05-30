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

class Node:
    def __init__(self,end = None):
        self.node=Flask(__name__)
        self.model_lock=defaultdict(lambda:threading.Lock())
        self.input_lock=defaultdict(lambda:threading.Lock())
        @self.node.route('/txion', methods = ['POST'])
        def transaction():
            addr = ""
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
                        f=request.files["json_file"]
                        sha=SHA256.new()
                        sha.update(f.read())
                        
                        f1=request.files["params_file"]
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
if __name__ == "__main__":
    node = Node()
    node.node.run(host='0.0.0.0',port=5000)
