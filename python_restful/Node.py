import hashlib
from collections import defaultdict
from flask import Flask, request, render_template_string, render_template, url_for, redirect, jsonify
import sys
from Crypto.Hash import SHA256
import numpy as np
import os
import json
import time
import random
import threading
import Inference
from gevent.pywsgi import WSGIServer
import convertimage2numpy as cn
from datetime import datetime


class Node:
    def __init__(self, end=None):
        self.model_lock = defaultdict(lambda: threading.Lock())
        self.input_lock = defaultdict(lambda: threading.Lock())
        self.inferObj = Inference.InferHelper(
            json.load(open("config.json", "r")))
        self.input_list_lock = threading.Lock()
        self.input_list = {}
        self.model_list_lock = threading.Lock()
        self.model_list = {}

    def transaction(self, request):
        try:
            if request.method == 'POST':
                # On each new POST request,
                # we extract the transaction data

                new_txion = json.loads(request.files["json"].read())
                # Then we add the transaction to our list
                # Because the transaction was successfully
                # submitted, we log it to our console
                info = {}
                contractType = new_txion["type"]

                print(contractType, flush=True)
                if contractType == "model_data":
                    f = request.files["params_file"]
                    sha = SHA256.new()
                    fr = f.read()
                    sha.update(fr)

                    f1 = request.files["json_file"]
                    fr1 = f1.read()
                    sha.update(fr1)
                    addr = "0x" + sha.hexdigest()
                    info = {
                        "Hash": addr,
                        "AuthorAddress": new_txion["author"],
                        "RawSize": (len(fr1) + len(fr)),
                        "InputShape": [3, 224, 224],
                        "OutputShape": [1],
                        "Gas": (len(fr1) + len(fr)) / 100
                    }
                    p = os.path.join("model", addr + "-0000.params")
                    p1 = os.path.join("model", addr + "-symbol.json")
                    f.stream.seek(0)
                    f1.stream.seek(0)
                    self.model_lock[addr].acquire()
                    try:
                        f.save(p)
                        f1.save(p1)
                    except Exception as e:
                        return jsonify({"msg": "error", "info": str(e)})
                    finally:
                        self.model_lock[addr].release()
                    # info["model_addr"]=addr
                    self.model_list_lock.acquire()
                    self.model_list[addr] = info
                    self.model_list_lock.release()
                #update param
                elif contractType == "list_input":
                    self.input_list_lock.acquire()
                    resp = jsonify({"msg": "ok", "info": self.input_list})
                    self.input_list_lock.release()
                    return resp
                elif contractType == "list_model":
                    self.model_list_lock.acquire()
                    resp = jsonify({"msg": "ok", "info": self.model_list})
                    self.model_list_lock.release()
                    return resp
                elif contractType == "input_data":
                    f = request.files["file"]
                    fr = f.read()
                    fr = cn.getImageFromFile(fr)
                    addr = "0x" + SHA256.new(data=fr.tobytes()).hexdigest()
                    info = {
                        "Hash": addr,
                        "AuthorAddress": new_txion["author"],
                        "RawSize": len(fr),
                        "Shape": [3, 224, 224]
                    }
                    print(addr, flush=True)
                    p = os.path.join("input_data", addr)
                    f.stream.seek(0)
                    self.input_lock[addr].acquire()
                    try:
                        np.save(open(p, "wb"), fr)
                    except Exception as e:
                        return jsonify({"msg": "error", "info": str(e)})
                    finally:
                        self.input_lock[addr].release()
                    self.input_list_lock.acquire()
                    self.input_list[addr] = info
                    self.input_list_lock.release()
                    # info["input_addr"]=addr
                return jsonify({"msg": "ok", "info": info})
        except Exception as e:
            return jsonify({"msg": "error", "info": str(e)})

    def infer(self, request):
        try:
            if request.method == 'POST':
                new_infer = request.get_json()
                print(datetime.now(), new_infer, file=sys.stderr)
                self.model_lock[new_infer["model_addr"]].acquire()
                model_addr = new_infer["model_addr"]
                if not os.path.exists(
                        os.path.join(
                            "model",
                            new_infer["model_addr"] + "-symbol.json")):
                    return jsonify({
                        "msg": "error",
                        "info": "json file does not exist"
                    })
                if not os.path.exists(
                        os.path.join(
                            "model",
                            new_infer["model_addr"] + "-0000.params")):
                    return jsonify({
                        "msg": "error",
                        "info": "params file does not exist"
                    })
                self.model_lock[new_infer["model_addr"]].release()

                self.input_lock[new_infer["input_addr"]].acquire()
                input_addr = new_infer["input_addr"]
                if not os.path.exists(
                        os.path.join("input_data", new_infer["input_addr"])):
                    return jsonify({
                        "msg": "error",
                        "info": "input file does not exist"
                    })
                self.input_lock[new_infer["input_addr"]].release()

                try:
                    res = self.inferObj.makeInfer(model_addr, input_addr)
                    if res["status"] == "error":
                        return jsonify({"msg": "error", "info": res})
                except Exception as ex:
                    print(datetime.now(), str(ex))
                    return jsonify({"msg": "error", "info": str(ex)})
                return jsonify({"msg": "ok", "info": str(res["info"])})
        except Exception as ex:
            return jsonify({"msg": "error", "info": str(ex)})


if __name__ == "__main__":
    node = Node()
    server = Flask(__name__)

    @server.route('/txion', methods=['POST'])
    def txion():
        return node.transaction(request)

    @server.route('/infer', methods=['POST'])
    def infer():
        return node.infer(request)

    server.run(host='0.0.0.0', port=5000, threaded=True)
