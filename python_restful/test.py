import requests
import json
import os
nonce = 0
miner = "0x0000000000000000000000000000000000000000001"
def getState():
    return requests.get("http://127.0.0.1:5000/getStates")
def getBlock():
    return requests.get("http://127.0.0.1:5000/getBlock")
def mine():
    return requests.post("http://127.0.0.1:5000/mine")
def tx(f,to,amount):
    global nonce
    nonce+=1
    return requests.post("http://127.0.0.1:5000/txion",json={"type":"tx","from":f,"to":to,"amount":str(amount),"nonce":nonce})
def infer(input_addr,model_addr):
    return requests.post("http://127.0.0.1:5000/infer",json={"input_addr":input_addr,"model_addr":model_addr}).json()
def uploadInput(f,filepath):
    (tt,tempfilename) = os.path.split(filepath)
    files = {
        'json': ("json", json.dumps({"type":"input_data","author":"0x0553b0185a35cd5bb6386747517ef7e53b15e287","filename":tempfilename}), 'application/json'),
        'file': (tempfilename, open(filepath, 'rb'), 'application/octet-stream')
    }
    return requests.post("http://127.0.0.1:5000/txion",files = files)
def uploadModel(json_filepath,params_file_path):
    (_,json_tempfilename) = os.path.split(json_filepath)
    (_,params_tempfilename) = os.path.split(params_file_path)
    files = {
        'json': ("json", json.dumps({"type":"model_data","author":"0x0553b0185a35cd5bb6386747517ef7e53b15e287"}), 'application/json'),
        'json_file': (json_tempfilename, open(json_filepath, 'rb'), 'application/octet-stream'),
        'params_file': (params_tempfilename, open(params_file_path, 'rb'), 'application/octet-stream')
    }
    return requests.post("http://127.0.0.1:5000/txion",files = files)
def uploadParam(f,filepath):
    (tt,tempfilename) = os.path.split(filepath)
    headers = {'Content-type': 'multipart/form-data'}
    files = {
        'json': ("json", json.dumps({"type":"param_data","filename":tempfilename}), 'application/json'),
        'file': (tempfilename, open(filepath, 'rb'), 'application/octet-stream')
    }
    return requests.post("http://127.0.0.1:5000/txion",files = files)
def createContract(f,model_address,param_address):
    global nonce
    nonce+=1
    return requests.post("http://127.0.0.1:5000/txion",json={"type":"contract_create","from":f,"model_address":model_address,"param_address":param_address,"nonce":nonce})
def callContract(f,input_address,contract_address):
    global nonce
    nonce+=1
    return requests.post("http://127.0.0.1:5000/txion",json={"type":"contract_call","from":f,"input_address":input_address,"contract_address":contract_address,"nonce":nonce})

if __name__ == "__main__":
    # mine()
    # tx(f=miner,to="0x0000000000000000000000000000000000000000001",amount=10)
    # model_info = uploadModel("/Users/cortex/go/src/shiotoli/Torrent/storrent/testfile/", "upload/Inception-BN-0126.params").json()
    # print(json.dumps(model_info),flush=True)
    input_info = uploadInput(miner,"upload/testing_machine.JPG").json()
    print(json.dumps(input_info),flush=True)
    # ii = json.loads(input_info["info"])
    # mi = json.loads(model_info["info"])
    # infer_info = infer(ii["Hash"],mi["Hash"])
    # print(infer_info)
    # model_info = uploadModel("upload/Inception-BN-symbol-2.json", "upload/Inception-BN-0126.params").json()
    # print(json.dumps(model_info),flush=True)
    # input_info = uploadInput(miner,"upload/data").json()
    # print(json.dumps(input_info),flush=True)
    # ii = json.loads(input_info["info"])
    # mi = json.loads(model_info["info"])
    # infer_info = infer(ii["Hash"],mi["Hash"])
    # print(infer_info)

    # mine()
    # contract_info = createContract(miner,model_address=model_info["info"]["model_addr"],param_address=param_info["info"]["param_addr"]).json()
    # mine()
    # print contract_info.json()
    # print getState().json()
    # callContract(miner,input_address=input_info["info"]["input_addr"],contract_address=contract_info["info"]["contract_addr"])
    # mine()
    # print getState().json()["result"][miner]
# print r.text
# r = requests.get("http://127.0.0.1:5000/getBlock")
# print r.text
# r = requests.post("http://127.0.0.1:5000/mine")
# r = requests.post("http://127.0.0.1:5000/txion",json={"type":"tx","from":"0x0000000000000000000000000000000000000000001","to":"0x0000000000000000000000000000000000000000002","amount":"100","nonce":0})
# r = requests.post("http://127.0.0.1:5000/mine")
# print r.text
