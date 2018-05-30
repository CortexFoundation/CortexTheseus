import json
from collections import defaultdict
import Inference
import DefaultContract
class CVM:
    def __init__(self,fname=None):
        #CVM state
        self.state = {
            "account":defaultdict(lambda: 10000),        #10000 CTXC for initial account
            "model_address":defaultdict(lambda:None),    #model address
            "param_address":defaultdict(lambda:None),    #param address 
            "input_address":defaultdict(lambda:None),    #input address
            "contract_address":defaultdict(lambda:None), #address where contract save
            "result":defaultdict(lambda:None),           #inference result
            "last_nonce":defaultdict(lambda:-1)          #last updated nonce
        }
        #load state from file
        if fname:
            temp_state = json.load(open(fname,"r"))
            for k in temp_state.keys():
                for k1,v1 in temp_state[k].items():
                    self.state[k][k1] = v1
        #inference config
        self.inference_config = json.load(open("config.json","r"))
        #inference object
        self.infer_obj = Inference.Inference(self.inference_config)
    #verify the transactions
    #Notice: this is a simple sample of Cortex Virtual Machine, we do not translate each instruction in the smart contract.
    def verify(self,unpacked_transactions):
        self.new_state = self.state.copy()
        account = self.new_state["account"]
        model_address = self.new_state["model_address"]
        param_address = self.new_state["param_address"]
        contract_address = self.new_state["contract_address"]
        input_address = self.new_state["input_address"]
        result = self.new_state["result"]
        last_nonce = self.new_state["last_nonce"]
        for t in unpacked_transactions:
            if t["type"] == "tx":
                #"transaction %s is invalid, not enough cortex"%t["tx_hash"]
                if account[t["from"]]<int(t["amount"]):
                    continue
		        #"transaction %s is invalid, amount need to be not negative"%t["tx_hash"]
                if int(t["amount"])<0:
                    continue
                #"transaction %s is invalid, nonce should be ascending "%t["tx_hash"]
                if int(t["nonce"])<=last_nonce[t["from"]]:
                    t["nonce"] = last_nonce[t["from"]]+1
                account[t["from"]]-=int(t["amount"])
                account[t["to"]]+=int(t["amount"])
            #handle different kinds of instructions
            if t["type"] == "model_data":
                model_address[t["model_addr"]] = t["model_path"]
            if t["type"] == "param_data":
                param_address[t["param_addr"]] = t["param_path"]
            if t["type"] == "input_data":
                input_address[t["input_addr"]] = t["input_path"]
            if t["type"] == "coinbase_tx":
                account[t["miner_addr"]]+=5
            if t["type"] == "contract_create":
                contract_address[t["contract_addr"]] = {
                    "model_address":t["model_address"],
                    "param_address":t["param_address"]
                }
            #this instrucion is for user-define input and model
            if t["type"] == "contract_call":
                result[t["from"]] = self.inference("./model_bind/"+t["contract_address"],
                input_address[t["input_address"]])
                
            #this instruction is for specific contract "Default Contract"
            if t["type"] == "call":
                DefaultContract.Run(account,t,self)
        return self.new_state,None
        
    #inference instruction
    def inference(self,model_path,data_path):
        return self.infer_obj.testSingleInference(model_path,data_path)
