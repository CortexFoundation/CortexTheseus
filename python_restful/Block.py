import hashlib as hasher
from Crypto.Hash import SHA256
import datetime as date
from CVM import CVM
import time
import json
import threading
import yaml
from collections import defaultdict
#define block
class Block:
    def __init__(self, index, previous_hash, timestamp, data, nonce):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.nonce = nonce
        self.hash = self.hash_block()
    #get block hash
    def hash_block(self):
        sha = SHA256.new((str(self.index) +
        str(self.timestamp) +
        str(self.data) +
        str(self.previous_hash) +
        str(self.nonce)).encode())
        return sha.hexdigest()

    #transfer block to dict
    def getDict(self):
        return {
            "index":self.index,
            "previous_hash":self.previous_hash,
            "timestamp":self.timestamp,
            "data":self.data,
            "nonce":self.nonce,
            "hash":self.hash
        }
#Blockchain
class Blockchain(object):
    def __init__(self,end=None):
        if not end:                 
            self._chain = [self.create_genesis_block()]
            self._chain_name = defaultdict(lambda:[])   #_chain_name save the blocks involved by each address
            self.CVM = CVM()
            json.dump(self._chain[-1].getDict(),open("./blockchain/%d.json"%(len(self._chain)-1),"w"),indent=2)
        else:       #if end!=None means load chain data from ./blockchain/0-end.json
            self._chain = []
            self._chain_name = defaultdict(lambda:[])
            for i in range(0,end+1):
                tmp_res = yaml.safe_load(open("./blockchain/"+str(i)+".json","r"))
                b = Block(tmp_res["index"],tmp_res["previous_hash"],tmp_res["timestamp"],tmp_res["data"],tmp_res["nonce"])
                b.hash = tmp_res["hash"]
                self._chain.append(b)
                visit = defaultdict(lambda:False)
                #add each involved block to self._chain_name[address]
                for t in b.data:
                    if "from" in t.keys() and not visit[t["from"]]:
                        self._chain_name[t["from"]].append(b)
                        visit[t["from"]] = True
                    if "to" in t.keys() and not visit[t["to"]]:
                        self._chain_name[t["to"]].append(b)
                        visit[t["to"]] = True
            self.CVM = CVM("./blockchain/"+str(end)+"_state.json")
             
        self.mutex = threading.Lock()
    # ...blockchain
    def create_genesis_block(self):
        # Manually construct a block with
        # index zero and arbitrary previous hash
        return Block(
                index =0,
                timestamp = int(time.time()*1000),
                data = [],
                previous_hash = "0",
                nonce = 0)

    # add new block
    def add_block(self,data,state):
        self.mutex.acquire()
        last_block = self._chain[-1]
        new_index = self._chain[-1].index+1
        new_timestamp = int(time.time()*1000)
        previous_hash = last_block.hash
        header = str(new_index) + str(new_timestamp) + str(data) + str(previous_hash)
        hash_result,nonce = self.proof_of_work(header)
        b = Block(
                index = new_index,
                timestamp = new_timestamp,
                data = data,
                previous_hash = previous_hash,
                nonce = nonce)
        self._chain.append(b)
        # update _chain_name
        visit = defaultdict(lambda:False)
        for t in data:
            if "from" in t.keys() and not visit[t["from"]]:
                self._chain_name[t["from"]].append(b)
                visit[t["from"]] = True
            if "to" in t.keys() and not visit[t["to"]]:
                self._chain_name[t["to"]].append(b)
                visit[t["to"]] = True
        json.dump(self._chain[-1].getDict(),open("./blockchain/%d.json"%(len(self._chain)-1),"w"),indent=2)
        json.dump(state,open("./blockchain/%d_state.json"%(len(self._chain)-1),"w"),indent=2)
        print("dump %d\n"%(len(self._chain)-1))
        self.mutex.release()
        return self._chain[-1]
    
    #verify and pack the transactions into a block
    def pack(self,unpaced_transactions, miner_address):
        temp_tx = [i for i in unpaced_transactions]
        temp_tx.append({"type":"coinbase_tx","miner_addr":miner_address,"reward":5})
        new_state, err = self.CVM.verify(temp_tx)
        if err:
            return None,err
        return self.add_block(temp_tx,new_state),None
        
    #miner
    def proof_of_work(self,header):
        target = 2**250
        nonce = 0
        while True:
            hash_result = SHA256.new(data=(str(header)+str(nonce)).encode()).hexdigest()
            if int(hash_result,16)<target:
                return hash_result, nonce
            nonce+=1
