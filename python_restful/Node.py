import hashlib
import imghdr
from flask import Flask, request, render_template_string, render_template,url_for,redirect,jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_user import login_required, UserManager, UserMixin
from flask_user.signals import user_registered
import sys
from Crypto.Hash import SHA256
from Block import Blockchain
import os
import json
import time
import random
import threading
import uuid


def get_addr():
    seed = uuid.uuid4()
    return SHA256.new(data=str(seed).encode()).hexdigest()

def cmp(a, b):
        return a["nonce"]-b["nonce"]

class ConfigClass(object):
    """ Flask application config """

    # Flask settings
    SECRET_KEY='9EXzI8LJl6PLhWtuNPum9EXzI8LJl6PLhWtuNPum9EXzI8LJl6PLhWtuNPum9EXzI8LJl6PLhWtuNPum9EXzI8LJl6PLhWtuNPum'

    # Flask-SQLAlchemy settings
    # File-based SQL database
    SQLALCHEMY_DATABASE_URI='mysql://root:123456@127.0.0.1/cortex'
    SQLALCHEMY_TRACK_MODIFICATIONS=False    # Avoids SQLAlchemy warning

    # Flask-User settings
    # Shown in and email templates and page footers
    USER_APP_NAME="Cortex chain sample"
    USER_ENABLE_EMAIL=False      # Disable email authentication
    USER_ENABLE_USERNAME=True    # Enable username authentication
    USER_REQUIRE_RETYPE_PASSWORD=True    # Simplify register form

def create_app():
    """ Flask application factory """

    # Create Flask app load app.config
    app=Flask(__name__,static_url_path='/static')
    app.config.from_object(__name__+'.ConfigClass')

    # Initialize Flask-SQLAlchemy
    db=SQLAlchemy(app)

    # Define the User data-model.
    # NB: Make sure to add flask_user UserMixin !!!
    class User(db.Model, UserMixin):
        __tablename__='miner_sample_users'
        id=db.Column(db.Integer, primary_key = True)
        active=db.Column('is_active', db.Boolean(),
                         nullable = False, server_default = '1')

        # User authentication information. The collation='NOCASE' is required
        # to search case insensitively when USER_IFIND_MODE is 'nocase_collation'.
        username=db.Column(db.String(100), nullable = False, unique = True)
        password=db.Column(db.String(255), nullable = False,
                           server_default = '')
        mineraddr=db.Column(db.String(255), nullable = True)

    
    @user_registered.connect_via(app)
    def _track_registrations(sender, user, **extra):
        user.mineraddr = get_addr()
        db.session.commit()
    # Create all database tables
    db.create_all()

    # Setup Flask-User and specify the User data-model
    user_manager=UserManager(app, db, User)

    # The Home page is accessible to anyone
    @app.route('/')
    def home_page():
        # String-based templates
        return render_template("home.html")

    # The Members page is only accessible to authenticated users via the @login_required decorator
    @app.route('/members')
    @login_required    # User must be authenticated
    def member_page():
        # String-based templates
        return render_template("member.html")

    return app

class Node:
    def __init__(self,end = None):
        self.node=create_app()
        # store unpacked trascations
        self.unpacked_trasactions=[]
        self.db=SQLAlchemy(self.node)
        self.memory_pool=[]
        self.blockchain=Blockchain(end)
        self.lock=threading.Lock()
        self.miner_address="0x0000000000000000000000000000000000000000001"
        @self.node.route('/account', methods = ['POST'])
        def getAccount():
            if request.method == "POST":
                addr = request.args.get("address")
                return jsonify({"status":"ok","account":self.blockchain.CVM.state["account"][addr]})
        @self.node.route('/txion', methods = ['POST'])
        def transaction():
            self.lock.acquire()
            if request.method == 'POST':
                # On each new POST request,
                # we extract the transaction data
                new_txion=request.get_json()
                if not new_txion:
                    new_txion=json.loads(request.form["parma"])
                print new_txion
                assert("nonce" in new_txion.keys())
                assert("from" in new_txion.keys())
                # print new_txion
                # Then we add the transaction to our list
                # Because the transaction was successfully
                # submitted, we log it to our console
                info={}
                contractType=new_txion["type"]
                new_txion["tx_hash"]=SHA256.new(
                    data = (str(long(time.time()*1000))+str(new_txion)+str(random.random())).encode()).hexdigest()

                info["tx_hash"] = new_txion["tx_hash"]
                new_txion["comment"] = ""
                #transfer ctxc
                if contractType == "tx":
                    assert("to" in new_txion.keys())
                    assert("amount" in new_txion.keys())
                #update model 
                elif contractType == "model_data":
                    assert("filename" in new_txion.keys())
                    f=request.files["file"]
                    p=os.path.join("model", new_txion["filename"])
                    model_addr=SHA256.new(
                        data = (str(long(time.time()*1000))).encode()).hexdigest()
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
                #call contract for user-define input_data
                elif contractType == "contract_call":
                    assert("input_address" in new_txion.keys())
                    assert("contract_address" in new_txion.keys())
                #call specific contract Default Contract
                elif contractType == "call":
                    
                    f = request.files["file"]
                    ff = f.read()
                    try:
                        ty = imghdr.what(None,h=ff)
                    except Exception as ex:
                        return jsonify({"msg": "err", "info": "must be image type"})
                    if ty!="png" and ty!="jpg" and ty!="jpeg":
                        return jsonify({"msg": "err", "info": "must be image type"})
                    img_key = hashlib.sha256(ff).hexdigest()
                    new_txion["input"] = img_key+"."+ty
                    p = os.path.join("input_data",new_txion["input"])
                    f.seek(0)
                    f.save(p)
                    assert("model" in new_txion.keys())
                #create new contract
                elif contractType == "contract_create":
                    assert("model_address" in new_txion.keys())
                    assert("param_address" in new_txion.keys())
                    contract_addr=SHA256.new(
                        data = (str(long(time.time()*1000))).encode()).hexdigest()
                    new_txion["contract_addr"]=contract_addr
                    os.symlink("../"+self.blockchain.CVM.state["model_address"]
                               [new_txion["model_address"]], "./model_bind/%s-symbol.json" % contract_addr)
                    os.symlink("../"+self.blockchain.CVM.state["param_address"]
                               [new_txion["param_address"]], "./model_bind/%s-0000.params" % contract_addr)
                    info["contract_addr"]=contract_addr
                else:
                    self.lock.release()
                    return json.dumps({"msg": "error, no such type"})
                    
                self.memory_pool.append(new_txion)
                self.lock.release()
                return jsonify({"msg": "ok", "info": info})
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
    #mine
    def mine(self):
        self.lock.acquire()
        # if len(self.unpacked_trasactions)==0:
            # self.lock.release()
            # return json.dumps({"msg":"error, there is no unpacked transaction"})
        self.unpacked_trasactions=sorted(self.memory_pool[:], cmp = cmp)
        block_hash, err=self.blockchain.pack(
            self.unpacked_trasactions, self.miner_address)
        if not err:
            self.memory_pool=self.memory_pool[len(
                self.unpacked_trasactions):]
            self.lock.release()
            return json.dumps({"msg": "ok"})
        else:
            self.lock.release()
            return json.dumps({"msg": "error, %s" % err})
# mine a new block. because the miner may use much resource, we only use a low difficulty and sleep for about 10 seconds. 
def mine(node):
    while True:
        node.mine()
        time.sleep(10)
if __name__ == "__main__":
    if len(sys.argv)==1:
    	node = Node()
    else:
        node = Node(int(sys.argv[1]))
    mine_thread = threading.Thread(target=mine,args=(node,))
    mine_thread.setDaemon(True)
    mine_thread.start()
    node.node.run(host='0.0.0.0',port=5001)
