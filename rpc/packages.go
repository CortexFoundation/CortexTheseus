/*
	This file is part of go-ethereum

	go-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	go-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with go-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/*

For each request type, define the following:

1. RpcRequest "To" method [message.go], which does basic validation and conversion to "Args" type via json.Decoder()
2. json.Decoder() calls "UnmarshalON" defined on each "Args" struct
3. EthereumApi method, taking the "Args" type and replying with an interface to be marshalled to ON

*/
package rpc

import (
	"fmt"
	"math/big"
	"strings"
	"sync"

	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/ethdb"
	"github.com/ethereum/go-ethereum/ethutil"
	"github.com/ethereum/go-ethereum/event/filter"
	"github.com/ethereum/go-ethereum/state"
	"github.com/ethereum/go-ethereum/xeth"
)

const (
	defaultGasPrice = "10000000000000"
	defaultGas      = "10000"
)

type EthereumApi struct {
	xeth          *xeth.XEth
	filterManager *filter.FilterManager

	logMut sync.RWMutex
	logs   map[int]state.Logs

	messagesMut sync.RWMutex
	messages    map[int][]xeth.WhisperMessage

	db ethutil.Database
}

func NewEthereumApi(eth *xeth.XEth) *EthereumApi {
	db, _ := ethdb.NewLDBDatabase("dapps")
	api := &EthereumApi{
		xeth:          eth,
		filterManager: filter.NewFilterManager(eth.Backend().EventMux()),
		logs:          make(map[int]state.Logs),
		messages:      make(map[int][]xeth.WhisperMessage),
		db:            db,
	}
	go api.filterManager.Start()

	return api
}

func (self *EthereumApi) NewFilter(args *FilterOptions, reply *interface{}) error {
	var id int
	filter := core.NewFilter(self.xeth.Backend())
	filter.SetOptions(toFilterOptions(args))
	filter.LogsCallback = func(logs state.Logs) {
		self.logMut.Lock()
		defer self.logMut.Unlock()

		self.logs[id] = append(self.logs[id], logs...)
	}
	id = self.filterManager.InstallFilter(filter)
	*reply = id

	return nil
}

func (self *EthereumApi) NewFilterString(args string, reply *interface{}) error {
	var id int
	filter := core.NewFilter(self.xeth.Backend())

	callback := func(block *types.Block) {
		self.logs[id] = append(self.logs[id], &state.StateLog{})
	}
	if args == "pending" {
		filter.PendingCallback = callback
	} else if args == "chain" {
		filter.BlockCallback = callback
	}

	id = self.filterManager.InstallFilter(filter)
	*reply = id

	return nil
}

func (self *EthereumApi) FilterChanged(id int, reply *interface{}) error {
	self.logMut.RLock()
	defer self.logMut.RUnlock()

	*reply = toLogs(self.logs[id])

	self.logs[id] = nil // empty the logs

	return nil
}

func (self *EthereumApi) Logs(id int, reply *interface{}) error {
	filter := self.filterManager.GetFilter(id)
	*reply = toLogs(filter.Find())

	return nil
}

func (p *EthereumApi) GetBlock(args *GetBlockArgs, reply *interface{}) error {
	err := args.requirements()
	if err != nil {
		return err
	}

	if args.BlockNumber > 0 {
		*reply = p.xeth.BlockByNumber(args.BlockNumber)
	} else {
		*reply = p.xeth.BlockByHash(args.Hash)
	}
	return nil
}

func (p *EthereumApi) Transact(args *NewTxArgs, reply *interface{}) error {
	if len(args.Gas) == 0 {
		args.Gas = defaultGas
	}

	if len(args.GasPrice) == 0 {
		args.GasPrice = defaultGasPrice
	}

	result, _ := p.xeth.Transact( /* TODO specify account */ args.To, args.Value, args.Gas, args.GasPrice, args.Data)
	*reply = result
	return nil
}

func (p *EthereumApi) Call(args *NewTxArgs, reply *interface{}) error {
	result, err := p.xeth.Call( /* TODO specify account */ args.To, args.Value, args.Gas, args.GasPrice, args.Data)
	if err != nil {
		return err
	}

	*reply = result
	return nil
}

func (p *EthereumApi) PushTx(args *PushTxArgs, reply *interface{}) error {
	err := args.requirementsPushTx()
	if err != nil {
		return err
	}
	result, _ := p.xeth.PushTx(args.Tx)
	*reply = result
	return nil
}

func (p *EthereumApi) GetStateAt(args *GetStateArgs, reply *interface{}) error {
	err := args.requirements()
	if err != nil {
		return err
	}

	state := p.xeth.State().SafeGet(args.Address)

	value := state.StorageString(args.Key)
	var hx string
	if strings.Index(args.Key, "0x") == 0 {
		hx = string([]byte(args.Key)[2:])
	} else {
		// Convert the incoming string (which is a bigint) into hex
		i, _ := new(big.Int).SetString(args.Key, 10)
		hx = ethutil.Bytes2Hex(i.Bytes())
	}
	rpclogger.Debugf("GetStateAt(%s, %s)\n", args.Address, hx)
	*reply = map[string]string{args.Key: value.Str()}
	return nil
}

func (p *EthereumApi) GetStorageAt(args *GetStorageArgs, reply *interface{}) error {
	err := args.requirements()
	if err != nil {
		return err
	}

	*reply = p.xeth.State().SafeGet(args.Address).Storage()
	return nil
}

func (p *EthereumApi) GetPeerCount(reply *interface{}) error {
	*reply = p.xeth.PeerCount()
	return nil
}

func (p *EthereumApi) GetIsListening(reply *interface{}) error {
	*reply = p.xeth.IsListening()
	return nil
}

func (p *EthereumApi) GetCoinbase(reply *interface{}) error {
	*reply = p.xeth.Coinbase()
	return nil
}

func (p *EthereumApi) Accounts(reply *interface{}) error {
	*reply = p.xeth.Accounts()
	return nil
}

func (p *EthereumApi) GetIsMining(reply *interface{}) error {
	*reply = p.xeth.IsMining()
	return nil
}

func (p *EthereumApi) BlockNumber(reply *interface{}) error {
	*reply = p.xeth.Backend().ChainManager().CurrentBlock().Number()
	return nil
}

func (p *EthereumApi) GetTxCountAt(args *GetTxCountArgs, reply *interface{}) error {
	err := args.requirements()
	if err != nil {
		return err
	}
	*reply = p.xeth.TxCountAt(args.Address)
	return nil
}

func (p *EthereumApi) GetBalanceAt(args *GetBalanceArgs, reply *interface{}) error {
	err := args.requirements()
	if err != nil {
		return err
	}
	state := p.xeth.State().SafeGet(args.Address)
	*reply = toHex(state.Balance().Bytes())
	return nil
}

func (p *EthereumApi) GetCodeAt(args *GetCodeAtArgs, reply *interface{}) error {
	err := args.requirements()
	if err != nil {
		return err
	}
	*reply = p.xeth.CodeAt(args.Address)
	return nil
}

func (p *EthereumApi) Sha3(args *Sha3Args, reply *interface{}) error {
	*reply = toHex(crypto.Sha3(fromHex(args.Data)))
	return nil
}

func (p *EthereumApi) DbPut(args *DbArgs, reply *interface{}) error {
	err := args.requirements()
	if err != nil {
		return err
	}

	p.db.Put([]byte(args.Database+args.Key), []byte(args.Value))
	*reply = true
	return nil
}

func (p *EthereumApi) DbGet(args *DbArgs, reply *interface{}) error {
	err := args.requirements()
	if err != nil {
		return err
	}

	res, _ := p.db.Get([]byte(args.Database + args.Key))
	*reply = string(res)
	return nil
}

func (p *EthereumApi) NewWhisperIdentity(reply *interface{}) error {
	*reply = p.xeth.Whisper().NewIdentity()
	return nil
}

func (p *EthereumApi) NewWhisperFilter(args *xeth.Options, reply *interface{}) error {
	var id int
	args.Fn = func(msg xeth.WhisperMessage) {
		p.messagesMut.Lock()
		defer p.messagesMut.Unlock()
		p.messages[id] = append(p.messages[id], msg)
	}
	id = p.xeth.Whisper().Watch(args)
	*reply = id
	return nil
}

func (self *EthereumApi) MessagesChanged(id int, reply *interface{}) error {
	self.messagesMut.RLock()
	defer self.messagesMut.RUnlock()

	*reply = self.messages[id]

	self.messages[id] = nil // empty the messages

	return nil
}

func (p *EthereumApi) WhisperPost(args *WhisperMessageArgs, reply *interface{}) error {
	err := p.xeth.Whisper().Post(args.Payload, args.To, args.From, args.Topics, args.Priority, args.Ttl)
	if err != nil {
		return err
	}

	*reply = true
	return nil
}

func (p *EthereumApi) HasWhisperIdentity(args string, reply *interface{}) error {
	*reply = p.xeth.Whisper().HasIdentity(args)
	return nil
}

func (p *EthereumApi) WhisperMessages(id int, reply *interface{}) error {
	*reply = p.xeth.Whisper().Messages(id)
	return nil
}

func (p *EthereumApi) GetRequestReply(req *RpcRequest, reply *interface{}) error {
	// Spec at https://github.com/ethereum/wiki/wiki/Generic-ON-RPC
	rpclogger.DebugDetailf("%T %s", req.Params, req.Params)
	switch req.Method {
	case "eth_coinbase":
		return p.GetCoinbase(reply)
	case "eth_listening":
		return p.GetIsListening(reply)
	case "eth_mining":
		return p.GetIsMining(reply)
	case "eth_peerCount":
		return p.GetPeerCount(reply)
	case "eth_number":
		return p.BlockNumber(reply)
	case "eth_accounts":
		return p.Accounts(reply)
	case "eth_countAt":
		args, err := req.ToGetTxCountArgs()
		if err != nil {
			return err
		}
		return p.GetTxCountAt(args, reply)
	case "eth_codeAt":
		args, err := req.ToGetCodeAtArgs()
		if err != nil {
			return err
		}
		return p.GetCodeAt(args, reply)
	case "eth_balanceAt":
		args, err := req.ToGetBalanceArgs()
		if err != nil {
			return err
		}
		return p.GetBalanceAt(args, reply)
	case "eth_stateAt":
		args, err := req.ToGetStateArgs()
		if err != nil {
			return err
		}
		return p.GetStateAt(args, reply)
	case "eth_storageAt":
		args, err := req.ToStorageAtArgs()
		if err != nil {
			return err
		}
		return p.GetStorageAt(args, reply)
	case "eth_blockByNumber", "eth_blockByHash":
		args, err := req.ToGetBlockArgs()
		if err != nil {
			return err
		}
		return p.GetBlock(args, reply)
	case "eth_transact":
		args, err := req.ToNewTxArgs()
		if err != nil {
			return err
		}
		return p.Transact(args, reply)
	case "eth_call":
		args, err := req.ToNewTxArgs()
		if err != nil {
			return err
		}
		return p.Call(args, reply)
	case "eth_newFilter":
		args, err := req.ToFilterArgs()
		if err != nil {
			return err
		}
		return p.NewFilter(args, reply)
	case "eth_newFilterString":
		args, err := req.ToFilterStringArgs()
		if err != nil {
			return err
		}
		return p.NewFilterString(args, reply)
	case "eth_changed":
		args, err := req.ToFilterChangedArgs()
		if err != nil {
			return err
		}
		return p.FilterChanged(args, reply)
	case "eth_gasPrice":
		*reply = defaultGasPrice
		return nil
	case "web3_sha3":
		args, err := req.ToSha3Args()
		if err != nil {
			return err
		}
		return p.Sha3(args, reply)
	case "db_put":
		args, err := req.ToDbPutArgs()
		if err != nil {
			return err
		}
		return p.DbPut(args, reply)
	case "db_get":
		args, err := req.ToDbGetArgs()
		if err != nil {
			return err
		}
		return p.DbGet(args, reply)
	case "shh_newIdentity":
		return p.NewWhisperIdentity(reply)
	case "shh_newFilter":
		args, err := req.ToWhisperFilterArgs()
		if err != nil {
			return err
		}
		return p.NewWhisperFilter(args, reply)
	case "shh_changed":
		args, err := req.ToWhisperIdArgs()
		if err != nil {
			return err
		}
		return p.MessagesChanged(args, reply)
	case "shh_post":
		args, err := req.ToWhisperPostArgs()
		if err != nil {
			return err
		}
		return p.WhisperPost(args, reply)
	case "shh_haveIdentity":
		args, err := req.ToWhisperHasIdentityArgs()
		if err != nil {
			return err
		}
		return p.HasWhisperIdentity(args, reply)
	case "shh_getMessages":
		args, err := req.ToWhisperIdArgs()
		if err != nil {
			return err
		}
		return p.WhisperMessages(args, reply)
	default:
		return NewErrorResponse(fmt.Sprintf("%v %s", ErrorNotImplemented, req.Method))
	}

	rpclogger.DebugDetailf("Reply: %T %s", reply, reply)
	return nil
}
