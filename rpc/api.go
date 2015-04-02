package rpc

import (
	"encoding/json"
	"math/big"
	"sync"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/xeth"
)

type EthereumApi struct {
	eth    *xeth.XEth
	xethMu sync.RWMutex
}

func NewEthereumApi(xeth *xeth.XEth) *EthereumApi {
	api := &EthereumApi{
		eth: xeth,
	}

	return api
}

func (api *EthereumApi) xeth() *xeth.XEth {
	api.xethMu.RLock()
	defer api.xethMu.RUnlock()

	return api.eth
}

func (api *EthereumApi) xethAtStateNum(num int64) *xeth.XEth {
	return api.xeth().AtStateNum(num)
}

func (api *EthereumApi) GetRequestReply(req *RpcRequest, reply *interface{}) error {
	// Spec at https://github.com/ethereum/wiki/wiki/JSON-RPC
	rpclogger.Debugf("%s %s", req.Method, req.Params)

	switch req.Method {
	case "web3_sha3":
		args := new(Sha3Args)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}
		*reply = common.ToHex(crypto.Sha3(common.FromHex(args.Data)))
	case "web3_clientVersion":
		*reply = api.xeth().ClientVersion()
	case "net_version":
		*reply = api.xeth().NetworkVersion()
	case "net_listening":
		*reply = api.xeth().IsListening()
	case "net_peerCount":
		v := api.xeth().PeerCount()
		*reply = common.ToHex(big.NewInt(int64(v)).Bytes())
	case "eth_protocolVersion":
		*reply = api.xeth().EthVersion()
	case "eth_coinbase":
		// TODO handling of empty coinbase due to lack of accounts
		res := api.xeth().Coinbase()
		if res == "0x" || res == "0x0" {
			*reply = nil
		} else {
			*reply = res
		}
	case "eth_mining":
		*reply = api.xeth().IsMining()
	case "eth_gasPrice":
		v := api.xeth().DefaultGas()
		*reply = common.ToHex(v.Bytes())
	case "eth_accounts":
		*reply = api.xeth().Accounts()
	case "eth_blockNumber":
		v := api.xeth().CurrentBlock().Number()
		*reply = common.ToHex(v.Bytes())
	case "eth_getBalance":
		args := new(GetBalanceArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		*reply = api.xethAtStateNum(args.BlockNumber).BalanceAt(args.Address)
		//v := api.xethAtStateNum(args.BlockNumber).State().SafeGet(args.Address).Balance()
		//*reply = common.ToHex(v.Bytes())
	case "eth_getStorage", "eth_storageAt":
		args := new(GetStorageArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		*reply = api.xethAtStateNum(args.BlockNumber).State().SafeGet(args.Address).Storage()
	case "eth_getStorageAt":
		args := new(GetStorageAtArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		*reply = api.xethAtStateNum(args.BlockNumber).StorageAt(args.Address, args.Key)
	case "eth_getTransactionCount":
		args := new(GetTxCountArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		count := api.xethAtStateNum(args.BlockNumber).TxCountAt(args.Address)
		*reply = common.ToHex(big.NewInt(int64(count)).Bytes())
	case "eth_getBlockTransactionCountByHash":
		args := new(GetBlockByHashArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		block := NewBlockRes(api.xeth().EthBlockByHash(args.BlockHash), false)
		*reply = common.ToHex(big.NewInt(int64(len(block.Transactions))).Bytes())
	case "eth_getBlockTransactionCountByNumber":
		args := new(GetBlockByNumberArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		block := NewBlockRes(api.xeth().EthBlockByNumber(args.BlockNumber), false)
		*reply = common.ToHex(big.NewInt(int64(len(block.Transactions))).Bytes())
	case "eth_getUncleCountByBlockHash":
		args := new(GetBlockByHashArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		block := api.xeth().EthBlockByHash(args.BlockHash)
		br := NewBlockRes(block, false)
		*reply = common.ToHex(big.NewInt(int64(len(br.Uncles))).Bytes())
	case "eth_getUncleCountByBlockNumber":
		args := new(GetBlockByNumberArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		block := api.xeth().EthBlockByNumber(args.BlockNumber)
		br := NewBlockRes(block, false)
		*reply = common.ToHex(big.NewInt(int64(len(br.Uncles))).Bytes())
	case "eth_getData", "eth_getCode":
		args := new(GetDataArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}
		*reply = api.xethAtStateNum(args.BlockNumber).CodeAt(args.Address)
	case "eth_sendTransaction", "eth_transact":
		args := new(NewTxArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		v, err := api.xeth().Transact(args.From, args.To, args.Value.String(), args.Gas.String(), args.GasPrice.String(), args.Data)
		if err != nil {
			return err
		}
		*reply = v
	case "eth_call":
		args := new(CallArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		v, err := api.xethAtStateNum(args.BlockNumber).Call(args.From, args.To, args.Value.String(), args.Gas.String(), args.GasPrice.String(), args.Data)
		if err != nil {
			return err
		}

		*reply = v
	case "eth_flush":
		return NewNotImplementedError(req.Method)
	case "eth_getBlockByHash":
		args := new(GetBlockByHashArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		block := api.xeth().EthBlockByHash(args.BlockHash)
		br := NewBlockRes(block, true)

		*reply = br
	case "eth_getBlockByNumber":
		args := new(GetBlockByNumberArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		block := api.xeth().EthBlockByNumber(args.BlockNumber)
		br := NewBlockRes(block, true)

		*reply = br
	case "eth_getTransactionByHash":
		// HashIndexArgs used, but only the "Hash" part we need.
		args := new(HashIndexArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
		}
		tx, bhash, bnum, txi := api.xeth().EthTransactionByHash(args.Hash)
		if tx != nil {
			v := NewTransactionRes(tx)
			v.BlockHash = newHexData(bhash)
			v.BlockNumber = newHexNum(bnum)
			v.TxIndex = newHexNum(txi)
			*reply = v
		}
	case "eth_getTransactionByBlockHashAndIndex":
		args := new(HashIndexArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		block := api.xeth().EthBlockByHash(args.Hash)
		br := NewBlockRes(block, true)

		if args.Index >= int64(len(br.Transactions)) || args.Index < 0 {
			return NewValidationError("Index", "does not exist")
		}
		*reply = br.Transactions[args.Index]
	case "eth_getTransactionByBlockNumberAndIndex":
		args := new(BlockNumIndexArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		block := api.xeth().EthBlockByNumber(args.BlockNumber)
		v := NewBlockRes(block, true)

		if args.Index >= int64(len(v.Transactions)) || args.Index < 0 {
			return NewValidationError("Index", "does not exist")
		}
		*reply = v.Transactions[args.Index]
	case "eth_getUncleByBlockHashAndIndex":
		args := new(HashIndexArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		br := NewBlockRes(api.xeth().EthBlockByHash(args.Hash), false)

		if args.Index >= int64(len(br.Uncles)) || args.Index < 0 {
			return NewValidationError("Index", "does not exist")
		}

		uhash := br.Uncles[args.Index]
		uncle := NewBlockRes(api.xeth().EthBlockByHash(uhash.String()), false)

		*reply = uncle
	case "eth_getUncleByBlockNumberAndIndex":
		args := new(BlockNumIndexArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		block := api.xeth().EthBlockByNumber(args.BlockNumber)
		v := NewBlockRes(block, true)

		if args.Index >= int64(len(v.Uncles)) || args.Index < 0 {
			return NewValidationError("Index", "does not exist")
		}

		uhash := v.Uncles[args.Index]
		uncle := NewBlockRes(api.xeth().EthBlockByHash(uhash.String()), false)

		*reply = uncle
	case "eth_getCompilers":
		c := []string{""}
		*reply = c
	case "eth_compileSolidity", "eth_compileLLL", "eth_compileSerpent":
		return NewNotImplementedError(req.Method)
	case "eth_newFilter":
		args := new(BlockFilterArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		id := api.xeth().RegisterFilter(args.Earliest, args.Latest, args.Skip, args.Max, args.Address, args.Topics)
		*reply = common.ToHex(big.NewInt(int64(id)).Bytes())
	case "eth_newBlockFilter":
		args := new(FilterStringArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}
		id := api.xeth().NewFilterString(args.Word)
		*reply = common.ToHex(big.NewInt(int64(id)).Bytes())
	case "eth_uninstallFilter":
		args := new(FilterIdArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}
		*reply = api.xeth().UninstallFilter(args.Id)
	case "eth_getFilterChanges":
		args := new(FilterIdArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}
		*reply = NewLogsRes(api.xeth().FilterChanged(args.Id))
	case "eth_getFilterLogs":
		args := new(FilterIdArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}
		*reply = NewLogsRes(api.xeth().Logs(args.Id))
	case "eth_getLogs":
		args := new(BlockFilterArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}
		*reply = NewLogsRes(api.xeth().AllLogs(args.Earliest, args.Latest, args.Skip, args.Max, args.Address, args.Topics))
	case "eth_getWork":
		api.xeth().SetMining(true)
		*reply = api.xeth().RemoteMining().GetWork()
	case "eth_submitWork":
		args := new(SubmitWorkArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}
		*reply = api.xeth().RemoteMining().SubmitWork(args.Nonce, common.HexToHash(args.Digest), common.HexToHash(args.Header))
	case "db_putString":
		args := new(DbArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		if err := args.requirements(); err != nil {
			return err
		}

		api.xeth().DbPut([]byte(args.Database+args.Key), args.Value)

		*reply = true
	case "db_getString":
		args := new(DbArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		if err := args.requirements(); err != nil {
			return err
		}

		res, _ := api.xeth().DbGet([]byte(args.Database + args.Key))
		*reply = string(res)
	case "db_putHex":
		args := new(DbHexArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		if err := args.requirements(); err != nil {
			return err
		}

		api.xeth().DbPut([]byte(args.Database+args.Key), args.Value)
		*reply = true
	case "db_getHex":
		args := new(DbHexArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		if err := args.requirements(); err != nil {
			return err
		}

		res, _ := api.xeth().DbGet([]byte(args.Database + args.Key))
		*reply = common.ToHex(res)
	case "shh_version":
		*reply = api.xeth().WhisperVersion()
	case "shh_post":
		args := new(WhisperMessageArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		err := api.xeth().Whisper().Post(args.Payload, args.To, args.From, args.Topics, args.Priority, args.Ttl)
		if err != nil {
			return err
		}

		*reply = true
	case "shh_newIdentity":
		*reply = api.xeth().Whisper().NewIdentity()
	// case "shh_removeIdentity":
	// 	args := new(WhisperIdentityArgs)
	// 	if err := json.Unmarshal(req.Params, &args); err != nil {
	// 		return err
	// 	}
	// 	*reply = api.xeth().Whisper().RemoveIdentity(args.Identity)
	case "shh_hasIdentity":
		args := new(WhisperIdentityArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}
		*reply = api.xeth().Whisper().HasIdentity(args.Identity)
	case "shh_newGroup", "shh_addToGroup":
		return NewNotImplementedError(req.Method)
	case "shh_newFilter":
		args := new(WhisperFilterArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}
		opts := new(xeth.Options)
		// opts.From = args.From
		opts.To = args.To
		opts.Topics = args.Topics
		id := api.xeth().NewWhisperFilter(opts)
		*reply = common.ToHex(big.NewInt(int64(id)).Bytes())
	case "shh_uninstallFilter":
		args := new(FilterIdArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}
		*reply = api.xeth().UninstallWhisperFilter(args.Id)
	case "shh_getFilterChanges":
		args := new(FilterIdArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}
		*reply = api.xeth().MessagesChanged(args.Id)
	case "shh_getMessages":
		args := new(FilterIdArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}
		*reply = api.xeth().Whisper().Messages(args.Id)

	// case "eth_register":
	// 	// Placeholder for actual type
	// 	args := new(HashIndexArgs)
	// 	if err := json.Unmarshal(req.Params, &args); err != nil {
	// 		return err
	// 	}
	// 	*reply = api.xeth().Register(args.Hash)
	// case "eth_unregister":
	// 	args := new(HashIndexArgs)
	// 	if err := json.Unmarshal(req.Params, &args); err != nil {
	// 		return err
	// 	}
	// 	*reply = api.xeth().Unregister(args.Hash)
	// case "eth_watchTx":
	// 	args := new(HashIndexArgs)
	// 	if err := json.Unmarshal(req.Params, &args); err != nil {
	// 		return err
	// 	}
	// 	*reply = api.xeth().PullWatchTx(args.Hash)
	default:
		return NewNotImplementedError(req.Method)
	}

	rpclogger.DebugDetailf("Reply: %T %s", reply, reply)
	return nil
}
