package rpc

import (
	"encoding/json"
	"math/big"
	"path"
	"sync"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/ethdb"
	"github.com/ethereum/go-ethereum/xeth"
)

type EthereumApi struct {
	eth    *xeth.XEth
	xethMu sync.RWMutex
	db     common.Database
}

func NewEthereumApi(xeth *xeth.XEth, dataDir string) *EthereumApi {
	// What about when dataDir is empty?
	db, err := ethdb.NewLDBDatabase(path.Join(dataDir, "dapps"))
	if err != nil {
		panic(err)
	}
	api := &EthereumApi{
		eth: xeth,
		db:  db,
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

func (api *EthereumApi) Close() {
	api.db.Close()
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
	case "eth_version":
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

		v := api.xethAtStateNum(args.BlockNumber).State().SafeGet(args.Address.Hex()).Balance()
		*reply = common.ToHex(v.Bytes())
	case "eth_getStorage", "eth_storageAt":
		args := new(GetStorageArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		*reply = api.xethAtStateNum(args.BlockNumber).State().SafeGet(args.Address.Hex()).Storage()
	case "eth_getStorageAt":
		args := new(GetStorageAtArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		state := api.xethAtStateNum(args.BlockNumber).State().SafeGet(args.Address.Hex())
		value := state.StorageString(args.Key.Hex())

		*reply = common.Bytes2Hex(value.Bytes())
	case "eth_getTransactionCount":
		args := new(GetTxCountArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		*reply = api.xethAtStateNum(args.BlockNumber).TxCountAt(args.Address.Hex())
	case "eth_getBlockTransactionCountByHash":
		args := new(GetBlockByHashArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		block := NewBlockRes(api.xeth().EthBlockByHash(args.BlockHash))
		*reply = common.ToHex(big.NewInt(int64(len(block.Transactions))).Bytes())
	case "eth_getBlockTransactionCountByNumber":
		args := new(GetBlockByNumberArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		block := NewBlockRes(api.xeth().EthBlockByNumber(args.BlockNumber))
		*reply = common.ToHex(big.NewInt(int64(len(block.Transactions))).Bytes())
	case "eth_getUncleCountByBlockHash":
		args := new(GetBlockByHashArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		block := api.xeth().EthBlockByHash(args.BlockHash)
		br := NewBlockRes(block)
		*reply = common.ToHex(big.NewInt(int64(len(br.Uncles))).Bytes())
	case "eth_getUncleCountByBlockNumber":
		args := new(GetBlockByNumberArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		block := api.xeth().EthBlockByNumber(args.BlockNumber)
		br := NewBlockRes(block)
		*reply = common.ToHex(big.NewInt(int64(len(br.Uncles))).Bytes())
	case "eth_getData", "eth_getCode":
		args := new(GetDataArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}
		*reply = api.xethAtStateNum(args.BlockNumber).CodeAt(args.Address.Hex())
	case "eth_sendTransaction", "eth_transact":
		args := new(NewTxArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		v, err := api.xeth().Transact(args.From.Hex(), args.To.Hex(), args.Value.String(), args.Gas.String(), args.GasPrice.String(), args.Data)
		if err != nil {
			return err
		}
		*reply = v
	case "eth_call":
		args := new(NewTxArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		v, err := api.xethAtStateNum(args.BlockNumber).Call(args.From.Hex(), args.To.Hex(), args.Value.String(), args.Gas.String(), args.GasPrice.String(), args.Data)
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
		br := NewBlockRes(block)
		br.fullTx = args.IncludeTxs

		*reply = br
	case "eth_getBlockByNumber":
		args := new(GetBlockByNumberArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		block := api.xeth().EthBlockByNumber(args.BlockNumber)
		br := NewBlockRes(block)
		br.fullTx = args.IncludeTxs

		*reply = br
	case "eth_getTransactionByHash":
		// HashIndexArgs used, but only the "Hash" part we need.
		args := new(HashIndexArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
		}
		tx := api.xeth().EthTransactionByHash(args.Hash)
		if tx != nil {
			*reply = NewTransactionRes(tx)
		}
	case "eth_getTransactionByBlockHashAndIndex":
		args := new(HashIndexArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		block := api.xeth().EthBlockByHash(args.Hash)
		br := NewBlockRes(block)
		br.fullTx = true

		if args.Index > int64(len(br.Transactions)) || args.Index < 0 {
			return NewValidationError("Index", "does not exist")
		}
		*reply = br.Transactions[args.Index]
	case "eth_getTransactionByBlockNumberAndIndex":
		args := new(BlockNumIndexArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		block := api.xeth().EthBlockByNumber(args.BlockNumber)
		v := NewBlockRes(block)
		v.fullTx = true

		if args.Index > int64(len(v.Transactions)) || args.Index < 0 {
			return NewValidationError("Index", "does not exist")
		}
		*reply = v.Transactions[args.Index]
	case "eth_getUncleByBlockHashAndIndex":
		args := new(HashIndexArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		br := NewBlockRes(api.xeth().EthBlockByHash(args.Hash))

		if args.Index > int64(len(br.Uncles)) || args.Index < 0 {
			return NewValidationError("Index", "does not exist")
		}

		uhash := br.Uncles[args.Index].Hex()
		uncle := NewBlockRes(api.xeth().EthBlockByHash(uhash))

		*reply = uncle
	case "eth_getUncleByBlockNumberAndIndex":
		args := new(BlockNumIndexArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		block := api.xeth().EthBlockByNumber(args.BlockNumber)
		v := NewBlockRes(block)
		v.fullTx = true

		if args.Index > int64(len(v.Uncles)) || args.Index < 0 {
			return NewValidationError("Index", "does not exist")
		}

		uhash := v.Uncles[args.Index].Hex()
		uncle := NewBlockRes(api.xeth().EthBlockByHash(uhash))

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

		opts := toFilterOptions(args)
		id := api.xeth().RegisterFilter(opts)
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
		opts := toFilterOptions(args)
		*reply = NewLogsRes(api.xeth().AllLogs(opts))
	case "eth_getWork":
		api.xeth().SetMining(true)
		*reply = api.xeth().RemoteMining().GetWork()
	case "eth_submitWork":
		args := new(SubmitWorkArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}
		*reply = api.xeth().RemoteMining().SubmitWork(args.Nonce, args.Digest, args.Header)
	case "db_putString":
		args := new(DbArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		if err := args.requirements(); err != nil {
			return err
		}

		api.db.Put([]byte(args.Database+args.Key), args.Value)
		*reply = true
	case "db_getString":
		args := new(DbArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		if err := args.requirements(); err != nil {
			return err
		}

		res, _ := api.db.Get([]byte(args.Database + args.Key))
		*reply = string(res)
	case "db_putHex":
		args := new(DbHexArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		if err := args.requirements(); err != nil {
			return err
		}

		api.db.Put([]byte(args.Database+args.Key), args.Value)
		*reply = true
	case "db_getHex":
		args := new(DbHexArgs)
		if err := json.Unmarshal(req.Params, &args); err != nil {
			return err
		}

		if err := args.requirements(); err != nil {
			return err
		}

		res, _ := api.db.Get([]byte(args.Database + args.Key))
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

func toFilterOptions(options *BlockFilterArgs) *core.FilterOptions {
	var opts core.FilterOptions

	// Convert optional address slice/string to byte slice
	if str, ok := options.Address.(string); ok {
		opts.Address = []common.Address{common.HexToAddress(str)}
	} else if slice, ok := options.Address.([]interface{}); ok {
		bslice := make([]common.Address, len(slice))
		for i, addr := range slice {
			if saddr, ok := addr.(string); ok {
				bslice[i] = common.HexToAddress(saddr)
			}
		}
		opts.Address = bslice
	}

	opts.Earliest = options.Earliest
	opts.Latest = options.Latest

	topics := make([][]common.Hash, len(options.Topics))
	for i, topicDat := range options.Topics {
		if slice, ok := topicDat.([]interface{}); ok {
			topics[i] = make([]common.Hash, len(slice))
			for j, topic := range slice {
				topics[i][j] = common.HexToHash(topic.(string))
			}
		} else if str, ok := topicDat.(string); ok {
			topics[i] = []common.Hash{common.HexToHash(str)}
		}
	}
	opts.Topics = topics

	return &opts
}

/*
	Work() chan<- *types.Block
	SetWorkCh(chan<- Work)
	Stop()
	Start()
	Rate() uint64
*/
