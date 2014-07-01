package ethpub

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"github.com/ethereum/eth-go/ethchain"
	"github.com/ethereum/eth-go/ethlog"
	"github.com/ethereum/eth-go/ethutil"
	"math/big"
	"strings"
	"sync/atomic"
)

var logger = ethlog.NewLogger("PUB")

type PEthereum struct {
	manager      ethchain.EthManager
	stateManager *ethchain.StateManager
	blockChain   *ethchain.BlockChain
	txPool       *ethchain.TxPool
}

func NewPEthereum(manager ethchain.EthManager) *PEthereum {
	return &PEthereum{
		manager,
		manager.StateManager(),
		manager.BlockChain(),
		manager.TxPool(),
	}
}

func (lib *PEthereum) GetBlock(hexHash string) *PBlock {
	hash := ethutil.FromHex(hexHash)
	block := lib.blockChain.GetBlock(hash)

	return NewPBlock(block)
}

func (lib *PEthereum) GetKey() *PKey {
	keyPair := ethutil.GetKeyRing().Get(0)

	return NewPKey(keyPair)
}

func (lib *PEthereum) GetStateObject(address string) *PStateObject {
	stateObject := lib.stateManager.CurrentState().GetStateObject(ethutil.FromHex(address))
	if stateObject != nil {
		return NewPStateObject(stateObject)
	}

	// See GetStorage for explanation on "nil"
	return NewPStateObject(nil)
}

func (lib *PEthereum) GetPeerCount() int {
	return lib.manager.PeerCount()
}

func (lib *PEthereum) GetPeers() []PPeer {
	var peers []PPeer
	for peer := lib.manager.Peers().Front(); peer != nil; peer = peer.Next() {
		p := peer.Value.(ethchain.Peer)
		// we only want connected peers
		if atomic.LoadInt32(p.Connected()) != 0 {
			peers = append(peers, *NewPPeer(p))
		}
	}

	return peers
}

func (lib *PEthereum) GetIsMining() bool {
	return lib.manager.IsMining()
}

func (lib *PEthereum) GetIsListening() bool {
	return lib.manager.IsListening()
}

func (lib *PEthereum) GetCoinBase() string {
	data, _ := ethutil.Config.Db.Get([]byte("KeyRing"))
	keyRing := ethutil.NewValueFromBytes(data)
	key := keyRing.Get(0).Bytes()

	return lib.SecretToAddress(hex.EncodeToString(key))
}

func (lib *PEthereum) GetTransactionsFor(address string, asJson bool) interface{} {
	sBlk := lib.manager.BlockChain().LastBlockHash
	blk := lib.manager.BlockChain().GetBlock(sBlk)
	addr := []byte(ethutil.FromHex(address))

	var txs []*PTx

	for ; blk != nil; blk = lib.manager.BlockChain().GetBlock(sBlk) {
		sBlk = blk.PrevHash

		// Loop through all transactions to see if we missed any while being offline
		for _, tx := range blk.Transactions() {
			if bytes.Compare(tx.Sender(), addr) == 0 || bytes.Compare(tx.Recipient, addr) == 0 {
				ptx := NewPTx(tx)
				//TODO: somehow move this to NewPTx
				ptx.Confirmations = int(lib.manager.BlockChain().LastBlockNumber - blk.BlockInfo().Number)
				txs = append(txs, ptx)
			}
		}
	}
	if asJson {
		txJson, err := json.Marshal(txs)
		if err != nil {
			return nil
		}
		return string(txJson)
	}
	return txs
}

func (lib *PEthereum) GetStorage(address, storageAddress string) string {
	return lib.GetStateObject(address).GetStorage(storageAddress)
}

func (lib *PEthereum) GetTxCountAt(address string) int {
	return lib.GetStateObject(address).Nonce()
}

func (lib *PEthereum) IsContract(address string) bool {
	return lib.GetStateObject(address).IsContract()
}

func (lib *PEthereum) SecretToAddress(key string) string {
	pair, err := ethutil.NewKeyPairFromSec(ethutil.FromHex(key))
	if err != nil {
		return ""
	}

	return ethutil.Hex(pair.Address())
}

func (lib *PEthereum) Transact(key, recipient, valueStr, gasStr, gasPriceStr, dataStr string) (*PReceipt, error) {
	return lib.createTx(key, recipient, valueStr, gasStr, gasPriceStr, dataStr)
}

func (lib *PEthereum) Create(key, valueStr, gasStr, gasPriceStr, script string) (*PReceipt, error) {
	return lib.createTx(key, "", valueStr, gasStr, gasPriceStr, script)
}

var namereg = ethutil.FromHex("bb5f186604d057c1c5240ca2ae0f6430138ac010")

func GetAddressFromNameReg(stateManager *ethchain.StateManager, name string) []byte {
	recp := new(big.Int).SetBytes([]byte(name))
	object := stateManager.CurrentState().GetStateObject(namereg)
	if object != nil {
		reg := object.GetStorage(recp)

		return reg.Bytes()
	}

	return nil
}
func (lib *PEthereum) createTx(key, recipient, valueStr, gasStr, gasPriceStr, scriptStr string) (*PReceipt, error) {
	var hash []byte
	var contractCreation bool
	if len(recipient) == 0 {
		contractCreation = true
	} else {
		// Check if an address is stored by this address
		addr := GetAddressFromNameReg(lib.stateManager, recipient)
		if len(addr) > 0 {
			hash = addr
		} else {
			hash = ethutil.FromHex(recipient)
		}
	}

	var keyPair *ethutil.KeyPair
	var err error
	if key[0:2] == "0x" {
		keyPair, err = ethutil.NewKeyPairFromSec([]byte(ethutil.FromHex(key[2:])))
	} else {
		keyPair, err = ethutil.NewKeyPairFromSec([]byte(ethutil.FromHex(key)))
	}

	if err != nil {
		return nil, err
	}

	value := ethutil.Big(valueStr)
	gas := ethutil.Big(gasStr)
	gasPrice := ethutil.Big(gasPriceStr)
	var tx *ethchain.Transaction
	// Compile and assemble the given data
	if contractCreation {
		var script []byte
		var err error
		if ethutil.IsHex(scriptStr) {
			script = ethutil.FromHex(scriptStr)
		} else {
			script, err = ethutil.Compile(scriptStr)
			if err != nil {
				return nil, err
			}
		}

		tx = ethchain.NewContractCreationTx(value, gas, gasPrice, script)
	} else {
		data := ethutil.StringToByteFunc(scriptStr, func(s string) (ret []byte) {
			slice := strings.Split(s, "\n")
			for _, dataItem := range slice {
				d := ethutil.FormatData(dataItem)
				ret = append(ret, d...)
			}
			return
		})

		tx = ethchain.NewTransactionMessage(hash, value, gas, gasPrice, data)
	}

	acc := lib.stateManager.TransState().GetOrNewStateObject(keyPair.Address())
	tx.Nonce = acc.Nonce
	acc.Nonce += 1
	lib.stateManager.TransState().UpdateStateObject(acc)

	tx.Sign(keyPair.PrivateKey)
	lib.txPool.QueueTransaction(tx)

	if contractCreation {
		logger.Infof("Contract addr %x", tx.CreationAddress())
	}

	return NewPReciept(contractCreation, tx.CreationAddress(), tx.Hash(), keyPair.Address()), nil
}
