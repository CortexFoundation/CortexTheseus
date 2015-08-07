// Copyright 2014 The go-ethereum Authors
// This file is part of the go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

// Package xeth is the interface to all Ethereum functionality.
package xeth

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math/big"
	"regexp"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/accounts"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/compiler"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/eth"
	"github.com/ethereum/go-ethereum/event/filter"
	"github.com/ethereum/go-ethereum/logger"
	"github.com/ethereum/go-ethereum/logger/glog"
	"github.com/ethereum/go-ethereum/miner"
	"github.com/ethereum/go-ethereum/rlp"
)

var (
	filterTickerTime = 5 * time.Minute
	defaultGasPrice  = big.NewInt(10000000000000) //150000000000
	defaultGas       = big.NewInt(90000)          //500000
	dappStorePre     = []byte("dapp-")
	addrReg          = regexp.MustCompile(`^(0x)?[a-fA-F0-9]{40}$`)
)

// byte will be inferred
const (
	UnknownFilterTy = iota
	BlockFilterTy
	TransactionFilterTy
	LogFilterTy
)

func DefaultGas() *big.Int { return new(big.Int).Set(defaultGas) }

func (self *XEth) DefaultGasPrice() *big.Int {
	if self.gpo == nil {
		self.gpo = eth.NewGasPriceOracle(self.backend)
	}
	return self.gpo.SuggestPrice()
}

type XEth struct {
	backend  *eth.Ethereum
	frontend Frontend

	state   *State
	whisper *Whisper

	quit          chan struct{}
	filterManager *filter.FilterManager

	logMu    sync.RWMutex
	logQueue map[int]*logQueue

	blockMu    sync.RWMutex
	blockQueue map[int]*hashQueue

	transactionMu    sync.RWMutex
	transactionQueue map[int]*hashQueue

	messagesMu sync.RWMutex
	messages   map[int]*whisperFilter

	// regmut   sync.Mutex
	// register map[string][]*interface{} // TODO improve return type

	agent *miner.RemoteAgent

	gpo *eth.GasPriceOracle
}

func NewTest(eth *eth.Ethereum, frontend Frontend) *XEth {
	return &XEth{
		backend:  eth,
		frontend: frontend,
	}
}

// New creates an XEth that uses the given frontend.
// If a nil Frontend is provided, a default frontend which
// confirms all transactions will be used.
func New(ethereum *eth.Ethereum, frontend Frontend) *XEth {
	xeth := &XEth{
		backend:          ethereum,
		frontend:         frontend,
		quit:             make(chan struct{}),
		filterManager:    filter.NewFilterManager(ethereum.EventMux()),
		logQueue:         make(map[int]*logQueue),
		blockQueue:       make(map[int]*hashQueue),
		transactionQueue: make(map[int]*hashQueue),
		messages:         make(map[int]*whisperFilter),
		agent:            miner.NewRemoteAgent(),
	}
	if ethereum.Whisper() != nil {
		xeth.whisper = NewWhisper(ethereum.Whisper())
	}
	ethereum.Miner().Register(xeth.agent)
	if frontend == nil {
		xeth.frontend = dummyFrontend{}
	}
	xeth.state = NewState(xeth, xeth.backend.ChainManager().State())

	go xeth.start()
	go xeth.filterManager.Start()

	return xeth
}

func (self *XEth) start() {
	timer := time.NewTicker(2 * time.Second)
done:
	for {
		select {
		case <-timer.C:
			self.logMu.Lock()
			for id, filter := range self.logQueue {
				if time.Since(filter.timeout) > filterTickerTime {
					self.filterManager.UninstallFilter(id)
					delete(self.logQueue, id)
				}
			}
			self.logMu.Unlock()

			self.blockMu.Lock()
			for id, filter := range self.blockQueue {
				if time.Since(filter.timeout) > filterTickerTime {
					self.filterManager.UninstallFilter(id)
					delete(self.blockQueue, id)
				}
			}
			self.blockMu.Unlock()

			self.transactionMu.Lock()
			for id, filter := range self.transactionQueue {
				if time.Since(filter.timeout) > filterTickerTime {
					self.filterManager.UninstallFilter(id)
					delete(self.transactionQueue, id)
				}
			}
			self.transactionMu.Unlock()

			self.messagesMu.Lock()
			for id, filter := range self.messages {
				if time.Since(filter.activity()) > filterTickerTime {
					self.Whisper().Unwatch(id)
					delete(self.messages, id)
				}
			}
			self.messagesMu.Unlock()
		case <-self.quit:
			break done
		}
	}
}

func (self *XEth) stop() {
	close(self.quit)
}

func cAddress(a []string) []common.Address {
	bslice := make([]common.Address, len(a))
	for i, addr := range a {
		bslice[i] = common.HexToAddress(addr)
	}
	return bslice
}

func cTopics(t [][]string) [][]common.Hash {
	topics := make([][]common.Hash, len(t))
	for i, iv := range t {
		topics[i] = make([]common.Hash, len(iv))
		for j, jv := range iv {
			topics[i][j] = common.HexToHash(jv)
		}
	}
	return topics
}

func (self *XEth) RemoteMining() *miner.RemoteAgent { return self.agent }

func (self *XEth) AtStateNum(num int64) *XEth {
	var st *state.StateDB
	switch num {
	case -2:
		st = self.backend.Miner().PendingState().Copy()
	default:
		if block := self.getBlockByHeight(num); block != nil {
			st = state.New(block.Root(), self.backend.StateDb())
		} else {
			st = state.New(self.backend.ChainManager().GetBlockByNumber(0).Root(), self.backend.StateDb())
		}
	}

	return self.WithState(st)
}

func (self *XEth) WithState(statedb *state.StateDB) *XEth {
	xeth := &XEth{
		backend:  self.backend,
		frontend: self.frontend,
		gpo:      self.gpo,
	}

	xeth.state = NewState(xeth, statedb)
	return xeth
}

func (self *XEth) State() *State { return self.state }

// subscribes to new head block events and
// waits until blockchain height is greater n at any time
// given the current head, waits for the next chain event
// sets the state to the current head
// loop is async and quit by closing the channel
// used in tests and JS console debug module to control advancing private chain manually
// Note: this is not threadsafe, only called in JS single process and tests
func (self *XEth) UpdateState() (wait chan *big.Int) {
	wait = make(chan *big.Int)
	go func() {
		sub := self.backend.EventMux().Subscribe(core.ChainHeadEvent{})
		var m, n *big.Int
		var ok bool
	out:
		for {
			select {
			case event := <-sub.Chan():
				ev, ok := event.(core.ChainHeadEvent)
				if ok {
					m = ev.Block.Number()
					if n != nil && n.Cmp(m) < 0 {
						wait <- n
						n = nil
					}
					statedb := state.New(ev.Block.Root(), self.backend.StateDb())
					self.state = NewState(self, statedb)
				}
			case n, ok = <-wait:
				if !ok {
					break out
				}
			}
		}
		sub.Unsubscribe()
	}()
	return
}

func (self *XEth) Whisper() *Whisper { return self.whisper }

func (self *XEth) getBlockByHeight(height int64) *types.Block {
	var num uint64

	switch height {
	case -2:
		return self.backend.Miner().PendingBlock()
	case -1:
		return self.CurrentBlock()
	default:
		if height < 0 {
			return nil
		}

		num = uint64(height)
	}

	return self.backend.ChainManager().GetBlockByNumber(num)
}

func (self *XEth) BlockByHash(strHash string) *Block {
	hash := common.HexToHash(strHash)
	block := self.backend.ChainManager().GetBlock(hash)

	return NewBlock(block)
}

func (self *XEth) EthBlockByHash(strHash string) *types.Block {
	hash := common.HexToHash(strHash)
	block := self.backend.ChainManager().GetBlock(hash)

	return block
}

func (self *XEth) EthTransactionByHash(hash string) (tx *types.Transaction, blhash common.Hash, blnum *big.Int, txi uint64) {
	// Due to increasing return params and need to determine if this is from transaction pool or
	// some chain, this probably needs to be refactored for more expressiveness
	data, _ := self.backend.ExtraDb().Get(common.FromHex(hash))
	if len(data) != 0 {
		dtx := new(types.Transaction)
		if err := rlp.DecodeBytes(data, dtx); err != nil {
			glog.V(logger.Error).Infoln(err)
			return
		}
		tx = dtx
	} else { // check pending transactions
		tx = self.backend.TxPool().GetTransaction(common.HexToHash(hash))
	}

	// meta
	var txExtra struct {
		BlockHash  common.Hash
		BlockIndex uint64
		Index      uint64
	}

	v, dberr := self.backend.ExtraDb().Get(append(common.FromHex(hash), 0x0001))
	// TODO check specifically for ErrNotFound
	if dberr != nil {
		return
	}
	r := bytes.NewReader(v)
	err := rlp.Decode(r, &txExtra)
	if err == nil {
		blhash = txExtra.BlockHash
		blnum = big.NewInt(int64(txExtra.BlockIndex))
		txi = txExtra.Index
	} else {
		glog.V(logger.Error).Infoln(err)
	}

	return
}

func (self *XEth) BlockByNumber(num int64) *Block {
	return NewBlock(self.getBlockByHeight(num))
}

func (self *XEth) EthBlockByNumber(num int64) *types.Block {
	return self.getBlockByHeight(num)
}

func (self *XEth) CurrentBlock() *types.Block {
	return self.backend.ChainManager().CurrentBlock()
}

func (self *XEth) GetBlockReceipts(bhash common.Hash) types.Receipts {
	return self.backend.BlockProcessor().GetBlockReceipts(bhash)
}

func (self *XEth) GetTxReceipt(txhash common.Hash) *types.Receipt {
	return core.GetReceipt(self.backend.ExtraDb(), txhash)
}

func (self *XEth) GasLimit() *big.Int {
	return self.backend.ChainManager().GasLimit()
}

func (self *XEth) Block(v interface{}) *Block {
	if n, ok := v.(int32); ok {
		return self.BlockByNumber(int64(n))
	} else if str, ok := v.(string); ok {
		return self.BlockByHash(str)
	} else if f, ok := v.(float64); ok { // JSON numbers are represented as float64
		return self.BlockByNumber(int64(f))
	}

	return nil
}

func (self *XEth) Accounts() []string {
	// TODO: check err?
	accounts, _ := self.backend.AccountManager().Accounts()
	accountAddresses := make([]string, len(accounts))
	for i, ac := range accounts {
		accountAddresses[i] = ac.Address.Hex()
	}
	return accountAddresses
}

// accessor for solidity compiler.
// memoized if available, retried on-demand if not
func (self *XEth) Solc() (*compiler.Solidity, error) {
	return self.backend.Solc()
}

// set in js console via admin interface or wrapper from cli flags
func (self *XEth) SetSolc(solcPath string) (*compiler.Solidity, error) {
	self.backend.SetSolc(solcPath)
	return self.Solc()
}

// store DApp value in extra database
func (self *XEth) DbPut(key, val []byte) bool {
	self.backend.ExtraDb().Put(append(dappStorePre, key...), val)
	return true
}

// retrieve DApp value from extra database
func (self *XEth) DbGet(key []byte) ([]byte, error) {
	val, err := self.backend.ExtraDb().Get(append(dappStorePre, key...))
	return val, err
}

func (self *XEth) PeerCount() int {
	return self.backend.PeerCount()
}

func (self *XEth) IsMining() bool {
	return self.backend.IsMining()
}

func (self *XEth) HashRate() int64 {
	return self.backend.Miner().HashRate()
}

func (self *XEth) EthVersion() string {
	return fmt.Sprintf("%d", self.backend.EthVersion())
}

func (self *XEth) NetworkVersion() string {
	return fmt.Sprintf("%d", self.backend.NetVersion())
}

func (self *XEth) WhisperVersion() string {
	return fmt.Sprintf("%d", self.backend.ShhVersion())
}

func (self *XEth) ClientVersion() string {
	return self.backend.ClientVersion()
}

func (self *XEth) SetMining(shouldmine bool, threads int) bool {
	ismining := self.backend.IsMining()
	if shouldmine && !ismining {
		err := self.backend.StartMining(threads)
		return err == nil
	}
	if ismining && !shouldmine {
		self.backend.StopMining()
	}
	return self.backend.IsMining()
}

func (self *XEth) IsListening() bool {
	return self.backend.IsListening()
}

func (self *XEth) Coinbase() string {
	eb, err := self.backend.Etherbase()
	if err != nil {
		return "0x0"
	}
	return eb.Hex()
}

func (self *XEth) NumberToHuman(balance string) string {
	b := common.Big(balance)

	return common.CurrencyToString(b)
}

func (self *XEth) StorageAt(addr, storageAddr string) string {
	return self.State().state.GetState(common.HexToAddress(addr), common.HexToHash(storageAddr)).Hex()
}

func (self *XEth) BalanceAt(addr string) string {
	return common.ToHex(self.State().state.GetBalance(common.HexToAddress(addr)).Bytes())
}

func (self *XEth) TxCountAt(address string) int {
	return int(self.State().state.GetNonce(common.HexToAddress(address)))
}

func (self *XEth) CodeAt(address string) string {
	return common.ToHex(self.State().state.GetCode(common.HexToAddress(address)))
}

func (self *XEth) CodeAtBytes(address string) []byte {
	return self.State().SafeGet(address).Code()
}

func (self *XEth) IsContract(address string) bool {
	return len(self.State().SafeGet(address).Code()) > 0
}

func (self *XEth) UninstallFilter(id int) bool {
	defer self.filterManager.UninstallFilter(id)

	if _, ok := self.logQueue[id]; ok {
		self.logMu.Lock()
		defer self.logMu.Unlock()
		delete(self.logQueue, id)
		return true
	}
	if _, ok := self.blockQueue[id]; ok {
		self.blockMu.Lock()
		defer self.blockMu.Unlock()
		delete(self.blockQueue, id)
		return true
	}
	if _, ok := self.transactionQueue[id]; ok {
		self.transactionMu.Lock()
		defer self.transactionMu.Unlock()
		delete(self.transactionQueue, id)
		return true
	}

	return false
}

func (self *XEth) NewLogFilter(earliest, latest int64, skip, max int, address []string, topics [][]string) int {
	self.logMu.Lock()
	defer self.logMu.Unlock()

	var id int
	filter := core.NewFilter(self.backend)
	filter.SetEarliestBlock(earliest)
	filter.SetLatestBlock(latest)
	filter.SetSkip(skip)
	filter.SetMax(max)
	filter.SetAddress(cAddress(address))
	filter.SetTopics(cTopics(topics))
	filter.LogsCallback = func(logs state.Logs) {
		self.logMu.Lock()
		defer self.logMu.Unlock()

		self.logQueue[id].add(logs...)
	}
	id = self.filterManager.InstallFilter(filter)
	self.logQueue[id] = &logQueue{timeout: time.Now()}

	return id
}

func (self *XEth) NewTransactionFilter() int {
	self.transactionMu.Lock()
	defer self.transactionMu.Unlock()

	var id int
	filter := core.NewFilter(self.backend)
	filter.TransactionCallback = func(tx *types.Transaction) {
		self.transactionMu.Lock()
		defer self.transactionMu.Unlock()

		self.transactionQueue[id].add(tx.Hash())
	}
	id = self.filterManager.InstallFilter(filter)
	self.transactionQueue[id] = &hashQueue{timeout: time.Now()}
	return id
}

func (self *XEth) NewBlockFilter() int {
	self.blockMu.Lock()
	defer self.blockMu.Unlock()

	var id int
	filter := core.NewFilter(self.backend)
	filter.BlockCallback = func(block *types.Block, logs state.Logs) {
		self.blockMu.Lock()
		defer self.blockMu.Unlock()

		self.blockQueue[id].add(block.Hash())
	}
	id = self.filterManager.InstallFilter(filter)
	self.blockQueue[id] = &hashQueue{timeout: time.Now()}
	return id
}

func (self *XEth) GetFilterType(id int) byte {
	if _, ok := self.blockQueue[id]; ok {
		return BlockFilterTy
	} else if _, ok := self.transactionQueue[id]; ok {
		return TransactionFilterTy
	} else if _, ok := self.logQueue[id]; ok {
		return LogFilterTy
	}

	return UnknownFilterTy
}

func (self *XEth) LogFilterChanged(id int) state.Logs {
	self.logMu.Lock()
	defer self.logMu.Unlock()

	if self.logQueue[id] != nil {
		return self.logQueue[id].get()
	}
	return nil
}

func (self *XEth) BlockFilterChanged(id int) []common.Hash {
	self.blockMu.Lock()
	defer self.blockMu.Unlock()

	if self.blockQueue[id] != nil {
		return self.blockQueue[id].get()
	}
	return nil
}

func (self *XEth) TransactionFilterChanged(id int) []common.Hash {
	self.blockMu.Lock()
	defer self.blockMu.Unlock()

	if self.transactionQueue[id] != nil {
		return self.transactionQueue[id].get()
	}
	return nil
}

func (self *XEth) Logs(id int) state.Logs {
	filter := self.filterManager.GetFilter(id)
	if filter != nil {
		return filter.Find()
	}

	return nil
}

func (self *XEth) AllLogs(earliest, latest int64, skip, max int, address []string, topics [][]string) state.Logs {
	filter := core.NewFilter(self.backend)
	filter.SetEarliestBlock(earliest)
	filter.SetLatestBlock(latest)
	filter.SetSkip(skip)
	filter.SetMax(max)
	filter.SetAddress(cAddress(address))
	filter.SetTopics(cTopics(topics))

	return filter.Find()
}

// NewWhisperFilter creates and registers a new message filter to watch for
// inbound whisper messages. All parameters at this point are assumed to be
// HEX encoded.
func (p *XEth) NewWhisperFilter(to, from string, topics [][]string) int {
	// Pre-define the id to be filled later
	var id int

	// Callback to delegate core whisper messages to this xeth filter
	callback := func(msg WhisperMessage) {
		p.messagesMu.RLock() // Only read lock to the filter pool
		defer p.messagesMu.RUnlock()
		p.messages[id].insert(msg)
	}
	// Initialize the core whisper filter and wrap into xeth
	id = p.Whisper().Watch(to, from, topics, callback)

	p.messagesMu.Lock()
	p.messages[id] = newWhisperFilter(id, p.Whisper())
	p.messagesMu.Unlock()

	return id
}

// UninstallWhisperFilter disables and removes an existing filter.
func (p *XEth) UninstallWhisperFilter(id int) bool {
	p.messagesMu.Lock()
	defer p.messagesMu.Unlock()

	if _, ok := p.messages[id]; ok {
		delete(p.messages, id)
		return true
	}
	return false
}

// WhisperMessages retrieves all the known messages that match a specific filter.
func (self *XEth) WhisperMessages(id int) []WhisperMessage {
	self.messagesMu.RLock()
	defer self.messagesMu.RUnlock()

	if self.messages[id] != nil {
		return self.messages[id].messages()
	}
	return nil
}

// WhisperMessagesChanged retrieves all the new messages matched by a filter
// since the last retrieval
func (self *XEth) WhisperMessagesChanged(id int) []WhisperMessage {
	self.messagesMu.RLock()
	defer self.messagesMu.RUnlock()

	if self.messages[id] != nil {
		return self.messages[id].retrieve()
	}
	return nil
}

// func (self *XEth) Register(args string) bool {
// 	self.regmut.Lock()
// 	defer self.regmut.Unlock()

// 	if _, ok := self.register[args]; ok {
// 		self.register[args] = nil // register with empty
// 	}
// 	return true
// }

// func (self *XEth) Unregister(args string) bool {
// 	self.regmut.Lock()
// 	defer self.regmut.Unlock()

// 	if _, ok := self.register[args]; ok {
// 		delete(self.register, args)
// 		return true
// 	}

// 	return false
// }

// // TODO improve return type
// func (self *XEth) PullWatchTx(args string) []*interface{} {
// 	self.regmut.Lock()
// 	defer self.regmut.Unlock()

// 	txs := self.register[args]
// 	self.register[args] = nil

// 	return txs
// }

type KeyVal struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

func (self *XEth) EachStorage(addr string) string {
	var values []KeyVal
	object := self.State().SafeGet(addr)
	it := object.Trie().Iterator()
	for it.Next() {
		values = append(values, KeyVal{common.ToHex(object.Trie().GetKey(it.Key)), common.ToHex(it.Value)})
	}

	valuesJson, err := json.Marshal(values)
	if err != nil {
		return ""
	}

	return string(valuesJson)
}

func (self *XEth) ToAscii(str string) string {
	padded := common.RightPadBytes([]byte(str), 32)

	return "0x" + common.ToHex(padded)
}

func (self *XEth) FromAscii(str string) string {
	if common.IsHex(str) {
		str = str[2:]
	}

	return string(bytes.Trim(common.FromHex(str), "\x00"))
}

func (self *XEth) FromNumber(str string) string {
	if common.IsHex(str) {
		str = str[2:]
	}

	return common.BigD(common.FromHex(str)).String()
}

func (self *XEth) PushTx(encodedTx string) (string, error) {
	tx := new(types.Transaction)
	err := rlp.DecodeBytes(common.FromHex(encodedTx), tx)
	if err != nil {
		glog.V(logger.Error).Infoln(err)
		return "", err
	}

	err = self.backend.TxPool().Add(tx)
	if err != nil {
		return "", err
	}

	if tx.To() == nil {
		from, err := tx.From()
		if err != nil {
			return "", err
		}

		addr := crypto.CreateAddress(from, tx.Nonce())
		glog.V(logger.Info).Infof("Tx(%x) created: %x\n", tx.Hash(), addr)
	} else {
		glog.V(logger.Info).Infof("Tx(%x) to: %x\n", tx.Hash(), tx.To())
	}

	return tx.Hash().Hex(), nil
}

func (self *XEth) Call(fromStr, toStr, valueStr, gasStr, gasPriceStr, dataStr string) (string, string, error) {
	statedb := self.State().State().Copy()
	var from *state.StateObject
	if len(fromStr) == 0 {
		accounts, err := self.backend.AccountManager().Accounts()
		if err != nil || len(accounts) == 0 {
			from = statedb.GetOrNewStateObject(common.Address{})
		} else {
			from = statedb.GetOrNewStateObject(accounts[0].Address)
		}
	} else {
		from = statedb.GetOrNewStateObject(common.HexToAddress(fromStr))
	}

	from.SetBalance(common.MaxBig)
	from.SetGasLimit(self.backend.ChainManager().GasLimit())
	msg := callmsg{
		from:     from,
		to:       common.HexToAddress(toStr),
		gas:      common.Big(gasStr),
		gasPrice: common.Big(gasPriceStr),
		value:    common.Big(valueStr),
		data:     common.FromHex(dataStr),
	}

	if msg.gas.Cmp(big.NewInt(0)) == 0 {
		msg.gas = DefaultGas()
	}

	if msg.gasPrice.Cmp(big.NewInt(0)) == 0 {
		msg.gasPrice = self.DefaultGasPrice()
	}

	header := self.CurrentBlock().Header()
	vmenv := core.NewEnv(statedb, self.backend.ChainManager(), msg, header)

	res, gas, err := core.ApplyMessage(vmenv, msg, from)
	return common.ToHex(res), gas.String(), err
}

func (self *XEth) ConfirmTransaction(tx string) bool {
	return self.frontend.ConfirmTransaction(tx)
}

func (self *XEth) doSign(from common.Address, hash common.Hash, didUnlock bool) ([]byte, error) {
	sig, err := self.backend.AccountManager().Sign(accounts.Account{Address: from}, hash.Bytes())
	if err == accounts.ErrLocked {
		if didUnlock {
			return nil, fmt.Errorf("signer account still locked after successful unlock")
		}
		if !self.frontend.UnlockAccount(from.Bytes()) {
			return nil, fmt.Errorf("could not unlock signer account")
		}
		// retry signing, the account should now be unlocked.
		return self.doSign(from, hash, true)
	} else if err != nil {
		return nil, err
	}
	return sig, nil
}

func (self *XEth) Sign(fromStr, hashStr string, didUnlock bool) (string, error) {
	var (
		from = common.HexToAddress(fromStr)
		hash = common.HexToHash(hashStr)
	)
	sig, err := self.doSign(from, hash, didUnlock)
	if err != nil {
		return "", err
	}
	return common.ToHex(sig), nil
}

func isAddress(addr string) bool {
	return addrReg.MatchString(addr)
}

func (self *XEth) Transact(fromStr, toStr, nonceStr, valueStr, gasStr, gasPriceStr, codeStr string) (string, error) {

	// this minimalistic recoding is enough (works for natspec.js)
	var jsontx = fmt.Sprintf(`{"params":[{"to":"%s","data": "%s"}]}`, toStr, codeStr)
	if !self.ConfirmTransaction(jsontx) {
		err := fmt.Errorf("Transaction not confirmed")
		return "", err
	}

	if len(toStr) > 0 && toStr != "0x" && !isAddress(toStr) {
		return "", errors.New("Invalid address")
	}

	var (
		from             = common.HexToAddress(fromStr)
		to               = common.HexToAddress(toStr)
		value            = common.Big(valueStr)
		gas              *big.Int
		price            *big.Int
		data             []byte
		contractCreation bool
	)

	if len(gasStr) == 0 {
		gas = DefaultGas()
	} else {
		gas = common.Big(gasStr)
	}

	if len(gasPriceStr) == 0 {
		price = self.DefaultGasPrice()
	} else {
		price = common.Big(gasPriceStr)
	}

	data = common.FromHex(codeStr)
	if len(toStr) == 0 {
		contractCreation = true
	}

	// 2015-05-18 Is this still needed?
	// TODO if no_private_key then
	//if _, exists := p.register[args.From]; exists {
	//	p.register[args.From] = append(p.register[args.From], args)
	//} else {
	/*
		account := accounts.Get(common.FromHex(args.From))
		if account != nil {
			if account.Unlocked() {
				if !unlockAccount(account) {
					return
				}
			}

			result, _ := account.Transact(common.FromHex(args.To), common.FromHex(args.Value), common.FromHex(args.Gas), common.FromHex(args.GasPrice), common.FromHex(args.Data))
			if len(result) > 0 {
				*reply = common.ToHex(result)
			}
		} else if _, exists := p.register[args.From]; exists {
			p.register[ags.From] = append(p.register[args.From], args)
		}
	*/

	// TODO: align default values to have the same type, e.g. not depend on
	// common.Value conversions later on
	var nonce uint64
	if len(nonceStr) != 0 {
		nonce = common.Big(nonceStr).Uint64()
	} else {
		state := self.backend.TxPool().State()
		nonce = state.GetNonce(from)
	}
	var tx *types.Transaction
	if contractCreation {
		tx = types.NewContractCreation(nonce, value, gas, price, data)
	} else {
		tx = types.NewTransaction(nonce, to, value, gas, price, data)
	}

	signed, err := self.sign(tx, from, false)
	if err != nil {
		return "", err
	}
	if err = self.backend.TxPool().Add(signed); err != nil {
		return "", err
	}

	if contractCreation {
		addr := crypto.CreateAddress(from, nonce)
		glog.V(logger.Info).Infof("Tx(%s) created: %s\n", signed.Hash().Hex(), addr.Hex())
	} else {
		glog.V(logger.Info).Infof("Tx(%s) to: %s\n", signed.Hash().Hex(), tx.To().Hex())
	}

	return signed.Hash().Hex(), nil
}

func (self *XEth) sign(tx *types.Transaction, from common.Address, didUnlock bool) (*types.Transaction, error) {
	hash := tx.SigHash()
	sig, err := self.doSign(from, hash, didUnlock)
	if err != nil {
		return tx, err
	}
	return tx.WithSignature(sig)
}

// callmsg is the message type used for call transations.
type callmsg struct {
	from          *state.StateObject
	to            common.Address
	gas, gasPrice *big.Int
	value         *big.Int
	data          []byte
}

// accessor boilerplate to implement core.Message
func (m callmsg) From() (common.Address, error) { return m.from.Address(), nil }
func (m callmsg) Nonce() uint64                 { return m.from.Nonce() }
func (m callmsg) To() *common.Address           { return &m.to }
func (m callmsg) GasPrice() *big.Int            { return m.gasPrice }
func (m callmsg) Gas() *big.Int                 { return m.gas }
func (m callmsg) Value() *big.Int               { return m.value }
func (m callmsg) Data() []byte                  { return m.data }

type logQueue struct {
	logs    state.Logs
	timeout time.Time
	id      int
}

func (l *logQueue) add(logs ...*state.Log) {
	l.logs = append(l.logs, logs...)
}

func (l *logQueue) get() state.Logs {
	l.timeout = time.Now()
	tmp := l.logs
	l.logs = nil
	return tmp
}

type hashQueue struct {
	hashes  []common.Hash
	timeout time.Time
	id      int
}

func (l *hashQueue) add(hashes ...common.Hash) {
	l.hashes = append(l.hashes, hashes...)
}

func (l *hashQueue) get() []common.Hash {
	l.timeout = time.Now()
	tmp := l.hashes
	l.hashes = nil
	return tmp
}
