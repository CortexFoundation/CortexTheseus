package core

import (
	"errors"
	"fmt"
	"sync"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/event"
	"github.com/ethereum/go-ethereum/logger"
)

var (
	txplogger = logger.NewLogger("TXP")

	ErrInvalidSender = errors.New("Invalid sender")
)

const txPoolQueueSize = 50

type TxPoolHook chan *types.Transaction
type TxMsg struct{ Tx *types.Transaction }

const (
	minGasPrice = 1000000
)

type TxProcessor interface {
	ProcessTransaction(tx *types.Transaction)
}

// The tx pool a thread safe transaction pool handler. In order to
// guarantee a non blocking pool we use a queue channel which can be
// independently read without needing access to the actual pool.
type TxPool struct {
	mu sync.RWMutex
	// Queueing channel for reading and writing incoming
	// transactions to
	queueChan chan *types.Transaction
	// Quiting channel
	quit chan bool
	// The actual pool
	//pool *list.List
	txs map[common.Hash]*types.Transaction

	SecondaryProcessor TxProcessor

	subscribers []chan TxMsg

	eventMux *event.TypeMux
}

func NewTxPool(eventMux *event.TypeMux) *TxPool {
	return &TxPool{
		txs:       make(map[common.Hash]*types.Transaction),
		queueChan: make(chan *types.Transaction, txPoolQueueSize),
		quit:      make(chan bool),
		eventMux:  eventMux,
	}
}

func (pool *TxPool) ValidateTransaction(tx *types.Transaction) error {
	// Validate sender
	if _, err := tx.From(); err != nil {
		return err
	}
	// Validate curve param
	v, _, _ := tx.Curve()
	if v > 28 || v < 27 {
		return fmt.Errorf("tx.v != (28 || 27) => %v", v)
	}
	return nil

	/* XXX this kind of validation needs to happen elsewhere in the gui when sending txs.
	   Other clients should do their own validation. Value transfer could throw error
	   but doesn't necessarily invalidate the tx. Gas can still be payed for and miner
	   can still be rewarded for their inclusion and processing.
	sender := pool.stateQuery.GetAccount(senderAddr)
	totAmount := new(big.Int).Set(tx.Value())
	// Make sure there's enough in the sender's account. Having insufficient
	// funds won't invalidate this transaction but simple ignores it.
	if sender.Balance().Cmp(totAmount) < 0 {
		return fmt.Errorf("Insufficient amount in sender's (%x) account", tx.From())
	}
	*/
}

func (self *TxPool) addTx(tx *types.Transaction) {
	self.txs[tx.Hash()] = tx
}

func (self *TxPool) add(tx *types.Transaction) error {
	if self.txs[tx.Hash()] != nil {
		return fmt.Errorf("Known transaction (%x)", tx.Hash()[0:4])
	}
	err := self.ValidateTransaction(tx)
	if err != nil {
		return err
	}

	self.addTx(tx)

	var toname string
	if to, valid := tx.To(); valid {
		toname = common.Bytes2Hex(to[:4])
	} else {
		toname = "[NEW_CONTRACT]"
	}
	// we can ignore the error here because From is
	// verified in ValidateTransaction.
	f, _ := tx.From()
	from := common.Bytes2Hex(f[:4])
	txplogger.Debugf("(t) %x => %s (%v) %x\n", from, to, tx.Value, tx.Hash())

	// Notify the subscribers
	go self.eventMux.Post(TxPreEvent{tx})

	return nil
}

func (self *TxPool) Size() int {
	return len(self.txs)
}

func (self *TxPool) Add(tx *types.Transaction) error {
	self.mu.Lock()
	defer self.mu.Unlock()
	return self.add(tx)
}

func (self *TxPool) AddTransactions(txs []*types.Transaction) {
	self.mu.Lock()
	defer self.mu.Unlock()

	for _, tx := range txs {
		if err := self.add(tx); err != nil {
			txplogger.Debugln(err)
		} else {
			txplogger.Debugf("tx %x\n", tx.Hash()[0:4])
		}
	}
}

func (self *TxPool) GetTransactions() (txs types.Transactions) {
	self.mu.RLock()
	defer self.mu.RUnlock()

	txs = make(types.Transactions, self.Size())
	i := 0
	for _, tx := range self.txs {
		txs[i] = tx
		i++
	}

	return
}

func (pool *TxPool) RemoveInvalid(query StateQuery) {
	pool.mu.Lock()

	var removedTxs types.Transactions
	for _, tx := range pool.txs {
		sender := query.GetAccount(tx.From())
		err := pool.ValidateTransaction(tx)
		if err != nil || sender.Nonce() >= tx.Nonce() {
			removedTxs = append(removedTxs, tx)
		}
	}
	pool.mu.Unlock()

	pool.RemoveSet(removedTxs)
}

func (self *TxPool) RemoveSet(txs types.Transactions) {
	self.mu.Lock()
	defer self.mu.Unlock()
	for _, tx := range txs {
		delete(self.txs, tx.Hash())
	}
}

func (pool *TxPool) Flush() {
	pool.txs = make(map[common.Hash]*types.Transaction)
}

func (pool *TxPool) Start() {
}

func (pool *TxPool) Stop() {
	pool.Flush()

	txplogger.Infoln("Stopped")
}
