package miner

import (
	"fmt"
	"math/big"
	"sort"

	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/eth"
	"github.com/ethereum/go-ethereum/ethutil"
	"github.com/ethereum/go-ethereum/event"
	"github.com/ethereum/go-ethereum/pow"
	"github.com/ethereum/go-ethereum/state"
	"gopkg.in/fatih/set.v0"
)

type environment struct {
	totalUsedGas *big.Int
	state        *state.StateDB
	coinbase     *state.StateObject
	block        *types.Block
	ancestors    *set.Set
	uncles       *set.Set
}

func env(block *types.Block, eth *eth.Ethereum) *environment {
	state := state.New(block.Root(), eth.Db())
	env := &environment{
		totalUsedGas: new(big.Int),
		state:        state,
		block:        block,
		ancestors:    set.New(),
		uncles:       set.New(),
		coinbase:     state.GetOrNewStateObject(block.Coinbase()),
	}
	for _, ancestor := range eth.ChainManager().GetAncestors(block, 7) {
		env.ancestors.Add(string(ancestor.Hash()))
	}

	return env
}

type Work struct {
	Number uint64
	Nonce  []byte
}

type Agent interface {
	Work() chan<- *types.Block
	SetNonceCh(chan<- Work)
	Stop()
	Pow() pow.PoW
}

type worker struct {
	agents []Agent
	recv   chan Work
	mux    *event.TypeMux
	quit   chan struct{}
	pow    pow.PoW

	eth      *eth.Ethereum
	chain    *core.ChainManager
	proc     *core.BlockProcessor
	coinbase []byte

	current *environment

	mining bool
}

func newWorker(coinbase []byte, eth *eth.Ethereum) *worker {
	return &worker{
		eth:      eth,
		mux:      eth.EventMux(),
		recv:     make(chan Work),
		chain:    eth.ChainManager(),
		proc:     eth.BlockProcessor(),
		coinbase: coinbase,
	}
}

func (self *worker) start() {
	self.mining = true

	self.quit = make(chan struct{})

	go self.update()
	go self.wait()
}

func (self *worker) stop() {
	self.mining = false

	close(self.quit)
}

func (self *worker) register(agent Agent) {
	self.agents = append(self.agents, agent)
	agent.SetNonceCh(self.recv)
}

func (self *worker) update() {
	events := self.mux.Subscribe(core.ChainEvent{}, core.TxPreEvent{})

out:
	for {
		select {
		case event := <-events.Chan():
			switch event := event.(type) {
			case core.ChainEvent:
				self.commitNewWork()
			case core.TxPreEvent:
				if err := self.commitTransaction(event.Tx); err != nil {
					self.push()
				}
			}
		case <-self.quit:
			break out
		}
	}

	events.Unsubscribe()
}

func (self *worker) wait() {
	for {
		for work := range self.recv {
			block := self.current.block
			if block.Number().Uint64() == work.Number && block.Nonce() == nil {
				self.current.block.Header().Nonce = work.Nonce

				self.chain.InsertChain(types.Blocks{self.current.block})
				self.mux.Post(core.NewMinedBlockEvent{self.current.block})
			}
			break
		}
	}
}

func (self *worker) push() {
	if self.mining {
		self.current.state.Update(ethutil.Big0)
		self.current.block.SetRoot(self.current.state.Root())

		for _, agent := range self.agents {
			agent.Work() <- self.current.block
		}
	}
}

func (self *worker) commitNewWork() {
	self.current = env(self.chain.NewBlock(self.coinbase), self.eth)
	parent := self.chain.GetBlock(self.current.block.ParentHash())
	self.current.coinbase.SetGasPool(core.CalcGasLimit(parent, self.current.block))

	transactions := self.eth.TxPool().GetTransactions()
	sort.Sort(types.TxByNonce{transactions})

	// Keep track of transactions which return errors so they can be removed
	var remove types.Transactions
	for _, tx := range transactions {
		err := self.commitTransaction(tx)
		switch {
		case core.IsNonceErr(err):
			remove = append(remove, tx)
		case core.IsGasLimitErr(err):
			// Break on gas limit
			break
		default:
			minerlogger.Infoln(err)
			remove = append(remove, tx)
		}
	}
	self.eth.TxPool().RemoveSet(remove)

	self.current.coinbase.AddAmount(core.BlockReward)

	self.push()
}

var (
	inclusionReward = new(big.Int).Div(core.BlockReward, big.NewInt(32))
	_uncleReward    = new(big.Int).Mul(core.BlockReward, big.NewInt(15))
	uncleReward     = new(big.Int).Div(_uncleReward, big.NewInt(16))
)

func (self *worker) commitUncle(uncle *types.Header) error {
	if self.current.uncles.Has(string(uncle.Hash())) {
		// Error not unique
		return core.UncleError("Uncle not unique")
	}
	self.current.uncles.Add(string(uncle.Hash()))

	if !self.current.ancestors.Has(string(uncle.ParentHash)) {
		return core.UncleError(fmt.Sprintf("Uncle's parent unknown (%x)", uncle.ParentHash[0:4]))
	}

	if !self.pow.Verify(types.NewBlockWithHeader(uncle)) {
		return core.ValidationError("Uncle's nonce is invalid (= %v)", ethutil.Bytes2Hex(uncle.Nonce))
	}

	uncleAccount := self.current.state.GetAccount(uncle.Coinbase)
	uncleAccount.AddAmount(uncleReward)

	self.current.coinbase.AddBalance(uncleReward)

	return nil
}

func (self *worker) commitTransaction(tx *types.Transaction) error {
	snapshot := self.current.state.Copy()
	receipt, txGas, err := self.proc.ApplyTransaction(self.current.coinbase, self.current.state, self.current.block, tx, self.current.totalUsedGas, true)
	if err != nil {
		self.current.state.Set(snapshot)

		return err
	}

	self.current.totalUsedGas.Add(self.current.totalUsedGas, txGas)
	self.current.block.AddTransaction(tx)
	self.current.block.AddReceipt(receipt)

	return nil
}
