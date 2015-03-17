package core

import (
	"math/big"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/state"
	"github.com/ethereum/go-ethereum/vm"
)

type VMEnv struct {
	state *state.StateDB
	block *types.Block
	msg   Message
	depth int
	chain *ChainManager
	typ   vm.Type
}

func NewEnv(state *state.StateDB, chain *ChainManager, msg Message, block *types.Block) *VMEnv {
	return &VMEnv{
		chain: chain,
		state: state,
		block: block,
		msg:   msg,
		typ:   vm.StdVmTy,
	}
}

func (self *VMEnv) Origin() common.Address   { return self.msg.From() }
func (self *VMEnv) BlockNumber() *big.Int    { return self.block.Number() }
func (self *VMEnv) Coinbase() common.Address { return self.block.Coinbase() }
func (self *VMEnv) Time() int64              { return self.block.Time() }
func (self *VMEnv) Difficulty() *big.Int     { return self.block.Difficulty() }
func (self *VMEnv) GasLimit() *big.Int       { return self.block.GasLimit() }
func (self *VMEnv) Value() *big.Int          { return self.msg.Value() }
func (self *VMEnv) State() *state.StateDB    { return self.state }
func (self *VMEnv) Depth() int               { return self.depth }
func (self *VMEnv) SetDepth(i int)           { self.depth = i }
func (self *VMEnv) VmType() vm.Type          { return self.typ }
func (self *VMEnv) SetVmType(t vm.Type)      { self.typ = t }
func (self *VMEnv) GetHash(n uint64) common.Hash {
	if block := self.chain.GetBlockByNumber(n); block != nil {
		return block.Hash()
	}

	return common.Hash{}
}
func (self *VMEnv) AddLog(log state.Log) {
	self.state.AddLog(log)
}
func (self *VMEnv) Transfer(from, to vm.Account, amount *big.Int) error {
	return vm.Transfer(from, to, amount)
}

func (self *VMEnv) vm(addr *common.Address, data []byte, gas, price, value *big.Int) *Execution {
	return NewExecution(self, addr, data, gas, price, value)
}

func (self *VMEnv) Call(me vm.ContextRef, addr common.Address, data []byte, gas, price, value *big.Int) ([]byte, error) {
	exe := self.vm(&addr, data, gas, price, value)
	return exe.Call(addr, me)
}
func (self *VMEnv) CallCode(me vm.ContextRef, addr common.Address, data []byte, gas, price, value *big.Int) ([]byte, error) {
	maddr := me.Address()
	exe := self.vm(&maddr, data, gas, price, value)
	return exe.Call(addr, me)
}

func (self *VMEnv) Create(me vm.ContextRef, data []byte, gas, price, value *big.Int) ([]byte, error, vm.ContextRef) {
	exe := self.vm(nil, data, gas, price, value)
	return exe.Create(me)
}
