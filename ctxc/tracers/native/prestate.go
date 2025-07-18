// Copyright 2022 The go-ethereum Authors
// This file is part of The go-ethereum library.
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
// along with The go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package native

import (
	"bytes"
	"encoding/json"
	"math/big"
	"sync/atomic"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/core/vm"
	"github.com/CortexFoundation/CortexTheseus/crypto"
	"github.com/CortexFoundation/CortexTheseus/ctxc/tracers"
)

//go:generate go run github.com/fjl/gencodec -type account -field-override accountMarshaling -out gen_account_json.go

func init() {
	register("prestateTracer", newPrestateTracer)
}

type state = map[common.Address]*account

type account struct {
	Balance *big.Int                    `json:"balance,omitempty"`
	Code    []byte                      `json:"code,omitempty"`
	Nonce   uint64                      `json:"nonce,omitempty"`
	Storage map[common.Hash]common.Hash `json:"storage,omitempty"`
	empty   bool
}

func (a *account) exists() bool {
	return a.Nonce > 0 || len(a.Code) > 0 || len(a.Storage) > 0 || (a.Balance != nil && a.Balance.Sign() != 0)
}

type accountMarshaling struct {
	Balance *hexutil.Big
	Code    hexutil.Bytes
}

type prestateTracer struct {
	env       *vm.CVM
	pre       state
	post      state
	create    bool
	to        common.Address
	gasLimit  uint64 // Amount of gas bought for the whole tx
	config    prestateTracerConfig
	interrupt uint32 // Atomic flag to signal execution interruption
	reason    error  // Textual reason for the interruption
	created   map[common.Address]bool
	deleted   map[common.Address]bool
}

type prestateTracerConfig struct {
	DiffMode       bool `json:"diffMode"`       // If true, this tracer will return state modifications
	DisableCode    bool `json:"disableCode"`    // If true, this tracer will not return the contract code
	DisableStorage bool `json:"disableStorage"` // If true, this tracer will not return the contract storage
}

func newPrestateTracer(ctx *tracers.Context, cfg json.RawMessage) (tracers.Tracer, error) {
	var config prestateTracerConfig
	if cfg != nil {
		if err := json.Unmarshal(cfg, &config); err != nil {
			return nil, err
		}
	}
	return &prestateTracer{
		pre:     state{},
		post:    state{},
		config:  config,
		created: make(map[common.Address]bool),
		deleted: make(map[common.Address]bool),
	}, nil
}

// CaptureStart implements the CVMLogger interface to initialize the tracing operation.
func (t *prestateTracer) CaptureStart(env *vm.CVM, from common.Address, to common.Address, create bool, input []byte, gas uint64, value *big.Int) {
	t.env = env
	t.create = create
	t.to = to

	t.lookupAccount(from)
	t.lookupAccount(to)
	t.lookupAccount(env.Context.Coinbase)

	// The recipient balance includes the value transferred.
	toBal := new(big.Int).Sub(t.pre[to].Balance, value)
	t.pre[to].Balance = toBal
	if env.ChainConfig().Rules(env.Context.BlockNumber, env.Context.Random != nil, env.Context.Time).IsEIP158 && create {
		t.pre[to].Nonce--
	}

	// The sender balance is after reducing: value and gasLimit.
	// We need to re-add them to get the pre-tx balance.
	fromBal := new(big.Int).Set(t.pre[from].Balance)
	gasPrice := env.TxContext.GasPrice
	consumedGas := new(big.Int).Mul(gasPrice, new(big.Int).SetUint64(t.gasLimit))
	fromBal.Add(fromBal, new(big.Int).Add(value, consumedGas))
	t.pre[from].Balance = fromBal
	t.pre[from].Nonce--

	if create && t.config.DiffMode {
		t.created[to] = true
	}
}

// CaptureEnd is called after the call finishes to finalize the tracing.
func (t *prestateTracer) CaptureEnd(output []byte, gasUsed uint64, err error) {
	if t.config.DiffMode {
		return
	}

	if t.create {
		// Keep existing account prior to contract creation at that address
		if s := t.pre[t.to]; s != nil && s.empty {
			// Exclude newly created contract.
			delete(t.pre, t.to)
		}
	}
}

// CaptureState implements the CVMLogger interface to trace a single step of VM execution.
func (t *prestateTracer) CaptureState(pc uint64, op vm.OpCode, gas, cost uint64, scope *vm.ScopeContext, rData []byte, depth int, err error) {
	if err != nil {
		return
	}
	stack := scope.Stack
	stackData := stack.Data()
	stackLen := len(stackData)
	caller := scope.Contract.Address()
	switch {
	case stackLen >= 1 && (op == vm.SLOAD || op == vm.SSTORE):
		slot := common.Hash(stackData[stackLen-1].Bytes32())
		t.lookupStorage(caller, slot)
	case stackLen >= 1 && (op == vm.EXTCODECOPY || op == vm.EXTCODEHASH || op == vm.EXTCODESIZE || op == vm.BALANCE || op == vm.SELFDESTRUCT):
		addr := common.Address(stackData[stackLen-1].Bytes20())
		t.lookupAccount(addr)
		if op == vm.SELFDESTRUCT {
			t.deleted[caller] = true
		}
	case stackLen >= 5 && (op == vm.DELEGATECALL || op == vm.CALL || op == vm.STATICCALL || op == vm.CALLCODE):
		addr := common.Address(stackData[stackLen-2].Bytes20())
		t.lookupAccount(addr)
	case op == vm.CREATE:
		nonce := t.env.StateDB.GetNonce(caller)
		addr := crypto.CreateAddress(caller, nonce)
		t.lookupAccount(addr)
		t.created[addr] = true
	case stackLen >= 4 && op == vm.CREATE2:
		offset := stackData[stackLen-2]
		size := stackData[stackLen-3]
		init := scope.Memory.GetCopy(offset.Uint64(), size.Uint64())
		inithash := crypto.Keccak256(init)
		salt := stackData[stackLen-4]
		addr := crypto.CreateAddress2(caller, salt.Bytes32(), inithash)
		t.lookupAccount(addr)
		t.created[addr] = true
	}
}

// CaptureFault implements the CVMLogger interface to trace an execution fault.
func (t *prestateTracer) CaptureFault(pc uint64, op vm.OpCode, gas, cost uint64, _ *vm.ScopeContext, depth int, err error) {
}

// CaptureEnter is called when CVM enters a new scope (via call, create or selfdestruct).
func (t *prestateTracer) CaptureEnter(typ vm.OpCode, from common.Address, to common.Address, input []byte, gas uint64, value *big.Int) {
}

// CaptureExit is called when CVM exits a scope, even if the scope didn't
// execute any code.
func (t *prestateTracer) CaptureExit(output []byte, gasUsed uint64, err error) {
}

func (t *prestateTracer) CaptureTxStart(gasLimit uint64) {
	t.gasLimit = gasLimit
}

func (t *prestateTracer) CaptureTxEnd(restGas uint64) {
	if !t.config.DiffMode {
		return
	}

	for addr, state := range t.pre {
		// The deleted account's state is pruned from `post` but kept in `pre`
		if _, ok := t.deleted[addr]; ok {
			continue
		}
		modified := false
		postAccount := &account{Storage: make(map[common.Hash]common.Hash)}
		newBalance := t.env.StateDB.GetBalance(addr)
		newNonce := t.env.StateDB.GetNonce(addr)

		if newBalance.Cmp(t.pre[addr].Balance) != 0 {
			modified = true
			postAccount.Balance = newBalance
		}
		if newNonce != t.pre[addr].Nonce {
			modified = true
			postAccount.Nonce = newNonce
		}
		if !t.config.DisableCode {
			newCode := t.env.StateDB.GetCode(addr)
			if !bytes.Equal(newCode, t.pre[addr].Code) {
				modified = true
				postAccount.Code = newCode
			}
		}

		if !t.config.DisableStorage {
			for key, val := range state.Storage {
				// don't include the empty slot
				if val == (common.Hash{}) {
					delete(t.pre[addr].Storage, key)
				}

				newVal := t.env.StateDB.GetState(addr, key)
				if val == newVal {
					// Omit unchanged slots
					delete(t.pre[addr].Storage, key)
				} else {
					modified = true
					if newVal != (common.Hash{}) {
						postAccount.Storage[key] = newVal
					}
				}
			}
		}

		if modified {
			t.post[addr] = postAccount
		} else {
			// if state is not modified, then no need to include into the pre state
			delete(t.pre, addr)
		}
	}

	// Remove accounts that were empty prior to execution. Unless
	// user requested to include empty accounts.
	for addr, s := range t.pre {
		if s.empty {
			delete(t.pre, addr)
		}
	}
}

// GetResult returns the json-encoded nested list of call traces, and any
// error arising from the encoding or forceful termination (via `Stop`).
func (t *prestateTracer) GetResult() (json.RawMessage, error) {
	var res []byte
	var err error
	if t.config.DiffMode {
		res, err = json.Marshal(struct {
			Post state `json:"post"`
			Pre  state `json:"pre"`
		}{t.post, t.pre})
	} else {
		res, err = json.Marshal(t.pre)
	}
	if err != nil {
		return nil, err
	}
	return json.RawMessage(res), t.reason
}

// Stop terminates execution of the tracer at the first opportune moment.
func (t *prestateTracer) Stop(err error) {
	t.reason = err
	atomic.StoreUint32(&t.interrupt, 1)
}

// lookupAccount fetches details of an account and adds it to the prestate
// if it doesn't exist there.
func (t *prestateTracer) lookupAccount(addr common.Address) {
	if _, ok := t.pre[addr]; ok {
		return
	}

	acc := &account{
		Balance: t.env.StateDB.GetBalance(addr),
		Nonce:   t.env.StateDB.GetNonce(addr),
		Code:    t.env.StateDB.GetCode(addr),
	}
	if !acc.exists() {
		acc.empty = true
	}
	// The code must be fetched first for the emptiness check.
	if t.config.DisableCode {
		acc.Code = nil
	}
	if !t.config.DisableStorage {
		acc.Storage = make(map[common.Hash]common.Hash)
	}
	t.pre[addr] = acc
}

// lookupStorage fetches the requested storage slot and adds
// it to the prestate of the given contract. It assumes `lookupAccount`
// has been performed on the contract before.
func (t *prestateTracer) lookupStorage(addr common.Address, key common.Hash) {
	if t.config.DisableStorage {
		return
	}
	if _, ok := t.pre[addr].Storage[key]; ok {
		return
	}
	t.pre[addr].Storage[key] = t.env.StateDB.GetState(addr, key)
}
