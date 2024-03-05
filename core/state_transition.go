// Copyright 2018 The go-ethereum Authors
// This file is part of the CortexFoundation library.
//
// The CortexFoundation library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexFoundation library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the CortexFoundation library. If not, see <http://www.gnu.org/licenses/>.

package core

import (
	"errors"
	"fmt"
	"math"
	"math/big"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/core/vm"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/CortexTheseus/security"
)

// ExecutionResult includes all output after executing given evm
// message no matter the execution itself is successful or not.
type ExecutionResult struct {
	UsedGas     uint64 // Total used gas, not including the refunded gas
	Quota       uint64
	RefundedGas uint64 // Total gas refunded after execution
	Err         error  // Any error encountered during the execution(listed in core/vm/errors.go)
	ReturnData  []byte // Returned data from evm(function result or data supplied with revert opcode)
}

// Unwrap returns the internal evm error which allows us for further
// analysis outside.
func (result *ExecutionResult) Unwrap() error {
	return result.Err
}

// Failed returns the indicator whether the execution is successful or not
func (result *ExecutionResult) Failed() bool { return result.Err != nil }

// Return is a helper function to help caller distinguish between revert reason
// and function return. Return returns the data after execution if no error occurs.
func (result *ExecutionResult) Return() []byte {
	if result.Err != nil {
		return nil
	}
	return common.CopyBytes(result.ReturnData)
}

// Revert returns the concrete revert reason if the execution is aborted by `REVERT`
// opcode. Note the reason can be nil if no data supplied with revert opcode.
func (result *ExecutionResult) Revert() []byte {
	if result.Err != vm.ErrExecutionReverted {
		return nil
	}
	return common.CopyBytes(result.ReturnData)
}

var (
	errInsufficientBalanceForGas = errors.New("insufficient balance to pay for gas")
	//PER_UPLOAD_BYTES             uint64 = 10 * 512 * 1024
)

/*
The State Transitioning Model

A state transition is a change made when a transaction is applied to the current world state
The state transitioning model does all the necessary work to work out a valid new state root.

1) Nonce handling
2) Pre pay gas
3) Create a new state object if the recipient is \0*32
4) Value transfer
== If contract creation ==

	4a) Attempt to run transaction data
	4b) If valid, use result as code for the new state object

== end ==
5) Run Script section
6) Derive new state root
*/

// StateTransition is the state of current tx in vm
type StateTransition struct {
	gp         *GasPool
	qp         *QuotaPool
	msg        *Message
	gas        uint64
	initialGas uint64
	state      vm.StateDB
	cvm        *vm.CVM
	modelGas   map[common.Address]uint64
}

type Message struct {
	To                *common.Address
	From              common.Address
	Nonce             uint64
	Value             *big.Int
	GasLimit          uint64
	GasPrice          *big.Int
	Data              []byte
	SkipAccountChecks bool
}

// XXX Rename message to something less arbitrary?
// TransactionToMessage converts a transaction into a Message.
func TransactionToMessage(tx *types.Transaction, s types.Signer) (*Message, error) {
	msg := &Message{
		Nonce:             tx.Nonce(),
		GasLimit:          tx.Gas(),
		GasPrice:          new(big.Int).Set(tx.GasPrice()),
		To:                tx.To(),
		Value:             tx.Value(),
		Data:              tx.Data(),
		SkipAccountChecks: false,
	}

	var err error
	msg.From, err = types.Sender(s, tx)
	return msg, err
}

// IntrinsicGas computes the 'intrinsic gas' for a message with the given data.
func IntrinsicGas(data []byte, contractCreation, upload, isHomestead, isEIP2028 bool) (uint64, error) {
	// Set the starting gas for the raw transaction
	var gas uint64
	if contractCreation && isHomestead {
		gas = params.TxGasContractCreation
	} else {
		if upload {
			gas = params.UploadGas
		} else {
			gas = params.TxGas
		}
	}
	// Bump the required gas by the amount of transactional data
	if len(data) > 0 {
		// Zero and non-zero bytes are priced differently
		var nz uint64
		for _, byt := range data {
			if byt != 0 {
				nz++
			}
		}
		// Make sure we don't exceed uint64 for all data combinations
		nonZeroGas := params.TxDataNonZeroGasFrontier
		if isEIP2028 {
			nonZeroGas = params.TxDataNonZeroGasEIP2028
		}
		if (math.MaxUint64-gas)/nonZeroGas < nz {
			return 0, vm.ErrOutOfGas
		}
		gas += nz * nonZeroGas

		z := uint64(len(data)) - nz
		if (math.MaxUint64-gas)/params.TxDataZeroGas < z {
			return 0, vm.ErrOutOfGas
		}
		gas += z * params.TxDataZeroGas
	}
	return gas, nil
}

// NewStateTransition initialises and returns a new state transition object.
func NewStateTransition(cvm *vm.CVM, msg *Message, gp *GasPool, qp *QuotaPool) *StateTransition {
	return &StateTransition{
		gp:    gp,
		qp:    qp,
		cvm:   cvm,
		msg:   msg,
		state: cvm.StateDB,
	}
}

// ApplyMessage computes the new state by applying the given message
// against the old state within the environment.
//
// ApplyMessage returns the bytes returned by any CVM execution (if it took place),
// the gas used (which includes gas refunds) and an error if it failed. An error always
// indicates a core error meaning that the message would always fail for that particular
// state and would never be accepted within a block.
func ApplyMessage(cvm *vm.CVM, msg *Message, gp *GasPool, qp *QuotaPool) (*ExecutionResult, error) {
	return NewStateTransition(cvm, msg, gp, qp).TransitionDb()
}

// to returns the recipient of the message.
func (st *StateTransition) to() common.Address {
	if st.msg == nil || st.msg.To == nil /* contract creation */ {
		return common.Address{}
	}
	return *st.msg.To
}

//func (st *StateTransition) useGas(amount uint64) error {
//	if st.gas < amount {
//		return vm.ErrOutOfGas
//	}
//	st.gas -= amount
//
//	return nil
//}

func (st *StateTransition) buyGas() error {
	mgval := new(big.Int).Mul(new(big.Int).SetUint64(st.msg.GasLimit), st.msg.GasPrice)
	if have, want := st.state.GetBalance(st.msg.From), mgval; have.Cmp(want) < 0 {
		return fmt.Errorf("%w: address %v have %v want %v gas %v price %v", errInsufficientBalanceForGas, st.msg.From.Hex(), have, want, st.msg.GasLimit, st.msg.GasPrice)
	}
	if err := st.gp.SubGas(st.msg.GasLimit); err != nil {
		return err
	}
	st.gas = st.msg.GasLimit

	st.initialGas = st.msg.GasLimit
	st.state.SubBalance(st.msg.From, mgval)
	return nil
}

// var confirmTime = params.CONFIRM_TIME * time.Second //-3600 * 24 * 30 * time.Second
func (st *StateTransition) preCheck() error {
	// Make sure this transaction's nonce is correct.
	if !st.msg.SkipAccountChecks {
		stNonce := st.state.GetNonce(st.msg.From)
		if msgNonce := st.msg.Nonce; stNonce < msgNonce {
			return fmt.Errorf("%w: address %v, tx: %d state: %d", ErrNonceTooHigh, st.msg.From.Hex(), msgNonce, stNonce)
		} else if stNonce > msgNonce {
			return fmt.Errorf("%w: address %v, tx: %d state: %d", ErrNonceTooLow, st.msg.From.Hex(), msgNonce, stNonce)
		} else if stNonce+1 < stNonce {
			return fmt.Errorf("%w: address %v, nonce: %d", ErrNonceMax, st.msg.From.Hex(), stNonce)
		}
	}

	if err := st.preQuotaCheck(); err != nil {
		return err
	}

	return st.buyGas()
}

/*const interv = 5

func (st *StateTransition) TorrentSync(meta common.Address, dir string, errCh chan error) {
	street := big.NewInt(0).Sub(st.cvm.PeekNumber, st.cvm.BlockNumber)
	point := big.NewInt(time.Now().Add(confirmTime).Unix())
	if point.Cmp(st.cvm.Context.Time) > 0 || street.Cmp(big.NewInt(params.CONFIRM_BLOCKS)) > 0 {
		duration := big.NewInt(0).Sub(big.NewInt(time.Now().Unix()), st.cvm.Context.Time)
		cost := big.NewInt(0)
		for i := 0; i < 3600 && duration.Cmp(cost) > 0; i++ {
			if !torrentfs.ExistTorrent(meta.Hex()) {
				log.Warn("Torrent synchronizing ... ...", "tvm", st.cvm.Context.Time, "duration", duration, "ago", common.PrettyDuration(time.Duration(duration.Uint64()*1000000000)), "level", i, "number", st.cvm.BlockNumber, "cost", cost, "peek", st.cvm.PeekNumber, "street", street)
				cost.Add(cost, big.NewInt(interv))
				time.Sleep(time.Second * interv)
				continue
			} else {
				log.Debug("Torrent has been found", "address", st.to(), "number", st.state.GetNum(st.to()), "current", st.cvm.BlockNumber, "meta", meta, "storage", dir, "level", i, "duration", duration, "ago", common.PrettyDuration(time.Duration(duration.Uint64()*1000000000)), "cost", cost)
				errCh <- nil
				return
			}
		}

		log.Error("Torrent synchronized timeout", "address", st.to(), "number", st.state.GetNum(st.to()), "current", st.cvm.BlockNumber, "meta", meta, "storage", dir, "street", street, "duration", duration, "cost", cost)
	} else {
		if !torrentfs.ExistTorrent(meta.Hex()) {
			log.Warn("Torrent not exist", "address", st.to(), "number", st.state.GetNum(st.to()), "current", st.cvm.BlockNumber, "meta", meta, "storage", dir)
			errCh <- ErrUnhandleTx
			return
		} else {
			errCh <- nil
			return
		}
	}

	if !torrentfs.ExistTorrent(meta.Hex()) {
		log.Error("Torrent synchronized failed", "address", st.to(), "number", st.state.GetNum(st.to()), "current", st.cvm.BlockNumber, "meta", meta, "storage", dir, "street", street)
		errCh <- ErrUnhandleTx
		return
	} else {
		errCr <- nil
		return
	}
}*/
// TransitionDb will transition the state by applying the current message and
// returning the result including the used gas. It returns an error if failed.
// An error indicates a consensus issue.
func (st *StateTransition) TransitionDb() (*ExecutionResult, error) {
	if err := st.preCheck(); err != nil {
		return nil, err
	}

	msg := st.msg
	sender := vm.AccountRef(msg.From)
	homestead := st.cvm.ChainConfig().IsHomestead(st.cvm.Context.BlockNumber)
	istanbul := st.cvm.ChainConfig().IsIstanbul(st.cvm.Context.BlockNumber)
	//matureBlockNumber := st.cvm.ChainConfig().GetMatureBlock()
	contractCreation := msg.To == nil

	/*if st.uploading() {
		if st.qp.Cmp(st.state.Upload(st.to())) < 0 {
			return nil, 0, big0, false,ErrQuotaLimitReached
		}
	}*/

	// Pay intrinsic gas
	gas, err := IntrinsicGas(msg.Data, contractCreation, st.uploading(), homestead, istanbul)
	if err != nil {
		return nil, err
	}
	if st.gas < gas {
		return nil, fmt.Errorf("%w: have %d, want %d", vm.ErrOutOfGas, st.gas, gas)
	}
	st.gas -= gas

	if msg.Value.Sign() > 0 && !st.cvm.Context.CanTransfer(st.state, msg.From, msg.Value) {
		return nil, fmt.Errorf("%w: address %v", ErrInsufficientFundsForTransfer, msg.From.Hex())
	}

	if blocked, num := security.IsBlocked(msg.From); blocked && st.cvm.Context.BlockNumber.Cmp(big.NewInt(num)) >= 0 {
		log.Debug("Bad address encounter!!", "addr", msg.From, "number", num)
		return nil, fmt.Errorf("%w: address %v", errors.New("Bad address encounter"), msg.From.Hex())
	}

	var (
		// vm errors do not effect consensus and are therefor
		// not assigned to err, except for insufficient balance
		// error.
		ret   []byte
		vmerr error
	)
	if contractCreation {
		ret, _, st.gas, st.modelGas, vmerr = st.cvm.Create(sender, msg.Data, st.gas, msg.Value)
	} else {
		// Increment the nonce for the next transaction
		//if pool.config.NoInfers && asm.HasInferOp(tx.Data()) {
		//	fmt.Println("Has INFER operation !!! continue ...")
		//}
		st.state.SetNonce(msg.From, st.state.GetNonce(sender.Address())+1)
		ret, st.gas, st.modelGas, vmerr = st.cvm.Call(sender, st.to(), msg.Data, st.gas, msg.Value)
	}

	if vmerr != nil {
		if vmerr == vm.ErrRuntime {
			return nil, vmerr
		}

		log.Debug("VM returned with error", "err", vmerr, "number", st.cvm.Context.BlockNumber, "from", msg.From.Hex())

		// The only possible consensus-error would be if there wasn't
		// sufficient balance to make the transfer happen. The first
		// balance transfer may never fail.
		if vmerr == vm.ErrInsufficientBalance {
			return nil, vmerr
		}

		//if vmerr == vm.ErrMetaInfoNotMature {
		//	return nil, 0, big0, false, vmerr
		//}
	}

	// cortex quota calculate
	var quota uint64 //default used 4 k quota every tx for testing
	if vmerr == nil {
		if quota, err = st.quotaCalculate(); err != nil {
			return nil, err
		}
	}

	// gas cost below this line
	var (
		gasRefund uint64
		gu        uint64
	)
	gasRefund = st.refundGas()

	// model gas calculate
	gu = st.gasUsed()
	//if (vmerr == nil || vmerr == vm.ErrOutOfGas) && st.modelGas != nil && len(st.modelGas) > 0 { //pay ctx to the model authors by the model gas * current price
	if vmerr == nil || (st.cvm.ChainConfig().ChainID.Uint64() == 21 && st.cvm.Context.BlockNumber.Cmp(big.NewInt(16000)) < 0 && vmerr == vm.ErrOutOfGas) {
		/*for addr, mgas := range st.modelGas {
			if mgas > params.MODEL_GAS_UP_LIMIT {
				continue
			}

			if gu < mgas {
				return nil, vm.ErrInsufficientBalance
			}

			gu -= mgas
			reward := new(big.Int).Mul(new(big.Int).SetUint64(mgas), msg.GasPrice)
			log.Debug("Model author reward", "author", addr.Hex(), "reward", reward, "number", st.cvm.Context.BlockNumber)
			st.state.AddBalance(addr, reward)
		}*/
		if gu, err = st.modelGasCalculate(gu); err != nil {
			return nil, err
		}
	}

	// normal gas
	st.state.AddBalance(st.cvm.Context.Coinbase, new(big.Int).Mul(new(big.Int).SetUint64(gu), msg.GasPrice))

	return &ExecutionResult{
		UsedGas:     st.gasUsed(),
		Quota:       quota,
		RefundedGas: gasRefund,
		Err:         vmerr,
		ReturnData:  ret,
	}, nil
}

// vote to model
func (st *StateTransition) uploading() bool {
	return st.msg != nil && st.msg.To != nil && st.msg.Value.Sign() == 0 && st.state.Uploading(st.to()) // && st.gas >= params.UploadGas
}

func (st *StateTransition) refundGas() uint64 {
	// Apply refund counter, capped to half of the used gas.
	refund := st.gasUsed() / 2
	if refund > st.state.GetRefund() {
		refund = st.state.GetRefund()
	}
	st.gas += refund

	// Return ETH for remaining gas, exchanged at the original rate.
	remaining := new(big.Int).Mul(new(big.Int).SetUint64(st.gas), st.msg.GasPrice)
	st.state.AddBalance(st.msg.From, remaining)

	// Also return remaining gas to the block gas counter so it is
	// available for the next transaction.
	st.gp.AddGas(st.gas)

	return refund
}

// gasUsed returns the amount of gas used up by the state transition.
func (st *StateTransition) gasUsed() uint64 {
	return st.initialGas - st.gas
}
