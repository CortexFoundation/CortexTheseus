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

package vm

import (
	"errors"
	_ "fmt"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/math"
	"github.com/CortexFoundation/CortexTheseus/params"
	// "github.com/CortexFoundation/CortexTheseus/core/types"
)

// memoryGasCosts calculates the quadratic gas for memory expansion. It does so
// only for the memory region that is expanded, not the total memory.
func memoryGasCost(mem *Memory, newMemSize uint64) (uint64, error) {

	if newMemSize == 0 {
		return 0, nil
	}
	// The maximum that will fit in a uint64 is max_word_count - 1
	// anything above that will result in an overflow.
	// Additionally, a newMemSize which results in a
	// newMemSizeWords larger than 0x7ffffffff will cause the square operation
	// to overflow.
	// The constant 0xffffffffe0 is the highest number that can be used without
	// overflowing the gas calculation
	if newMemSize > 0x1FFFFFFFE0 {
		return 0, ErrGasUintOverflow
	}

	newMemSizeWords := toWordSize(newMemSize)
	newMemSize = newMemSizeWords * 32
	if newMemSize > uint64(mem.Len()) {
		square := newMemSizeWords * newMemSizeWords
		linCoef := newMemSizeWords * params.MemoryGas
		quadCoef := square / params.QuadCoeffDiv
		newTotalFee := linCoef + quadCoef

		fee := newTotalFee - mem.lastGasCost
		mem.lastGasCost = newTotalFee

		return fee, nil
	}
	return 0, nil
}

func constGasFunc(gas uint64) gasFunc {
	return func(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
		return gas, nil
	}
}

// memoryCopierGas creates the gas functions for the following opcodes, and takes
// the stack position of the operand which determines the size of the data to copy
// as argument:
// CALLDATACOPY (stack position 2)
// CODECOPY (stack position 2)
// EXTCODECOPY (stack poition 3)
// RETURNDATACOPY (stack position 2)
func memoryCopierGas(stackpos int) gasFunc {
	return func(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
		// Gas for expanding the memory
		gas, err := memoryGasCost(mem, memorySize)
		if err != nil {
			return 0, err
		}
		var overflow bool
		if gas, overflow = math.SafeAdd(gas, GasFastestStep); overflow {
			return 0, ErrGasUintOverflow
		}
		// And gas for copying data, charged per word at param.CopyGas
		words, overflow := stack.Back(stackpos).Uint64WithOverflow()
		if overflow {
			return 0, ErrGasUintOverflow
		}

		if words, overflow = math.SafeMul(toWordSize(words), params.CopyGas); overflow {
			return 0, ErrGasUintOverflow
		}

		if gas, overflow = math.SafeAdd(gas, words); overflow {
			return 0, ErrGasUintOverflow
		}
		return gas, nil
	}
}

var (
	// gasCallDataCopy = memoryCopierGas(2)
	// gasCodeCopy     = memoryCopierGas(2)
	// gasExtCodeCopy    = memoryCopierGas(3)
	// gasReturnDataCopy = memoryCopierGas(2)
	gasMcopy = memoryCopierGas(2)
)

func gasCallDataCopy(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	gas, err := memoryGasCost(mem, memorySize)
	if err != nil {
		return 0, err
	}

	var overflow bool
	if gas, overflow = math.SafeAdd(gas, GasFastestStep); overflow {
		return 0, ErrGasUintOverflow
	}

	words, overflow := stack.Back(2).Uint64WithOverflow()
	if overflow {
		return 0, ErrGasUintOverflow
	}

	if words, overflow = math.SafeMul(toWordSize(words), params.CopyGas); overflow {
		return 0, ErrGasUintOverflow
	}

	if gas, overflow = math.SafeAdd(gas, words); overflow {
		return 0, ErrGasUintOverflow
	}
	return gas, nil
}

func gasReturnDataCopy(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	gas, err := memoryGasCost(mem, memorySize)
	if err != nil {
		return 0, err
	}

	var overflow bool
	if gas, overflow = math.SafeAdd(gas, GasFastestStep); overflow {
		return 0, ErrGasUintOverflow
	}

	words, overflow := stack.Back(2).Uint64WithOverflow()
	if overflow {
		return 0, ErrGasUintOverflow
	}

	if words, overflow = math.SafeMul(toWordSize(words), params.CopyGas); overflow {
		return 0, ErrGasUintOverflow
	}

	if gas, overflow = math.SafeAdd(gas, words); overflow {
		return 0, ErrGasUintOverflow
	}
	return gas, nil
}

func gasSStore(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	var (
		y, x    = stack.Back(1), stack.Back(0)
		current = cvm.StateDB.GetState(contract.Address(), x.Bytes32())
	)
	// The legacy gas metering only takes into consideration the current state
	// Legacy rules should be applied if we are in Petersburg (removal of EIP-1283)
	// OR Constantinople is not active
	if cvm.chainRules.IsPetersburg || !cvm.chainRules.IsConstantinople {
		// This checks for 3 scenario's and calculates gas accordingly:
		//
		// 1. From a zero-value address to a non-zero value         (NEW VALUE)
		// 2. From a non-zero value address to a zero-value address (DELETE)
		// 3. From a non-zero to a non-zero                         (CHANGE)
		switch {
		case current == (common.Hash{}) && y.Sign() != 0: // 0 => non 0
			return params.SstoreSetGas, nil
		case current != (common.Hash{}) && y.Sign() == 0: // non 0 => 0
			cvm.StateDB.AddRefund(params.SstoreRefundGas)
			return params.SstoreClearGas, nil
		default: // non 0 => non 0 (or 0 => 0)
			return params.SstoreResetGas, nil
		}
	}
	// The new gas metering is based on net gas costs (EIP-1283):
	//
	// 1. If current value equals new value (this is a no-op), 200 gas is deducted.
	// 2. If current value does not equal new value
	//   2.1. If original value equals current value (this storage slot has not been changed by the current execution context)
	//     2.1.1. If original value is 0, 20000 gas is deducted.
	// 	   2.1.2. Otherwise, 5000 gas is deducted. If new value is 0, add 15000 gas to refund counter.
	// 	2.2. If original value does not equal current value (this storage slot is dirty), 200 gas is deducted. Apply both of the following clauses.
	// 	  2.2.1. If original value is not 0
	//       2.2.1.1. If current value is 0 (also means that new value is not 0), remove 15000 gas from refund counter. We can prove that refund counter will never go below 0.
	//       2.2.1.2. If new value is 0 (also means that current value is not 0), add 15000 gas to refund counter.
	// 	  2.2.2. If original value equals new value (this storage slot is reset)
	//       2.2.2.1. If original value is 0, add 19800 gas to refund counter.
	// 	     2.2.2.2. Otherwise, add 4800 gas to refund counter.
	value := common.Hash(y.Bytes32())
	if current == value { // noop (1)
		return params.NetSstoreNoopGas, nil
	}
	original := cvm.StateDB.GetCommittedState(contract.Address(), x.Bytes32())
	if original == current {
		if original == (common.Hash{}) { // create slot (2.1.1)
			return params.NetSstoreInitGas, nil
		}
		if value == (common.Hash{}) { // delete slot (2.1.2b)
			cvm.StateDB.AddRefund(params.NetSstoreClearRefund)
		}
		return params.NetSstoreCleanGas, nil // write existing slot (2.1.2)
	}
	if original != (common.Hash{}) {
		if current == (common.Hash{}) { // recreate slot (2.2.1.1)
			cvm.StateDB.SubRefund(params.NetSstoreClearRefund)
		} else if value == (common.Hash{}) { // delete slot (2.2.1.2)
			cvm.StateDB.AddRefund(params.NetSstoreClearRefund)
		}
	}
	if original == value {
		if original == (common.Hash{}) { // reset to original inexistent slot (2.2.2.1)
			cvm.StateDB.AddRefund(params.NetSstoreResetClearRefund)
		} else { // reset to original existing slot (2.2.2.2)
			cvm.StateDB.AddRefund(params.NetSstoreResetRefund)
		}
	}
	return params.NetSstoreDirtyGas, nil
}

//  0. If *gasleft* is less than or equal to 2300, fail the current call.
//  1. If current value equals new value (this is a no-op), SLOAD_GAS is deducted.
//  2. If current value does not equal new value:
//     2.1. If original value equals current value (this storage slot has not been changed by the current execution context):
//     2.1.1. If original value is 0, SSTORE_SET_GAS (20K) gas is deducted.
//     2.1.2. Otherwise, SSTORE_RESET_GAS gas is deducted. If new value is 0, add SSTORE_CLEARS_SCHEDULE to refund counter.
//     2.2. If original value does not equal current value (this storage slot is dirty), SLOAD_GAS gas is deducted. Apply both of the following clauses:
//     2.2.1. If original value is not 0:
//     2.2.1.1. If current value is 0 (also means that new value is not 0), subtract SSTORE_CLEARS_SCHEDULE gas from refund counter.
//     2.2.1.2. If new value is 0 (also means that current value is not 0), add SSTORE_CLEARS_SCHEDULE gas to refund counter.
//     2.2.2. If original value equals new value (this storage slot is reset):
//     2.2.2.1. If original value is 0, add SSTORE_INIT_REFUND to refund counter.
//     2.2.2.2. Otherwise, add SSTORE_CLEAN_REFUND gas to refund counter.
func gasSStoreEIP2200(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	// If we fail the minimum gas availability invariant, fail (0)
	if contract.Gas <= params.SstoreSentryGasEIP2200 {
		return 0, errors.New("not enough gas for reentrancy sentry")
	}
	// Gas sentry honoured, do the actual gas calculation based on the stored value
	var (
		y, x    = stack.Back(1), stack.Back(0)
		current = cvm.StateDB.GetState(contract.Address(), x.Bytes32())
	)
	value := common.Hash(y.Bytes32())

	if current == value { // noop (1)
		return params.SloadGasEIP2200, nil
	}
	original := cvm.StateDB.GetCommittedState(contract.Address(), x.Bytes32())
	if original == current {
		if original == (common.Hash{}) { // create slot (2.1.1)
			return params.SstoreSetGasEIP2200, nil
		}
		if value == (common.Hash{}) { // delete slot (2.1.2b)
			cvm.StateDB.AddRefund(params.SstoreClearsScheduleRefundEIP2200)
		}
		return params.SstoreResetGasEIP2200, nil // write existing slot (2.1.2)
	}
	if original != (common.Hash{}) {
		if current == (common.Hash{}) { // recreate slot (2.2.1.1)
			cvm.StateDB.SubRefund(params.SstoreClearsScheduleRefundEIP2200)
		} else if value == (common.Hash{}) { // delete slot (2.2.1.2)
			cvm.StateDB.AddRefund(params.SstoreClearsScheduleRefundEIP2200)
		}
	}
	if original == value {
		if original == (common.Hash{}) { // reset to original inexistent slot (2.2.2.1)
			cvm.StateDB.AddRefund(params.SstoreSetGasEIP2200 - params.SloadGasEIP2200)
		} else { // reset to original existing slot (2.2.2.2)
			cvm.StateDB.AddRefund(params.SstoreResetGasEIP2200 - params.SloadGasEIP2200)
		}
	}
	return params.SloadGasEIP2200, nil // dirty update (2.2)
}

func makeGasLog(n uint64) gasFunc {
	return func(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
		requestedSize, overflow := stack.Back(1).Uint64WithOverflow()
		if overflow {
			return 0, ErrGasUintOverflow
		}

		gas, err := memoryGasCost(mem, memorySize)
		if err != nil {
			return 0, err
		}

		if gas, overflow = math.SafeAdd(gas, params.LogGas); overflow {
			return 0, ErrGasUintOverflow
		}
		if gas, overflow = math.SafeAdd(gas, n*params.LogTopicGas); overflow {
			return 0, ErrGasUintOverflow
		}

		var memorySizeGas uint64
		if memorySizeGas, overflow = math.SafeMul(requestedSize, params.LogDataGas); overflow {
			return 0, ErrGasUintOverflow
		}
		if gas, overflow = math.SafeAdd(gas, memorySizeGas); overflow {
			return 0, ErrGasUintOverflow
		}
		return gas, nil
	}
}

func gasKeccak256(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	var overflow bool
	gas, err := memoryGasCost(mem, memorySize)
	if err != nil {
		return 0, err
	}

	if gas, overflow = math.SafeAdd(gas, params.Sha3Gas); overflow {
		return 0, ErrGasUintOverflow
	}

	wordGas, overflow := stack.Back(1).Uint64WithOverflow()
	if overflow {
		return 0, ErrGasUintOverflow
	}
	if wordGas, overflow = math.SafeMul(toWordSize(wordGas), params.Sha3WordGas); overflow {
		return 0, ErrGasUintOverflow
	}
	if gas, overflow = math.SafeAdd(gas, wordGas); overflow {
		return 0, ErrGasUintOverflow
	}
	return gas, nil
}

func gasCodeCopy(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	gas, err := memoryGasCost(mem, memorySize)
	if err != nil {
		return 0, err
	}

	var overflow bool
	if gas, overflow = math.SafeAdd(gas, GasFastestStep); overflow {
		return 0, ErrGasUintOverflow
	}

	wordGas, overflow := stack.Back(2).Uint64WithOverflow()
	if overflow {
		return 0, ErrGasUintOverflow
	}
	if wordGas, overflow = math.SafeMul(toWordSize(wordGas), params.CopyGas); overflow {
		return 0, ErrGasUintOverflow
	}
	if gas, overflow = math.SafeAdd(gas, wordGas); overflow {
		return 0, ErrGasUintOverflow
	}
	return gas, nil
}

func gasExtCodeCopy(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	gas, err := memoryGasCost(mem, memorySize)
	if err != nil {
		return 0, err
	}

	var overflow bool
	if gas, overflow = math.SafeAdd(gas, gt.ExtcodeCopy); overflow {
		return 0, ErrGasUintOverflow
	}

	wordGas, overflow := stack.Back(3).Uint64WithOverflow()
	if overflow {
		return 0, ErrGasUintOverflow
	}

	if wordGas, overflow = math.SafeMul(toWordSize(wordGas), params.CopyGas); overflow {
		return 0, ErrGasUintOverflow
	}

	if gas, overflow = math.SafeAdd(gas, wordGas); overflow {
		return 0, ErrGasUintOverflow
	}
	return gas, nil
}

func gasExtCodeHash(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	return gt.ExtcodeHash, nil
}

func gasMLoad(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	var overflow bool
	gas, err := memoryGasCost(mem, memorySize)
	if err != nil {
		return 0, ErrGasUintOverflow
	}
	if gas, overflow = math.SafeAdd(gas, GasFastestStep); overflow {
		return 0, ErrGasUintOverflow
	}
	return gas, nil
}

func gasMStore8(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	var overflow bool
	gas, err := memoryGasCost(mem, memorySize)
	if err != nil {
		return 0, ErrGasUintOverflow
	}
	if gas, overflow = math.SafeAdd(gas, GasFastestStep); overflow {
		return 0, ErrGasUintOverflow
	}
	return gas, nil
}

func gasMStore(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	var overflow bool
	gas, err := memoryGasCost(mem, memorySize)
	if err != nil {
		return 0, ErrGasUintOverflow
	}
	if gas, overflow = math.SafeAdd(gas, GasFastestStep); overflow {
		return 0, ErrGasUintOverflow
	}
	return gas, nil
}

func gasCreate(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	var overflow bool
	gas, err := memoryGasCost(mem, memorySize)
	if err != nil {
		return 0, err
	}
	if gas, overflow = math.SafeAdd(gas, params.CreateGas); overflow {
		return 0, ErrGasUintOverflow
	}
	return gas, nil
}

func gasCreate2(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	var overflow bool
	gas, err := memoryGasCost(mem, memorySize)
	if err != nil {
		return 0, err
	}

	if gas, overflow = math.SafeAdd(gas, params.Create2Gas); overflow {
		return 0, ErrGasUintOverflow
	}

	wordGas, overflow := stack.Back(2).Uint64WithOverflow()
	if overflow {
		return 0, ErrGasUintOverflow
	}
	if wordGas, overflow = math.SafeMul(toWordSize(wordGas), params.Sha3WordGas); overflow {
		return 0, ErrGasUintOverflow
	}
	if gas, overflow = math.SafeAdd(gas, wordGas); overflow {
		return 0, ErrGasUintOverflow
	}

	return gas, nil
}

func gasExpFrontier(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	expByteLen := uint64((stack.data[stack.len()-2].BitLen() + 7) / 8)

	var (
		gas      = expByteLen * params.ExpByteFrontier // no overflow check required. Max is 256 * ExpByte gas
		overflow bool
	)
	if gas, overflow = math.SafeAdd(gas, params.ExpGas); overflow {
		return 0, ErrGasUintOverflow
	}
	return gas, nil
}

func gasExpEIP158(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	expByteLen := uint64((stack.data[stack.len()-2].BitLen() + 7) / 8)

	var (
		gas      = expByteLen * params.ExpByteEIP158 // no overflow check required. Max is 256 * ExpByte gas
		overflow bool
	)
	if gas, overflow = math.SafeAdd(gas, params.ExpGas); overflow {
		return 0, ErrGasUintOverflow
	}
	return gas, nil
}

func gasBalance(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	return gt.Balance, nil
}

func gasExtCodeSize(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	return gt.ExtcodeSize, nil
}

func gasSLoad(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	return gt.SLoad, nil
}

func gasExp(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	expByteLen := uint64((stack.data[stack.len()-2].BitLen() + 7) / 8)

	var (
		gas      = expByteLen * gt.ExpByte // no overflow check required. Max is 256 * ExpByte gas
		overflow bool
	)
	if gas, overflow = math.SafeAdd(gas, GasSlowStep); overflow {
		return 0, ErrGasUintOverflow
	}
	return gas, nil
}

func gasCall(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	var (
		gas            = gt.Calls
		transfersValue = !stack.Back(2).IsZero()
		address        = common.Address(stack.Back(1).Bytes20())
	)
	if cvm.chainRules.IsEIP158 {
		if transfersValue && cvm.StateDB.Empty(address) {
			gas += params.CallNewAccountGas
		}
	} else if !cvm.StateDB.Exist(address) {
		gas += params.CallNewAccountGas
	}
	if transfersValue {
		gas += params.CallValueTransferGas
	}
	memoryGas, err := memoryGasCost(mem, memorySize)
	if err != nil {
		return 0, err
	}
	var overflow bool
	if gas, overflow = math.SafeAdd(gas, memoryGas); overflow {
		return 0, ErrGasUintOverflow
	}

	cvm.callGasTemp, err = callGas(cvm.chainRules.IsEIP150, contract.Gas, gas, stack.Back(0))
	if err != nil {
		return 0, err
	}
	if gas, overflow = math.SafeAdd(gas, cvm.callGasTemp); overflow {
		return 0, ErrGasUintOverflow
	}
	return gas, nil
}

func gasCallCode(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	gas := gt.Calls
	if stack.Back(2).Sign() != 0 {
		gas += params.CallValueTransferGas
	}
	memoryGas, err := memoryGasCost(mem, memorySize)
	if err != nil {
		return 0, err
	}
	var overflow bool
	if gas, overflow = math.SafeAdd(gas, memoryGas); overflow {
		return 0, ErrGasUintOverflow
	}

	cvm.callGasTemp, err = callGas(cvm.chainRules.IsEIP150, contract.Gas, gas, stack.Back(0))
	if err != nil {
		return 0, err
	}
	if gas, overflow = math.SafeAdd(gas, cvm.callGasTemp); overflow {
		return 0, ErrGasUintOverflow
	}
	return gas, nil
}

func gasReturn(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	return memoryGasCost(mem, memorySize)
}

func gasRevert(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	return memoryGasCost(mem, memorySize)
}

func gasSuicide(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	var gas uint64
	// EIP150 homestead gas reprice fork:
	if cvm.chainRules.IsEIP150 {
		gas = gt.Suicide
		var (
			address = common.Address(stack.Back(0).Bytes20())
			eip158  = cvm.chainRules.IsEIP158
		)

		if eip158 {
			// if empty and transfers value
			if cvm.StateDB.Empty(address) && cvm.StateDB.GetBalance(contract.Address()).Sign() != 0 {
				gas += gt.CreateBySuicide
			}
		} else if !cvm.StateDB.Exist(address) {
			gas += gt.CreateBySuicide
		}
	}

	if !cvm.StateDB.HasSelfDestructed(contract.Address()) {
		cvm.StateDB.AddRefund(params.SuicideRefundGas)
	}
	return gas, nil
}

func gasDelegateCall(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	gas, err := memoryGasCost(mem, memorySize)
	if err != nil {
		return 0, err
	}
	var overflow bool
	if gas, overflow = math.SafeAdd(gas, gt.Calls); overflow {
		return 0, ErrGasUintOverflow
	}

	cvm.callGasTemp, err = callGas(cvm.chainRules.IsEIP150, contract.Gas, gas, stack.Back(0))
	if err != nil {
		return 0, err
	}
	if gas, overflow = math.SafeAdd(gas, cvm.callGasTemp); overflow {
		return 0, ErrGasUintOverflow
	}
	return gas, nil
}

func gasStaticCall(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	gas, err := memoryGasCost(mem, memorySize)
	if err != nil {
		return 0, err
	}
	var overflow bool
	if gas, overflow = math.SafeAdd(gas, gt.Calls); overflow {
		return 0, ErrGasUintOverflow
	}

	cvm.callGasTemp, err = callGas(cvm.chainRules.IsEIP150, contract.Gas, gas, stack.Back(0))
	if err != nil {
		return 0, err
	}
	if gas, overflow = math.SafeAdd(gas, cvm.callGasTemp); overflow {
		return 0, ErrGasUintOverflow
	}
	return gas, nil
}

func gasPush(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	return GasFastestStep, nil
}

func gasSwap(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	return GasFastestStep, nil
}

func gasDup(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	return GasFastestStep, nil
}
