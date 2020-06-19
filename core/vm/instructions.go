// Copyright 2019 The CortexTheseus Authors
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
	"fmt"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
	torrentfs "github.com/CortexFoundation/torrentfs/types"
	"github.com/holiman/uint256"
	"golang.org/x/crypto/sha3"
)

var (
	errWriteProtection       = errors.New("cvm: write protection")
	errReturnDataOutOfBounds = errors.New("cvm: return data out of bounds")
	errMetaInfoBlockNum      = errors.New("cvm: meta info blocknum <= 0")
	ErrMetaInfoNotMature     = errors.New("cvm: errMetaInfoNotMature")
	errMetaShapeNotMatch     = errors.New("cvm: model and input shape not matched")
	errMetaInfoExpired       = errors.New("cvm: errMetaInfoExpired")
	errMaxCodeSizeExceeded   = errors.New("cvm: max code size exceeded")
	errInvalidJump           = errors.New("cvm: invalid jump destination")
)

func opAdd(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	y.Add(&x, y)
	return nil, nil
}

func opSub(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	y.Sub(&x, y)
	return nil, nil
}

func opMul(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	y.Mul(&x, y)

	return nil, nil
}

func opDiv(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	y.Div(&x, y)
	return nil, nil
}

func opSdiv(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	y.SDiv(&x, y)
	return nil, nil
}

func opMod(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	y.Mod(&x, y)
	return nil, nil
}

func opSmod(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	y.SMod(&x, y)
	return nil, nil
}

func opExp(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	base, exponent := callContext.stack.pop(), callContext.stack.peek()
	exponent.Exp(&base, exponent)
	return nil, nil
}

func opSignExtend(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	back, num := callContext.stack.pop(), callContext.stack.peek()
	num.ExtendSign(num, &back)
	return nil, nil
}

func opNot(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x := callContext.stack.peek()
	x.Not(x)
	return nil, nil
}

func opLt(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	if x.Lt(y) {
		y.SetOne()
	} else {
		y.Clear()
	}
	return nil, nil
}

func opGt(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	if x.Gt(y) {
		y.SetOne()
	} else {
		y.Clear()
	}
	return nil, nil
}

func opSlt(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	if x.Slt(y) {
		y.SetOne()
	} else {
		y.Clear()
	}
	return nil, nil
}

func opSgt(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	if x.Sgt(y) {
		y.SetOne()
	} else {
		y.Clear()
	}
	return nil, nil
}

func opEq(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	if x.Eq(y) {
		y.SetOne()
	} else {
		y.Clear()
	}
	return nil, nil
}

func opIszero(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x := callContext.stack.peek()
	if x.IsZero() {
		x.SetOne()
	} else {
		x.Clear()
	}
	return nil, nil
}

func opAnd(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	y.And(&x, y)
	return nil, nil
}

func opOr(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	y.Or(&x, y)
	return nil, nil
}

func opXor(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	y.Xor(&x, y)
	return nil, nil
}

func opByte(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	th, val := callContext.stack.pop(), callContext.stack.peek()
	val.Byte(&th)
	return nil, nil
}

func opAddmod(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y, z := callContext.stack.pop(), callContext.stack.pop(), callContext.stack.peek()
	if z.IsZero() {
		z.Clear()
	} else {
		z.AddMod(&x, &y, z)
	}
	return nil, nil
}

func opMulmod(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y, z := callContext.stack.pop(), callContext.stack.pop(), callContext.stack.peek()
	z.MulMod(&x, &y, z)
	return nil, nil
}

// opSHL implements Shift Left
// The SHL instruction (shift left) pops 2 values from the stack, first arg1 and then arg2,
// and pushes on the stack arg2 shifted to the left by arg1 number of bits.
func opSHL(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	// Note, second operand is left in the stack; accumulate result into it, and no need to push it afterwards
	shift, value := callContext.stack.pop(), callContext.stack.peek()
	if shift.LtUint64(256) {
		value.Lsh(value, uint(shift.Uint64()))
	} else {
		value.Clear()
	}
	return nil, nil
}

// opSHR implements Logical Shift Right
// The SHR instruction (logical shift right) pops 2 values from the stack, first arg1 and then arg2,
// and pushes on the stack arg2 shifted to the right by arg1 number of bits with zero fill.
func opSHR(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	// Note, second operand is left in the stack; accumulate result into it, and no need to push it afterwards
	shift, value := callContext.stack.pop(), callContext.stack.peek()
	if shift.LtUint64(256) {
		value.Rsh(value, uint(shift.Uint64()))
	} else {
		value.Clear()
	}
	return nil, nil
}

// opSAR implements Arithmetic Shift Right
// The SAR instruction (arithmetic shift right) pops 2 values from the stack, first arg1 and then arg2,
// and pushes on the stack arg2 shifted to the right by arg1 number of bits with sign extension.
func opSAR(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	shift, value := callContext.stack.pop(), callContext.stack.peek()
	if shift.GtUint64(256) {
		if value.Sign() >= 0 {
			value.Clear()
		} else {
			// Max negative shift: all bits set
			value.SetAllOne()
		}
		return nil, nil
	}
	n := uint(shift.Uint64())
	value.SRsh(value, n)
	return nil, nil
}

func opSha3(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	offset, size := callContext.stack.pop(), callContext.stack.peek()
	data := callContext.memory.GetPtr(int64(offset.Uint64()), int64(size.Uint64()))

	if interpreter.hasher == nil {
		interpreter.hasher = sha3.NewLegacyKeccak256().(keccakState)
	} else {
		interpreter.hasher.Reset()
	}
	interpreter.hasher.Write(data)
	interpreter.hasher.Read(interpreter.hasherBuf[:])

	cvm := interpreter.cvm
	if cvm.vmConfig.EnablePreimageRecording {
		cvm.StateDB.AddPreimage(interpreter.hasherBuf, data)
	}

	size.SetBytes(interpreter.hasherBuf[:])
	return nil, nil
}

func opAddress(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(new(uint256.Int).SetBytes(callContext.contract.Address().Bytes()))
	return nil, nil
}

func opBalance(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	slot := callContext.stack.peek()
	address := common.Address(slot.Bytes20())
	slot.SetFromBig(interpreter.cvm.StateDB.GetBalance(address))
	return nil, nil
}

func opOrigin(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(new(uint256.Int).SetBytes(interpreter.cvm.Origin.Bytes()))
	return nil, nil
}

func opCaller(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(new(uint256.Int).SetBytes(callContext.contract.Caller().Bytes()))
	return nil, nil
}

func opCallValue(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	v, _ := uint256.FromBig(callContext.contract.value)
	callContext.stack.push(v)
	return nil, nil
}

func opCallDataLoad(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x := callContext.stack.peek()
	if offset, overflow := x.Uint64WithOverflow(); !overflow {
		data := getData(callContext.contract.Input, offset, 32)
		x.SetBytes(data)
	} else {
		x.Clear()
	}
	return nil, nil
}

func opCallDataSize(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(new(uint256.Int).SetUint64(uint64(len(callContext.contract.Input))))
	return nil, nil
}

func opCallDataCopy(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	var (
		memOffset  = callContext.stack.pop()
		dataOffset = callContext.stack.pop()
		length     = callContext.stack.pop()
	)
	dataOffset64, overflow := dataOffset.Uint64WithOverflow()
	if overflow {
		dataOffset64 = 0xffffffffffffffff
	}
	// These values are checked for overflow during gas cost calculation
	memOffset64 := memOffset.Uint64()
	length64 := length.Uint64()
	callContext.memory.Set(memOffset64, length64, getData(callContext.contract.Input, dataOffset64, length64))
	return nil, nil
}

func opReturnDataSize(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(new(uint256.Int).SetUint64(uint64(len(interpreter.returnData))))
	return nil, nil
}

func opReturnDataCopy(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	var (
		memOffset  = callContext.stack.pop()
		dataOffset = callContext.stack.pop()
		length     = callContext.stack.pop()
	)

	offset64, overflow := dataOffset.Uint64WithOverflow()
	if overflow {
		return nil, ErrReturnDataOutOfBounds
	}
	// we can reuse dataOffset now (aliasing it for clarity)
	var end = dataOffset
	end.Add(&dataOffset, &length)
	end64, overflow := end.Uint64WithOverflow()
	if overflow || uint64(len(interpreter.returnData)) < end64 {
		return nil, ErrReturnDataOutOfBounds
	}
	callContext.memory.Set(memOffset.Uint64(), length.Uint64(), interpreter.returnData[offset64:end64])
	return nil, nil
}

func opExtCodeSize(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	slot := callContext.stack.peek()
	slot.SetUint64(uint64(interpreter.cvm.StateDB.GetCodeSize(common.Address(slot.Bytes20()))))

	return nil, nil
}

func opCodeSize(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	l := new(uint256.Int)
	l.SetUint64(uint64(len(callContext.contract.Code)))
	callContext.stack.push(l)

	return nil, nil
}

func opCodeCopy(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	var (
		memOffset  = callContext.stack.pop()
		codeOffset = callContext.stack.pop()
		length     = callContext.stack.pop()
	)
	uint64CodeOffset, overflow := codeOffset.Uint64WithOverflow()
	if overflow {
		uint64CodeOffset = 0xffffffffffffffff
	}
	codeCopy := getData(callContext.contract.Code, uint64CodeOffset, length.Uint64())
	callContext.memory.Set(memOffset.Uint64(), length.Uint64(), codeCopy)
	return nil, nil
}

func opExtCodeCopy(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	var (
		stack      = callContext.stack
		a          = stack.pop()
		memOffset  = stack.pop()
		codeOffset = stack.pop()
		length     = stack.pop()
	)
	uint64CodeOffset, overflow := codeOffset.Uint64WithOverflow()
	if overflow {
		uint64CodeOffset = 0xffffffffffffffff
	}
	addr := common.Address(a.Bytes20())
	codeCopy := getData(interpreter.cvm.StateDB.GetCode(addr), uint64CodeOffset, length.Uint64())
	callContext.memory.Set(memOffset.Uint64(), length.Uint64(), codeCopy)
	return nil, nil
}

// opExtCodeHash returns the code hash of a specified account.
// There are several cases when the function is called, while we can relay everything
// to `state.GetCodeHash` function to ensure the correctness.
//   (1) Caller tries to get the code hash of a normal contract account, state
// should return the relative code hash and set it as the result.
//
//   (2) Caller tries to get the code hash of a non-existent account, state should
// return common.Hash{} and zero will be set as the result.
//
//   (3) Caller tries to get the code hash for an account without contract code,
// state should return emptyCodeHash(0xc5d246...) as the result.
//
//   (4) Caller tries to get the code hash of a precompiled account, the result
// should be zero or emptyCodeHash.
//
// It is worth noting that in order to avoid unnecessary create and clean,
// all precompile accounts on mainnet have been transferred 1 wei, so the return
// here should be emptyCodeHash.
// If the precompile account is not transferred any amount on a private or
// customized chain, the return value will be zero.
//
//   (5) Caller tries to get the code hash for an account which is marked as suicided
// in the current transaction, the code hash of this account should be returned.
//
//   (6) Caller tries to get the code hash for an account which is marked as deleted,
// this account should be regarded as a non-existent account and zero should be returned.
func opExtCodeHash(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	slot := callContext.stack.peek()
	address := common.Address(slot.Bytes20())
	if interpreter.cvm.StateDB.Empty(address) {
		slot.Clear()
	} else {
		slot.SetBytes(interpreter.cvm.StateDB.GetCodeHash(address).Bytes())
	}
	return nil, nil
}

func opGasprice(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	v, _ := uint256.FromBig(interpreter.cvm.GasPrice)
	callContext.stack.push(v)
	return nil, nil
}

func opBlockhash(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	num := callContext.stack.peek()
	num64, overflow := num.Uint64WithOverflow()
	if overflow {
		num.Clear()
		return nil, nil
	}
	var upper, lower uint64
	upper = interpreter.cvm.BlockNumber.Uint64()
	if upper < 257 {
		lower = 0
	} else {
		lower = upper - 256
	}
	if num64 >= lower && num64 < upper {
		num.SetBytes(interpreter.cvm.GetHash(num64).Bytes())
	} else {
		num.Clear()
	}
	return nil, nil
}

func opCoinbase(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(new(uint256.Int).SetBytes(interpreter.cvm.Coinbase.Bytes()))
	return nil, nil
}

func opTimestamp(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	v, _ := uint256.FromBig(interpreter.cvm.Time)
	callContext.stack.push(v)
	return nil, nil
}

func opNumber(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	v, _ := uint256.FromBig(interpreter.cvm.BlockNumber)
	callContext.stack.push(v)
	return nil, nil
}

func opDifficulty(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	v, _ := uint256.FromBig(interpreter.cvm.Difficulty)
	callContext.stack.push(v)
	return nil, nil
}

func opGasLimit(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(new(uint256.Int).SetUint64(interpreter.cvm.GasLimit))
	return nil, nil
}

func opPop(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.pop()
	return nil, nil
}

func opMload(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	v := callContext.stack.peek()
	offset := int64(v.Uint64())
	v.SetBytes(callContext.memory.GetPtr(offset, 32))
	return nil, nil
}

func opMstore(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	// pop value of the stack
	mStart, val := callContext.stack.pop(), callContext.stack.pop()
	callContext.memory.Set32(mStart.Uint64(), &val)
	return nil, nil
}

func opMstore8(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	off, val := callContext.stack.pop(), callContext.stack.pop()
	callContext.memory.store[off.Uint64()] = byte(val.Uint64())

	return nil, nil
}

func opSload(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	loc := callContext.stack.peek()
	hash := common.Hash(loc.Bytes32())
	val := interpreter.cvm.StateDB.GetState(callContext.contract.Address(), hash)
	loc.SetBytes(val.Bytes())
	return nil, nil
}

func opSstore(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	loc := callContext.stack.pop()
	val := callContext.stack.pop()
	interpreter.cvm.StateDB.SetState(callContext.contract.Address(),
		common.Hash(loc.Bytes32()), common.Hash(val.Bytes32()))
	return nil, nil
}

func opJump(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	pos := callContext.stack.pop()
	if !callContext.contract.validJumpdest(&pos) {
		return nil, ErrInvalidJump
	}
	*pc = pos.Uint64()
	return nil, nil
}

func opJumpi(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	pos, cond := callContext.stack.pop(), callContext.stack.pop()
	if !cond.IsZero() {
		if !callContext.contract.validJumpdest(&pos) {
			return nil, ErrInvalidJump
		}
		*pc = pos.Uint64()
	} else {
		*pc++
	}
	return nil, nil
}

func opJumpdest(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	return nil, nil
}

func opBeginSub(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	return nil, ErrInvalidSubroutineEntry
}

func opJumpSub(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	if len(callContext.rstack.data) >= 1023 {
		return nil, ErrReturnStackExceeded
	}
	pos := callContext.stack.pop()
	if !pos.IsUint64() {
		return nil, ErrInvalidJump
	}
	posU64 := pos.Uint64()
	if !callContext.contract.validJumpSubdest(posU64) {
		return nil, ErrInvalidJump
	}
	callContext.rstack.push(*pc)
	*pc = posU64 + 1
	return nil, nil
}

func opReturnSub(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	if len(callContext.rstack.data) == 0 {
		return nil, ErrInvalidRetsub
	}
	// Other than the check that the return stack is not empty, there is no
	// need to validate the pc from 'returns', since we only ever push valid
	//values onto it via jumpsub.
	*pc = callContext.rstack.pop() + 1
	return nil, nil
}

func opPc(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(new(uint256.Int).SetUint64(*pc))
	return nil, nil
}

func opMsize(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(new(uint256.Int).SetUint64(uint64(callContext.memory.Len())))
	return nil, nil
}

func opGas(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(new(uint256.Int).SetUint64(callContext.contract.Gas))
	return nil, nil
}

var (
//confirmTime = params.CONFIRM_TIME * time.Second //-3600 * 24 * 30 * time.Second
)

func opInfer(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	_modelAddr, _inputAddr, _outputOffset := callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop()
	modelAddr := common.Address(_modelAddr.Bytes20())
	inputAddr := common.Address(_inputAddr.Bytes20())
	var (
		modelMeta *torrentfs.ModelMeta
		inputMeta *torrentfs.InputMeta
	)
	modelMeta, modelErr := checkModel(interpreter.cvm, callContext.stack, modelAddr)
	if modelErr != nil {
		callContext.stack.push(new(uint256.Int).Clear())
		return nil, modelErr
	}

	inputMeta, inputErr := checkInputMeta(interpreter.cvm, callContext.stack, inputAddr)
	if inputErr != nil {
		callContext.stack.push(new(uint256.Int).Clear())
		return nil, inputErr
	}

	log.Debug("interpreter check shape 1", "modelMeta", modelMeta, "inputMeta", inputMeta)
	// Model&Input shape should match
	if len(modelMeta.InputShape) != len(inputMeta.Shape) {
		callContext.stack.push(new(uint256.Int).Clear())
		if interpreter.cvm.vmConfig.DebugInferVM {
			fmt.Println("modelmeta: ", modelMeta.InputShape, " inputmeta: ", inputMeta.Shape)
		}
		return nil, errMetaShapeNotMatch
	}
	log.Debug("interpreter check shape 2", "modelMeta", modelMeta, "inputMeta", inputMeta)
	for idx, modelShape := range modelMeta.InputShape {
		if modelShape != inputMeta.Shape[idx] || modelShape <= 0 || inputMeta.Shape[idx] <= 0 {
			callContext.stack.push(new(uint256.Int).Clear())
			if interpreter.cvm.vmConfig.DebugInferVM {
				fmt.Println("modelmeta: ", modelMeta.InputShape, " inputmeta: ", inputMeta.Shape)
			}
			return nil, errMetaShapeNotMatch
		}
	}

	//todo model & input tfs validation
	output, err := interpreter.cvm.Infer(modelMeta.Hash.Hex(), inputMeta.Hash.Hex(), modelMeta.RawSize, inputMeta.RawSize)
	if interpreter.cvm.vmConfig.DebugInferVM {
		fmt.Println("DebugInferVM ", "output: ", output, " err: ", err, "model = ", modelMeta.Hash.Hex(), "input = ", inputMeta.Hash.Hex())
	}
	if err != nil {
		callContext.stack.push(new(uint256.Int).Clear())
		return nil, err
	}
	if err := callContext.memory.WriteSolidityUint256Array(int64(_outputOffset.Uint64()), output); err != nil {
		callContext.stack.push(new(uint256.Int).Clear())
		return nil, err
	}
	callContext.stack.push(new(uint256.Int).SetOne())

	return nil, nil
}

func checkModel(cvm *CVM, stack *Stack, modelAddr common.Address) (*torrentfs.ModelMeta, error) {
	var (
		modelMeta *torrentfs.ModelMeta
	)
	var err error
	if modelMeta, err = cvm.GetModelMeta(modelAddr); err != nil {
		return nil, err
	}
	// Model Meta is validation
	if cvm.StateDB.Uploading(modelAddr) {
		return nil, errors.New("MODEL IS NOT UPLOADED ERROR")
	}

	matureBlockNumber := cvm.ChainConfig().GetMatureBlock()
	log.Debug("checkModel", "modelAddr blocknum", cvm.StateDB.GetNum(modelAddr), "modelMeta", modelMeta)
	if cvm.StateDB.GetNum(modelAddr).Int64() <= 0 {
		return nil, errMetaInfoBlockNum
	}
	if cvm.StateDB.GetNum(modelAddr).Int64() > cvm.BlockNumber.Int64()-matureBlockNumber {
		log.Debug("instructions", "modelAddr", modelAddr, "modelAddrBlkNum", cvm.StateDB.GetNum(modelAddr), "Current", cvm.BlockNumber, "MB", matureBlockNumber)
		return nil, ErrMetaInfoNotMature
	}

	if cvm.StateDB.GetNum(modelAddr).Int64() < cvm.BlockNumber.Int64()-params.ExpiredBlks {
		return nil, errMetaInfoExpired
	}

	if modelMeta.Gas > params.MODEL_GAS_LIMIT {
		//return nil, errExecutionReverted
		return nil, errors.New("INVALID MODEL GAS LIMIT ERROR")
	}
	return modelMeta, nil
}

func checkInputMeta(cvm *CVM, stack *Stack, inputAddr common.Address) (*torrentfs.InputMeta, error) {
	var (
		inputMeta *torrentfs.InputMeta
		err       error
	)
	if inputMeta, err = cvm.GetInputMeta(inputAddr); err != nil {
		return nil, err
	}
	// Model Meta is validation
	if cvm.StateDB.Uploading(inputAddr) {
		return nil, errors.New("MODEL IS NOT UPLOADED ERROR")
	}

	log.Debug("checkInput", "modelAddr blocknum", cvm.StateDB.GetNum(inputAddr), "inputMeta", inputMeta)
	if cvm.StateDB.GetNum(inputAddr).Int64() <= 0 {
		return nil, errMetaInfoBlockNum
	}

	matureBlockNumber := cvm.ChainConfig().GetMatureBlock()
	if cvm.StateDB.GetNum(inputAddr).Int64() > cvm.BlockNumber.Int64()-matureBlockNumber {
		log.Debug("instructions", "inputAddr", inputAddr, "inputAddrBlkNum", cvm.StateDB.GetNum(inputAddr), "Current", cvm.BlockNumber, "Uploading", cvm.StateDB.Uploading(inputAddr), "MB", matureBlockNumber)
		return nil, ErrMetaInfoNotMature
	}

	if cvm.StateDB.GetNum(inputAddr).Int64() < cvm.BlockNumber.Int64()-params.ExpiredBlks {
		return nil, errMetaInfoExpired
	}

	return inputMeta, nil
}

func opInferArray(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	_modelAddr, _inputHeaderOffset, _outputOffset := callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop()
	// fmt.Println(fmt.Sprintf("%d, %d, %d", _modelAddr, _inputHeaderOffset, _outputOffset))
	inputBuff, inputError := interpreter.cvm.StateDB.GetSolidityBytes(callContext.contract.Address(), common.Hash(_inputHeaderOffset.Bytes32()))
	if inputError != nil {
		return nil, inputError
	}
	inputSize := uint256.NewInt().SetUint64(uint64(len(inputBuff)))
	modelAddr := common.Address(_modelAddr.Bytes20())
	// log.Debug(fmt.Sprintf("_input = %v, payload = %v ", inputSize, inputBuff))

	modelMeta, modelErr := checkModel(interpreter.cvm, callContext.stack, modelAddr)
	if modelErr != nil {
		callContext.stack.push(new(uint256.Int).Clear())
		return nil, modelErr
	}

	if false {
		//TODO(tian) omit input shape for infer array
		var dataSize uint64 = 1
		for _, modelShape := range modelMeta.InputShape {
			dataSize *= modelShape
		}
		if dataSize != inputSize.Uint64() {
			callContext.stack.push(new(uint256.Int).Clear())
			if interpreter.cvm.vmConfig.DebugInferVM {
				fmt.Println("modelmeta: ", modelMeta.InputShape, "datasize: ", dataSize, "inputSize: ", inputSize)
			}
			return nil, errMetaShapeNotMatch
		}
	}
	var output []byte
	var err error
	output, err = interpreter.cvm.InferArray(modelMeta.Hash.Hex(),
		inputBuff, modelMeta.RawSize)
	if err != nil {
		callContext.stack.push(new(uint256.Int).Clear())
		return nil, err
	}
	if interpreter.cvm.vmConfig.DebugInferVM {
		fmt.Println("output", output)
	}
	if err := callContext.memory.WriteSolidityUint256Array(int64(_outputOffset.Uint64()), output); err != nil {
		callContext.stack.push(new(uint256.Int).Clear())
		return nil, err
	}
	// interpreter.intPool.get().SetUint64
	callContext.stack.push(new(uint256.Int).SetOne())

	return nil, nil
}

func opCreate(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	var (
		value        = callContext.stack.pop()
		offset, size = callContext.stack.pop(), callContext.stack.pop()
		input        = callContext.memory.GetCopy(int64(offset.Uint64()), int64(size.Uint64()))
		gas          = callContext.contract.Gas
	)
	if interpreter.cvm.chainRules.IsEIP150 {
		gas -= gas / 64
	}
	// reuse size int for stackvalue
	stackvalue := size

	callContext.contract.UseGas(gas)
	res, addr, returnGas, modelGas, suberr := interpreter.cvm.Create(callContext.contract, input, gas, value.ToBig())
	// Push item on the stack based on the returned error. If the ruleset is
	// homestead we must check for CodeStoreOutOfGasError (homestead only
	// rule) and treat as an error, if the ruleset is frontier we must
	// ignore this error and pretend the operation was successful.
	if interpreter.cvm.chainRules.IsHomestead && suberr == ErrCodeStoreOutOfGas {
		stackvalue.Clear()
	} else if suberr != nil && suberr != ErrCodeStoreOutOfGas {
		stackvalue.Clear()
	} else {
		stackvalue.SetBytes(addr.Bytes())
	}
	callContext.stack.push(&stackvalue)
	callContext.contract.Gas += returnGas

	for addr, mGas := range modelGas {
		callContext.contract.ModelGas[addr] += mGas
	}
	if suberr == ErrExecutionReverted {
		return res, nil
	}
	return nil, nil
}

func opCreate2(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	var (
		endowment    = callContext.stack.pop()
		offset, size = callContext.stack.pop(), callContext.stack.pop()
		salt         = callContext.stack.pop()
		input        = callContext.memory.GetCopy(int64(offset.Uint64()), int64(size.Uint64()))
		gas          = callContext.contract.Gas
	)

	// Apply EIP150
	gas -= gas / 64
	callContext.contract.UseGas(gas)
	// reuse size int for stackvalue
	stackvalue := size
	res, addr, returnGas, modelGas, suberr := interpreter.cvm.Create2(callContext.contract, input, gas,
		endowment.ToBig(), salt.ToBig())
	// Push item on the stack based on the returned error.
	if suberr != nil {
		stackvalue.Clear()
	} else {
		stackvalue.SetBytes(addr.Bytes())
	}
	callContext.stack.push(&stackvalue)
	callContext.contract.Gas += returnGas

	for addr, mGas := range modelGas {
		callContext.contract.ModelGas[addr] += mGas
	}

	if suberr == ErrExecutionReverted {
		return res, nil
	}
	return nil, nil
}

func opCall(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	stack := callContext.stack
	// Pop gas. The actual gas in interpreter.cvm.callGasTemp.
	// We can use this as a temporary value
	temp := stack.pop()
	gas := interpreter.cvm.callGasTemp
	// Pop other call parameters.
	addr, value, inOffset, inSize, retOffset, retSize := stack.pop(), stack.pop(), stack.pop(), stack.pop(), stack.pop(), stack.pop()
	toAddr := common.Address(addr.Bytes20())
	// Get the arguments from the memory.
	args := callContext.memory.GetPtr(int64(inOffset.Uint64()), int64(inSize.Uint64()))

	if !value.IsZero() {
		gas += params.CallStipend
	}
	ret, returnGas, modelGas, err := interpreter.cvm.Call(callContext.contract, toAddr, args, gas, value.ToBig())
	if err != nil {
		temp.Clear()
	} else {
		temp.SetOne()
	}
	stack.push(&temp)
	if err == nil || err == ErrExecutionReverted {
		callContext.memory.Set(retOffset.Uint64(), retSize.Uint64(), ret)
	}
	callContext.contract.Gas += returnGas
	for addr, mGas := range modelGas {
		callContext.contract.ModelGas[addr] += mGas
	}
	return ret, nil
}

func opCallCode(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	// Pop gas. The actual gas is in interpreter.cvm.callGasTemp.
	stack := callContext.stack
	// We use it as a temporary value
	temp := stack.pop()
	gas := interpreter.cvm.callGasTemp
	// Pop other call parameters.
	addr, value, inOffset, inSize, retOffset, retSize := stack.pop(), stack.pop(), stack.pop(), stack.pop(), stack.pop(), stack.pop()
	toAddr := common.Address(addr.Bytes20())
	// Get arguments from the memory.
	args := callContext.memory.GetPtr(int64(inOffset.Uint64()), int64(inSize.Uint64()))

	if !value.IsZero() {
		gas += params.CallStipend
	}
	ret, returnGas, modelGas, err := interpreter.cvm.CallCode(callContext.contract, toAddr, args, gas, value.ToBig())
	if err != nil {
		temp.Clear()
	} else {
		temp.SetOne()
	}
	stack.push(&temp)
	if err == nil || err == ErrExecutionReverted {
		callContext.memory.Set(retOffset.Uint64(), retSize.Uint64(), ret)
	}
	callContext.contract.Gas += returnGas
	for addr, mGas := range modelGas {
		callContext.contract.ModelGas[addr] += mGas
	}
	return ret, nil
}

func opDelegateCall(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	stack := callContext.stack
	// Pop gas. The actual gas is in interpreter.cvm.callGasTemp.
	// We use it as a temporary value
	temp := stack.pop()
	gas := interpreter.cvm.callGasTemp
	// Pop other call parameters.
	addr, inOffset, inSize, retOffset, retSize := stack.pop(), stack.pop(), stack.pop(), stack.pop(), stack.pop()
	toAddr := common.Address(addr.Bytes20())
	// Get arguments from the memory.
	args := callContext.memory.GetPtr(int64(inOffset.Uint64()), int64(inSize.Uint64()))

	ret, returnGas, modelGas, err := interpreter.cvm.DelegateCall(callContext.contract, toAddr, args, gas)
	if err != nil {
		temp.Clear()
	} else {
		temp.SetOne()
	}
	stack.push(&temp)
	if err == nil || err == ErrExecutionReverted {
		callContext.memory.Set(retOffset.Uint64(), retSize.Uint64(), ret)
	}
	callContext.contract.Gas += returnGas
	for addr, mGas := range modelGas {
		callContext.contract.ModelGas[addr] += mGas
	}
	return ret, nil
}

func opStaticCall(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	// Pop gas. The actual gas is in interpreter.cvm.callGasTemp.
	stack := callContext.stack
	// We use it as a temporary value
	temp := stack.pop()
	gas := interpreter.cvm.callGasTemp
	// Pop other call parameters.
	addr, inOffset, inSize, retOffset, retSize := stack.pop(), stack.pop(), stack.pop(), stack.pop(), stack.pop()
	toAddr := common.Address(addr.Bytes20())
	// Get arguments from the memory.
	args := callContext.memory.GetPtr(int64(inOffset.Uint64()), int64(inSize.Uint64()))

	ret, returnGas, modelGas, err := interpreter.cvm.StaticCall(callContext.contract, toAddr, args, gas)
	if err != nil {
		temp.Clear()
	} else {
		temp.SetOne()
	}
	stack.push(&temp)
	if err == nil || err == ErrExecutionReverted {
		callContext.memory.Set(retOffset.Uint64(), retSize.Uint64(), ret)
	}
	callContext.contract.Gas += returnGas
	for addr, mGas := range modelGas {
		callContext.contract.ModelGas[addr] += mGas
	}
	return ret, nil
}

func opReturn(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	offset, size := callContext.stack.pop(), callContext.stack.pop()
	ret := callContext.memory.GetPtr(int64(offset.Uint64()), int64(size.Uint64()))

	return ret, nil
}

func opRevert(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	offset, size := callContext.stack.pop(), callContext.stack.pop()
	ret := callContext.memory.GetPtr(int64(offset.Uint64()), int64(size.Uint64()))

	return ret, nil
}

func opStop(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	return nil, nil
}

func opSuicide(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	beneficiary := callContext.stack.pop()
	balance := interpreter.cvm.StateDB.GetBalance(callContext.contract.Address())
	interpreter.cvm.StateDB.AddBalance(common.Address(beneficiary.Bytes20()), balance)
	interpreter.cvm.StateDB.Suicide(callContext.contract.Address())
	return nil, nil
}

// make log instruction function
func makeLog(size int) executionFunc {
	return func(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
		topics := make([]common.Hash, size)
		stack := callContext.stack
		mStart, mSize := stack.pop(), stack.pop()
		for i := 0; i < size; i++ {
			addr := stack.pop()
			topics[i] = common.Hash(addr.Bytes32())
		}

		d := callContext.memory.GetCopy(int64(mStart.Uint64()), int64(mSize.Uint64()))
		interpreter.cvm.StateDB.AddLog(&types.Log{
			Address: callContext.contract.Address(),
			Topics:  topics,
			Data:    d,
			// This is a non-consensus field, but assigned here because
			// core/state doesn't know the current block number.
			BlockNumber: interpreter.cvm.BlockNumber.Uint64(),
		})

		return nil, nil
	}
}

// opPush1 is a specialized version of pushN
func opPush1(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	var (
		codeLen = uint64(len(callContext.contract.Code))
		integer = new(uint256.Int)
	)
	*pc += 1
	if *pc < codeLen {
		callContext.stack.push(integer.SetUint64(uint64(callContext.contract.Code[*pc])))
	} else {
		callContext.stack.push(integer.Clear())
	}
	return nil, nil
}

// make push instruction function
func makePush(size uint64, pushByteSize int) executionFunc {
	return func(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
		codeLen := len(callContext.contract.Code)

		startMin := codeLen
		if int(*pc+1) < startMin {
			startMin = int(*pc + 1)
		}

		endMin := codeLen
		if startMin+pushByteSize < endMin {
			endMin = startMin + pushByteSize
		}

		integer := new(uint256.Int)
		callContext.stack.push(integer.SetBytes(common.RightPadBytes(
			callContext.contract.Code[startMin:endMin], pushByteSize)))

		*pc += size
		return nil, nil
	}
}

// make dup instruction function
func makeDup(size int64) executionFunc {
	return func(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
		callContext.stack.dup(int(size))
		return nil, nil
	}
}

// make swap instruction function
func makeSwap(size int64) executionFunc {
	// switch n + 1 otherwise n would be swapped with n
	size++
	return func(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
		callContext.stack.swap(int(size))
		return nil, nil
	}
}
