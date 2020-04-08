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
	"math/big"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/math"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
	"golang.org/x/crypto/sha3"
)

var (
	bigZero                  = new(big.Int)
	tt255                    = math.BigPow(2, 255)
	errWriteProtection       = errors.New("cvm: write protection")
	errReturnDataOutOfBounds = errors.New("cvm: return data out of bounds")
	errExecutionReverted     = errors.New("cvm: execution reverted")
	errMetaInfoBlockNum      = errors.New("cvm: meta info blocknum <= 0")
	ErrMetaInfoNotMature     = errors.New("cvm: errMetaInfoNotMature")
	errMetaShapeNotMatch     = errors.New("cvm: model and input shape not matched")
	errMetaInfoExpired       = errors.New("cvm: errMetaInfoExpired")
	errMaxCodeSizeExceeded   = errors.New("cvm: max code size exceeded")
	errInvalidJump           = errors.New("cvm: invalid jump destination")

	big0  = big.NewInt(0)
	big31 = big.NewInt(31)
)

func opAdd(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	math.U256(y.Add(x, y))

	interpreter.intPool.putOne(x)
	return nil, nil
}

func opSub(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	math.U256(y.Sub(x, y))

	interpreter.intPool.putOne(x)
	return nil, nil
}

func opMul(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.pop()
	callContext.stack.push(math.U256(x.Mul(x, y)))

	interpreter.intPool.putOne(y)

	return nil, nil
}

func opDiv(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	if y.Sign() != 0 {
		math.U256(y.Div(x, y))
	} else {
		y.SetUint64(0)
	}
	interpreter.intPool.putOne(x)
	return nil, nil
}

func opSdiv(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := math.S256(callContext.stack.pop()), math.S256(callContext.stack.pop())
	res := interpreter.intPool.getZero()

	if y.Sign() == 0 || x.Sign() == 0 {
		callContext.stack.push(res)
	} else {
		if x.Sign() != y.Sign() {
			res.Div(x.Abs(x), y.Abs(y))
			res.Neg(res)
		} else {
			res.Div(x.Abs(x), y.Abs(y))
		}
		callContext.stack.push(math.U256(res))
	}
	interpreter.intPool.put(x, y)
	return nil, nil
}

func opMod(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.pop()
	if y.Sign() == 0 {
		callContext.stack.push(x.SetUint64(0))
	} else {
		callContext.stack.push(math.U256(x.Mod(x, y)))
	}
	interpreter.intPool.putOne(y)
	return nil, nil
}

func opSmod(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := math.S256(callContext.stack.pop()), math.S256(callContext.stack.pop())
	res := interpreter.intPool.getZero()

	if y.Sign() == 0 {
		callContext.stack.push(res)
	} else {
		if x.Sign() < 0 {
			res.Mod(x.Abs(x), y.Abs(y))
			res.Neg(res)
		} else {
			res.Mod(x.Abs(x), y.Abs(y))
		}
		callContext.stack.push(math.U256(res))
	}
	interpreter.intPool.put(x, y)
	return nil, nil
}

func opExp(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	base, exponent := callContext.stack.pop(), callContext.stack.pop()
	// some shortcuts
	cmpToOne := exponent.Cmp(big1)
	if cmpToOne < 0 { // Exponent is zero
		// x ^ 0 == 1
		callContext.stack.push(base.SetUint64(1))
	} else if base.Sign() == 0 {
		// 0 ^ y, if y != 0, == 0
		callContext.stack.push(base.SetUint64(0))
	} else if cmpToOne == 0 { // Exponent is one
		// x ^ 1 == x
		callContext.stack.push(base)
	} else {
		callContext.stack.push(math.Exp(base, exponent))
		interpreter.intPool.putOne(base)
	}
	interpreter.intPool.putOne(exponent)

	return nil, nil
}

func opSignExtend(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	back := callContext.stack.pop()
	if back.Cmp(big.NewInt(31)) < 0 {
		bit := uint(back.Uint64()*8 + 7)
		num := callContext.stack.pop()
		mask := back.Lsh(common.Big1, bit)
		mask.Sub(mask, common.Big1)
		if num.Bit(int(bit)) > 0 {
			num.Or(num, mask.Not(mask))
		} else {
			num.And(num, mask)
		}

		callContext.stack.push(math.U256(num))
	}

	interpreter.intPool.putOne(back)
	return nil, nil
}

func opNot(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x := callContext.stack.peek()
	math.U256(x.Not(x))
	return nil, nil
}

func opLt(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	if x.Cmp(y) < 0 {
		y.SetUint64(1)
	} else {
		y.SetUint64(0)
	}
	interpreter.intPool.putOne(x)
	return nil, nil
}

func opGt(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	if x.Cmp(y) > 0 {
		y.SetUint64(1)
	} else {
		y.SetUint64(0)
	}
	interpreter.intPool.putOne(x)
	return nil, nil
}

func opSlt(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()

	xSign := x.Cmp(tt255)
	ySign := y.Cmp(tt255)

	switch {
	case xSign >= 0 && ySign < 0:
		y.SetUint64(1)

	case xSign < 0 && ySign >= 0:
		y.SetUint64(0)

	default:
		if x.Cmp(y) < 0 {
			y.SetUint64(1)
		} else {
			y.SetUint64(0)
		}
	}
	interpreter.intPool.putOne(x)
	return nil, nil
}

func opSgt(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()

	xSign := x.Cmp(tt255)
	ySign := y.Cmp(tt255)

	switch {
	case xSign >= 0 && ySign < 0:
		y.SetUint64(0)

	case xSign < 0 && ySign >= 0:
		y.SetUint64(1)

	default:
		if x.Cmp(y) > 0 {
			y.SetUint64(1)
		} else {
			y.SetUint64(0)
		}
	}
	interpreter.intPool.putOne(x)
	return nil, nil
}

func opEq(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	if x.Cmp(y) == 0 {
		y.SetUint64(1)
	} else {
		y.SetUint64(0)
	}
	interpreter.intPool.putOne(x)
	return nil, nil
}

func opIszero(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x := callContext.stack.peek()
	if x.Sign() > 0 {
		x.SetUint64(0)
	} else {
		x.SetUint64(1)
	}
	return nil, nil
}

func opAnd(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.pop()
	callContext.stack.push(x.And(x, y))

	interpreter.intPool.putOne(y)
	return nil, nil
}

func opOr(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	y.Or(x, y)

	interpreter.intPool.putOne(x)
	return nil, nil
}

func opXor(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y := callContext.stack.pop(), callContext.stack.peek()
	y.Xor(x, y)

	interpreter.intPool.putOne(x)
	return nil, nil
}

func opByte(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	th, val := callContext.stack.pop(), callContext.stack.peek()
	if th.Cmp(common.Big32) < 0 {
		b := math.Byte(val, 32, int(th.Int64()))
		val.SetUint64(uint64(b))
	} else {
		val.SetUint64(0)
	}
	interpreter.intPool.putOne(th)
	return nil, nil
}

func opAddmod(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y, z := callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop()
	if z.Cmp(bigZero) > 0 {
		x.Add(x, y)
		x.Mod(x, z)
		callContext.stack.push(math.U256(x))
	} else {
		callContext.stack.push(x.SetUint64(0))
	}
	interpreter.intPool.put(y, z)
	return nil, nil
}

func opMulmod(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	x, y, z := callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop()
	if z.Cmp(bigZero) > 0 {
		x.Mul(x, y)
		x.Mod(x, z)
		callContext.stack.push(math.U256(x))
	} else {
		callContext.stack.push(x.SetUint64(0))
	}
	interpreter.intPool.put(y, z)
	return nil, nil
}

// opSHL implements Shift Left
// The SHL instruction (shift left) pops 2 values from the stack, first arg1 and then arg2,
// and pushes on the stack arg2 shifted to the left by arg1 number of bits.
func opSHL(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	// Note, second operand is left in the stack; accumulate result into it, and no need to push it afterwards
	shift, value := math.U256(callContext.stack.pop()), math.U256(callContext.stack.peek())
	defer interpreter.intPool.putOne(shift) // First operand back into the pool

	if shift.Cmp(common.Big256) >= 0 {
		value.SetUint64(0)
		return nil, nil
	}
	n := uint(shift.Uint64())
	math.U256(value.Lsh(value, n))

	return nil, nil
}

// opSHR implements Logical Shift Right
// The SHR instruction (logical shift right) pops 2 values from the stack, first arg1 and then arg2,
// and pushes on the stack arg2 shifted to the right by arg1 number of bits with zero fill.
func opSHR(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	// Note, second operand is left in the stack; accumulate result into it, and no need to push it afterwards
	shift, value := math.U256(callContext.stack.pop()), math.U256(callContext.stack.peek())
	defer interpreter.intPool.putOne(shift) // First operand back into the pool

	if shift.Cmp(common.Big256) >= 0 {
		value.SetUint64(0)
		return nil, nil
	}
	n := uint(shift.Uint64())
	math.U256(value.Rsh(value, n))

	return nil, nil
}

// opSAR implements Arithmetic Shift Right
// The SAR instruction (arithmetic shift right) pops 2 values from the stack, first arg1 and then arg2,
// and pushes on the stack arg2 shifted to the right by arg1 number of bits with sign extension.
func opSAR(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	// Note, S256 returns (potentially) a new bigint, so we're popping, not peeking this one
	shift, value := math.U256(callContext.stack.pop()), math.S256(callContext.stack.pop())
	defer interpreter.intPool.putOne(shift) // First operand back into the pool

	if shift.Cmp(common.Big256) >= 0 {
		if value.Sign() >= 0 {
			value.SetUint64(0)
		} else {
			value.SetInt64(-1)
		}
		callContext.stack.push(math.U256(value))
		return nil, nil
	}
	n := uint(shift.Uint64())
	value.Rsh(value, n)
	callContext.stack.push(math.U256(value))

	return nil, nil
}

func opSha3(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	offset, size := callContext.stack.pop(), callContext.stack.pop()
	data := callContext.memory.GetPtr(offset.Int64(), size.Int64())

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
	callContext.stack.push(interpreter.intPool.get().SetBytes(interpreter.hasherBuf[:]))

	interpreter.intPool.put(offset, size)
	return nil, nil
}

func opAddress(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	//callContext.stack.push(callContext.contract.Address().Big())
	callContext.stack.push(callContext.contract.Address().SetBig(interpreter.intPool.get()))
	return nil, nil
}

func opBalance(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	slot := callContext.stack.peek()
	slot.Set(interpreter.cvm.StateDB.GetBalance(common.BigToAddress(slot)))
	return nil, nil
}

func opOrigin(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	//callContext.stack.push(interpreter.cvm.Origin.Big())
	callContext.stack.push(interpreter.cvm.Origin.SetBig(interpreter.intPool.get()))
	return nil, nil
}

func opCaller(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	//callContext.stack.push(contract.Caller().Big())
	callContext.stack.push(callContext.contract.Caller().SetBig(interpreter.intPool.get()))
	return nil, nil
}

func opCallValue(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(interpreter.intPool.get().Set(callContext.contract.value))
	return nil, nil
}

func opCallDataLoad(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(interpreter.intPool.get().SetBytes(getDataBig(callContext.contract.Input, callContext.stack.pop(), big32)))
	return nil, nil
}

func opCallDataSize(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(interpreter.intPool.get().SetInt64(int64(len(callContext.contract.Input))))
	return nil, nil
}

func opCallDataCopy(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	var (
		memOffset  = callContext.stack.pop()
		dataOffset = callContext.stack.pop()
		length     = callContext.stack.pop()
	)
	callContext.memory.Set(memOffset.Uint64(), length.Uint64(), getDataBig(callContext.contract.Input, dataOffset, length))

	interpreter.intPool.put(memOffset, dataOffset, length)
	return nil, nil
}

func opReturnDataSize(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(interpreter.intPool.get().SetUint64(uint64(len(interpreter.returnData))))
	return nil, nil
}

func opReturnDataCopy(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	var (
		memOffset  = callContext.stack.pop()
		dataOffset = callContext.stack.pop()
		length     = callContext.stack.pop()

		end = interpreter.intPool.get().Add(dataOffset, length)
	)
	defer interpreter.intPool.put(memOffset, dataOffset, length, end)

	//if end.BitLen() > 64 || uint64(len(interpreter.returnData)) < end.Uint64() {
	if !end.IsUint64() || uint64(len(interpreter.returnData)) < end.Uint64() {
		return nil, errReturnDataOutOfBounds
	}
	callContext.memory.Set(memOffset.Uint64(), length.Uint64(), interpreter.returnData[dataOffset.Uint64():end.Uint64()])

	return nil, nil
}

func opExtCodeSize(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	slot := callContext.stack.peek()
	slot.SetUint64(uint64(interpreter.cvm.StateDB.GetCodeSize(common.BigToAddress(slot))))

	return nil, nil
}

func opCodeSize(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	l := interpreter.intPool.get().SetInt64(int64(len(callContext.contract.Code)))
	callContext.stack.push(l)

	return nil, nil
}

func opCodeCopy(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	var (
		memOffset  = callContext.stack.pop()
		codeOffset = callContext.stack.pop()
		length     = callContext.stack.pop()
	)
	codeCopy := getDataBig(callContext.contract.Code, codeOffset, length)
	callContext.memory.Set(memOffset.Uint64(), length.Uint64(), codeCopy)

	interpreter.intPool.put(memOffset, codeOffset, length)
	return nil, nil
}

func opExtCodeCopy(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	var (
		addr       = common.BigToAddress(callContext.stack.pop())
		memOffset  = callContext.stack.pop()
		codeOffset = callContext.stack.pop()
		length     = callContext.stack.pop()
	)
	codeCopy := getDataBig(interpreter.cvm.StateDB.GetCode(addr), codeOffset, length)
	callContext.memory.Set(memOffset.Uint64(), length.Uint64(), codeCopy)

	interpreter.intPool.put(memOffset, codeOffset, length)
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
	slot.SetBytes(interpreter.cvm.StateDB.GetCodeHash(common.BigToAddress(slot)).Bytes())
	return nil, nil
}

func opGasprice(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(interpreter.intPool.get().Set(interpreter.cvm.GasPrice))
	return nil, nil
}

func opBlockhash(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	num := callContext.stack.pop()

	n := interpreter.intPool.get().Sub(interpreter.cvm.BlockNumber, common.Big257)
	if num.Cmp(n) > 0 && num.Cmp(interpreter.cvm.BlockNumber) < 0 {
		callContext.stack.push(interpreter.cvm.GetHash(num.Uint64()).Big())
	} else {
		callContext.stack.push(interpreter.intPool.getZero())
	}
	interpreter.intPool.put(num, n)
	return nil, nil
}

func opCoinbase(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(interpreter.cvm.Coinbase.Big())
	return nil, nil
}

func opTimestamp(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(math.U256(interpreter.intPool.get().Set(interpreter.cvm.Time)))
	return nil, nil
}

func opNumber(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(math.U256(interpreter.intPool.get().Set(interpreter.cvm.BlockNumber)))
	return nil, nil
}

func opDifficulty(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(math.U256(interpreter.intPool.get().Set(interpreter.cvm.Difficulty)))
	return nil, nil
}

func opGasLimit(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(math.U256(interpreter.intPool.get().SetUint64(interpreter.cvm.GasLimit)))
	return nil, nil
}

func opPop(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	interpreter.intPool.putOne(callContext.stack.pop())
	return nil, nil
}

func opMload(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	v := callContext.stack.peek()
	offset := v.Int64()
	v.SetBytes(callContext.memory.GetPtr(offset, 32))
	return nil, nil
}

func opMstore(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	// pop value of the stack
	mStart, val := callContext.stack.pop(), callContext.stack.pop()
	callContext.memory.Set32(mStart.Uint64(), val)

	interpreter.intPool.put(mStart, val)
	return nil, nil
}

func opMstore8(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	off, val := callContext.stack.pop().Int64(), callContext.stack.pop().Int64()
	callContext.memory.store[off] = byte(val & 0xff)

	return nil, nil
}

func opSload(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	loc := callContext.stack.peek()
	val := interpreter.cvm.StateDB.GetState(callContext.contract.Address(), common.BigToHash(loc))
	loc.SetBytes(val.Bytes())
	return nil, nil
}

func opSstore(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	loc := common.BigToHash(callContext.stack.pop())
	val := callContext.stack.pop()
	interpreter.cvm.StateDB.SetState(callContext.contract.Address(), loc, common.BigToHash(val))

	interpreter.intPool.putOne(val)
	return nil, nil
}

func opJump(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	pos := callContext.stack.pop()
	if !callContext.contract.validJumpdest(pos) {
		return nil, errInvalidJump
	}
	*pc = pos.Uint64()

	interpreter.intPool.putOne(pos)
	return nil, nil
}

func opJumpi(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	pos, cond := callContext.stack.pop(), callContext.stack.pop()
	if cond.Sign() != 0 {
		if !callContext.contract.validJumpdest(pos) {
			return nil, errInvalidJump
		}
		*pc = pos.Uint64()
	} else {
		*pc++
	}

	interpreter.intPool.put(pos, cond)
	return nil, nil
}

func opJumpdest(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	return nil, nil
}

func opPc(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(interpreter.intPool.get().SetUint64(*pc))
	return nil, nil
}

func opMsize(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(interpreter.intPool.get().SetInt64(int64(callContext.memory.Len())))
	return nil, nil
}

func opGas(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	callContext.stack.push(interpreter.intPool.get().SetUint64(callContext.contract.Gas))
	return nil, nil
}

var (
//confirmTime = params.CONFIRM_TIME * time.Second //-3600 * 24 * 30 * time.Second
)

func opInfer(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	_modelAddr, _inputAddr, _outputOffset := callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop()
	modelAddr := common.BigToAddress(_modelAddr)
	inputAddr := common.BigToAddress(_inputAddr)
	var (
		modelMeta *types.ModelMeta
		inputMeta *types.InputMeta
	)
	modelMeta, modelErr := checkModel(interpreter.cvm, callContext.stack, modelAddr)
	if modelErr != nil {
		callContext.stack.push(interpreter.intPool.getZero())
		return nil, modelErr
	}

	inputMeta, inputErr := checkInputMeta(interpreter.cvm, callContext.stack, inputAddr)
	if inputErr != nil {
		callContext.stack.push(interpreter.intPool.getZero())
		return nil, inputErr
	}

	log.Debug("interpreter check shape 1", "modelMeta", modelMeta, "inputMeta", inputMeta)
	// Model&Input shape should match
	if len(modelMeta.InputShape) != len(inputMeta.Shape) {
		callContext.stack.push(interpreter.intPool.getZero())
		if interpreter.cvm.vmConfig.DebugInferVM {
			fmt.Println("modelmeta: ", modelMeta.InputShape, " inputmeta: ", inputMeta.Shape)
		}
		return nil, errMetaShapeNotMatch
	}
	log.Debug("interpreter check shape 2", "modelMeta", modelMeta, "inputMeta", inputMeta)
	for idx, modelShape := range modelMeta.InputShape {
		if modelShape != inputMeta.Shape[idx] || modelShape <= 0 || inputMeta.Shape[idx] <= 0 {
			callContext.stack.push(interpreter.intPool.getZero())
			if interpreter.cvm.vmConfig.DebugInferVM {
				fmt.Println("modelmeta: ", modelMeta.InputShape, " inputmeta: ", inputMeta.Shape)
			}
			return nil, errMetaShapeNotMatch
		}
	}

	// /*if interpreter.cvm.Context.Time.Cmp(big.NewInt(time.Now().Add(confirmTime).Unix())) <= 0 {
	// 	logs := interpreter.cvm.StateDB.GetCurrentLogs()
	// 	if logs != nil && len(logs) > 0 {
	// 		for _, log := range logs {
	// 			topics := log.Topics
	// 			//todo
	// 			if topics != nil && len(topics) == 4 && topics[0].Big().Cmp(modelMeta.Hash.Big()) == 0 && topics[1].Big().Cmp(inputMeta.Hash.Big()) == 0 {
	// 				if topics[3].Big().Cmp(big.NewInt(0)) == 0 {
	// 					//consensus
	// 					interpreter.cvm.StateDB.SetNum(modelAddr, big.NewInt(0).Sub(interpreter.cvm.BlockNumber, big.NewInt(params.MatureBlks+1)))
	// 					interpreter.cvm.StateDB.SetNum(inputAddr, big.NewInt(0).Sub(interpreter.cvm.BlockNumber, big.NewInt(params.MatureBlks+1)))
	// 					ret := topics[2].Big().Uint64()
	// 					callContext.stack.push(interpreter.intPool.get().SetUint64(ret))
	// 				} else {
	// 					callContext.stack.push(interpreter.intPool.getZero())
	// 					return nil, errAiRuntime
	// 				}
	// 				return nil, nil
	// 			} else {
	// 			}
	// 		}
	// 	} else {
	// 	}
	// } else {
	// }*/

	// log.Debug("interpreter infer<", "modelMeta", modelMeta, "inputMeta", inputMeta)
	//todo model & input tfs validation
	output, err := interpreter.cvm.Infer(modelMeta.Hash.Hex(), inputMeta.Hash.Hex(), modelMeta.RawSize, inputMeta.RawSize)
	if interpreter.cvm.vmConfig.DebugInferVM {
		fmt.Println("DebugInferVM ", "output: ", output, " err: ", err, "model = ", modelMeta.Hash.Hex(), "input = ", inputMeta.Hash.Hex())
	}
	// log.Debug("interpreter infer>", "modelMeta", modelMeta, "inputMeta", inputMeta, "output", output)
	if err != nil {
		callContext.stack.push(interpreter.intPool.getZero())
		// if !synapse.CheckBuiltInTorrentFsError(err) {
		//consensus
		//makeAiLog(common.BigToHash(modelMeta.Hash.Big()), common.BigToHash(inputMeta.Hash.Big()), 0, err, interpreter, contract)
		//}
		return nil, err
	}
	//consensus
	//matureBlockNumber := interpreter.cvm.ChainConfig().GetMatureBlock()
	//interpreter.cvm.StateDB.SetNum(modelAddr, new(big.Int).Sub(interpreter.cvm.BlockNumber, big.NewInt(matureBlockNumber+1)))
	//interpreter.cvm.StateDB.SetNum(inputAddr, new(big.Int).Sub(interpreter.cvm.BlockNumber, big.NewInt(matureBlockNumber+1)))
	// interpreter.intPool.get().SetUint64(output)
	if err := callContext.memory.WriteSolidityUint256Array(_outputOffset.Int64(), output); err != nil {
		callContext.stack.push(interpreter.intPool.getZero())
		return nil, err
	}
	callContext.stack.push(interpreter.intPool.get().SetUint64(1))
	//consensus
	//makeAiLog(common.BigToHash(modelMeta.Hash.Big()), common.BigToHash(inputMeta.Hash.Big()), output, nil, interpreter, contract)

	return nil, nil
}

func checkModel(cvm *CVM, stack *Stack, modelAddr common.Address) (*types.ModelMeta, error) {
	var (
		modelMeta *types.ModelMeta
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
	if cvm.StateDB.GetNum(modelAddr).Cmp(big0) <= 0 {
		return nil, errMetaInfoBlockNum
	}
	if cvm.StateDB.GetNum(modelAddr).Cmp(new(big.Int).Sub(cvm.BlockNumber, big.NewInt(matureBlockNumber))) > 0 {
		log.Debug("instructions", "modelAddr", modelAddr, "modelAddrBlkNum", cvm.StateDB.GetNum(modelAddr), "Current", cvm.BlockNumber, "MB", matureBlockNumber)
		return nil, ErrMetaInfoNotMature
	}

	if cvm.StateDB.GetNum(modelAddr).Cmp(new(big.Int).Sub(cvm.BlockNumber, big.NewInt(params.ExpiredBlks))) < 0 {
		return nil, errMetaInfoExpired
	}

	if modelMeta.Gas > params.MODEL_GAS_LIMIT {
		//return nil, errExecutionReverted
		return nil, errors.New("INVALID MODEL GAS LIMIT ERROR")
	}
	return modelMeta, nil
}

func checkInputMeta(cvm *CVM, stack *Stack, inputAddr common.Address) (*types.InputMeta, error) {
	var (
		inputMeta *types.InputMeta
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
	if cvm.StateDB.GetNum(inputAddr).Cmp(big0) <= 0 {
		return nil, errMetaInfoBlockNum
	}

	matureBlockNumber := cvm.ChainConfig().GetMatureBlock()
	if cvm.StateDB.GetNum(inputAddr).Cmp(new(big.Int).Sub(cvm.BlockNumber, big.NewInt(matureBlockNumber))) > 0 {
		log.Debug("instructions", "inputAddr", inputAddr, "inputAddrBlkNum", cvm.StateDB.GetNum(inputAddr), "Current", cvm.BlockNumber, "Uploading", cvm.StateDB.Uploading(inputAddr), "MB", matureBlockNumber)
		return nil, ErrMetaInfoNotMature
	}

	if cvm.StateDB.GetNum(inputAddr).Cmp(new(big.Int).Sub(cvm.BlockNumber, big.NewInt(params.ExpiredBlks))) < 0 {
		return nil, errMetaInfoExpired
	}

	return inputMeta, nil
}

func opInferArray(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	_modelAddr, _inputHeaderOffset, _outputOffset := callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop()
	// fmt.Println(fmt.Sprintf("%d, %d, %d", _modelAddr, _inputHeaderOffset, _outputOffset))
	inputBuff, inputError := interpreter.cvm.StateDB.GetSolidityBytes(callContext.contract.Address(), common.BigToHash(_inputHeaderOffset))
	if inputError != nil {
		return nil, inputError
	}
	inputSize := big.NewInt(int64(len(inputBuff)))
	modelAddr := common.BigToAddress(_modelAddr)
	// log.Debug(fmt.Sprintf("_input = %v, payload = %v ", inputSize, inputBuff))

	modelMeta, modelErr := checkModel(interpreter.cvm, callContext.stack, modelAddr)
	if modelErr != nil {
		// log.Error("opInferArray", "modelErr", modelErr)
		callContext.stack.push(interpreter.intPool.getZero())
		return nil, modelErr
	}

	if false {
		//TODO(tian) omit input shape for infer array
		var dataSize uint64 = 1
		for _, modelShape := range modelMeta.InputShape {
			dataSize *= modelShape
		}
		if dataSize != inputSize.Uint64() {
			callContext.stack.push(interpreter.intPool.getZero())
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
	// output = big.NewInt(2147483647).Bytes()
	if err != nil {
		callContext.stack.push(interpreter.intPool.getZero())
		return nil, err
	}
	if interpreter.cvm.vmConfig.DebugInferVM {
		fmt.Println("output", output)
	}
	if err := callContext.memory.WriteSolidityUint256Array(_outputOffset.Int64(), output); err != nil {
		callContext.stack.push(interpreter.intPool.getZero())
		return nil, err
	}
	// interpreter.intPool.get().SetUint64
	callContext.stack.push(interpreter.intPool.get().SetUint64(1))

	//matureBlockNumber := interpreter.cvm.ChainConfig().GetMatureBlock()
	//update model status
	//interpreter.cvm.StateDB.SetNum(modelAddr, new(big.Int).Sub(interpreter.cvm.BlockNumber, big.NewInt(matureBlockNumber+1)))
	return nil, nil
}

func opCreate(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	var (
		value        = callContext.stack.pop()
		offset, size = callContext.stack.pop(), callContext.stack.pop()
		input        = callContext.memory.Get(offset.Int64(), size.Int64())
		gas          = callContext.contract.Gas
	)
	if interpreter.cvm.ChainConfig().IsEIP150(interpreter.cvm.BlockNumber) {
		gas -= gas / 64
	}

	callContext.contract.UseGas(gas)
	res, addr, returnGas, modelGas, suberr := interpreter.cvm.Create(callContext.contract, input, gas, value)
	// Push item on the stack based on the returned error. If the ruleset is
	// homestead we must check for CodeStoreOutOfGasError (homestead only
	// rule) and treat as an error, if the ruleset is frontier we must
	// ignore this error and pretend the operation was successful.
	if interpreter.cvm.ChainConfig().IsHomestead(interpreter.cvm.BlockNumber) && suberr == ErrCodeStoreOutOfGas {
		callContext.stack.push(interpreter.intPool.getZero())
	} else if suberr != nil && suberr != ErrCodeStoreOutOfGas {
		callContext.stack.push(interpreter.intPool.getZero())
	} else {
		callContext.stack.push(addr.Big())
	}
	callContext.contract.Gas += returnGas
	for addr, mGas := range modelGas {
		callContext.contract.ModelGas[addr] += mGas
	}
	interpreter.intPool.put(value, offset, size)

	if suberr == errExecutionReverted {
		return res, nil
	}
	return nil, nil
}

func opCreate2(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	var (
		endowment    = callContext.stack.pop()
		offset, size = callContext.stack.pop(), callContext.stack.pop()
		salt         = callContext.stack.pop()
		input        = callContext.memory.Get(offset.Int64(), size.Int64())
		gas          = callContext.contract.Gas
	)

	// Apply EIP150
	gas -= gas / 64
	callContext.contract.UseGas(gas)
	res, addr, returnGas, modelGas, suberr := interpreter.cvm.Create2(callContext.contract, input, gas, endowment, salt)
	if interpreter.cvm.ChainConfig().IsHomestead(interpreter.cvm.BlockNumber) && suberr == ErrCodeStoreOutOfGas {
		callContext.stack.push(interpreter.intPool.getZero())
	} else if suberr != nil && suberr != ErrCodeStoreOutOfGas {
		callContext.stack.push(interpreter.intPool.getZero())
	} else {
		callContext.stack.push(addr.Big())
	}
	callContext.contract.Gas += returnGas
	for addr, mGas := range modelGas {
		callContext.contract.ModelGas[addr] += mGas
	}
	interpreter.intPool.put(endowment, offset, size)
	// Push item on the stack based on the returned error.
	if suberr != nil {
		callContext.stack.push(interpreter.intPool.getZero())
	} else {
		callContext.stack.push(addr.Big())
	}
	callContext.contract.Gas += returnGas
	interpreter.intPool.put(endowment, offset, size, salt)

	if suberr == errExecutionReverted {
		return res, nil
	}
	return nil, nil
}

func opCall(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	// Pop gas. The actual gas in in interpreter.cvm.callGasTemp.
	interpreter.intPool.putOne(callContext.stack.pop())
	gas := interpreter.cvm.callGasTemp
	// Pop other call parameters.
	addr, value, inOffset, inSize, retOffset, retSize := callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop()
	toAddr := common.BigToAddress(addr)
	value = math.U256(value)
	// Get the arguments from the memory.
	args := callContext.memory.GetPtr(inOffset.Int64(), inSize.Int64())

	if value.Sign() != 0 {
		gas += params.CallStipend
	}
	ret, returnGas, modelGas, err := interpreter.cvm.Call(callContext.contract, toAddr, args, gas, value)
	if err != nil {
		callContext.stack.push(interpreter.intPool.getZero())
	} else {
		callContext.stack.push(interpreter.intPool.get().SetUint64(1))
	}
	if err == nil || err == errExecutionReverted {
		callContext.memory.Set(retOffset.Uint64(), retSize.Uint64(), ret)
	}
	callContext.contract.Gas += returnGas
	for addr, mGas := range modelGas {
		callContext.contract.ModelGas[addr] += mGas
	}

	interpreter.intPool.put(addr, value, inOffset, inSize, retOffset, retSize)

	return ret, nil
}

func opCallCode(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	// Pop gas. The actual gas is in interpreter.cvm.callGasTemp.
	interpreter.intPool.putOne(callContext.stack.pop())
	gas := interpreter.cvm.callGasTemp
	// Pop other call parameters.
	addr, value, inOffset, inSize, retOffset, retSize := callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop()
	toAddr := common.BigToAddress(addr)
	value = math.U256(value)
	// Get arguments from the memory.
	args := callContext.memory.GetPtr(inOffset.Int64(), inSize.Int64())

	if value.Sign() != 0 {
		gas += params.CallStipend
	}
	ret, returnGas, modelGas, err := interpreter.cvm.CallCode(callContext.contract, toAddr, args, gas, value)
	if err != nil {
		callContext.stack.push(interpreter.intPool.getZero())
	} else {
		callContext.stack.push(interpreter.intPool.get().SetUint64(1))
	}
	if err == nil || err == errExecutionReverted {
		callContext.memory.Set(retOffset.Uint64(), retSize.Uint64(), ret)
	}
	callContext.contract.Gas += returnGas
	for addr, mGas := range modelGas {
		callContext.contract.ModelGas[addr] += mGas
	}

	interpreter.intPool.put(addr, value, inOffset, inSize, retOffset, retSize)
	return ret, nil
}

func opDelegateCall(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	// Pop gas. The actual gas is in interpreter.cvm.callGasTemp.
	interpreter.intPool.putOne(callContext.stack.pop())
	gas := interpreter.cvm.callGasTemp
	// Pop other call parameters.
	addr, inOffset, inSize, retOffset, retSize := callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop()
	toAddr := common.BigToAddress(addr)
	// Get arguments from the memory.
	args := callContext.memory.GetPtr(inOffset.Int64(), inSize.Int64())

	ret, returnGas, modelGas, err := interpreter.cvm.DelegateCall(callContext.contract, toAddr, args, gas)
	if err != nil {
		callContext.stack.push(interpreter.intPool.getZero())
	} else {
		callContext.stack.push(interpreter.intPool.get().SetUint64(1))
	}
	if err == nil || err == errExecutionReverted {
		callContext.memory.Set(retOffset.Uint64(), retSize.Uint64(), ret)
	}
	callContext.contract.Gas += returnGas
	for addr, mGas := range modelGas {
		callContext.contract.ModelGas[addr] += mGas
	}

	interpreter.intPool.put(addr, inOffset, inSize, retOffset, retSize)
	return ret, nil
}

func opStaticCall(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	// Pop gas. The actual gas is in interpreter.cvm.callGasTemp.
	interpreter.intPool.putOne(callContext.stack.pop())
	gas := interpreter.cvm.callGasTemp
	// Pop other call parameters.
	addr, inOffset, inSize, retOffset, retSize := callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop(), callContext.stack.pop()
	toAddr := common.BigToAddress(addr)
	// Get arguments from the memory.
	args := callContext.memory.GetPtr(inOffset.Int64(), inSize.Int64())

	ret, returnGas, modelGas, err := interpreter.cvm.StaticCall(callContext.contract, toAddr, args, gas)
	if err != nil {
		callContext.stack.push(interpreter.intPool.getZero())
	} else {
		callContext.stack.push(interpreter.intPool.get().SetUint64(1))
	}
	if err == nil || err == errExecutionReverted {
		callContext.memory.Set(retOffset.Uint64(), retSize.Uint64(), ret)
	}
	callContext.contract.Gas += returnGas
	for addr, mGas := range modelGas {
		callContext.contract.ModelGas[addr] += mGas
	}

	interpreter.intPool.put(addr, inOffset, inSize, retOffset, retSize)
	return ret, nil
}

func opReturn(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	offset, size := callContext.stack.pop(), callContext.stack.pop()
	ret := callContext.memory.GetPtr(offset.Int64(), size.Int64())

	interpreter.intPool.put(offset, size)
	return ret, nil
}

func opRevert(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	offset, size := callContext.stack.pop(), callContext.stack.pop()
	ret := callContext.memory.GetPtr(offset.Int64(), size.Int64())

	interpreter.intPool.put(offset, size)
	return ret, nil
}

func opStop(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	return nil, nil
}

func opSuicide(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
	balance := interpreter.cvm.StateDB.GetBalance(callContext.contract.Address())
	interpreter.cvm.StateDB.AddBalance(common.BigToAddress(callContext.stack.pop()), balance)

	interpreter.cvm.StateDB.Suicide(callContext.contract.Address())
	return nil, nil
}

// make log instruction function
func makeLog(size int) executionFunc {
	return func(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
		topics := make([]common.Hash, size)
		mStart, mSize := callContext.stack.pop(), callContext.stack.pop()
		for i := 0; i < size; i++ {
			topics[i] = common.BigToHash(callContext.stack.pop())
		}

		d := callContext.memory.Get(mStart.Int64(), mSize.Int64())
		interpreter.cvm.StateDB.AddLog(&types.Log{
			Address: callContext.contract.Address(),
			Topics:  topics,
			Data:    d,
			// This is a non-consensus field, but assigned here because
			// core/state doesn't know the current block number.
			BlockNumber: interpreter.cvm.BlockNumber.Uint64(),
		})

		interpreter.intPool.put(mStart, mSize)
		return nil, nil
	}
}

/*func makeAiLog(model common.Hash, input common.Hash, ai uint64, err error, interpreter *CVMInterpreter, contract *Contract) ([]byte, error) {
	topics := make([]common.Hash, 4)
	topics[0] = model
	topics[1] = input
	topics[2] = common.BigToHash(big.NewInt(0).SetUint64(ai))

	if err != nil && ai == 0{
		topics[3] = common.BigToHash(big.NewInt(1))
	} else {
		topics[3] = common.BigToHash(big.NewInt(0))
	}
	interpreter.cvm.StateDB.AddLog(&types.Log{
		Address: callContext.contract.Address(),
		Topics:  topics,
		//Data:    nil,
		// This is a non-consensus field, but assigned here because
		// core/state doesn't know the current block number.
		BlockNumber: interpreter.cvm.BlockNumber.Uint64(),
		Removed: false,
	})
	return nil, nil
}*/

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

		integer := interpreter.intPool.get()
		callContext.stack.push(integer.SetBytes(common.RightPadBytes(callContext.contract.Code[startMin:endMin], pushByteSize)))

		*pc += size
		return nil, nil
	}
}

// make dup instruction function
func makeDup(size int64) executionFunc {
	return func(pc *uint64, interpreter *CVMInterpreter, callContext *callCtx) ([]byte, error) {
		callContext.stack.dup(interpreter.intPool, int(size))
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
