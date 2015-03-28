package vm

import (
	"fmt"
	"math/big"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/crypto"
)

type Vm struct {
	env Environment

	logTy  byte
	logStr string

	err error
	// For logging
	debug bool

	BreakPoints []int64
	Stepping    bool
	Fn          string

	Recoverable bool

	// Will be called before the vm returns
	After func(*Context, error)
}

func New(env Environment) *Vm {
	lt := LogTyPretty

	return &Vm{debug: Debug, env: env, logTy: lt, Recoverable: true}
}

func (self *Vm) Run(context *Context, callData []byte) (ret []byte, err error) {
	self.env.SetDepth(self.env.Depth() + 1)
	defer self.env.SetDepth(self.env.Depth() - 1)

	var (
		caller = context.caller
		code   = context.Code
		value  = context.value
		price  = context.Price
	)

	self.Printf("(%d) (%x) %x (code=%d) gas: %v (d) %x", self.env.Depth(), caller.Address().Bytes()[:4], context.Address(), len(code), context.Gas, callData).Endl()

	// User defer pattern to check for an error and, based on the error being nil or not, use all gas and return.
	defer func() {
		if self.After != nil {
			self.After(context, err)
		}

		if err != nil {
			self.Printf(" %v", err).Endl()
			// In case of a VM exception (known exceptions) all gas consumed (panics NOT included).
			context.UseGas(context.Gas)

			ret = context.Return(nil)
		}
	}()

	if context.CodeAddr != nil {
		if p := Precompiled[context.CodeAddr.Str()]; p != nil {
			return self.RunPrecompiled(p, callData, context)
		}
	}

	var (
		op OpCode

		destinations        = analyseJumpDests(context.Code)
		mem                 = NewMemory()
		stack               = newStack()
		pc           uint64 = 0
		step                = 0
		statedb             = self.env.State()

		jump = func(from uint64, to *big.Int) error {
			p := to.Uint64()

			nop := context.GetOp(p)
			if !destinations.Has(p) {
				return fmt.Errorf("invalid jump destination (%v) %v", nop, p)
			}

			self.Printf(" ~> %v", to)
			pc = to.Uint64()

			self.Endl()

			return nil
		}
	)

	// Don't bother with the execution if there's no code.
	if len(code) == 0 {
		return context.Return(nil), nil
	}

	for {
		// The base for all big integer arithmetic
		base := new(big.Int)

		step++
		// Get the memory location of pc
		op = context.GetOp(pc)

		self.Printf("(pc) %-3d -o- %-14s (m) %-4d (s) %-4d ", pc, op.String(), mem.Len(), stack.len())
		newMemSize, gas, err := self.calculateGasAndSize(context, caller, op, statedb, mem, stack)
		if err != nil {
			return nil, err
		}

		self.Printf("(g) %-3v (%v)", gas, context.Gas)

		if !context.UseGas(gas) {
			self.Endl()

			tmp := new(big.Int).Set(context.Gas)

			context.UseGas(context.Gas)

			return context.Return(nil), OOG(gas, tmp)
		}

		mem.Resize(newMemSize.Uint64())

		switch op {
		// 0x20 range
		case ADD:
			x, y := stack.pop(), stack.pop()
			self.Printf(" %v + %v", y, x)

			base.Add(x, y)

			U256(base)

			self.Printf(" = %v", base)
			// pop result back on the stack
			stack.push(base)
		case SUB:
			x, y := stack.pop(), stack.pop()
			self.Printf(" %v - %v", y, x)

			base.Sub(x, y)

			U256(base)

			self.Printf(" = %v", base)
			// pop result back on the stack
			stack.push(base)
		case MUL:
			x, y := stack.pop(), stack.pop()
			self.Printf(" %v * %v", y, x)

			base.Mul(x, y)

			U256(base)

			self.Printf(" = %v", base)
			// pop result back on the stack
			stack.push(base)
		case DIV:
			x, y := stack.pop(), stack.pop()
			self.Printf(" %v / %v", x, y)

			if y.Cmp(common.Big0) != 0 {
				base.Div(x, y)
			}

			U256(base)

			self.Printf(" = %v", base)
			// pop result back on the stack
			stack.push(base)
		case SDIV:
			x, y := S256(stack.pop()), S256(stack.pop())

			self.Printf(" %v / %v", x, y)

			if y.Cmp(common.Big0) == 0 {
				base.Set(common.Big0)
			} else {
				n := new(big.Int)
				if new(big.Int).Mul(x, y).Cmp(common.Big0) < 0 {
					n.SetInt64(-1)
				} else {
					n.SetInt64(1)
				}

				base.Div(x.Abs(x), y.Abs(y)).Mul(base, n)

				U256(base)
			}

			self.Printf(" = %v", base)
			stack.push(base)
		case MOD:
			x, y := stack.pop(), stack.pop()

			self.Printf(" %v %% %v", x, y)

			if y.Cmp(common.Big0) == 0 {
				base.Set(common.Big0)
			} else {
				base.Mod(x, y)
			}

			U256(base)

			self.Printf(" = %v", base)
			stack.push(base)
		case SMOD:
			x, y := S256(stack.pop()), S256(stack.pop())

			self.Printf(" %v %% %v", x, y)

			if y.Cmp(common.Big0) == 0 {
				base.Set(common.Big0)
			} else {
				n := new(big.Int)
				if x.Cmp(common.Big0) < 0 {
					n.SetInt64(-1)
				} else {
					n.SetInt64(1)
				}

				base.Mod(x.Abs(x), y.Abs(y)).Mul(base, n)

				U256(base)
			}

			self.Printf(" = %v", base)
			stack.push(base)

		case EXP:
			x, y := stack.pop(), stack.pop()

			self.Printf(" %v ** %v", x, y)

			base.Exp(x, y, Pow256)

			U256(base)

			self.Printf(" = %v", base)

			stack.push(base)
		case SIGNEXTEND:
			back := stack.pop()
			if back.Cmp(big.NewInt(31)) < 0 {
				bit := uint(back.Uint64()*8 + 7)
				num := stack.pop()
				mask := new(big.Int).Lsh(common.Big1, bit)
				mask.Sub(mask, common.Big1)
				if common.BitTest(num, int(bit)) {
					num.Or(num, mask.Not(mask))
				} else {
					num.And(num, mask)
				}

				num = U256(num)

				self.Printf(" = %v", num)

				stack.push(num)
			}
		case NOT:
			stack.push(U256(new(big.Int).Not(stack.pop())))
			//base.Sub(Pow256, stack.pop()).Sub(base, common.Big1)
			//base = U256(base)
			//stack.push(base)
		case LT:
			x, y := stack.pop(), stack.pop()
			self.Printf(" %v < %v", x, y)
			// x < y
			if x.Cmp(y) < 0 {
				stack.push(common.BigTrue)
			} else {
				stack.push(common.BigFalse)
			}
		case GT:
			x, y := stack.pop(), stack.pop()
			self.Printf(" %v > %v", x, y)

			// x > y
			if x.Cmp(y) > 0 {
				stack.push(common.BigTrue)
			} else {
				stack.push(common.BigFalse)
			}

		case SLT:
			x, y := S256(stack.pop()), S256(stack.pop())
			self.Printf(" %v < %v", x, y)
			// x < y
			if x.Cmp(S256(y)) < 0 {
				stack.push(common.BigTrue)
			} else {
				stack.push(common.BigFalse)
			}
		case SGT:
			x, y := S256(stack.pop()), S256(stack.pop())
			self.Printf(" %v > %v", x, y)

			// x > y
			if x.Cmp(y) > 0 {
				stack.push(common.BigTrue)
			} else {
				stack.push(common.BigFalse)
			}

		case EQ:
			x, y := stack.pop(), stack.pop()
			self.Printf(" %v == %v", y, x)

			// x == y
			if x.Cmp(y) == 0 {
				stack.push(common.BigTrue)
			} else {
				stack.push(common.BigFalse)
			}
		case ISZERO:
			x := stack.pop()
			if x.Cmp(common.BigFalse) > 0 {
				stack.push(common.BigFalse)
			} else {
				stack.push(common.BigTrue)
			}

			// 0x10 range
		case AND:
			x, y := stack.pop(), stack.pop()
			self.Printf(" %v & %v", y, x)

			stack.push(base.And(x, y))
		case OR:
			x, y := stack.pop(), stack.pop()
			self.Printf(" %v | %v", x, y)

			stack.push(base.Or(x, y))
		case XOR:
			x, y := stack.pop(), stack.pop()
			self.Printf(" %v ^ %v", x, y)

			stack.push(base.Xor(x, y))
		case BYTE:
			th, val := stack.pop(), stack.pop()

			if th.Cmp(big.NewInt(32)) < 0 {
				byt := big.NewInt(int64(common.LeftPadBytes(val.Bytes(), 32)[th.Int64()]))

				base.Set(byt)
			} else {
				base.Set(common.BigFalse)
			}

			self.Printf(" => 0x%x", base.Bytes())

			stack.push(base)
		case ADDMOD:
			x := stack.pop()
			y := stack.pop()
			z := stack.pop()

			if z.Cmp(Zero) > 0 {
				add := new(big.Int).Add(x, y)
				base.Mod(add, z)

				base = U256(base)
			}

			self.Printf(" %v + %v %% %v = %v", x, y, z, base)

			stack.push(base)
		case MULMOD:
			x := stack.pop()
			y := stack.pop()
			z := stack.pop()

			if z.Cmp(Zero) > 0 {
				mul := new(big.Int).Mul(x, y)
				base.Mod(mul, z)

				U256(base)
			}

			self.Printf(" %v + %v %% %v = %v", x, y, z, base)

			stack.push(base)

			// 0x20 range
		case SHA3:
			offset, size := stack.pop(), stack.pop()
			data := crypto.Sha3(mem.Get(offset.Int64(), size.Int64()))

			stack.push(common.BigD(data))

			self.Printf(" => (%v) %x", size, data)
			// 0x30 range
		case ADDRESS:
			stack.push(common.Bytes2Big(context.Address().Bytes()))

			self.Printf(" => %x", context.Address())
		case BALANCE:
			addr := common.BigToAddress(stack.pop())
			balance := statedb.GetBalance(addr)

			stack.push(balance)

			self.Printf(" => %v (%x)", balance, addr)
		case ORIGIN:
			origin := self.env.Origin()

			stack.push(origin.Big())

			self.Printf(" => %x", origin)
		case CALLER:
			caller := context.caller.Address()
			stack.push(common.Bytes2Big(caller.Bytes()))

			self.Printf(" => %x", caller)
		case CALLVALUE:
			stack.push(value)

			self.Printf(" => %v", value)
		case CALLDATALOAD:
			var (
				offset  = stack.pop()
				data    = make([]byte, 32)
				lenData = big.NewInt(int64(len(callData)))
			)

			if lenData.Cmp(offset) >= 0 {
				length := new(big.Int).Add(offset, common.Big32)
				length = common.BigMin(length, lenData)

				copy(data, callData[offset.Int64():length.Int64()])
			}

			self.Printf(" => 0x%x", data)

			stack.push(common.BigD(data))
		case CALLDATASIZE:
			l := int64(len(callData))
			stack.push(big.NewInt(l))

			self.Printf(" => %d", l)
		case CALLDATACOPY:
			var (
				mOff = stack.pop()
				cOff = stack.pop()
				l    = stack.pop()
			)
			data := getData(callData, cOff, l)

			mem.Set(mOff.Uint64(), l.Uint64(), data)

			self.Printf(" => [%v, %v, %v]", mOff, cOff, l)
		case CODESIZE, EXTCODESIZE:
			var code []byte
			if op == EXTCODESIZE {
				addr := common.BigToAddress(stack.pop())

				code = statedb.GetCode(addr)
			} else {
				code = context.Code
			}

			l := big.NewInt(int64(len(code)))
			stack.push(l)

			self.Printf(" => %d", l)
		case CODECOPY, EXTCODECOPY:
			var code []byte
			if op == EXTCODECOPY {
				addr := common.BigToAddress(stack.pop())
				code = statedb.GetCode(addr)
			} else {
				code = context.Code
			}

			var (
				mOff = stack.pop()
				cOff = stack.pop()
				l    = stack.pop()
			)

			codeCopy := getData(code, cOff, l)

			mem.Set(mOff.Uint64(), l.Uint64(), codeCopy)

			self.Printf(" => [%v, %v, %v] %x", mOff, cOff, l, codeCopy)
		case GASPRICE:
			stack.push(context.Price)

			self.Printf(" => %x", context.Price)

			// 0x40 range
		case BLOCKHASH:
			num := stack.pop()

			n := new(big.Int).Sub(self.env.BlockNumber(), common.Big257)
			if num.Cmp(n) > 0 && num.Cmp(self.env.BlockNumber()) < 0 {
				stack.push(self.env.GetHash(num.Uint64()).Big())
			} else {
				stack.push(common.Big0)
			}

			self.Printf(" => 0x%x", stack.peek().Bytes())
		case COINBASE:
			coinbase := self.env.Coinbase()

			stack.push(coinbase.Big())

			self.Printf(" => 0x%x", coinbase)
		case TIMESTAMP:
			time := self.env.Time()

			stack.push(big.NewInt(time))

			self.Printf(" => 0x%x", time)
		case NUMBER:
			number := self.env.BlockNumber()

			stack.push(U256(number))

			self.Printf(" => 0x%x", number.Bytes())
		case DIFFICULTY:
			difficulty := self.env.Difficulty()

			stack.push(difficulty)

			self.Printf(" => 0x%x", difficulty.Bytes())
		case GASLIMIT:
			self.Printf(" => %v", self.env.GasLimit())

			stack.push(self.env.GasLimit())

			// 0x50 range
		case PUSH1, PUSH2, PUSH3, PUSH4, PUSH5, PUSH6, PUSH7, PUSH8, PUSH9, PUSH10, PUSH11, PUSH12, PUSH13, PUSH14, PUSH15, PUSH16, PUSH17, PUSH18, PUSH19, PUSH20, PUSH21, PUSH22, PUSH23, PUSH24, PUSH25, PUSH26, PUSH27, PUSH28, PUSH29, PUSH30, PUSH31, PUSH32:
			a := uint64(op - PUSH1 + 1)
			byts := context.GetRangeValue(pc+1, a)
			// push value to stack
			stack.push(common.BigD(byts))
			pc += a

			step += int(op) - int(PUSH1) + 1

			self.Printf(" => 0x%x", byts)
		case POP:
			stack.pop()
		case DUP1, DUP2, DUP3, DUP4, DUP5, DUP6, DUP7, DUP8, DUP9, DUP10, DUP11, DUP12, DUP13, DUP14, DUP15, DUP16:
			n := int(op - DUP1 + 1)
			stack.dup(n)

			self.Printf(" => [%d] 0x%x", n, stack.peek().Bytes())
		case SWAP1, SWAP2, SWAP3, SWAP4, SWAP5, SWAP6, SWAP7, SWAP8, SWAP9, SWAP10, SWAP11, SWAP12, SWAP13, SWAP14, SWAP15, SWAP16:
			n := int(op - SWAP1 + 2)
			stack.swap(n)

			self.Printf(" => [%d]", n)
		case LOG0, LOG1, LOG2, LOG3, LOG4:
			n := int(op - LOG0)
			topics := make([]common.Hash, n)
			mStart, mSize := stack.pop(), stack.pop()
			for i := 0; i < n; i++ {
				topics[i] = common.BigToHash(stack.pop()) //common.LeftPadBytes(stack.pop().Bytes(), 32)
			}

			data := mem.Get(mStart.Int64(), mSize.Int64())
			log := &Log{context.Address(), topics, data, self.env.BlockNumber().Uint64()}
			self.env.AddLog(log)

			self.Printf(" => %v", log)
		case MLOAD:
			offset := stack.pop()
			val := common.BigD(mem.Get(offset.Int64(), 32))
			stack.push(val)

			self.Printf(" => 0x%x", val.Bytes())
		case MSTORE: // Store the value at stack top-1 in to memory at location stack top
			// pop value of the stack
			mStart, val := stack.pop(), stack.pop()
			mem.Set(mStart.Uint64(), 32, common.BigToBytes(val, 256))

			self.Printf(" => 0x%x", val)
		case MSTORE8:
			off, val := stack.pop().Int64(), stack.pop().Int64()

			mem.store[off] = byte(val & 0xff)

			self.Printf(" => [%v] 0x%x", off, mem.store[off])
		case SLOAD:
			loc := common.BigToHash(stack.pop())
			val := common.Bytes2Big(statedb.GetState(context.Address(), loc))
			stack.push(val)

			self.Printf(" {0x%x : 0x%x}", loc, val.Bytes())
		case SSTORE:
			loc := common.BigToHash(stack.pop())
			val := stack.pop()

			statedb.SetState(context.Address(), loc, val)

			self.Printf(" {0x%x : 0x%x}", loc, val.Bytes())
		case JUMP:
			if err := jump(pc, stack.pop()); err != nil {
				return nil, err
			}

			continue
		case JUMPI:
			pos, cond := stack.pop(), stack.pop()

			if cond.Cmp(common.BigTrue) >= 0 {
				if err := jump(pc, pos); err != nil {
					return nil, err
				}

				continue
			}

			self.Printf(" ~> false")

		case JUMPDEST:
		case PC:
			stack.push(big.NewInt(int64(pc)))
		case MSIZE:
			stack.push(big.NewInt(int64(mem.Len())))
		case GAS:
			stack.push(context.Gas)

			self.Printf(" => %x", context.Gas)
			// 0x60 range
		case CREATE:

			var (
				value        = stack.pop()
				offset, size = stack.pop(), stack.pop()
				input        = mem.Get(offset.Int64(), size.Int64())
				gas          = new(big.Int).Set(context.Gas)
				addr         common.Address
			)
			self.Endl()

			context.UseGas(context.Gas)
			ret, suberr, ref := self.env.Create(context, input, gas, price, value)
			if suberr != nil {
				stack.push(common.BigFalse)

				self.Printf(" (*) 0x0 %v", suberr)
			} else {
				// gas < len(ret) * CreateDataGas == NO_CODE
				dataGas := big.NewInt(int64(len(ret)))
				dataGas.Mul(dataGas, GasCreateByte)
				if context.UseGas(dataGas) {
					ref.SetCode(ret)
				}
				addr = ref.Address()

				stack.push(addr.Big())

			}

		case CALL, CALLCODE:
			gas := stack.pop()
			// pop gas and value of the stack.
			addr, value := stack.pop(), stack.pop()
			value = U256(value)
			// pop input size and offset
			inOffset, inSize := stack.pop(), stack.pop()
			// pop return size and offset
			retOffset, retSize := stack.pop(), stack.pop()

			address := common.BigToAddress(addr)
			self.Printf(" => %x", address).Endl()

			// Get the arguments from the memory
			args := mem.Get(inOffset.Int64(), inSize.Int64())

			if len(value.Bytes()) > 0 {
				gas.Add(gas, GasStipend)
			}

			var (
				ret []byte
				err error
			)
			if op == CALLCODE {
				ret, err = self.env.CallCode(context, address, args, gas, price, value)
			} else {
				ret, err = self.env.Call(context, address, args, gas, price, value)
			}

			if err != nil {
				stack.push(common.BigFalse)

				self.Printf("%v").Endl()
			} else {
				stack.push(common.BigTrue)

				mem.Set(retOffset.Uint64(), retSize.Uint64(), ret)
			}
			self.Printf("resume %x (%v)", context.Address(), context.Gas)
		case RETURN:
			offset, size := stack.pop(), stack.pop()
			ret := mem.Get(offset.Int64(), size.Int64())

			self.Printf(" => [%v, %v] (%d) 0x%x", offset, size, len(ret), ret).Endl()

			return context.Return(ret), nil
		case SUICIDE:
			receiver := statedb.GetOrNewStateObject(common.BigToAddress(stack.pop()))
			balance := statedb.GetBalance(context.Address())

			self.Printf(" => (%x) %v", receiver.Address().Bytes()[:4], balance)

			receiver.AddBalance(balance)

			statedb.Delete(context.Address())

			fallthrough
		case STOP: // Stop the context
			self.Endl()

			return context.Return(nil), nil
		default:
			self.Printf("(pc) %-3v Invalid opcode %x\n", pc, op).Endl()

			return nil, fmt.Errorf("Invalid opcode %x", op)
		}

		pc++

		self.Endl()
	}
}

func (self *Vm) calculateGasAndSize(context *Context, caller ContextRef, op OpCode, statedb *state.StateDB, mem *Memory, stack *stack) (*big.Int, *big.Int, error) {
	var (
		gas                 = new(big.Int)
		newMemSize *big.Int = new(big.Int)
	)
	err := baseCheck(op, stack, gas)
	if err != nil {
		return nil, nil, err
	}

	// stack Check, memory resize & gas phase
	switch op {
	case SWAP1, SWAP2, SWAP3, SWAP4, SWAP5, SWAP6, SWAP7, SWAP8, SWAP9, SWAP10, SWAP11, SWAP12, SWAP13, SWAP14, SWAP15, SWAP16:
		n := int(op - SWAP1 + 2)
		err := stack.require(n)
		if err != nil {
			return nil, nil, err
		}
		gas.Set(GasFastestStep)
	case DUP1, DUP2, DUP3, DUP4, DUP5, DUP6, DUP7, DUP8, DUP9, DUP10, DUP11, DUP12, DUP13, DUP14, DUP15, DUP16:
		n := int(op - DUP1 + 1)
		err := stack.require(n)
		if err != nil {
			return nil, nil, err
		}
		gas.Set(GasFastestStep)
	case LOG0, LOG1, LOG2, LOG3, LOG4:
		n := int(op - LOG0)
		err := stack.require(n + 2)
		if err != nil {
			return nil, nil, err
		}

		mSize, mStart := stack.data[stack.len()-2], stack.data[stack.len()-1]

		gas.Add(gas, GasLogBase)
		gas.Add(gas, new(big.Int).Mul(big.NewInt(int64(n)), GasLogTopic))
		gas.Add(gas, new(big.Int).Mul(mSize, GasLogByte))

		newMemSize = calcMemSize(mStart, mSize)
	case EXP:
		gas.Add(gas, new(big.Int).Mul(big.NewInt(int64(len(stack.data[stack.len()-2].Bytes()))), GasExpByte))
	case SSTORE:
		err := stack.require(2)
		if err != nil {
			return nil, nil, err
		}

		var g *big.Int
		y, x := stack.data[stack.len()-2], stack.data[stack.len()-1]
		val := statedb.GetState(context.Address(), common.BigToHash(x))
		if len(val) == 0 && len(y.Bytes()) > 0 {
			// 0 => non 0
			g = GasStorageAdd
		} else if len(val) > 0 && len(y.Bytes()) == 0 {
			statedb.Refund(self.env.Origin(), RefundStorage)

			g = GasStorageMod
		} else {
			// non 0 => non 0 (or 0 => 0)
			g = GasStorageMod
		}
		gas.Set(g)
	case SUICIDE:
		if !statedb.IsDeleted(context.Address()) {
			statedb.Refund(self.env.Origin(), RefundSuicide)
		}
	case MLOAD:
		newMemSize = calcMemSize(stack.peek(), u256(32))
	case MSTORE8:
		newMemSize = calcMemSize(stack.peek(), u256(1))
	case MSTORE:
		newMemSize = calcMemSize(stack.peek(), u256(32))
	case RETURN:
		newMemSize = calcMemSize(stack.peek(), stack.data[stack.len()-2])
	case SHA3:
		newMemSize = calcMemSize(stack.peek(), stack.data[stack.len()-2])

		words := toWordSize(stack.data[stack.len()-2])
		gas.Add(gas, words.Mul(words, GasSha3Word))
	case CALLDATACOPY:
		newMemSize = calcMemSize(stack.peek(), stack.data[stack.len()-3])

		words := toWordSize(stack.data[stack.len()-3])
		gas.Add(gas, words.Mul(words, GasCopyWord))
	case CODECOPY:
		newMemSize = calcMemSize(stack.peek(), stack.data[stack.len()-3])

		words := toWordSize(stack.data[stack.len()-3])
		gas.Add(gas, words.Mul(words, GasCopyWord))
	case EXTCODECOPY:
		newMemSize = calcMemSize(stack.data[stack.len()-2], stack.data[stack.len()-4])

		words := toWordSize(stack.data[stack.len()-4])
		gas.Add(gas, words.Mul(words, GasCopyWord))

	case CREATE:
		newMemSize = calcMemSize(stack.data[stack.len()-2], stack.data[stack.len()-3])
	case CALL, CALLCODE:
		gas.Add(gas, stack.data[stack.len()-1])

		if op == CALL {
			if self.env.State().GetStateObject(common.BigToAddress(stack.data[stack.len()-2])) == nil {
				gas.Add(gas, GasCallNewAccount)
			}
		}

		if len(stack.data[stack.len()-3].Bytes()) > 0 {
			gas.Add(gas, GasCallValueTransfer)
		}

		x := calcMemSize(stack.data[stack.len()-6], stack.data[stack.len()-7])
		y := calcMemSize(stack.data[stack.len()-4], stack.data[stack.len()-5])

		newMemSize = common.BigMax(x, y)
	}

	if newMemSize.Cmp(common.Big0) > 0 {
		newMemSizeWords := toWordSize(newMemSize)
		newMemSize.Mul(newMemSizeWords, u256(32))

		if newMemSize.Cmp(u256(int64(mem.Len()))) > 0 {
			oldSize := toWordSize(big.NewInt(int64(mem.Len())))
			pow := new(big.Int).Exp(oldSize, common.Big2, Zero)
			linCoef := new(big.Int).Mul(oldSize, GasMemWord)
			quadCoef := new(big.Int).Div(pow, GasQuadCoeffDenom)
			oldTotalFee := new(big.Int).Add(linCoef, quadCoef)

			pow.Exp(newMemSizeWords, common.Big2, Zero)
			linCoef = new(big.Int).Mul(newMemSizeWords, GasMemWord)
			quadCoef = new(big.Int).Div(pow, GasQuadCoeffDenom)
			newTotalFee := new(big.Int).Add(linCoef, quadCoef)

			gas.Add(gas, new(big.Int).Sub(newTotalFee, oldTotalFee))
		}
	}

	return newMemSize, gas, nil
}

func (self *Vm) RunPrecompiled(p *PrecompiledAccount, callData []byte, context *Context) (ret []byte, err error) {
	gas := p.Gas(len(callData))
	if context.UseGas(gas) {
		ret = p.Call(callData)
		self.Printf("NATIVE_FUNC => %x", ret)
		self.Endl()

		return context.Return(ret), nil
	} else {
		self.Printf("NATIVE_FUNC => failed").Endl()

		tmp := new(big.Int).Set(context.Gas)

		return nil, OOG(gas, tmp)
	}
}

func (self *Vm) Printf(format string, v ...interface{}) VirtualMachine {
	if self.debug {
		if self.logTy == LogTyPretty {
			self.logStr += fmt.Sprintf(format, v...)
		}
	}

	return self
}

func (self *Vm) Endl() VirtualMachine {
	if self.debug {
		if self.logTy == LogTyPretty {
			vmlogger.Infoln(self.logStr)
			self.logStr = ""
		}
	}

	return self
}

func (self *Vm) Env() Environment {
	return self.env
}
