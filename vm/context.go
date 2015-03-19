package vm

import (
	"math"
	"math/big"

	"github.com/ethereum/go-ethereum/common"
)

type ContextRef interface {
	ReturnGas(*big.Int, *big.Int)
	Address() common.Address
	SetCode([]byte)
}

type Context struct {
	caller ContextRef
	self   ContextRef

	Code     []byte
	CodeAddr *common.Address

	value, Gas, UsedGas, Price *big.Int

	Args []byte
}

// Create a new context for the given data items
func NewContext(caller ContextRef, object ContextRef, value, gas, price *big.Int) *Context {
	c := &Context{caller: caller, self: object, Args: nil}

	// Gas should be a pointer so it can safely be reduced through the run
	// This pointer will be off the state transition
	c.Gas = gas //new(big.Int).Set(gas)
	c.value = new(big.Int).Set(value)
	// In most cases price and value are pointers to transaction objects
	// and we don't want the transaction's values to change.
	c.Price = new(big.Int).Set(price)
	c.UsedGas = new(big.Int)

	return c
}

func (c *Context) GetOp(n uint64) OpCode {
	return OpCode(c.GetByte(n))
}

func (c *Context) GetByte(n uint64) byte {
	if n < uint64(len(c.Code)) {
		return c.Code[n]
	}

	return 0
}

func (c *Context) GetBytes(x, y int) []byte {
	return c.GetRangeValue(uint64(x), uint64(y))
}

func (c *Context) GetRangeValue(x, size uint64) []byte {
	x = uint64(math.Min(float64(x), float64(len(c.Code))))
	y := uint64(math.Min(float64(x+size), float64(len(c.Code))))

	return common.RightPadBytes(c.Code[x:y], int(size))
}

func (c *Context) GetCode(x, size uint64) []byte {
	return getData(c.Code, x, size)
}

func (c *Context) Return(ret []byte) []byte {
	// Return the remaining gas to the caller
	c.caller.ReturnGas(c.Gas, c.Price)

	return ret
}

/*
 * Gas functions
 */
func (c *Context) UseGas(gas *big.Int) bool {
	if c.Gas.Cmp(gas) < 0 {
		return false
	}

	// Sub the amount of gas from the remaining
	c.Gas.Sub(c.Gas, gas)
	c.UsedGas.Add(c.UsedGas, gas)

	return true
}

// Implement the caller interface
func (c *Context) ReturnGas(gas, price *big.Int) {
	// Return the gas to the context
	c.Gas.Add(c.Gas, gas)
	c.UsedGas.Sub(c.UsedGas, gas)
}

/*
 * Set / Get
 */
func (c *Context) Address() common.Address {
	return c.self.Address()
}

func (self *Context) SetCode(code []byte) {
	self.Code = code
}

func (self *Context) SetCallCode(addr *common.Address, code []byte) {
	self.Code = code
	self.CodeAddr = addr
}
