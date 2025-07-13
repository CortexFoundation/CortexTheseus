// Copyright 2019 The go-ethereum Authors
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

package vm

import (
	"fmt"
	"sort"

	"github.com/holiman/uint256"

	"github.com/CortexFoundation/CortexTheseus/params"
)

var activators = map[int]func(*JumpTable){
	5656: enable5656,
	3855: enable3855,
	2200: enable2200,
	1884: enable1884,
	1344: enable1344,
	7939: enable7939,
}

// EnableEIP enables the given EIP on the config.
// This operation writes in-place, and callers need to ensure that the globally
// defined jump tables are not polluted.
func EnableEIP(eipNum int, jt *JumpTable) error {
	enablerFn, ok := activators[eipNum]
	if !ok {
		return fmt.Errorf("undefined eip %d", eipNum)
	}
	enablerFn(jt)
	return nil
}

func ValidEip(eipNum int) bool {
	_, ok := activators[eipNum]
	return ok
}
func ActivateableEips() []string {
	var nums []string
	for k := range activators {
		nums = append(nums, fmt.Sprintf("%d", k))
	}
	sort.Strings(nums)
	return nums
}

// opCLZ implements the CLZ opcode (count leading zero bytes)
func opCLZ(pc *uint64, interpreter *CVMInterpreter, scope *ScopeContext) ([]byte, error) {
	x := scope.Stack.peek()
	x.SetUint64(256 - uint64(x.BitLen()))
	return nil, nil
}

func enable7939(jt *JumpTable) {
	jt[CLZ] = &operation{
		execute:       opCLZ,
		gasCost:       constGasFunc(GasFastStep),
		validateStack: makeStackFunc(1, 1),
	}
}

// enable5656 enables EIP-5656 (MCOPY opcode)
// https://eips.cortex.org/EIPS/eip-5656
func enable5656(jt *JumpTable) {
	jt[MCOPY] = &operation{
		execute: opMcopy,
		//gasCost:       constGasFunc(GasFastestStep),
		gasCost:       gasMcopy,
		validateStack: makeStackFunc(3, 0),
		memorySize:    memoryMcopy,
	}
}

// opMcopy implements the MCOPY opcode (https://eips.cortex.org/EIPS/eip-5656)
func opMcopy(pc *uint64, interpreter *CVMInterpreter, scope *ScopeContext) ([]byte, error) {
	var (
		dst    = scope.Stack.pop()
		src    = scope.Stack.pop()
		length = scope.Stack.pop()
	)
	// These values are checked for overflow during memory expansion calculation
	// (the memorySize function on the opcode).
	scope.Memory.Copy(dst.Uint64(), src.Uint64(), length.Uint64())
	return nil, nil
}

// enable1884 applies EIP-1884 to the given jump table:
// - Increase cost of BALANCE to 700
// - Increase cost of EXTCODEHASH to 700
// - Increase cost of SLOAD to 800
// - Define SELFBALANCE, with cost GasFastStep (5)
func enable1884(jt *JumpTable) {
	jt[BALANCE].gasCost = constGasFunc(params.BalanceGasEIP1884)
	jt[EXTCODEHASH].gasCost = constGasFunc(params.ExtcodeHashGasEIP1884)
	jt[SLOAD].gasCost = constGasFunc(params.SloadGasEIP1884)

	// New opcode
	jt[SELFBALANCE] = &operation{
		execute:       opSelfBalance,
		gasCost:       constGasFunc(GasFastestStep),
		validateStack: makeStackFunc(0, 1),
	}
}

func opSelfBalance(pc *uint64, interpreter *CVMInterpreter, scope *ScopeContext) ([]byte, error) {
	balance, _ := uint256.FromBig(interpreter.cvm.StateDB.GetBalance(scope.Contract.Address()))
	scope.Stack.push(balance)
	return nil, nil
}

// enable1344 applies EIP-1344 (ChainID Opcode)
// - Adds an opcode that returns the current chain’s EIP-155 unique identifier
func enable1344(jt *JumpTable) {
	// New opcode
	jt[CHAINID] = &operation{
		execute: opChainID,
		gasCost: constGasFunc(GasQuickStep),
		//minStack:    minStack(0, 1),
		//maxStack:    maxStack(0, 1),
		validateStack: makeStackFunc(0, 1),
	}
}

// opChainID implements CHAINID opcode
func opChainID(pc *uint64, interpreter *CVMInterpreter, scope *ScopeContext) ([]byte, error) {
	chainId, _ := uint256.FromBig(interpreter.cvm.chainConfig.ChainID)
	scope.Stack.push(chainId)
	return nil, nil
}

// enable2200 applies EIP-2200 (Rebalance net-metered SSTORE)
func enable2200(jt *JumpTable) {
	jt[SLOAD].gasCost = constGasFunc(params.SloadGasEIP2200)
	jt[SSTORE].gasCost = gasSStoreEIP2200
}

// enable3855 applies EIP-3855 (PUSH0 opcode)
func enable3855(jt *JumpTable) {
	// New opcode
	jt[PUSH0] = &operation{
		execute:       opPush0,
		gasCost:       constGasFunc(GasQuickStep),
		validateStack: makeStackFunc(0, 1),
	}
}

// opPush0 implements the PUSH0 opcode
func opPush0(pc *uint64, interpreter *CVMInterpreter, scope *ScopeContext) ([]byte, error) {
	scope.Stack.push(new(uint256.Int))
	return nil, nil
}
