// Copyright 2019 The CortexTheseus Authors
// This file is part of the CortexTheseus library.
//
// The CortexTheseus library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexTheseus library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the CortexTheseus library. If not, see <http://www.gnu.org/licenses/>.

package vm

import (
	"fmt"

	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/holiman/uint256"
)

var activators = map[int]func(*JumpTable){
	3855: enable3855,
	2200: enable2200,
	1884: enable1884,
	1344: enable1344,
}

// EnableEIP enables the given EIP on the config.
// This operation writes in-place, and callers need to ensure that the globally
// defined jump tables are not polluted.
func EnableEIP(eipNum int, jt *JumpTable) error {
	switch eipNum {
	case 2200:
		enable2200(jt)
	case 1884:
		enable1884(jt)
	case 1344:
		enable1344(jt)
	default:
		return fmt.Errorf("undefined eip %d", eipNum)
	}
	return nil
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

func opSelfBalance(pc *uint64, interpreter *CVMInterpreter, callContext *ScopeContext) ([]byte, error) {
	balance, _ := uint256.FromBig(interpreter.cvm.StateDB.GetBalance(callContext.Contract.Address()))
	callContext.Stack.push(balance)
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
func opChainID(pc *uint64, interpreter *CVMInterpreter, callContext *ScopeContext) ([]byte, error) {
	chainId, _ := uint256.FromBig(interpreter.cvm.chainConfig.ChainID)
	callContext.Stack.push(chainId)
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
func opPush0(pc *uint64, interpreter *CVMInterpreter, callContext *ScopeContext) ([]byte, error) {
	callContext.Stack.push(new(uint256.Int))
	return nil, nil
}
