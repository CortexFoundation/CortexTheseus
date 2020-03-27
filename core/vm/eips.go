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
)

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
func enable1884(instructionSet *JumpTable) {
	instructionSet[BALANCE].gasCost = constGasFunc(params.BalanceGasEIP1884)
	instructionSet[EXTCODEHASH].gasCost = constGasFunc(params.ExtcodeHashGasEIP1884)
	instructionSet[SLOAD].gasCost = constGasFunc(params.SloadGasEIP1884)

	// New opcode
	instructionSet[SELFBALANCE] = operation{
		execute:       opSelfBalance,
		gasCost:       constGasFunc(GasFastestStep),
		validateStack: makeStackFunc(0, 1),
		valid:         true,
	}
}

func opSelfBalance(pc *uint64, interpreter *CVMInterpreter, contract *Contract, memory *Memory, stack *Stack) ([]byte, error) {
	balance := interpreter.intPool.get().Set(interpreter.cvm.StateDB.GetBalance(contract.Address()))
	stack.push(balance)
	return nil, nil
}

// enable1344 applies EIP-1344 (ChainID Opcode)
// - Adds an opcode that returns the current chain’s EIP-155 unique identifier
func enable1344(instructionSet *JumpTable) {
	// New opcode
	instructionSet[CHAINID] = operation{
		execute: opChainID,
		gasCost: constGasFunc(GasQuickStep),
		//minStack:    minStack(0, 1),
		//maxStack:    maxStack(0, 1),
		validateStack: makeStackFunc(0, 1),
		valid:         true,
	}
}

// opChainID implements CHAINID opcode
func opChainID(pc *uint64, interpreter *CVMInterpreter, contract *Contract, memory *Memory, stack *Stack) ([]byte, error) {
	chainId := interpreter.intPool.get().Set(interpreter.cvm.chainConfig.ChainID)
	stack.push(chainId)
	return nil, nil
}

// enable2200 applies EIP-2200 (Rebalance net-metered SSTORE)
func enable2200(instructionSet *JumpTable) {
	instructionSet[SLOAD].gasCost = constGasFunc(params.SloadGasEIP2200)
	instructionSet[SSTORE].gasCost = gasSStoreEIP2200
}
