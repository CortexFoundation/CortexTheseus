// Copyright 2023 The CortexTheseus Authors
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
	_ "fmt"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/math"
	"github.com/CortexFoundation/CortexTheseus/params"
)

func gasInfer(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	modelAddr := common.Address(stack.Back(0).Bytes20())
	inputAddr := common.Address(stack.Back(1).Bytes20())
	_, modelErr := checkModel(cvm, stack, modelAddr)
	if modelErr != nil {
		return 0, modelErr
	}
	_, inputErr := checkInputMeta(cvm, stack, inputAddr)
	if inputErr != nil {
		return 0, inputErr
	}

	gas, err := memoryGasCost(mem, 0)
	if err != nil {
		return 0, err
	}
	modelOps, errOps := cvm.OpsInfer(modelAddr)
	if errOps != nil {
		return 0, errOps
	}
	modelGas := modelOps / params.InferOpsPerGas
	var overflow bool
	if gas, overflow = math.SafeAdd(gas, modelGas); overflow {
		return 0, ErrGasUintOverflow
	}
	return gas, nil
}

func gasInferArray(gt params.GasTable, cvm *CVM, contract *Contract, stack *Stack, mem *Memory, memorySize uint64) (uint64, error) {
	modelAddr := common.Address(stack.Back(0).Bytes20())
	_, modelErr := checkModel(cvm, stack, modelAddr)
	if modelErr != nil {
		return 0, modelErr
	}
	gas, err := memoryGasCost(mem, 0)
	if err != nil {
		return 0, err
	}
	modelOps, errOps := cvm.OpsInfer(modelAddr)
	if errOps != nil {
		return 0, errOps
	}
	modelGas := modelOps / params.InferOpsPerGas
	if modelGas < params.CallInferGas {
		modelGas = params.CallInferGas
	}
	var overflow bool
	if gas, overflow = math.SafeAdd(gas, modelGas); overflow {
		return 0, ErrGasUintOverflow
	}
	return gas, nil
}
