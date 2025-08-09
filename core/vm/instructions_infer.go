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
	"errors"
	"fmt"

	torrentfs "github.com/CortexFoundation/torrentfs/types"
	"github.com/holiman/uint256"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
)

func handleInferError(stack *Stack, err error) error {
	stack.push(new(uint256.Int).Clear())
	return err
}

func checkModel(cvm *CVM, modelAddr common.Address) (*torrentfs.ModelMeta, error) {
	modelMeta, err := cvm.GetModelMeta(modelAddr)
	if err != nil {
		return nil, err
	}
	if cvm.StateDB.Uploading(modelAddr) {
		return nil, errors.New("MODEL IS NOT UPLOADED ERROR")
	}

	blockNum := cvm.StateDB.GetNum(modelAddr)
	if blockNum.Int64() <= 0 {
		return nil, errMetaInfoBlockNum
	}

	matureBlockNumber := cvm.ChainConfig().GetMatureBlock()
	if blockNum.Int64() > cvm.Context.BlockNumber.Int64()-matureBlockNumber {
		log.Debug("instructions", "modelAddr", modelAddr, "modelAddrBlkNum", blockNum, "Current", cvm.Context.BlockNumber, "MB", matureBlockNumber)
		return nil, ErrMetaInfoNotMature
	}

	if blockNum.Int64() < cvm.Context.BlockNumber.Int64()-params.ExpiredBlks {
		return nil, errMetaInfoExpired
	}

	if modelMeta.Gas > params.MODEL_GAS_LIMIT {
		return nil, errors.New("INVALID MODEL GAS LIMIT ERROR")
	}
	return modelMeta, nil
}

func checkInputMeta(cvm *CVM, inputAddr common.Address) (*torrentfs.InputMeta, error) {
	inputMeta, err := cvm.GetInputMeta(inputAddr)
	if err != nil {
		return nil, err
	}
	if cvm.StateDB.Uploading(inputAddr) {
		return nil, errors.New("MODEL IS NOT UPLOADED ERROR")
	}

	blockNum := cvm.StateDB.GetNum(inputAddr)
	if blockNum.Int64() <= 0 {
		return nil, errMetaInfoBlockNum
	}

	matureBlockNumber := cvm.ChainConfig().GetMatureBlock()
	if blockNum.Int64() > cvm.Context.BlockNumber.Int64()-matureBlockNumber {
		log.Debug("instructions", "inputAddr", inputAddr, "inputAddrBlkNum", blockNum, "Current", cvm.Context.BlockNumber, "Uploading", cvm.StateDB.Uploading(inputAddr), "MB", matureBlockNumber)
		return nil, ErrMetaInfoNotMature
	}

	if blockNum.Int64() < cvm.Context.BlockNumber.Int64()-params.ExpiredBlks {
		return nil, errMetaInfoExpired
	}
	return inputMeta, nil
}

func opInfer(pc *uint64, cvm *CVM, scope *ScopeContext) ([]byte, error) {
	_modelAddr, _inputAddr, _outputOffset := scope.Stack.pop(), scope.Stack.pop(), scope.Stack.pop()
	modelAddr := common.Address(_modelAddr.Bytes20())
	inputAddr := common.Address(_inputAddr.Bytes20())

	modelMeta, err := checkModel(cvm, modelAddr)
	if err != nil {
		return nil, handleInferError(scope.Stack, err)
	}

	inputMeta, err := checkInputMeta(cvm, inputAddr)
	if err != nil {
		return nil, handleInferError(scope.Stack, err)
	}

	if len(modelMeta.InputShape) != len(inputMeta.Shape) {
		if cvm.vmConfig.DebugInferVM {
			fmt.Println("modelmeta: ", modelMeta.InputShape, " inputmeta: ", inputMeta.Shape)
		}
		return nil, handleInferError(scope.Stack, errMetaShapeNotMatch)
	}

	for idx, modelShape := range modelMeta.InputShape {
		if modelShape != inputMeta.Shape[idx] || modelShape == 0 || inputMeta.Shape[idx] == 0 {
			if cvm.vmConfig.DebugInferVM {
				fmt.Println("modelmeta: ", modelMeta.InputShape, " inputmeta: ", inputMeta.Shape)
			}
			return nil, handleInferError(scope.Stack, errMetaShapeNotMatch)
		}
	}

	output, err := cvm.Infer(modelMeta.Hash.Hex(), inputMeta.Hash.Hex(), modelMeta.RawSize, inputMeta.RawSize)
	if err != nil {
		return nil, handleInferError(scope.Stack, err)
	}

	if err := scope.Memory.WriteSolidityUint256Array(int64(_outputOffset.Uint64()), output); err != nil {
		return nil, handleInferError(scope.Stack, err)
	}

	scope.Stack.push(new(uint256.Int).SetOne())
	return nil, nil
}

func opInferArray(pc *uint64, cvm *CVM, scope *ScopeContext) ([]byte, error) {
	_modelAddr, _inputHeaderOffset, _outputOffset := scope.Stack.pop(), scope.Stack.pop(), scope.Stack.pop()
	modelAddr := common.Address(_modelAddr.Bytes20())

	inputBuff, inputError := cvm.StateDB.GetSolidityBytes(scope.Contract.Address(), common.Hash(_inputHeaderOffset.Bytes32()))
	if inputError != nil {
		return nil, inputError
	}

	modelMeta, err := checkModel(cvm, modelAddr)
	if err != nil {
		return nil, handleInferError(scope.Stack, err)
	}

	//TODO(tian) enable input shape check for infer array
	// var dataSize uint64 = 1
	// for _, modelShape := range modelMeta.InputShape {
	// 	dataSize *= modelShape
	// }
	// if dataSize != uint64(len(inputBuff)) {
	// 	return nil, handleInferError(scope.Stack, errMetaShapeNotMatch)
	// }

	output, err := cvm.InferArray(modelMeta.Hash.Hex(), inputBuff, modelMeta.RawSize)
	if err != nil {
		return nil, handleInferError(scope.Stack, err)
	}

	if err := scope.Memory.WriteSolidityUint256Array(int64(_outputOffset.Uint64()), output); err != nil {
		return nil, handleInferError(scope.Stack, err)
	}

	scope.Stack.push(new(uint256.Int).SetOne())
	return nil, nil
}
