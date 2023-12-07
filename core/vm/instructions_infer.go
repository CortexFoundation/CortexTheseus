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

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
	torrentfs "github.com/CortexFoundation/torrentfs/types"
	"github.com/holiman/uint256"
)

func opInfer(pc *uint64, interpreter *CVMInterpreter, callContext *ScopeContext) ([]byte, error) {
	_modelAddr, _inputAddr, _outputOffset := callContext.Stack.pop(), callContext.Stack.pop(), callContext.Stack.pop()
	modelAddr := common.Address(_modelAddr.Bytes20())
	inputAddr := common.Address(_inputAddr.Bytes20())
	var (
		modelMeta *torrentfs.ModelMeta
		inputMeta *torrentfs.InputMeta
	)
	modelMeta, modelErr := checkModel(interpreter.cvm, callContext.Stack, modelAddr)
	if modelErr != nil {
		callContext.Stack.push(new(uint256.Int).Clear())
		return nil, modelErr
	}

	inputMeta, inputErr := checkInputMeta(interpreter.cvm, callContext.Stack, inputAddr)
	if inputErr != nil {
		callContext.Stack.push(new(uint256.Int).Clear())
		return nil, inputErr
	}

	log.Debug("interpreter check shape 1", "modelMeta", modelMeta, "inputMeta", inputMeta)
	// Model&Input shape should match
	if len(modelMeta.InputShape) != len(inputMeta.Shape) {
		callContext.Stack.push(new(uint256.Int).Clear())
		if interpreter.cvm.vmConfig.DebugInferVM {
			fmt.Println("modelmeta: ", modelMeta.InputShape, " inputmeta: ", inputMeta.Shape)
		}
		return nil, errMetaShapeNotMatch
	}
	log.Debug("interpreter check shape 2", "modelMeta", modelMeta, "inputMeta", inputMeta)
	for idx, modelShape := range modelMeta.InputShape {
		if modelShape != inputMeta.Shape[idx] || modelShape == 0 || inputMeta.Shape[idx] == 0 {
			callContext.Stack.push(new(uint256.Int).Clear())
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
		callContext.Stack.push(new(uint256.Int).Clear())
		return nil, err
	}
	if err := callContext.Memory.WriteSolidityUint256Array(int64(_outputOffset.Uint64()), output); err != nil {
		callContext.Stack.push(new(uint256.Int).Clear())
		return nil, err
	}
	callContext.Stack.push(new(uint256.Int).SetOne())

	return nil, nil
}

func checkModel(cvm *CVM, stack *Stack, modelAddr common.Address) (*torrentfs.ModelMeta, error) {
	var (
		modelMeta *torrentfs.ModelMeta
		err       error
	)
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
	if cvm.StateDB.GetNum(modelAddr).Int64() > cvm.Context.BlockNumber.Int64()-matureBlockNumber {
		log.Debug("instructions", "modelAddr", modelAddr, "modelAddrBlkNum", cvm.StateDB.GetNum(modelAddr), "Current", cvm.Context.BlockNumber, "MB", matureBlockNumber)
		return nil, ErrMetaInfoNotMature
	}

	if cvm.StateDB.GetNum(modelAddr).Int64() < cvm.Context.BlockNumber.Int64()-params.ExpiredBlks {
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
	if cvm.StateDB.GetNum(inputAddr).Int64() > cvm.Context.BlockNumber.Int64()-matureBlockNumber {
		log.Debug("instructions", "inputAddr", inputAddr, "inputAddrBlkNum", cvm.StateDB.GetNum(inputAddr), "Current", cvm.Context.BlockNumber, "Uploading", cvm.StateDB.Uploading(inputAddr), "MB", matureBlockNumber)
		return nil, ErrMetaInfoNotMature
	}

	if cvm.StateDB.GetNum(inputAddr).Int64() < cvm.Context.BlockNumber.Int64()-params.ExpiredBlks {
		return nil, errMetaInfoExpired
	}

	return inputMeta, nil
}

func opInferArray(pc *uint64, interpreter *CVMInterpreter, callContext *ScopeContext) ([]byte, error) {
	_modelAddr, _inputHeaderOffset, _outputOffset := callContext.Stack.pop(), callContext.Stack.pop(), callContext.Stack.pop()
	// fmt.Println(fmt.Sprintf("%d, %d, %d", _modelAddr, _inputHeaderOffset, _outputOffset))
	inputBuff, inputError := interpreter.cvm.StateDB.GetSolidityBytes(callContext.Contract.Address(), common.Hash(_inputHeaderOffset.Bytes32()))
	if inputError != nil {
		return nil, inputError
	}
	inputSize := uint256.NewInt(uint64(len(inputBuff)))
	modelAddr := common.Address(_modelAddr.Bytes20())
	// log.Debug(fmt.Sprintf("_input = %v, payload = %v ", inputSize, inputBuff))

	modelMeta, modelErr := checkModel(interpreter.cvm, callContext.Stack, modelAddr)
	if modelErr != nil {
		callContext.Stack.push(new(uint256.Int).Clear())
		return nil, modelErr
	}

	if false {
		//TODO(tian) omit input shape for infer array
		var dataSize uint64 = 1
		for _, modelShape := range modelMeta.InputShape {
			dataSize *= modelShape
		}
		if dataSize != inputSize.Uint64() {
			callContext.Stack.push(new(uint256.Int).Clear())
			if interpreter.cvm.vmConfig.DebugInferVM {
				fmt.Println("modelmeta: ", modelMeta.InputShape, "datasize: ", dataSize, "inputSize: ", inputSize)
			}
			return nil, errMetaShapeNotMatch
		}
	}
	var (
		output []byte
		err    error
	)
	output, err = interpreter.cvm.InferArray(modelMeta.Hash.Hex(),
		inputBuff, modelMeta.RawSize)
	if err != nil {
		callContext.Stack.push(new(uint256.Int).Clear())
		return nil, err
	}
	if interpreter.cvm.vmConfig.DebugInferVM {
		fmt.Println("output", output)
	}
	if err := callContext.Memory.WriteSolidityUint256Array(int64(_outputOffset.Uint64()), output); err != nil {
		callContext.Stack.push(new(uint256.Int).Clear())
		return nil, err
	}
	// interpreter.intPool.get().SetUint64
	callContext.Stack.push(new(uint256.Int).SetOne())

	return nil, nil
}
