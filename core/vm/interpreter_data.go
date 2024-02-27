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
	"math/big"

	"github.com/CortexFoundation/inference/synapse"
	torrentfs "github.com/CortexFoundation/torrentfs/types"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
)

func (in *CVMInterpreter) prepareData(contract *Contract, input []byte) ([]byte, error) {
	if in.cvm.vmConfig.RPC_GetInternalTransaction {
		return nil, nil
	}

	if input != nil {
		log.Debug("Readonly for input meta")
		return nil, nil
	}

	if in.cvm.category.IsModel {
		var modelMeta torrentfs.ModelMeta
		if err := modelMeta.DecodeRLP(contract.Code); err != nil {
			log.Error("Failed decode model meta", "code", contract.Code, "err", err)
			return nil, err
		} else {
			log.Debug("Model meta",
				"meta", modelMeta,
				"modelMeta.RawSize", modelMeta.RawSize,
				"Upload", in.cvm.StateDB.Upload(contract.Address()),
				"params.MODEL_MIN_UPLOAD_BYTES", params.MODEL_MIN_UPLOAD_BYTES)
			if modelMeta.BlockNum.Sign() == 0 {
				if modelMeta.RawSize > params.MODEL_MIN_UPLOAD_BYTES && modelMeta.RawSize <= params.MODEL_MAX_UPLOAD_BYTES { // 1Byte ~ 1TB
					if modelMeta.RawSize > params.DEFAULT_UPLOAD_BYTES {
						in.cvm.StateDB.SetUpload(contract.Address(), new(big.Int).SetUint64(modelMeta.RawSize-params.DEFAULT_UPLOAD_BYTES))
					}
				} else {
					return nil, ErrInvalidMetaRawSize
				}

				if !common.IsHexAddress(modelMeta.AuthorAddress.String()) {
					return nil, ErrInvalidMetaAuthor
				}

				if modelMeta.Gas == uint64(0) {
					//modelMeta.SetGas(params.MODEL_GAS_LIMIT)
					modelMeta.SetGas(0)
				} else if modelMeta.Gas > params.MODEL_GAS_UP_LIMIT {
					modelMeta.SetGas(params.MODEL_GAS_LIMIT)
				} else if int64(modelMeta.Gas) < 0 {
					modelMeta.SetGas(0)
				}

				in.cvm.StateDB.SetNum(contract.Address(), in.cvm.Context.BlockNumber)
				modelMeta.SetBlockNum(*in.cvm.Context.BlockNumber)
				if tmpCode, err := modelMeta.ToBytes(); err != nil {
					return nil, err
				} else {
					contract.Code = append([]byte{0, 1}, tmpCode...)
				}
				log.Debug("Model created", "size", modelMeta.RawSize, "hash", modelMeta.Hash.Hex(), "author", modelMeta.AuthorAddress.Hex(), "gas", modelMeta.Gas, "birth", modelMeta.BlockNum.Uint64())
			} else {
				log.Debug("Invalid model meta", "size", modelMeta.RawSize, "hash", modelMeta.Hash.Hex(), "author", modelMeta.AuthorAddress.Hex(), "gas", modelMeta.Gas, "birth", modelMeta.BlockNum.Uint64())
			}
			info := common.StorageEntry{
				Hash: modelMeta.Hash.Hex(),
				Size: 0,
			}
			if err := synapse.Engine().Download(info); err != nil {
				return nil, err
			}
			return contract.Code, nil
		}
	} else if in.cvm.category.IsInput {
		var inputMeta torrentfs.InputMeta
		if err := inputMeta.DecodeRLP(contract.Code); err != nil {
			log.Error("Failed decode input meta", "code", contract.Code, "err", err)
			return nil, err
		} else {
			if inputMeta.BlockNum.Sign() == 0 {
				if inputMeta.RawSize > 0 {
					if inputMeta.RawSize > params.DEFAULT_UPLOAD_BYTES {
						in.cvm.StateDB.SetUpload(contract.Address(), new(big.Int).SetUint64(inputMeta.RawSize-params.DEFAULT_UPLOAD_BYTES))
					}
				} else {
					return nil, ErrInvalidMetaRawSize
				}

				inputMeta.SetBlockNum(*in.cvm.Context.BlockNumber)
				in.cvm.StateDB.SetNum(contract.Address(), in.cvm.Context.BlockNumber)
				if tmpCode, err := inputMeta.ToBytes(); err != nil {
					return nil, err
				} else {
					contract.Code = append([]byte{0, 2}, tmpCode...)
				}
				//log.Info("Input meta created", "size", inputMeta.RawSize, "author", inputMeta.AuthorAddress)
			} else {
				log.Warn("Invalid input meta", "size", inputMeta.RawSize, "hash", inputMeta.Hash.Hex(), "birth", inputMeta.BlockNum.Uint64())
			}
			info := common.StorageEntry{
				Hash: inputMeta.Hash.Hex(),
				Size: 0,
			}
			if err := synapse.Engine().Download(info); err != nil {
				return nil, err
			}
			return contract.Code, nil
		}
	}

	return nil, nil
}
