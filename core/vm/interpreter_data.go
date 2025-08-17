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

func (in *CVM) prepareData(contract *Contract, input []byte) ([]byte, error) {
	if in.vmConfig.RPC_GetInternalTransaction || input != nil {
		return nil, nil
	}

	if in.category.IsModel {
		var modelMeta torrentfs.ModelMeta
		if err := modelMeta.DecodeRLP(contract.Code); err != nil {
			log.Error("Failed decode model meta", "code", contract.Code, "err", err)
			return nil, err
		}

		if modelMeta.BlockNum.Sign() == 0 {
			if modelMeta.RawSize > params.MODEL_MIN_UPLOAD_BYTES && modelMeta.RawSize <= params.MODEL_MAX_UPLOAD_BYTES {
				if modelMeta.RawSize > params.DEFAULT_UPLOAD_BYTES {
					in.StateDB.SetUpload(contract.Address(), new(big.Int).SetUint64(modelMeta.RawSize-params.DEFAULT_UPLOAD_BYTES))
				}
			} else {
				return nil, ErrInvalidMetaRawSize
			}

			if !common.IsHexAddress(modelMeta.AuthorAddress.String()) {
				return nil, ErrInvalidMetaAuthor
			}

			if modelMeta.Gas == uint64(0) {
				modelMeta.SetGas(0)
			} else if modelMeta.Gas > params.MODEL_GAS_UP_LIMIT {
				modelMeta.SetGas(params.MODEL_GAS_LIMIT)
			} else if int64(modelMeta.Gas) < 0 {
				modelMeta.SetGas(0)
			}

			in.StateDB.SetNum(contract.Address(), in.Context.BlockNumber)
			modelMeta.SetBlockNum(*in.Context.BlockNumber)

			tmpCode, err := modelMeta.ToBytes()
			if err != nil {
				return nil, err
			}
			contract.Code = append([]byte{0, 1}, tmpCode...)
			log.Debug("Model created", "size", modelMeta.RawSize, "hash", modelMeta.Hash.Hex(), "author", modelMeta.AuthorAddress.Hex(), "gas", modelMeta.Gas, "birth", modelMeta.BlockNum.Uint64())
		} else {
			log.Debug("Invalid model meta", "size", modelMeta.RawSize, "hash", modelMeta.Hash.Hex(), "author", modelMeta.AuthorAddress.Hex(), "gas", modelMeta.Gas, "birth", modelMeta.BlockNum.Uint64())
		}

		info := common.StorageEntry{Hash: modelMeta.Hash.Hex(), Size: 0}
		if err := synapse.Engine().Download(info); err != nil {
			return nil, err
		}
		return contract.Code, nil
	}

	if in.category.IsInput {
		var inputMeta torrentfs.InputMeta
		if err := inputMeta.DecodeRLP(contract.Code); err != nil {
			log.Error("Failed decode input meta", "code", contract.Code, "err", err)
			return nil, err
		}

		if inputMeta.BlockNum.Sign() == 0 {
			if inputMeta.RawSize == 0 {
				return nil, ErrInvalidMetaRawSize
			}
			if inputMeta.RawSize > params.DEFAULT_UPLOAD_BYTES {
				in.StateDB.SetUpload(contract.Address(), new(big.Int).SetUint64(inputMeta.RawSize-params.DEFAULT_UPLOAD_BYTES))
			}

			inputMeta.SetBlockNum(*in.Context.BlockNumber)
			in.StateDB.SetNum(contract.Address(), in.Context.BlockNumber)

			tmpCode, err := inputMeta.ToBytes()
			if err != nil {
				return nil, err
			}
			contract.Code = append([]byte{0, 2}, tmpCode...)
		} else {
			log.Warn("Invalid input meta", "size", inputMeta.RawSize, "hash", inputMeta.Hash.Hex(), "birth", inputMeta.BlockNum.Uint64())
		}

		info := common.StorageEntry{Hash: inputMeta.Hash.Hex(), Size: 0}
		if err := synapse.Engine().Download(info); err != nil {
			return nil, err
		}
		return contract.Code, nil
	}

	return nil, nil
}
