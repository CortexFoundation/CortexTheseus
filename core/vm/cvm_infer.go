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
	_ "encoding/hex"
	"fmt"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/inference/synapse"
	torrentfs "github.com/CortexFoundation/torrentfs/types"
)

type Category struct {
	IsCode, IsModel, IsInput bool
}

func (cvm *CVM) IsCode(code []byte) bool {
	if len(code) < 2 {
		return false
	}
	return code[0]|code[1] == 0x00
}

func (cvm *CVM) IsModel(code []byte) bool {
	if len(code) < 2 {
		return false
	}
	return code[0]^0x00|code[1]^0x01 == 0x00
}

func (cvm *CVM) IsInput(code []byte) bool {
	if len(code) < 2 {
		return false
	}
	return code[0]^0x00|code[1]^0x02 == 0x00
}

/*const interv = 5

func (cvm *CVM) DataSync(meta common.Address, dir string, errCh chan error) {
	street := big.NewInt(0).Sub(cvm.PeekNumber, cvm.BlockNumber)
	point := big.NewInt(time.Now().Add(confirmTime).Unix())
	if point.Cmp(cvm.Context.Time) > 0 || street.Cmp(big.NewInt(params.CONFIRM_BLOCKS)) > 0 {
		cost := big.NewInt(0)
		duration := big.NewInt(0).Sub(big.NewInt(time.Now().Unix()), cvm.Context.Time)
		for i := 0; i < 3600 && duration.Cmp(cost) > 0; i++ {
			if !torrentfs.ExistTorrent(meta.String()) {
				log.Warn("Inference synchronizing ... ...", "point", point, "tvm", cvm.Context.Time, "ago", common.PrettyDuration(time.Duration(duration.Uint64()*1000000000)), "level", i, "number", cvm.BlockNumber, "street", street)
				cost.Add(cost, big.NewInt(interv))
				time.Sleep(time.Second * interv)
				continue
			} else {
				errCh <- nil
				return
			}
		}
		log.Error("Torrent synchronized timeout", "address", meta.Hex(), "number", cvm.BlockNumber, "meta", meta, "storage", dir, "street", street, "duration", duration, "cost", cost)
	} else {
		if !torrentfs.Exist(meta.String()) {
			log.Warn("Data not exist", "address", meta.Hex(), "number", cvm.BlockNumber, "current", cvm.BlockNumber, "meta", meta, "storage", dir)
			errCh <- synapse.ErrModelFileNotExist
			return
		} else {
			errCh <- nil
			return
		}
	}

	if !torrentfs.Exist(meta.String()) {
		log.Warn("Data not exist", "address", meta.Hex(), "number", cvm.BlockNumber, "current", cvm.BlockNumber, "meta", meta, "storage", dir)
		errCh <- synapse.ErrModelFileNotExist
		return
	} else {
		errCh <- nil
		return
	}

	//log.Error("Torrent synchronized timeout", "address", meta.Hex(), "number", cvm.BlockNumber, "meta", meta, "storage", dir, "street", street)
	//errCh <- synapse.ErrModelFileNotExist
	//return
}*/

// infer function that returns an int64 as output, can be used a categorical output
func (cvm *CVM) Infer(modelInfoHash, inputInfoHash string, modelRawSize, inputRawSize uint64) ([]byte, error) {
	var (
		inferRes []byte
		errRes   error
	)

	start := mclock.Now()

	cvmVersion := synapse.CVMVersion(cvm.chainConfig, cvm.Context.BlockNumber)
	model := common.StorageEntry{
		Hash: modelInfoHash,
		Size: modelRawSize,
	}
	input := common.StorageEntry{
		Hash: inputInfoHash,
		Size: inputRawSize,
	}
	inferRes, errRes = synapse.Engine().InferByInfoHashWithSize(model, input, cvmVersion, cvm.chainConfig.ChainID.Int64())

	elapsed := time.Duration(mclock.Now()) - time.Duration(start)

	if errRes == nil {
		log.Debug("[hash ] succeed", "label", inferRes, "model", modelInfoHash, "input", inputInfoHash, "number", cvm.Context.BlockNumber, "elapsed", common.PrettyDuration(elapsed))
	}
	// ret := synapse.ArgMax(inferRes)
	if cvm.vmConfig.DebugInferVM {
		fmt.Println("infer result: ", inferRes, errRes)
	}
	return inferRes, errRes
}

// infer function that returns an int64 as output, can be used a categorical output
func (cvm *CVM) InferArray(modelInfoHash string, inputArray []byte, modelRawSize uint64) ([]byte, error) {
	log.Trace("Detail", "Input Content", hexutil.Encode(inputArray))

	if cvm.vmConfig.DebugInferVM {
		fmt.Println("Model Hash", modelInfoHash, "number", cvm.Context.BlockNumber, "Input Content", hexutil.Encode(inputArray))
	}

	var (
		inferRes []byte
		errRes   error
	)

	start := mclock.Now()

	cvmVersion := synapse.CVMVersion(cvm.chainConfig, cvm.Context.BlockNumber)
	model := common.StorageEntry{
		Hash: modelInfoHash,
		Size: modelRawSize,
	}
	inferRes, errRes = synapse.Engine().InferByInputContentWithSize(model, inputArray, cvmVersion, cvm.chainConfig.ChainID.Int64())
	elapsed := time.Duration(mclock.Now()) - time.Duration(start)

	if errRes == nil {
		log.Debug("[array] succeed", "label", inferRes, "model", modelInfoHash, "array", inputArray, "number", cvm.Context.BlockNumber, "elapsed", common.PrettyDuration(elapsed))
	}
	return inferRes, errRes
}

// infer function that returns an int64 as output, can be used a categorical output
func (cvm *CVM) OpsInfer(addr common.Address) (opsRes uint64, errRes error) {
	modelMeta, err := cvm.GetModelMeta(addr)
	if err != nil {
		return 0, err
	}

	start := mclock.Now()
	model := common.StorageEntry{
		Hash: modelMeta.Hash.Hex(),
		Size: modelMeta.RawSize,
	}
	opsRes, errRes = synapse.Engine().GetGasByInfoHashWithSize(model, cvm.chainConfig.ChainID.Int64())

	elapsed := time.Duration(mclock.Now()) - time.Duration(start)

	if errRes == nil {
		log.Debug("[ops  ] succeed", "ops", opsRes, "addr", addr, "elapsed", common.PrettyDuration(elapsed))
	}

	return opsRes, errRes
}

func (cvm *CVM) GetModelMeta(addr common.Address) (meta *torrentfs.ModelMeta, err error) {
	modelMetaRaw := cvm.StateDB.GetCode(addr)
	var modelMeta torrentfs.ModelMeta
	if err := modelMeta.DecodeRLP(modelMetaRaw); err != nil {
		return nil, err
	} else {
		return &modelMeta, nil
	}
}

func (cvm *CVM) GetInputMeta(addr common.Address) (meta *torrentfs.InputMeta, err error) {
	inputMetaRaw := cvm.StateDB.GetCode(addr)
	var inputMeta torrentfs.InputMeta
	if err := inputMeta.DecodeRLP(inputMetaRaw); err != nil {
		return nil, err
	} else {
		return &inputMeta, nil
	}
}
