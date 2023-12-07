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

package core

import (
	"github.com/CortexFoundation/CortexTheseus/common"
	math2 "github.com/CortexFoundation/CortexTheseus/common/math"
	"github.com/CortexFoundation/CortexTheseus/core/vm"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/inference/synapse"
	torrentfs "github.com/CortexFoundation/torrentfs/types"

	"math/big"
)

func (st *StateTransition) preQuotaCheck() error {
	if st.uploading() {
		// log.Debug("state_transition", "uploading", st.uploading(), "st.state.GetNum(st.to())", st.state.GetNum(st.to()))
		if st.state.GetNum(st.to()).Cmp(big0) <= 0 {
			log.Warn("Uploading block number is zero", "address", st.to(), "number", st.state.GetNum(st.to()), "current", st.cvm.Context.BlockNumber)
			return ErrUnhandleTx
		}

		if st.state.GetNum(st.to()).Cmp(new(big.Int).Sub(st.cvm.Context.BlockNumber, big.NewInt(params.SeedingBlks))) > 0 {
			log.Warn("Not ready for seeding", "address", st.to(), "number", st.state.GetNum(st.to()), "current", st.cvm.Context.BlockNumber, "seeding", params.SeedingBlks)
			return ErrUnhandleTx
		}
		cost := math2.Uint64Min(params.PER_UPLOAD_BYTES, st.state.Upload(st.to()).Uint64())
		// log.Debug("state_transition",
		//                                "new(big.Int).SetUint64(params.PER_UPLOAD_BYTES)", new(big.Int).SetUint64(params.PER_UPLOAD_BYTES),
		//                                      "st.state.Upload(st.to())", st.state.Upload(st.to()), "cost", cost, "st.qp", st.qp)
		if err := st.qp.SubQuota(cost); err != nil {
			log.Warn("Quota waiting ... ...", "quotapool", st.qp.String(), "cost", st.state.Upload(st.to()), "current", st.cvm.Context.BlockNumber)
			return ErrQuotaLimitReached
		}

		//meta, err := st.cvm.GetMetaHash(st.to())
		//if err != nil {
		//      log.Warn("Uploading meta is not exist", "address", st.to(), "number", st.state.GetNum(st.to()), "current", st.cvm.BlockNumber)
		//      return ErrUnhandleTx
		//}

		//errCh := make(chan error)
		//go st.TorrentSync(meta, st.cvm.Config().StorageDir, errCh)
		//select {
		//case err := <-errCh:
		//      if err != nil {
		//              return err
		//      }
		//}
	}

	return nil
}

func (st *StateTransition) quotaCalculate() (quota uint64, err error) {
	if st.uploading() {
		cur := st.state.Upload(st.to()).Uint64()
		if cur > 0 {
			quota = math2.Uint64Min(params.PER_UPLOAD_BYTES, cur)

			var (
				remain  uint64
				ih      string
				request uint64
			)

			remain = st.state.SubUpload(st.to(), new(big.Int).SetUint64(quota)).Uint64()
			if remain == 0 {
				st.state.SetNum(st.to(), st.cvm.Context.BlockNumber)
				log.Debug("Upload OK", "address", st.to().Hex(), "number", st.cvm.Context.BlockNumber, "nonce", st.msg.Nonce)
			} else {
				log.Debug("Waiting ...", "address", st.to().Hex(), "number", st.cvm.Context.BlockNumber, "remain", remain)
			}

			raw := st.state.GetCode(st.to())
			if st.cvm.IsModel(raw) {
				var modelMeta torrentfs.ModelMeta
				if err = modelMeta.DecodeRLP(raw); err == nil {
					ih = modelMeta.Hash.Hex()
					request = modelMeta.RawSize - remain
				}
			} else if st.cvm.IsInput(raw) {
				var inputMeta torrentfs.InputMeta
				if err = inputMeta.DecodeRLP(raw); err == nil {
					ih = inputMeta.Hash.Hex()
					request = inputMeta.RawSize - remain
				}
			} else {
				return 0, vm.ErrRuntime
			}

			if err != nil {
				return 0, vm.ErrRuntime
			}

			if err = synapse.Engine().Download(common.StorageEntry{
				Hash: ih,
				Size: request,
			}); err != nil {
				return 0, err
			}
		}
	}

	return
}

func (st *StateTransition) modelGasCalculate(gu uint64) (uint64, error) {
	for addr, mgas := range st.modelGas {
		if mgas > params.MODEL_GAS_UP_LIMIT {
			continue
		}

		if gu < mgas {
			return 0, vm.ErrInsufficientBalance
		}

		gu -= mgas
		reward := new(big.Int).Mul(new(big.Int).SetUint64(mgas), st.msg.GasPrice)
		log.Debug("Model author reward", "author", addr.Hex(), "reward", reward, "number", st.cvm.Context.BlockNumber)
		st.state.AddBalance(addr, reward)
	}
	return gu, nil
}
