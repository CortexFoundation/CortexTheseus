// Copyright 2023 The CortexTheseus Authors
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

package robot

import (
	"context"
	"errors"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/types"
	"time"
)

func (m *Monitor) parseFileMeta(tx *types.Transaction, meta *types.FileMeta, b *types.Block) error {
	log.Debug("Monitor", "FileMeta", meta)

	receipt, err := m.getReceipt(tx.Hash.String())
	if err != nil {
		return err
	}

	if receipt.ContractAddr == nil {
		log.Warn("contract address is nil, waiting for indexing", "tx.Hash.String()", tx.Hash.String())
		return errors.New("contract address is nil")
	}

	log.Debug("Transaction Receipt", "address", receipt.ContractAddr.String(), "gas", receipt.GasUsed, "status", receipt.Status) //, "tx", receipt.TxHash.String())

	if receipt.Status != 1 {
		log.Warn("receipt.Status is wrong", "receipt.Status", receipt.Status)
		return nil
	}

	log.Debug("Meta data", "meta", meta)

	info := m.fs.NewFileInfo(meta)

	info.LeftSize = meta.RawSize
	info.ContractAddr = receipt.ContractAddr
	info.Relate = append(info.Relate, *info.ContractAddr)
	op, update, err := m.fs.AddFile(info)
	if err != nil {
		log.Warn("Create file failed", "err", err)
		return err
	}
	if update && op == 1 {
		log.Debug("Create new file", "ih", meta.InfoHash, "op", op)

		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		if m.mode == params.FULL {
			return m.download(ctx, meta.InfoHash, 512*1024)
		} else {
			return m.download(ctx, meta.InfoHash, 0)
		}
	}
	return nil
}

func (m *Monitor) parseBlockTorrentInfo(b *types.Block) (bool, error) {
	var (
		record bool
		start  = mclock.Now()
		final  []types.Transaction
	)

	for _, tx := range b.Txs {
		if meta := tx.Parse(); meta != nil {
			log.Debug("Data encounter", "ih", meta.InfoHash, "number", b.Number, "meta", meta)
			if err := m.parseFileMeta(&tx, meta, b); err != nil {
				log.Error("Parse file meta error", "err", err, "number", b.Number)
				return false, err
			}
			record = true
			final = append(final, tx)
		} else if tx.IsFlowControl() {
			if tx.Recipient == nil {
				continue
			}
			file := m.fs.GetFileByAddr(*tx.Recipient)
			if file == nil {
				continue
			}
			receipt, err := m.getReceipt(tx.Hash.String())
			if err != nil {
				return false, err
			}
			if receipt.Status != 1 || receipt.GasUsed != params.UploadGas {
				continue
			}
			remainingSize, err := m.getRemainingSize((*tx.Recipient).String())
			if err != nil {
				log.Error("Get remain failed", "err", err, "addr", (*tx.Recipient).String())
				return false, err
			}
			if file.LeftSize > remainingSize {
				file.LeftSize = remainingSize
				if _, progress, err := m.fs.AddFile(file); err != nil {
					return false, err
				} else if progress {
					log.Debug("Update storage success", "ih", file.Meta.InfoHash, "left", file.LeftSize)
					var bytesRequested uint64
					if file.Meta.RawSize > file.LeftSize {
						bytesRequested = file.Meta.RawSize - file.LeftSize
					}
					if file.LeftSize == 0 {
						log.Debug("Data processing completed !!!", "ih", file.Meta.InfoHash, "addr", (*tx.Recipient).String(), "remain", common.StorageSize(remainingSize), "request", common.StorageSize(bytesRequested), "raw", common.StorageSize(file.Meta.RawSize), "number", b.Number)
					} else {
						log.Debug("Data processing ...", "ih", file.Meta.InfoHash, "addr", (*tx.Recipient).String(), "remain", common.StorageSize(remainingSize), "request", common.StorageSize(bytesRequested), "raw", common.StorageSize(file.Meta.RawSize), "number", b.Number)
					}
					ctx, cancel := context.WithTimeout(context.Background(), timeout)
					defer cancel()
					if err := m.download(ctx, file.Meta.InfoHash, bytesRequested); err != nil {
						return false, err
					}
				}
			}
			record = true
			final = append(final, tx)
		}
	}

	if len(final) > 0 && len(final) < len(b.Txs) {
		log.Debug("Final txs layout", "total", len(b.Txs), "final", len(final), "num", b.Number, "txs", m.fs.Txs())
		b.Txs = final
	}

	if record {
		if err := m.fs.AddBlock(b); err == nil {
			log.Debug("Root has been changed", "number", b.Number, "hash", b.Hash, "root", m.fs.Root())
		} else {
			log.Warn("Block added failed", "number", b.Number, "hash", b.Hash, "root", m.fs.Root(), "err", err)
		}
	}

	if len(b.Txs) > 0 {
		elapsed := time.Duration(mclock.Now()) - time.Duration(start)
		log.Trace("Transactions scanning", "count", len(b.Txs), "number", b.Number, "elapsed", common.PrettyDuration(elapsed))
	}

	return record, nil
}
