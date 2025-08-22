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
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/types"
)

func (m *Monitor) parseFileMeta(tx *types.Transaction, meta *types.FileMeta, b *types.Block) error {
	log.Debug("Parsing FileMeta", "infoHash", meta.InfoHash)

	// Step 1: Get transaction receipt
	receipt, err := m.getReceipt(tx.Hash.String())
	if err != nil {
		log.Error("Failed to get receipt", "txHash", tx.Hash.String(), "err", err)
		return err
	}

	// Step 2: Validate receipt
	if receipt.ContractAddr == nil {
		// More descriptive error message is better
		err = errors.New("contract address is nil")
		log.Warn("Contract address is nil, unable to proceed", "txHash", tx.Hash.String())
		return err
	}

	if receipt.Status != 1 {
		log.Warn("Transaction receipt status is not successful", "txHash", tx.Hash.String(), "status", receipt.Status)
		return nil // Return nil for unsuccessful transactions as it's a valid state
	}

	log.Debug("Transaction receipt OK", "address", receipt.ContractAddr.String(), "gas", receipt.GasUsed)

	// Step 3: Create and add file information
	info := m.fs.NewFileInfo(meta)
	info.LeftSize = meta.RawSize
	info.ContractAddr = receipt.ContractAddr
	info.Relate = append(info.Relate, *info.ContractAddr)

	op, update, err := m.fs.AddFile(info)
	if err != nil {
		log.Warn("Failed to add file to filesystem", "infoHash", meta.InfoHash, "err", err)
		return err
	}

	// Step 4: Handle file download if it's a new file
	if update && op == 1 {
		log.Debug("New file created, initiating download", "infoHash", meta.InfoHash)

		sizeLimit := uint64(0)
		if m.mode == params.FULL {
			sizeLimit = 512 * 1024 // Set a specific size limit for full mode
		}

		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		return m.download(ctx, meta.InfoHash, sizeLimit)
	}

	return nil
}

func (m *Monitor) parseBlockTorrentInfo(b *types.Block) (bool, error) {
	start := mclock.Now()
	var (
		record bool
		final  []types.Transaction
	)

	for _, tx := range b.Txs {
		// Attempt to parse transaction for file metadata
		if meta := tx.Parse(); meta != nil {
			log.Debug("Data encounter", "infoHash", meta.InfoHash, "blockNumber", b.Number)
			if err := m.parseFileMeta(&tx, meta, b); err != nil {
				log.Error("Parse file meta failed", "error", err, "blockNumber", b.Number, "txHash", tx.Hash)
				return false, err
			}
			record = true
			final = append(final, tx)
			continue
		}

		// Handle flow control transactions
		if tx.IsFlowControl() {
			// Use guard clauses to reduce nesting
			if tx.Recipient == nil {
				continue
			}

			file := m.fs.GetFileByAddr(*tx.Recipient)
			if file == nil {
				continue
			}

			receipt, err := m.getReceipt(tx.Hash.String())
			if err != nil {
				log.Error("Failed to get receipt for flow control tx", "error", err, "txHash", tx.Hash)
				return false, err
			}

			if receipt.Status != 1 || receipt.GasUsed != params.UploadGas {
				continue
			}

			// All checks passed, process the flow control transaction
			remainingSize, err := m.getRemainingSize((*tx.Recipient).String())
			if err != nil {
				log.Error("Get remaining size failed", "error", err, "addr", (*tx.Recipient).String())
				return false, err
			}

			if file.LeftSize > remainingSize {
				file.LeftSize = remainingSize
				_, progress, err := m.fs.AddFile(file)
				if err != nil {
					return false, err
				}

				if progress {
					bytesRequested := uint64(0)
					if file.Meta.RawSize > file.LeftSize {
						bytesRequested = file.Meta.RawSize - file.LeftSize
					}

					logMsg := "Data processing..."
					if file.LeftSize == 0 {
						logMsg = "Data processing completed!"
					}

					log.Debug(logMsg,
						"infoHash", file.Meta.InfoHash,
						"addr", (*tx.Recipient).String(),
						"remain", common.StorageSize(remainingSize),
						"request", common.StorageSize(bytesRequested),
						"raw", common.StorageSize(file.Meta.RawSize),
						"blockNumber", b.Number)

					ctx, cancel := context.WithTimeout(context.Background(), timeout)
					if err := m.download(ctx, file.Meta.InfoHash, bytesRequested); err != nil {
						cancel() // Ensure cancel is called on error
						return false, err
					}
					cancel() // Call cancel when download is successful
				}
			}
			record = true
			final = append(final, tx)
		}
	}

	// Update block transactions if necessary
	if len(final) > 0 && len(final) < len(b.Txs) {
		log.Debug("Txs filtered", "total", len(b.Txs), "final", len(final), "blockNumber", b.Number)
		b.Txs = final
	}

	// Add block to filesystem if any relevant transactions were found
	if record {
		if err := m.fs.AddBlock(b); err != nil {
			log.Warn("Block added failed", "blockNumber", b.Number, "blockHash", b.Hash, "root", m.fs.Root(), "error", err)
			return false, err // Return the error if adding the block fails
		}
		log.Debug("Block added to filesystem", "blockNumber", b.Number, "blockHash", b.Hash, "root", m.fs.Root())
	}

	// Log transaction scanning time
	if len(b.Txs) > 0 {
		elapsed := time.Duration(mclock.Now()) - time.Duration(start)
		log.Trace("Transaction scanning complete", "count", len(b.Txs), "blockNumber", b.Number, "elapsed", common.PrettyDuration(elapsed))
	}

	return record, nil
}
