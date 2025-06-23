// Copyright 2019 The go-ethereum Authors
// This file is part of The go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with The go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package core

import (
	"runtime"
	"sync/atomic"

	"github.com/CortexFoundation/CortexTheseus/consensus"
	"github.com/CortexFoundation/CortexTheseus/core/state"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/core/vm"
	"github.com/CortexFoundation/CortexTheseus/params"

	"golang.org/x/sync/errgroup"
)

// statePrefetcher is a basic Prefetcher, which blindly executes a block on top
// of an arbitrary state with the goal of prefetching potentially useful state
// data from disk before the main block processor start executing.
type statePrefetcher struct {
	config *params.ChainConfig // Chain configuration options
	bc     *BlockChain         // Canonical block chain
	engine consensus.Engine    // Consensus engine used for block rewards
}

// newStatePrefetcher initialises a new statePrefetcher.
func newStatePrefetcher(config *params.ChainConfig, bc *BlockChain, engine consensus.Engine) *statePrefetcher {
	return &statePrefetcher{
		config: config,
		bc:     bc,
		engine: engine,
	}
}

// Prefetch processes the state changes according to the Cortex rules by running
// the transaction messages using the statedb, but any changes are discarded. The
// only goal is to pre-cache transaction signatures and state trie nodes.
func (p *statePrefetcher) Prefetch(block *types.Block, statedb *state.StateDB, cfg vm.Config, interrupt *atomic.Bool) {
	var (
		fails     atomic.Int64
		header    = block.Header()
		gaspool   = new(GasPool).AddGas(block.GasLimit())
		signer    = types.MakeSigner(p.config, header.Number, header.Time)
		workers   errgroup.Group
		quotaPool = NewQuotaPool(header.Quota)
	)

	workers.SetLimit(max(1, 4*runtime.NumCPU()/5)) // Aggressively run the prefetching

	if err := quotaPool.SubQuota(header.QuotaUsed); err != nil {
		return
	}

	// Iterate over and process the individual transactions
	for i, tx := range block.Transactions() {
		stateCpy := statedb.Copy() // closure
		workers.Go(func() error {
			// If block precaching was interrupted, abort
			if interrupt != nil && interrupt.Load() {
				return nil
			}
			// We attempt to apply a transaction. The goal is not to execute
			// the transaction successfully, rather to warm up touched data slots.
			cvm := vm.NewCVM(NewCVMBlockContext(header, p.bc, nil), stateCpy, p.config, cfg)

			// Convert the transaction into an executable message and pre-cache its sender
			msg, err := TransactionToMessage(tx, signer)
			if err != nil {
				fails.Add(1)
				return nil // Also invalid block, bail out
			}
			// Disable the nonce check
			msg.SkipNonceChecks = true

			stateCpy.SetTxContext(tx.Hash(), i)

			if _, err := ApplyMessage(cvm, msg, gaspool, quotaPool); err != nil {
				fails.Add(1)
				return nil // Ugh, something went horribly wrong, bail out
			}
			// If we're pre-byzantium, pre-load trie nodes for the intermediate root
			stateCpy.IntermediateRoot(true)

			return nil
		})
	}
	workers.Wait()
	//blockPrefetchTxsValidMeter.Mark(int64(len(block.Transactions())) - fails.Load())
	//blockPrefetchTxsInvalidMeter.Mark(fails.Load())
	return
}
