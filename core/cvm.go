// Copyright 2018 The CortexTheseus Authors
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
	"math/big"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/consensus"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/core/vm"
	"github.com/CortexFoundation/CortexTheseus/log"
)

// ChainContext supports retrieving headers and consensus parameters from the
// current blockchain to be used during transaction processing.
type ChainContext interface {
	// Engine retrieves the chain's consensus engine.
	Engine() consensus.Engine

	// GetHeader returns the header corresponding to the hash/number argument pair.
	GetHeader(common.Hash, uint64) *types.Header
}

// NewCVMContext creates a new context for use in the CVM.
func NewCVMBlockContext(header *types.Header, chain ChainContext, author *common.Address) vm.BlockContext {
	// If we don't have an explicit author (i.e. not mining), extract from the header
	var (
		beneficiary common.Address
		random      *common.Hash
	)
	if author == nil {
		beneficiary, _ = chain.Engine().Author(header) // Ignore error, we're past header validation
	} else {
		beneficiary = *author
	}
	if header.Difficulty.Cmp(common.Big0) == 0 {
		random = &header.MixDigest
	}
	return vm.BlockContext{
		CanTransfer: CanTransfer,
		Transfer:    Transfer,
		GetHash:     GetHashFn(header, chain),
		Coinbase:    beneficiary,
		BlockNumber: new(big.Int).Set(header.Number),
		Time:        header.Time,
		Difficulty:  new(big.Int).Set(header.Difficulty),
		GasLimit:    header.GasLimit,
		Quota:       header.Quota,
		Random:      random,
	}
}

// NewCVMTxContext creates a new transaction context for a single transaction.
func NewCVMTxContext(msg *Message) vm.TxContext {
	return vm.TxContext{
		Origin:   msg.From,
		GasPrice: new(big.Int).Set(msg.GasPrice),
	}
}

// GetHashFn returns a GetHashFunc which retrieves header hashes by number
func GetHashFn(ref *types.Header, chain ChainContext) func(n uint64) common.Hash {
	var cache []common.Hash

	return func(n uint64) common.Hash {
		// If there's no hash cache yet, make one
		if len(cache) == 0 {
			cache = append(cache, ref.ParentHash)
		}
		// Try to fulfill the request from the cache
		if idx := ref.Number.Uint64() - n - 1; idx < uint64(len(cache)) {
			return cache[idx]
		}
		// No luck in the cache, but we can start iterating from the last element we already know
		lastKnownHash := cache[len(cache)-1]
		lastKnownNumber := ref.Number.Uint64() - uint64(len(cache))
		for {
			header := chain.GetHeader(lastKnownHash, lastKnownNumber)
			if header == nil {
				break
			}
			cache = append(cache, header.ParentHash)
			lastKnownHash = header.ParentHash
			lastKnownNumber = header.Number.Uint64() - 1
			if n == lastKnownNumber {
				return lastKnownHash
			}
		}
		return common.Hash{}
	}
}

// CanTransfer checks whether there are enough funds in the address' account to make a transfer.
// This does not take the necessary gas in to account to make the transfer valid.
func CanTransfer(db vm.StateDB, addr common.Address, amount *big.Int) bool {
	return db.GetBalance(addr).Cmp(amount) >= 0
}

// Transfer subtracts amount from sender and adds amount to recipient using the given Db
func Transfer(db vm.StateDB, sender, recipient common.Address, amount *big.Int) {
	db.SubBalance(sender, amount)
	db.AddBalance(recipient, amount)
	log.Trace("Transfer subtracts amount from sender and adds amount to recipient using the given Db", "sender", sender, "recipient", recipient, "amount", amount)
}
