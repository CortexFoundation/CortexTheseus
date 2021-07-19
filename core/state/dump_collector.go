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

// Package state provides a caching layer atop the Cortex state trie.

package state

import (
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/rlp"
	"github.com/CortexFoundation/CortexTheseus/trie"
)

type DumpCollector interface {
	OnRoot(common.Hash)
	OnAccount(common.Address, DumpAccount)
}

type DumpConfig struct {
	SkipCode          bool
	SkipStorage       bool
	OnlyWithAddresses bool
	Start             []byte
	Max               uint64
}

// DumpToCollector iterates the state according to the given options and inserts
// the items into a collector for aggregation or serialization.
func (s *StateDB)DumpToCollector( c DumpCollector, conf *DumpConfig) (nextKey []byte) {
	// Sanitize the input to allow nil configs
	if conf == nil {
		conf = new(DumpConfig)
	}
	var (
		missingPreimages int
		accounts         uint64
		start            = time.Now()
		logged           = time.Now()
	)
	log.Info("Trie dumping started", "root", s.trie.Hash())
	c.OnRoot(s.trie.Hash())

	it := trie.NewIterator(s.trie.NodeIterator(conf.Start))
	for it.Next() {
		var data Account
		if err := rlp.DecodeBytes(it.Value, &data); err != nil {
			panic(err)
		}
		account := DumpAccount{
			Balance:   data.Balance.String(),
			Nonce:     data.Nonce,
			Root:      string(data.Root[:]),
			CodeHash:  string(data.CodeHash),
			SecureKey: it.Key,
		}
		addrBytes := s.trie.GetKey(it.Key)
		if addrBytes == nil {
			// Preimage missing
			missingPreimages++
			if conf.OnlyWithAddresses {
				continue
			}
			account.SecureKey = it.Key
		}
		addr := common.BytesToAddress(addrBytes)
		obj := newObject(s, addr, data)
		if !conf.SkipCode {
			account.Code = string(obj.Code(s.db))
		}
		if !conf.SkipStorage {
			account.Storage = make(map[common.Hash]string)
			storageIt := trie.NewIterator(obj.getTrie(s.db).NodeIterator(nil))
			for storageIt.Next() {
				_, content, _, err := rlp.Split(storageIt.Value)
				if err != nil {
					log.Error("Failed to decode the value returned by iterator", "error", err)
					continue
				}
				account.Storage[common.BytesToHash(s.trie.GetKey(storageIt.Key))] = common.Bytes2Hex(content)
			}
		}
		c.OnAccount(addr, account)
		accounts++
		if time.Since(logged) > 8*time.Second {
			log.Info("Trie dumping in progress", "at", it.Key, "accounts", accounts,
				"elapsed", common.PrettyDuration(time.Since(start)))
			logged = time.Now()
		}
		if conf.Max > 0 && accounts >= conf.Max {
			if it.Next() {
				nextKey = it.Key
			}
			break
		}
	}
	if missingPreimages > 0 {
		log.Warn("Dump incomplete due to missing preimages", "missing", missingPreimages)
	}
	log.Info("Trie dumping complete", "accounts", accounts,
		"elapsed", common.PrettyDuration(time.Since(start)))

	return nextKey
}
