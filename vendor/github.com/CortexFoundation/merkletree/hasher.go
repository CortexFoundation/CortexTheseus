// Copyright 2021 The CortexTheseus Authors
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

// Package core implements the Cortex consensus protocol

package merkletree

import (
	"hash"
	"sync"

	"golang.org/x/crypto/sha3"
)

type keccakState interface {
	hash.Hash
	Read([]byte) (int, error)
}

type hashNode []byte

type hasher struct {
	sha keccakState
}

var hasherPool = sync.Pool{
	New: func() interface{} {
		return &hasher{
			sha: sha3.NewLegacyKeccak256().(keccakState),
		}
	},
}

func (h *hasher) sum(data []byte) []byte {
	n := make([]byte, 32)
	h.sha.Reset()
	h.sha.Write(data)
	h.sha.Read(n)
	return n
}

func newHasher() *hasher {
	h := hasherPool.Get().(*hasher)
	return h
}

func returnHasherToPool(h *hasher) {
	hasherPool.Put(h)
}
