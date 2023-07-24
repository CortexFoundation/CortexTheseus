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

type BlockContent struct {
	x    string
	n    uint64
	hash []byte
}

func (t *BlockContent) cache() []byte {
	return t.hash
}

func (t *BlockContent) CalculateHash() ([]byte, error) {
	if cache := t.cache(); cache != nil {
		return cache, nil
	}

	h := newHasher()
	defer returnHasherToPool(h)
	t.hash = h.sum([]byte(t.x))
	return t.hash, nil
}

func (t *BlockContent) Equals(other Content) (bool, error) {
	return t.x == other.(*BlockContent).x && t.n == other.(*BlockContent).n, nil
}

func NewContent(x string, n uint64) *BlockContent {
	return &BlockContent{x: x, n: n}
}

func (t *BlockContent) N() uint64 {
	return t.n
}
