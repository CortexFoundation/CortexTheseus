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

package state

import (
	"math/big"

	"github.com/CortexFoundation/CortexTheseus/common"
)

type (
	uploadChange struct {
		account common.Address
		prev    *big.Int
	}
	numChange struct {
		account common.Address
		prev    *big.Int
	}
)

func (ch uploadChange) copy() journalEntry {
	return uploadChange{
		account: ch.account,
		prev:    new(big.Int).Set(ch.prev),
	}
}

func (ch numChange) copy() journalEntry {
	return numChange{
		account: ch.account,
		prev:    new(big.Int).Set(ch.prev),
	}
}

func (ch uploadChange) revert(s *StateDB) {
	s.getStateObject(ch.account).setUpload(ch.prev)
}

func (ch uploadChange) dirtied() *common.Address {
	return &ch.account
}

func (ch numChange) revert(s *StateDB) {
	s.getStateObject(ch.account).setNum(ch.prev)
}

func (ch numChange) dirtied() *common.Address {
	return &ch.account
}

func (j *journal) uploadChange(addr common.Address, previous *big.Int) {
	j.append(uploadChange{
		account: addr,
		prev:    new(big.Int).Set(previous),
	})
}

func (j *journal) numChange(addr common.Address, previous *big.Int) {
	j.append(numChange{
		account: addr,
		prev:    new(big.Int).Set(previous),
	})
}
