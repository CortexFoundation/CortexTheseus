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
)

var big0 = big.NewInt(0)

func (s *stateObject) SubUpload(amount *big.Int) *big.Int {
	if amount.Sign() == 0 {
		return s.Upload()
	}
	var ret *big.Int = big0
	if s.Upload().Cmp(amount) > 0 {
		ret = new(big.Int).Sub(s.Upload(), amount)
	}
	s.SetUpload(ret)
	return ret
}

func (s *stateObject) SetUpload(amount *big.Int) {
	s.db.journal.append(uploadChange{
		account: &s.address,
		prev:    new(big.Int).Set(s.data.Upload),
	})
	s.setUpload(amount)
}

func (s *stateObject) setNum(num *big.Int) {
	s.data.Num = num
}

func (s *stateObject) SetNum(num *big.Int) {
	s.db.journal.append(numChange{
		account: &s.address,
		prev:    new(big.Int).Set(s.data.Num),
	})
	s.setNum(num)
}

func (s *stateObject) setUpload(amount *big.Int) {
	s.data.Upload = amount
}

func (s *stateObject) Upload() *big.Int {
	return s.data.Upload
}
func (s *stateObject) Num() *big.Int {
	return s.data.Num
}
