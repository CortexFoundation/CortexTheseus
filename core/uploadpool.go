// Copyright 2015 The go-ethereum Authors
// This file is part of the go-ethereum library.
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
// along with the go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package core

import (
	"fmt"
	"math"
)

// GasPool tracks the amount of gas available during execution of the transactions
// in a block. The zero value is a pool with zero gas available.
type UploadPool uint64

// AddGas makes gas available for execution.
func (gp *UploadPool) AddUpload(amount uint64) *UploadPool {
	if uint64(*gp) > math.MaxUint64-amount {
		panic("gas pool pushed above uint64")
	}
	*(*uint64)(gp) += amount
	return gp
}

// SubGas deducts the given amount from the pool if enough gas is
// available and returns an error otherwise.
func (gp *UploadPool) subUpload(amount uint64) error {
	if uint64(*gp) < amount {
		return ErrUploadLimitReached
	}
	*(*uint64)(gp) -= amount
	return nil
}

func (gp *UploadPool) useUpload() (uint64, error) {
	if uint64(*gp) < 0 {
		return uint64(0), ErrUploadLimitReached
	}
	*(*uint64)(gp) = uint64(*gp) / 2
	return uint64(*gp), nil
}

// Gas returns the amount of gas remaining in the pool.
func (gp *UploadPool) Upload() uint64 {
	return uint64(*gp)
}

func (gp *UploadPool) String() string {
	return fmt.Sprintf("%d", *gp)
}
