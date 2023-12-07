// Copyright 2019 The go-ethereum Authors
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
	"fmt"
	"math"
)

// QuotaPool tracks the amount of gas available during execution of the transactions
// in a block. The zero value is a pool with zero gas available.
type QuotaPool uint64

// AddGas makes gas available for execution.
func (qp *QuotaPool) AddQuota(amount uint64) *QuotaPool {
	if uint64(*qp) > math.MaxUint64-amount {
		panic("quota pool pushed above uint64")
	}
	*(*uint64)(qp) += amount
	return qp
}

// SubGas deducts the given amount from the pool if enough gas is
// available and returns an error otherwise.
func (qp *QuotaPool) SubQuota(amount uint64) error {
	if uint64(*qp) < amount {
		return ErrQuotaLimitReached
	}
	*(*uint64)(qp) -= amount
	return nil
}

// Gas returns the amount of gas remaining in the pool.
func (qp *QuotaPool) Quota() uint64 {
	return uint64(*qp)
}

func (qp *QuotaPool) String() string {
	return fmt.Sprintf("%d", *qp)
}

func NewQuotaPool(x uint64) *QuotaPool {
	var qp = new(QuotaPool)
	*(*uint64)(qp) = x
	return qp
}
