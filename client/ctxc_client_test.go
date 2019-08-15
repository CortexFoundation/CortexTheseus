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

package ctxcclient

import "github.com/CortexFoundation/CortexTheseus"

// Verify that Client implements the cortex interfaces.
var (
	_ = cortex.ChainReader(&Client{})
	_ = cortex.TransactionReader(&Client{})
	_ = cortex.ChainStateReader(&Client{})
	_ = cortex.ChainSyncReader(&Client{})
	_ = cortex.ContractCaller(&Client{})
	_ = cortex.GasEstimator(&Client{})
	_ = cortex.GasPricer(&Client{})
	_ = cortex.LogFilterer(&Client{})
	_ = cortex.PendingStateReader(&Client{})
	// _ = cortex.PendingStateEventer(&Client{})
	_ = cortex.PendingContractCaller(&Client{})
)
