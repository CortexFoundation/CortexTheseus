// Copyright 2019 The CortexTheseus Authors
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

// Contains the metrics collected by the downloader.

package downloader

import (
	"github.com/CortexFoundation/CortexTheseus/metrics"
)

var (
	headerInMeter      = metrics.NewRegisteredMeter("ctxc/downloader/headers/in", nil)
	headerReqTimer     = metrics.NewRegisteredTimer("ctxc/downloader/headers/req", nil)
	headerDropMeter    = metrics.NewRegisteredMeter("ctxc/downloader/headers/drop", nil)
	headerTimeoutMeter = metrics.NewRegisteredMeter("ctxc/downloader/headers/timeout", nil)

	bodyInMeter      = metrics.NewRegisteredMeter("ctxc/downloader/bodies/in", nil)
	bodyReqTimer     = metrics.NewRegisteredTimer("ctxc/downloader/bodies/req", nil)
	bodyDropMeter    = metrics.NewRegisteredMeter("ctxc/downloader/bodies/drop", nil)
	bodyTimeoutMeter = metrics.NewRegisteredMeter("ctxc/downloader/bodies/timeout", nil)

	receiptInMeter      = metrics.NewRegisteredMeter("ctxc/downloader/receipts/in", nil)
	receiptReqTimer     = metrics.NewRegisteredTimer("ctxc/downloader/receipts/req", nil)
	receiptDropMeter    = metrics.NewRegisteredMeter("ctxc/downloader/receipts/drop", nil)
	receiptTimeoutMeter = metrics.NewRegisteredMeter("ctxc/downloader/receipts/timeout", nil)

	stateInMeter   = metrics.NewRegisteredMeter("ctxc/downloader/states/in", nil)
	stateDropMeter = metrics.NewRegisteredMeter("ctxc/downloader/states/drop", nil)

	throttleCounter = metrics.NewRegisteredCounter("ctxc/downloader/throttle", nil)
)
