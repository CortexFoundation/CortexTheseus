// Copyright 2023 The CortexTheseus Authors
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

package robot

import (
	"github.com/CortexFoundation/CortexTheseus/metrics"
	"time"
)

const (
	batch   = 4096 * 2 //params.SyncBatch
	delay   = 12       //params.Delay
	timeout = 30 * time.Second
)

var (
	rpcBlockMeter   = metrics.NewRegisteredMeter("torrent/block/call", nil)
	rpcCurrentMeter = metrics.NewRegisteredMeter("torrent/current/call", nil)
	rpcUploadMeter  = metrics.NewRegisteredMeter("torrent/upload/call", nil)
	rpcReceiptMeter = metrics.NewRegisteredMeter("torrent/receipt/call", nil)
)
