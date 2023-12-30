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

package backend

import (
	"github.com/CortexFoundation/CortexTheseus/metrics"
	"github.com/CortexFoundation/torrentfs/params"
)

const (
	bucket          = params.Bucket //it is best size is 1/3 full nodes
	group           = params.Group
	taskChanBuffer  = 1 //params.SyncBatch
	torrentChanSize = 8

	block = int64(params.PER_UPLOAD_BYTES)
	//loops = 30

	torrentTypeOnChain = 0
	torrentTypeLocal   = 1

	TORRENT = "torrent"

	SEED_PRE = "s-"
)

var (
	server         bool = false
	enableWorm     bool = false
	getfileMeter        = metrics.NewRegisteredMeter("torrent/getfile/call", nil)
	availableMeter      = metrics.NewRegisteredMeter("torrent/available/call", nil)
	diskReadMeter       = metrics.NewRegisteredMeter("torrent/disk/read", nil)

	downloadMeter = metrics.NewRegisteredMeter("torrent/download/call", nil)
	updateMeter   = metrics.NewRegisteredMeter("torrent/update/call", nil)

	memcacheHitMeter  = metrics.NewRegisteredMeter("torrent/memcache/hit", nil)
	memcacheReadMeter = metrics.NewRegisteredMeter("torrent/memcache/read", nil)

	memcacheMissMeter  = metrics.NewRegisteredMeter("torrent/memcache/miss", nil)
	memcacheWriteMeter = metrics.NewRegisteredMeter("torrent/memcache/write", nil)
)
