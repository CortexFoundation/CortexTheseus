// Copyright 2020 The CortexTheseus Authors
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
package torrentfs

const (
	ProtocolName         = "nas"
	ProtocolVersion      = uint64(1)
	NumberOfMessageCodes = uint64(0)
	ProtocolVersionStr   = "1.0"
	statusCode           = 0

	Bucket    = 1024
	Group     = 32
	SyncBatch = 4096
	Delay     = 12
	//Scope     = 4
	TIER  = 3
	LEAFS = 32768
)

var (
	MainnetTrackers = []string{
		"://tracker.cortexlabs.ai:5008",
	}

	BernardTrackers = MainnetTrackers

	TorrentBoostNodes = []string{
		"http://storage.cortexlabs.ai:7881",
	}
)
