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

package params

import (
	"time"
)

const (
	ProtocolName         = "nas"
	ProtocolVersion      = uint64(4)
	NumberOfMessageCodes = 128
	ProtocolVersionStr   = "4.0"

	DefaultMaxMessageSize = uint32(1024)

	StatusCode = 0
	QueryCode  = 1
	MsgCode    = 2

	PeerStateCycle = time.Second * 300

	ExpirationCycle   = time.Second
	TransmissionCycle = 300 * time.Millisecond
	HandshakeTimeout  = 60 * time.Second
)
