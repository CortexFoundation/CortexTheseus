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

// Contains the metrics collected by the fetcher.

package fetcher

import (
	"github.com/CortexFoundation/CortexTheseus/metrics"
)

var (
	// Useful block announcements == announcements we did not already know about
	propAnnounceInMeter       = metrics.NewRegisteredMeter("ctxc/fetcher/prop/announces/in", nil)
	propAnnounceUsefulInMeter = metrics.NewRegisteredMeter("ctxc/fetcher/prop/announces/useful", nil)

	propAnnounceOutTimer  = metrics.NewRegisteredTimer("ctxc/fetcher/prop/announces/out", nil)
	propAnnounceDropMeter = metrics.NewRegisteredMeter("ctxc/fetcher/prop/announces/drop", nil)
	propAnnounceDOSMeter  = metrics.NewRegisteredMeter("ctxc/fetcher/prop/announces/dos", nil)

	// All useful incoming block broadcasts == broadcasts we did not already have
	propBroadcastInMeter     = metrics.NewRegisteredMeter("ctxc/fetcher/prop/broadcasts/in", nil)
	propBroadcastUsefulMeter = metrics.NewRegisteredMeter("ctxc/fetcher/prop/broadcasts/useful", nil)

	propBroadcastOutTimer  = metrics.NewRegisteredTimer("ctxc/fetcher/prop/broadcasts/out", nil)
	propBroadcastDropMeter = metrics.NewRegisteredMeter("ctxc/fetcher/prop/broadcasts/drop", nil)
	propBroadcastDOSMeter  = metrics.NewRegisteredMeter("ctxc/fetcher/prop/broadcasts/dos", nil)

	headerFetchMeter = metrics.NewRegisteredMeter("ctxc/fetcher/fetch/headers", nil)
	bodyFetchMeter   = metrics.NewRegisteredMeter("ctxc/fetcher/fetch/bodies", nil)

	headerFilterInMeter  = metrics.NewRegisteredMeter("ctxc/fetcher/filter/headers/in", nil)
	headerFilterOutMeter = metrics.NewRegisteredMeter("ctxc/fetcher/filter/headers/out", nil)
	bodyFilterInMeter    = metrics.NewRegisteredMeter("ctxc/fetcher/filter/bodies/in", nil)
	bodyFilterOutMeter   = metrics.NewRegisteredMeter("ctxc/fetcher/filter/bodies/out", nil)
)
