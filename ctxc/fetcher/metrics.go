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

	txAnnounceInMeter          = metrics.NewRegisteredMeter("ctxc/fetcher/transaction/announces/in", nil)
	txAnnounceKnownMeter       = metrics.NewRegisteredMeter("ctxc/fetcher/transaction/announces/known", nil)
	txAnnounceUnderpricedMeter = metrics.NewRegisteredMeter("ctxc/fetcher/transaction/announces/underpriced", nil)
	txAnnounceOnchainMeter     = metrics.NewRegisteredMeter("eth/fetcher/transaction/announces/onchain", nil)
	txAnnounceDOSMeter         = metrics.NewRegisteredMeter("ctxc/fetcher/transaction/announces/dos", nil)

	txBroadcastInMeter          = metrics.NewRegisteredMeter("ctxc/fetcher/transaction/broadcasts/in", nil)
	txBroadcastKnownMeter       = metrics.NewRegisteredMeter("ctxc/fetcher/transaction/broadcasts/known", nil)
	txBroadcastUnderpricedMeter = metrics.NewRegisteredMeter("ctxc/fetcher/transaction/broadcasts/underpriced", nil)
	txBroadcastOtherRejectMeter = metrics.NewRegisteredMeter("ctxc/fetcher/transaction/broadcasts/otherreject", nil)

	txRequestOutMeter     = metrics.NewRegisteredMeter("ctxc/fetcher/transaction/request/out", nil)
	txRequestFailMeter    = metrics.NewRegisteredMeter("ctxc/fetcher/transaction/request/fail", nil)
	txRequestDoneMeter    = metrics.NewRegisteredMeter("ctxc/fetcher/transaction/request/done", nil)
	txRequestTimeoutMeter = metrics.NewRegisteredMeter("ctxc/fetcher/transaction/request/timeout", nil)

	txReplyInMeter          = metrics.NewRegisteredMeter("ctxc/fetcher/transaction/replies/in", nil)
	txReplyKnownMeter       = metrics.NewRegisteredMeter("ctxc/fetcher/transaction/replies/known", nil)
	txReplyUnderpricedMeter = metrics.NewRegisteredMeter("ctxc/fetcher/transaction/replies/underpriced", nil)
	txReplyOtherRejectMeter = metrics.NewRegisteredMeter("ctxc/fetcher/transaction/replies/otherreject", nil)

	txFetcherWaitingPeers   = metrics.NewRegisteredGauge("ctxc/fetcher/transaction/waiting/peers", nil)
	txFetcherWaitingHashes  = metrics.NewRegisteredGauge("ctxc/fetcher/transaction/waiting/hashes", nil)
	txFetcherQueueingPeers  = metrics.NewRegisteredGauge("ctxc/fetcher/transaction/queueing/peers", nil)
	txFetcherQueueingHashes = metrics.NewRegisteredGauge("ctxc/fetcher/transaction/queueing/hashes", nil)
	txFetcherFetchingPeers  = metrics.NewRegisteredGauge("ctxc/fetcher/transaction/fetching/peers", nil)
	txFetcherFetchingHashes = metrics.NewRegisteredGauge("ctxc/fetcher/transaction/fetching/hashes", nil)

	txFetcherSlowPeers = metrics.NewRegisteredGauge("eth/fetcher/transaction/slow/peers", nil)
	txFetcherSlowWait  = metrics.NewRegisteredHistogram("eth/fetcher/transaction/slow/wait", nil, metrics.NewExpDecaySample(1028, 0.015))
)
