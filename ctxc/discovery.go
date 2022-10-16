// Copyright 2019 The CortexTheseus Authors
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

package ctxc

import (
	"github.com/CortexFoundation/CortexTheseus/core"
	"github.com/CortexFoundation/CortexTheseus/core/forkid"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/p2p/dnsdisc"
	"github.com/CortexFoundation/CortexTheseus/p2p/enode"
	"github.com/CortexFoundation/CortexTheseus/rlp"
)

// ctxcEntry is the "ctxc" ENR entry which advertises ctxc protocol
// on the discovery network.
type ctxcEntry struct {
	ForkID forkid.ID // Fork identifier per EIP-2124

	// Ignore additional fields (for forward compatibility).
	Rest []rlp.RawValue `rlp:"tail"`
}

// ENRKey implements enr.Entry.
func (e ctxcEntry) ENRKey() string {
	return "ctxc"
}

// startCtxcEntryUpdate starts the ENR updater loop.
func (ctxc *Cortex) startCtxcEntryUpdate(ln *enode.LocalNode) {
	var newHead = make(chan core.ChainHeadEvent, 10)
	sub := ctxc.blockchain.SubscribeChainHeadEvent(newHead)

	go func() {
		defer sub.Unsubscribe()
		for {
			select {
			case <-newHead:
				ln.Set(ctxc.currentCtxcEntry())
			case err := <-sub.Err():
				// Would be nice to sync with ctxc.Stop, but there is no
				// good way to do that.
				log.Warn("Local dns discovery quit", "err", err)
				return
			}
		}
	}()
}

func (ctxc *Cortex) currentCtxcEntry() *ctxcEntry {
	return &ctxcEntry{
		ForkID: forkid.NewID(ctxc.blockchain.Config(), ctxc.blockchain.Genesis().Hash(),
			ctxc.blockchain.CurrentHeader().Number.Uint64())}
}

// setupDiscovery creates the node discovery source for the ctxc protocol.
func (ctxc *Cortex) setupDiscovery() (enode.Iterator, error) {
	if len(ctxc.config.DiscoveryURLs) == 0 {
		return nil, nil
	}
	client := dnsdisc.NewClient(dnsdisc.Config{})
	return client.NewIterator(ctxc.config.DiscoveryURLs...)
}
