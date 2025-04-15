// Copyright 2019 The go-ethereum Authors
// This file is part of The go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with The go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package ctxc

import (
	"github.com/CortexFoundation/CortexTheseus/core"
	"github.com/CortexFoundation/CortexTheseus/core/forkid"
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
func (ctxc *Cortex) startENRUpdater(ln *enode.LocalNode) {
	var newHead = make(chan core.ChainHeadEvent, 10)
	sub := ctxc.blockchain.SubscribeChainHeadEvent(newHead)

	ln.Set(ctxc.currentCtxcEntry(ctxc.blockchain))
	go func() {
		defer sub.Unsubscribe()
		for {
			select {
			case <-newHead:
				ln.Set(ctxc.currentCtxcEntry(ctxc.blockchain))
			case <-sub.Err():
				// Would be nice to sync with ctxc.Stop, but there is no
				// good way to do that.
				return
			}
		}
	}()
}

func (ctxc *Cortex) currentCtxcEntry(chain *core.BlockChain) *ctxcEntry {
	head := chain.CurrentHeader()
	return &ctxcEntry{
		ForkID: forkid.NewID(chain.Config(), chain.Genesis(), head.Number.Uint64(), head.Time),
	}
}

// NewNodeFilter returns a filtering function that returns whether the provided
// enode advertises a forkid compatible with the current chain.
func NewNodeFilter(chain *core.BlockChain) func(*enode.Node) bool {
	filter := forkid.NewFilter(chain)
	return func(n *enode.Node) bool {
		var entry ctxcEntry
		if err := n.Load(&entry); err != nil {
			return false
		}
		err := filter(entry.ForkID)
		return err == nil
	}
}
