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

import (
	"fmt"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	mapset "github.com/ucwong/golang-set"
	"sync"
)

type Peer struct {
	host *TorrentFS
	peer *p2p.Peer
	ws   p2p.MsgReadWriter

	trusted bool

	known mapset.Set // Messages already known by the peer to avoid wasting bandwidth
	quit  chan struct{}

	wg sync.WaitGroup
}

func newPeer(host *TorrentFS, remote *p2p.Peer, rw p2p.MsgReadWriter) *Peer {
	return &Peer{
		host:    host,
		peer:    remote,
		ws:      rw,
		trusted: false,
		known:   mapset.NewSet(),
		quit:    make(chan struct{}),
	}
}

func (p *Peer) Start() error {
	return nil
}

func (peer *Peer) handshake() error {
	//log.Info("Nas handshake", "peer", *peer.peer)
	errc := make(chan error, 1)
	peer.wg.Add(1)
	go func() {
		defer peer.wg.Done()

		errc <- p2p.SendItems(peer.ws, statusCode, ProtocolVersion, 0, nil, false)
	}()
	if err := <-errc; err != nil {
		return fmt.Errorf("peer [%x] failed to send status packet: %v", peer.ID(), err)
	}
	return nil
}

func (p *Peer) Stop() error {
	close(p.quit)
	p.wg.Wait()
	return nil
}
func (peer *Peer) ID() []byte {
	id := peer.peer.ID()
	return id[:]
}
