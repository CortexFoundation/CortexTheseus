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
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/CortexFoundation/CortexTheseus/rlp"
	mapset "github.com/ucwong/golang-set"
	"sync"
	"time"
)

type Peer struct {
	id   string
	host *TorrentFS
	peer *p2p.Peer
	ws   p2p.MsgReadWriter

	trusted bool

	known mapset.Set // Messages already known by the peer to avoid wasting bandwidth
	quit  chan struct{}

	wg sync.WaitGroup

	version uint64

	peerInfo *PeerInfo

	//listen uint64
	//root   string
	//files  uint64
	//leafs  uint64
}

type PeerInfo struct {
	Listen uint64      `json:"listen"`
	Root   common.Hash `json:"root"` // SHA3 hash of the peer's best owned block
	Files  uint64      `json:"files"`
	Leafs  uint64      `json:"leafs"`
}

func newPeer(id string, host *TorrentFS, remote *p2p.Peer, rw p2p.MsgReadWriter) *Peer {
	return &Peer{
		id:      id,
		host:    host,
		peer:    remote,
		ws:      rw,
		trusted: false,
		known:   mapset.NewSet(),
		quit:    make(chan struct{}),
	}
}

func (peer *Peer) Info() *PeerInfo {
	return peer.peerInfo
}

func (peer *Peer) start() error {
	peer.wg.Add(1)
	go peer.update()
	return nil
}

func (peer *Peer) update() {
	defer peer.wg.Done()
	stateTicker := time.NewTicker(peerStateCycle)
	defer stateTicker.Stop()

	// Loop and transmit until termination is requested
	for {
		select {
		case query := <-peer.host.queryChan:
			if err := peer.broadcast(query); err != nil {
				log.Trace("broadcast failed", "reason", err, "peer", peer.ID())
				return
			}
		case <-stateTicker.C:
			if err := peer.state(); err != nil {
				log.Trace("broadcast failed", "reason", err, "peer", peer.ID())
				return
			}

		case <-peer.quit:
			return
		}
	}
}

func (peer *Peer) state() error {
	state := PeerInfo{
		Listen: uint64(peer.host.LocalPort()),
		Root:   peer.host.chain().Root(),
		Files:  uint64(peer.host.Congress()),
		Leafs:  uint64(len(peer.host.chain().Blocks())),
	}
	if err := p2p.Send(peer.ws, statusCode, &state); err != nil {
		return err
	}
	return nil
}

type Query struct {
	Hash string `json:"hash"`
	Size uint64 `json:"size"`
}

func (peer *Peer) broadcast(query Query) error {
	//filter
	if err := p2p.Send(peer.ws, messagesCode, &query); err != nil {
		return err
	}
	return nil
}

func (peer *Peer) expire() {
	unmark := make(map[common.Hash]struct{})
	peer.known.Each(func(v interface{}) bool {
		return true
	})
	// Dump all known but no longer cached
	for hash := range unmark {
		peer.known.Remove(hash)
	}
}

func (peer *Peer) handshake() error {
	log.Debug("Nas handshake", "peer", peer.ID())
	errc := make(chan error, 2)
	peer.wg.Add(1)
	go func() {
		defer peer.wg.Done()
		log.Debug("Nas send items", "status", statusCode, "version", ProtocolVersion)
		errc <- p2p.SendItems(peer.ws, statusCode, ProtocolVersion, &PeerInfo{Listen: uint64(peer.host.LocalPort()), Root: peer.host.chain().Root(), Files: uint64(peer.host.Congress()), Leafs: uint64(len(peer.host.chain().Blocks()))})
		log.Debug("Nas send items OK", "status", statusCode, "version", ProtocolVersion, "len", len(errc))
	}()
	// Fetch the remote status packet and verify protocol match
	peer.wg.Add(1)
	go func() {
		defer peer.wg.Done()
		errc <- peer.readStatus()
	}()

	timeout := time.NewTimer(handshakeTimeout)
	defer timeout.Stop()
	for i := 0; i < 2; i++ {
		select {
		case err := <-errc:
			if err != nil {
				return fmt.Errorf("peer [%x] failed to send status packet: %v", peer.ID(), err)
			}
		case <-timeout.C:
			log.Info("Handshake timeout")
			return fmt.Errorf("peer [%x] timeout", peer.ID())
		}
	}

	log.Debug("Nas p2p hanshake success", "id", peer.ID())
	return nil
}

func (peer *Peer) readStatus() error {
	packet, err := peer.ws.ReadMsg()
	if err != nil {
		return err
	}

	defer packet.Discard()

	if packet.Code != statusCode {
		return fmt.Errorf("peer [%x] sent packet %x before status packet", peer.ID(), packet.Code)
	}
	s := rlp.NewStream(packet.Payload, uint64(packet.Size))
	_, err = s.List()
	if err != nil {
		return fmt.Errorf("peer [%x] sent bad status message: %v", peer.ID(), err)
	}
	peerVersion, err := s.Uint()
	if err != nil {
		return fmt.Errorf("peer [%x] sent bad status message (unable to decode version): %v", peer.ID(), err)
	}
	if peerVersion != ProtocolVersion {
		return fmt.Errorf("peer [%x]: protocol version mismatch %d != %d", peer.ID(), peerVersion, ProtocolVersion)
	}

	err = s.Decode(&peer.peerInfo)
	if err != nil {
		return fmt.Errorf("peer [%x] failed to send peer info packet: %v", peer.ID(), err)
	}

	peer.version = peerVersion
	return nil
}

func (peer *Peer) stop() error {
	close(peer.quit)
	peer.wg.Wait()
	return nil
}
func (peer *Peer) ID() []byte {
	id := peer.peer.ID()
	return id[:]
}
