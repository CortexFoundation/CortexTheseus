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
	"reflect"
	"sync"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/CortexFoundation/CortexTheseus/rlp"
	mapset "github.com/deckarep/golang-set/v2"

	"github.com/CortexFoundation/torrentfs/params"
)

type Peer struct {
	id       string
	host     *TorrentFS
	peer     *p2p.Peer
	ws       p2p.MsgReadWriter
	trusted  bool
	known    mapset.Set[string]
	quit     chan any
	wg       sync.WaitGroup
	version  uint64
	peerInfo *PeerInfo

	msgChan chan any
	seeding mapset.Set[string]

	once sync.Once
}

type PeerInfo struct {
	Listen uint64      `json:"listen"`
	Root   common.Hash `json:"root"` // SHA3 hash of the peer's best owned block
	Files  uint64      `json:"files"`
	Leafs  uint64      `json:"leafs"`
}

type MsgInfo struct {
	Desc string `json:"desc"`
}

func newPeer(id string, host *TorrentFS, remote *p2p.Peer, rw p2p.MsgReadWriter) *Peer {
	p := &Peer{
		id:      id,
		host:    host,
		peer:    remote,
		ws:      rw,
		known:   mapset.NewSet[string](),
		trusted: false,
		quit:    make(chan any),
		msgChan: make(chan any, 1),
		seeding: mapset.NewSet[string](),
	}
	return p
}

func (peer *Peer) Info() *PeerInfo {
	return peer.peerInfo
}

func (peer *Peer) start() error {
	peer.wg.Add(1)
	go peer.update()
	peer.wg.Add(1)
	go peer.calling()
	return nil
}

func (peer *Peer) expire() {
	// Use a slice to store keys to be removed, which is safer if the map's Each method
	// doesn't guarantee safe concurrent modification.
	var toRemove []string

	// Iterate over all known keys.
	peer.known.Each(func(key string) bool {
		if _, err := peer.host.Envelopes().Get(key); err != nil {
			toRemove = append(toRemove, key)
		}
		return true // Continue the iteration.
	})

	// Remove all keys collected in the first step.
	for _, key := range toRemove {
		peer.known.Remove(key)
	}
}

func (peer *Peer) update() {
	// Defer statements to ensure resources are cleaned up on function exit.
	defer peer.wg.Done()
	stateTicker := time.NewTicker(params.PeerStateCycle)
	defer stateTicker.Stop()
	transmitTicker := time.NewTicker(params.TransmissionCycle)
	defer transmitTicker.Stop()
	expireTicker := time.NewTicker(params.ExpirationCycle)
	defer expireTicker.Stop()

	// The main event loop for peer operations.
	for {
		select {
		case <-expireTicker.C:
			peer.expire()

		case <-transmitTicker.C:
			// Check for neighbors before attempting to broadcast to avoid unnecessary logs.
			if peer.host.Neighbors() == 0 {
				log.Warn("No neighbors found, skipping transmission", "peer", peer.ID())
				continue
			}

			if err := peer.broadcast(); err != nil {
				// Use Trace for expected, non-critical failures.
				log.Trace("Transmit broadcast failed", "reason", err, "peer", peer.ID())
				// Return here as the failure might be critical.
				return
			}

		case <-stateTicker.C:
			if err := peer.state(); err != nil {
				log.Trace("State broadcast failed", "reason", err, "peer", peer.ID())
				return
			}
		case <-peer.quit:
			log.Debug("Peer update loop terminated", "peer", peer.ID())
			return
		}
	}
}

func (peer *Peer) state() error {
	state := PeerInfo{
		Listen: uint64(peer.host.LocalPort()),
		Root:   peer.host.monitor.DB().Root(),
		Files:  uint64(peer.host.Congress()),
		Leafs:  uint64(len(peer.host.monitor.DB().Blocks())),
	}
	if err := p2p.Send(peer.ws, params.StatusCode, &state); err != nil {
		return err
	}
	return nil
}

type Query struct {
	Hash string `json:"hash"`
	Size uint64 `json:"size"`
}

func (peer *Peer) seen(hash string) {
	if !peer.seeding.Contains(hash) {
		peer.seeding.Add(hash)
	}
}

func (peer *Peer) mark(hash string) {
	peer.known.Add(hash)
}

func (peer *Peer) marked(hash string) bool {
	return peer.known.Contains(hash)
}

func (peer *Peer) broadcast() error {
	keys := peer.host.Envelopes().Keys()

	for _, k := range keys {
		// Ensure the key is a string and handle potential type assertion failures.
		keyStr, ok := k.Interface().(string)
		if !ok {
			// Log a warning if the key is not the expected type.
			log.Warn("Envelopes key is not a string, skipping", "keyType", reflect.TypeOf(k.Interface()).String())
			continue
		}

		// Check if the peer has already processed this key.
		if peer.marked(keyStr) {
			continue
		}

		// Get the value and handle potential errors.
		v, err := peer.host.Envelopes().Get(keyStr)
		if err != nil {
			log.Warn("Failed to get envelope value, skipping", "key", keyStr, "err", err)
			continue
		}

		// Construct the query object.
		query := Query{
			Hash: keyStr,
			Size: v.Value().(uint64), // Assuming the value's type assertion is safe.
		}

		// Send the query and handle any transmission errors.
		if err := p2p.Send(peer.ws, params.QueryCode, &query); err != nil {
			// Return immediately on a send failure.
			return err
		}

		// Update metrics and mark the key as sent.
		peer.host.sent.Add(1)
		peer.mark(keyStr)
	}

	return nil
}

func (peer *Peer) call(msg any) {
	peer.msgChan <- msg
}

func (peer *Peer) calling() {
	defer peer.wg.Done()
	for {
		select {
		case msg := <-peer.msgChan:
			if err := p2p.Send(peer.ws, params.MsgCode, &msg); err != nil {
				log.Warn("Msg sending failed", "msg", msg, "id", peer.id, "err", err)
				return
			}
			log.Info("Msg sending", "msg", msg, "id", peer.id)
		case <-peer.quit:
			return
		}
	}
}

func (peer *Peer) handshake() error {
	log.Debug("Nas handshake", "peer", peer.ID())
	errc := make(chan error, 2)
	peer.wg.Add(1)
	go func() {
		defer peer.wg.Done()
		log.Debug("Nas send items", "status", params.StatusCode, "version", params.ProtocolVersion)
		info := PeerInfo{
			Listen: uint64(peer.host.LocalPort()),
			Root:   peer.host.monitor.DB().Root(),
			Files:  uint64(peer.host.Congress()),
			Leafs:  uint64(len(peer.host.monitor.DB().Blocks())),
		}
		select {
		case errc <- p2p.SendItems(peer.ws, params.StatusCode, params.ProtocolVersion, &info):
		case <-peer.quit:
		}
		log.Debug("Nas send items OK", "status", params.StatusCode, "version", params.ProtocolVersion, "len", len(errc))
	}()
	// Fetch the remote status packet and verify protocol match
	peer.wg.Add(1)
	go func() {
		defer peer.wg.Done()
		select {
		case errc <- peer.readStatus():
		case <-peer.quit:
		}
	}()

	timeout := time.NewTimer(params.HandshakeTimeout)
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
		case <-peer.quit:
			return nil
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

	if packet.Code != params.StatusCode {
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
	if peerVersion != params.ProtocolVersion {
		return fmt.Errorf("peer [%x]: protocol version mismatch %d != %d", peer.ID(), peerVersion, params.ProtocolVersion)
	}

	err = s.Decode(&peer.peerInfo)
	if err != nil {
		return fmt.Errorf("peer [%x] failed to send peer info packet: %v", peer.ID(), err)
	}

	peer.version = peerVersion
	return nil
}

func (peer *Peer) stop() error {
	peer.once.Do(func() {
		close(peer.quit)
		peer.wg.Wait()
	})
	return nil
}

func (peer *Peer) ID() string {
	id := peer.peer.ID()
	return id.String()
}
