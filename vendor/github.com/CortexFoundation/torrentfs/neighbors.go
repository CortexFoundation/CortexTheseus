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
// along with the CortexTheseus library. If not, see <http://www.gnu.org/licenses/>

package torrentfs

import (
	"context"
	"errors"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/CortexFoundation/torrentfs/params"

	"github.com/ucwong/go-ttlmap"
)

func (fs *TorrentFS) MaxMessageSize() uint32 {
	return params.DefaultMaxMessageSize
}

/*func (fs *TorrentFS) find(ih string) (*Peer, error) {
        for s, p := range fs.peers {
                if p.seeding.Contains(ih) {
                        // TODO
                        log.Debug("Seed found !!!", "from", s, "ih", ih)
                        return p, nil
                }
        }

        log.Debug("Seed not found !!!", "neighbors", len(fs.peers), "ih", ih)
        return nil, nil
}*/

func (fs *TorrentFS) HandlePeer(peer *p2p.Peer, rw p2p.MsgReadWriter) error {
	//tfsPeer := newPeer(fmt.Sprintf("%x", peer.ID().Bytes()[:8]), fs, peer, rw)
	tfsPeer := newPeer(peer.ID().String(), fs, peer, rw)

	fs.peerMu.Lock()
	fs.peers[tfsPeer.id] = tfsPeer
	fs.in.Add(1)
	fs.peerMu.Unlock()

	defer func() {
		fs.peerMu.Lock()
		delete(fs.peers, tfsPeer.id)
		fs.out.Add(1)
		fs.peerMu.Unlock()
	}()

	if err := tfsPeer.handshake(); err != nil {
		return err
	}

	fs.record(peer.ID().String())

	tfsPeer.start()
	defer tfsPeer.stop()

	return fs.runMessageLoop(tfsPeer)
}

func (fs *TorrentFS) runMessageLoop(p *Peer) error {
	for {
		if err := fs.handleMsg(p); err != nil {
			return err
		}
	}
}

func (fs *TorrentFS) handleMsg(p *Peer) error {
	packet, err := p.ws.ReadMsg()
	if err != nil {
		log.Debug("message loop", "peer", p.peer.ID(), "err", err)
		return err
	}

	if packet.Size > fs.MaxMessageSize() {
		log.Warn("oversized message received", "peer", p.peer.ID())
		return errors.New("oversized message received")
	}

	defer packet.Discard()

	log.Debug("Nas "+params.ProtocolVersionStr+" package", "size", packet.Size, "code", packet.Code, "peer", p.peer.ID())

	switch packet.Code {
	case params.StatusCode:
		var info *PeerInfo
		if err := packet.Decode(&info); err != nil {
			log.Warn("failed to decode peer state, peer will be disconnected", "peer", p.peer.ID(), "err", err)
			return errors.New("invalid peer state")
		}
		p.peerInfo = info
	case params.QueryCode:
		if params.ProtocolVersion >= 4 {
			var info *Query
			if err := packet.Decode(&info); err != nil {
				log.Warn("failed to decode msg, peer will be disconnected", "peer", p.peer.ID(), "err", err)
				return errors.New("invalid msg")
			}

			if !common.IsHexAddress(info.Hash) {
				return errors.New("invalid address")
			}

			if ok := fs.collapse(info.Hash, info.Size); ok {
				return nil
			}

			if err := fs.wakeup(context.Background(), info.Hash); err == nil {
				if err := fs.traverse(info.Hash, info.Size); err == nil {
					fs.received.Add(1)
				}
			}
		}
	case params.MsgCode:
		if params.ProtocolVersion > 4 {
			var info *MsgInfo
			if err := packet.Decode(&info); err != nil {
				log.Warn("failed to decode msg, peer will be disconnected", "peer", p.peer.ID(), "err", err)
				return errors.New("invalid msg")
			}
			log.Warn("Nas msg testing", "code", params.MsgCode, "desc", info.Desc)
		}
	default:
		log.Warn("Encounter package code", "code", packet.Code)
		return errors.New("invalid code")
	}

	// TODO

	return nil
}

func (fs *TorrentFS) collapse(ih string, rawSize uint64) bool {
	if s, err := fs.tunnel.Get(ih); err == nil && s.Value().(uint64) >= rawSize {
		return true
	}

	return false
}

func (fs *TorrentFS) traverse(ih string, rawSize uint64) error {
	if err := fs.tunnel.Set(ih, ttlmap.NewItem(rawSize, ttlmap.WithTTL(60*time.Second)), nil); err == nil {
		log.Trace("Wormhole traverse", "ih", ih, "size", common.StorageSize(rawSize))
	} else {
		return err
	}
	return nil
}

func (fs *TorrentFS) broadcast(ih string, rawSize uint64) bool {
	if !common.IsHexAddress(ih) {
		return false
	}

	//if s, err := fs.tunnel.Get(ih); err == nil && s.Value().(uint64) >= rawSize {
	if fs.collapse(ih, rawSize) {
		return false
	}

	//fs.tunnel.Set(ih, ttlmap.NewItem(rawSize, ttlmap.WithTTL(60*time.Second)), nil)
	if err := fs.traverse(ih, rawSize); err != nil {
		return false
	}

	return true
}

func (fs *TorrentFS) Envelopes() *ttlmap.Map {
	fs.peerMu.RLock()
	defer fs.peerMu.RUnlock()

	return fs.tunnel
}

func (fs *TorrentFS) Neighbors() int {
	if fs.net != nil {
		return fs.net.PeerCount()
	}

	return len(fs.peers)
}
