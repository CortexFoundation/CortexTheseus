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

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/ucwong/go-ttlmap"

	"github.com/CortexFoundation/torrentfs/params"
)

func (fs *TorrentFS) MaxMessageSize() uint32 {
	return params.DefaultMaxMessageSize
}

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
		// Log the error and return immediately on read failure.
		log.Debug("Failed to read message", "peer", p.ID(), "err", err)
		return err
	}

	// Ensure the packet is discarded after processing. This handles all exit paths.
	defer packet.Discard()

	// Check for oversized messages immediately after a successful read.
	if packet.Size > fs.MaxMessageSize() {
		log.Warn("Received oversized message", "peer", p.ID(), "size", packet.Size)
		return errors.New("oversized message received")
	}

	log.Debug("Handling Nas package", "protocol", params.ProtocolVersionStr, "size", packet.Size, "code", packet.Code, "peer", p.ID())

	// Use a helper function for consistent error handling.
	handleDecodeError := func(err error) error {
		log.Warn("Failed to decode package", "peer", p.ID(), "code", packet.Code, "err", err)
		return errors.New("invalid package format")
	}

	switch packet.Code {
	case params.StatusCode:
		var info *PeerInfo
		if err := packet.Decode(&info); err != nil {
			return handleDecodeError(err)
		}
		p.peerInfo = info
		log.Debug("Peer status received", "peer", p.ID(), "root", info.Root)

	case params.QueryCode:
		if params.ProtocolVersion < 4 {
			log.Warn("Protocol version too low for query", "peer", p.ID(), "version", params.ProtocolVersion)
			return errors.New("protocol version not supported")
		}
		var queryInfo *Query
		if err := packet.Decode(&queryInfo); err != nil {
			return handleDecodeError(err)
		}

		if !common.IsHexAddress(queryInfo.Hash) {
			log.Warn("Received invalid hash address", "peer", p.ID(), "hash", queryInfo.Hash)
			return errors.New("invalid hash address")
		}

		// If the file is already being handled, return.
		if fs.collapse(queryInfo.Hash, queryInfo.Size) {
			log.Debug("Query for file already in progress", "peer", p.ID(), "hash", queryInfo.Hash)
			return nil
		}

		// Handle the new query.
		if err := fs.wakeup(context.Background(), queryInfo.Hash); err == nil {
			if err := fs.traverse(queryInfo.Hash, queryInfo.Size); err == nil {
				fs.received.Add(1)
				log.Debug("Query processed successfully", "peer", p.ID(), "hash", queryInfo.Hash)
			}
		}

	case params.MsgCode:
		if params.ProtocolVersion <= 5 {
			log.Warn("Protocol version too low for message", "peer", p.ID(), "version", params.ProtocolVersion)
			return errors.New("protocol version not supported")
		}
		var msgInfo *MsgInfo
		if err := packet.Decode(&msgInfo); err != nil {
			return handleDecodeError(err)
		}
		log.Warn("Nas message received", "code", params.MsgCode, "desc", msgInfo.Desc, "peer", p.ID())

	default:
		log.Warn("Encountered unknown package code", "peer", p.ID(), "code", packet.Code)
		return errors.New("invalid package code")
	}

	// If the message was handled successfully, return nil.
	return nil
}

func (fs *TorrentFS) Neighbors() int {
	if fs.net != nil {
		return fs.net.PeerCount()
	}

	return len(fs.peers)
}

func (fs *TorrentFS) Envelopes() *ttlmap.Map {
	fs.peerMu.RLock()
	defer fs.peerMu.RUnlock()

	return fs.tunnel
}
