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
// along with the CortexTheseus library. If not, see <http://www.gnu.org/licenses/>

package torrentfs

import (
	"context"
	"errors"
	"fmt"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/CortexFoundation/CortexTheseus/p2p/enode"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	lru "github.com/hashicorp/golang-lru"
	"sync"
	//"time"
)

// TorrentFS contains the torrent file system internals.
type TorrentFS struct {
	protocol p2p.Protocol // Protocol description and parameters
	config   *Config
	monitor  *Monitor

	peerMu sync.RWMutex     // Mutex to sync the active peer set
	peers  map[string]*Peer // Set of currently active peers

	//queryChan chan Query

	nasCache   *lru.Cache
	queryCache *lru.Cache
}

func (t *TorrentFS) storage() *TorrentManager {
	return t.monitor.dl
}

func (t *TorrentFS) chain() *ChainDB {
	return t.monitor.fs
}

var inst *TorrentFS = nil

func GetStorage() CortexStorage {
	//if inst == nil {
	//inst, _ = New(&DefaultConfig, true, false, false)
	//}
	return inst //GetTorrentInstance()
}

// New creates a new torrentfs instance with the given configuration.
func New(config *Config, cache, compress, listen bool) (*TorrentFS, error) {
	if inst != nil {
		return inst, nil
	}

	monitor, moErr := NewMonitor(config, cache, compress, listen)
	if moErr != nil {
		log.Error("Failed create monitor")
		return nil, moErr
	}

	inst = &TorrentFS{
		config:  config,
		monitor: monitor,
		peers:   make(map[string]*Peer),
		//queryChan: make(chan Query, 128),
	}

	inst.nasCache, _ = lru.New(25)
	inst.queryCache, _ = lru.New(25)

	inst.protocol = p2p.Protocol{
		Name:    ProtocolName,
		Version: uint(ProtocolVersion),
		Length:  NumberOfMessageCodes,
		Run:     inst.HandlePeer,
		NodeInfo: func() interface{} {
			return map[string]interface{}{
				"version": ProtocolVersion,
				"status": map[string]interface{}{
					"dht":            !config.DisableDHT,
					"port":           inst.LocalPort(),
					"root":           inst.chain().Root().Hex(),
					"files":          inst.Congress(),
					"active":         inst.Candidate(),
					"leafs":          len(inst.chain().Blocks()),
					"number":         monitor.currentNumber,
					"maxMessageSize": inst.MaxMessageSize(),
					//					"listen":         monitor.listen,
				},
			}
		},
		PeerInfo: func(id enode.ID) interface{} {
			inst.peerMu.Lock()
			defer inst.peerMu.Unlock()
			if p := inst.peers[fmt.Sprintf("%x", id[:8])]; p != nil {
				return map[string]interface{}{
					"version": p.version,
					"listen":  p.Info().Listen,
					"root":    p.Info().Root.Hex(),
					"files":   p.Info().Files,
					"leafs":   p.Info().Leafs,
				}
			}
			return nil
		},
	}

	return inst, nil
}

func (tfs *TorrentFS) MaxMessageSize() uint32 {
	return DefaultMaxMessageSize
}

func (tfs *TorrentFS) HandlePeer(peer *p2p.Peer, rw p2p.MsgReadWriter) error {
	tfsPeer := newPeer(fmt.Sprintf("%x", peer.ID().Bytes()[:8]), tfs, peer, rw)

	tfs.peerMu.Lock()
	tfs.peers[tfsPeer.id] = tfsPeer
	tfs.peerMu.Unlock()

	defer func() {
		tfs.peerMu.Lock()
		delete(tfs.peers, tfsPeer.id)
		tfs.peerMu.Unlock()
	}()

	if err := tfsPeer.handshake(); err != nil {
		return err
	}

	tfsPeer.start()
	defer func() {
		tfsPeer.stop()
	}()

	return tfs.runMessageLoop(tfsPeer, rw)
}
func (tfs *TorrentFS) runMessageLoop(p *Peer, rw p2p.MsgReadWriter) error {
	for {
		// fetch the next packet
		packet, err := rw.ReadMsg()
		if err != nil {
			log.Debug("message loop", "peer", p.peer.ID(), "err", err)
			return err
		}

		if packet.Size > tfs.MaxMessageSize() {
			log.Warn("oversized message received", "peer", p.peer.ID())
			return errors.New("oversized message received")
		}

		log.Debug("Nas package", "size", packet.Size)

		switch packet.Code {
		case statusCode:
			var info *PeerInfo
			if err := packet.Decode(&info); err != nil {
				log.Warn("failed to decode peer state, peer will be disconnected", "peer", p.peer.ID(), "err", err)
				return errors.New("invalid peer state")
			}
			p.peerInfo = info
		case queryCode:
			if ProtocolVersion > 1 {
				var info *Query
				if err := packet.Decode(&info); err != nil {
					log.Warn("failed to decode msg, peer will be disconnected", "peer", p.peer.ID(), "err", err)
					return errors.New("invalid msg")
				}
				if suc := tfs.queryCache.Contains(info.Hash); !suc {
					log.Info("Nas msg received", "ih", info.Hash, "size", common.StorageSize(float64(info.Size)))
					if progress, e := tfs.chain().GetTorrent(info.Hash); e == nil && progress >= info.Size {
						if err := tfs.storage().Search(context.Background(), info.Hash, info.Size, nil); err != nil {
							log.Error("Nas 2.0 error", "err", err)
							return err
						}
					}
					tfs.queryCache.Add(info.Hash, info.Size)
				}
			}
		case msgCode:
		default:
			log.Warn("Encounter package code", "code", packet.Code)
			return errors.New("invalid code")
		}
		packet.Discard()
	}
}

// Protocols implements the node.Service interface.
func (tfs *TorrentFS) Protocols() []p2p.Protocol { return []p2p.Protocol{tfs.protocol} }

// APIs implements the node.Service interface.
func (tfs *TorrentFS) APIs() []rpc.API {
	return []rpc.API{
		{
			Namespace: ProtocolName,
			Version:   ProtocolVersionStr,
			Service:   NewPublicTorrentAPI(tfs),
			Public:    false,
		},
	}
}

func (tfs *TorrentFS) Version() uint {
	return tfs.protocol.Version
}

// Start starts the data collection thread and the listening server of the dashboard.
// Implements the node.Service interface.
func (tfs *TorrentFS) Start(server *p2p.Server) error {
	log.Info("Started nas v.1.0", "config", tfs, "mode", tfs.config.Mode)
	if tfs == nil || tfs.monitor == nil {
		return nil
	}
	return tfs.monitor.Start()
}

// Stop stops the data collection thread and the connection listener of the dashboard.
// Implements the node.Service interface.
func (tfs *TorrentFS) Stop() error {
	if tfs == nil || tfs.monitor == nil {
		return nil
	}
	// Wait until every goroutine terminates.
	tfs.monitor.stop()

	if tfs.nasCache != nil {
		tfs.nasCache.Purge()
	}

	if tfs.queryCache != nil {
		tfs.queryCache.Purge()
	}
	return nil
}

// Available is used to check the file status
func (fs *TorrentFS) Available(ctx context.Context, infohash string, rawSize uint64) (bool, error) {
	ret, f, cost, err := fs.storage().available(infohash, rawSize)
	if fs.config.Mode == LAZY {
		if errors.Is(err, ErrInactiveTorrent) {
			if progress, e := fs.chain().GetTorrent(infohash); e == nil {
				log.Debug("Lazy mode, restarting", "ih", infohash, "request", progress)
				if e := fs.storage().Search(ctx, infohash, progress, nil); e == nil {
					log.Warn("Torrent wake up", "ih", infohash, "progress", progress, "err", err, "available", ret, "raw", rawSize, "err", err)
				}
			} else {
				log.Warn("Unregister file", "ih", infohash, "size", common.StorageSize(float64(rawSize)))
			}
		} else if errors.Is(err, ErrUnfinished) {
			if suc := fs.nasCache.Contains(infohash); !suc {
				if f == 0 {
					var speed float64
					if cost > 0 {
						t := float64(cost) / (1000 * 1000 * 1000)
						speed = float64(f) / t
					}
					log.Info("Nas 2.0 query", "ih", infohash, "raw", common.StorageSize(float64(rawSize)), "finish", f, "cost", common.PrettyDuration(cost), "speed", common.StorageSize(speed), "peers", len(fs.peers), "cache", fs.nasCache.Len(), "err", err)
					fs.nasCache.Add(infohash, rawSize)
				}
			}

			log.Debug("Torrent sync downloading", "ih", infohash, "available", ret, "raw", rawSize, "finish", f, "err", err)
		}
	}
	return ret, err
}

func (fs *TorrentFS) GetFileWithSize(ctx context.Context, infohash string, rawSize uint64, subpath string) ([]byte, error) {
	if available, err := fs.Available(ctx, infohash, rawSize); err != nil || !available {
		return nil, err
	}

	return fs.GetFile(ctx, infohash, subpath)
}

// GetFile is used to get file from storage, current this will not be call after available passed
func (fs *TorrentFS) GetFile(ctx context.Context, infohash, subpath string) ([]byte, error) {
	ret, f, err := fs.storage().getFile(infohash, subpath)

	if err != nil {
		log.Warn("Not avaialble err in getFile", "err", err, "ret", ret, "ih", infohash, "progress", f)
	}

	return ret, err
}

//Download is used to download file with request
func (fs *TorrentFS) Download(ctx context.Context, ih string, request uint64) error {
	update, p, err := fs.chain().SetTorrent(ih, request)
	if err != nil {
		return err
	}

	if update {
		log.Debug("Search in fs download", "ih", ih, "request", p)
		if err := fs.storage().Search(ctx, ih, p, nil); err != nil {
			return err
		}
	}

	return nil
}

func (fs *TorrentFS) LocalPort() int {
	return fs.storage().LocalPort()
}

func (fs *TorrentFS) Congress() int {
	return fs.storage().Congress()
}

func (fs *TorrentFS) Candidate() int {
	return fs.storage().Candidate()
}
