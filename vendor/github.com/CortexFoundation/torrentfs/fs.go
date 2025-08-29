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
	"sync"
	"sync/atomic"

	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/CortexFoundation/CortexTheseus/p2p/dnsdisc"
	"github.com/CortexFoundation/CortexTheseus/p2p/enode"
	params1 "github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"github.com/CortexFoundation/robot"
	mapset "github.com/deckarep/golang-set/v2"
	"github.com/ucwong/go-ttlmap"

	"github.com/CortexFoundation/torrentfs/backend"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/types"
)

// TorrentFS contains the torrent file system internals.
type TorrentFS struct {
	protocol p2p.Protocol // Protocol description and parameters
	config   *params.Config
	monitor  robot.IMonitor
	handler  *backend.TorrentManager
	//db       *backend.ChainDB

	peerMu sync.RWMutex     // Mutex to sync the active peer set
	peers  map[string]*Peer // Set of currently active peers

	//queryChan chan Query

	//nasCache *lru.Cache
	//queryCache *lru.Cache
	nasCounter uint64

	received atomic.Uint64
	sent     atomic.Uint64
	retry    atomic.Uint64

	in  atomic.Uint64
	out atomic.Uint64

	// global file hash & score
	//scoreTable map[string]int

	//seedingNotify chan string
	closeAll chan any
	wg       sync.WaitGroup
	stopOnce sync.Once
	worm     mapset.Set[string]
	history  mapset.Set[string]

	tunnel *ttlmap.Map

	callback chan any
	//ttlchan  chan any

	net *p2p.Server

	initOnce sync.Once
}

func (t *TorrentFS) storage() *backend.TorrentManager {
	return t.handler
}

var (
	inst    *TorrentFS = nil
	mut     sync.RWMutex
	newOnce sync.Once
)

func GetStorage() CortexStorage {
	mut.RLock()
	defer mut.RUnlock()

	if inst == nil {
		//inst, _ = New(&DefaultConfig, true, false, false)
		log.Warn("Storage instance get failed, should new it first")
	}
	return inst //GetTorrentInstance()
}

// New creates a new torrentfs instance with the given configuration.
func New(config *params.Config, cache, compress, listen bool) (t *TorrentFS, err error) {
	newOnce.Do(func() {
		mut.Lock()
		defer mut.Unlock()
		t, err = create(config, cache, compress, listen)
	})

	if t == nil {
		t = inst
	}

	return
}

func create(config *params.Config, cache, compress, listen bool) (*TorrentFS, error) {
	if inst != nil {
		log.Warn("Storage has been already inited", "storage", inst, "config", config, "cache", cache, "compress", compress, "listen", listen)
		return inst, nil
	}

	/*db, err := backend.NewChainDB(config)
	if err != nil {
		log.Error("file storage failed", "err", err)
		return nil, err
	}
	log.Info("File storage initialized")

	handler, err := backend.NewTorrentManager(config, db.ID(), cache, compress)
	if err != nil || handler == nil {
		log.Error("fs manager failed", "err", err)
		return nil, errors.New("fs download manager initialise failed")
	}
	log.Info("Fs manager initialized")*/

	_callback := make(chan any, 1024)
	monitor, err := robot.New(config, cache, compress, listen, _callback)
	if err != nil {
		log.Error("Failed create monitor", "err", err)
		return nil, err
	}

	log.Info("Fs monitor initialized")

	handler, err := backend.NewTorrentManager(config, monitor.ID(), cache, compress)
	if err != nil || handler == nil {
		log.Error("fs manager failed", "err", err)
		return nil, errors.New("fs download manager initialise failed")
	}
	log.Info("Fs manager initialized")

	inst = &TorrentFS{
		config:  config,
		monitor: monitor,
		handler: handler,
		//db:      db,
		peers: make(map[string]*Peer),
		//queryChan: make(chan Query, 128),
	}

	//inst.nasCache, _ = lru.New(32)
	inst.callback = _callback
	//inst.ttlchan = make(chan any, 1024)
	//inst.queryCache, _ = lru.New(25)

	//inst.scoreTable = make(map[string]int)
	//inst.seedingNotify = make(chan string, 32)

	inst.worm = mapset.NewSet[string]()
	inst.history = mapset.NewSet[string]()

	inst.sent.Store(0)
	inst.received.Store(0)
	inst.retry.Store(0)
	inst.protocol = p2p.Protocol{
		Name:    params.ProtocolName,
		Version: uint(params.ProtocolVersion),
		Length:  params.NumberOfMessageCodes,
		Run:     inst.HandlePeer,
		NodeInfo: func() any {
			return map[string]any{
				"version": params.ProtocolVersion,
				"status": map[string]any{
					"dht":            !config.DisableDHT,
					"tcp":            !config.DisableTCP,
					"utp":            !config.DisableUTP,
					"port":           inst.LocalPort(),
					"root":           inst.monitor.DB().Root().Hex(),
					"files":          inst.Congress(),
					"active":         inst.Candidate(),
					"nominee":        inst.Nominee(),
					"leafs":          len(inst.monitor.DB().Blocks()),
					"number":         inst.monitor.CurrentNumber(),
					"maxMessageSize": inst.MaxMessageSize(),
					//					"listen":         monitor.listen,
					"metrics": inst.NasCounter(),
					//"neighbours": inst.Neighbors(),
					"received": inst.received.Load(),
					"sent":     inst.sent.Load(),
				},
				//"score": inst.scoreTable,
				"worm": inst.worm,
			}
		},
		PeerInfo: func(id enode.ID) any {
			inst.peerMu.RLock()
			defer inst.peerMu.RUnlock()
			if p := inst.peers[id.String()]; p != nil {
				if p.Info() != nil {
					return map[string]any{
						"version": p.version,
						"listen":  p.Info().Listen,
						"root":    p.Info().Root.Hex(),
						"files":   p.Info().Files,
						"leafs":   p.Info().Leafs,
					}
				} else {

					return map[string]any{
						"version": p.version,
					}
				}
			}
			return nil
		},
	}

	//add dns discovery
	if inst.protocol.DialCandidates == nil {
		log.Info("Nas dial candidates", "version", params.ProtocolVersion)
		client := dnsdisc.NewClient(dnsdisc.Config{})
		s, err := client.NewIterator([]string{params1.KnownDNSNetwork(params1.MainnetGenesisHash, "all")}...)
		if err == nil {
			inst.protocol.DialCandidates = s
		} else {
			log.Warn("Protocol dial candidates failed", "err", err)
		}
	}

	options := &ttlmap.Options{
		InitialCapacity: 1024,
		OnWillExpire: func(key string, item ttlmap.Item) {
			log.Debug("Nas msg expired", "ih", key, "v", item.Value())
		},
		OnWillEvict: func(key string, item ttlmap.Item) {
			log.Debug("Nas msg evicted", "ih", key, "v", item.Value())
		},
	}

	inst.tunnel = ttlmap.New(options)

	inst.closeAll = make(chan any)

	//inst.wg.Add(1)
	//go inst.listen()
	//inst.wg.Add(1)
	//go inst.process()
	//inst.init()

	log.Info("Fs instance created")

	return inst, nil
}

func (fs *TorrentFS) listen() {
	log.Info("Bitsflow listener starting ...")
	defer fs.wg.Done()
	//ttl := time.NewTimer(3 * time.Second)
	//ticker := time.NewTimer(300 * time.Second)
	//defer ttl.Stop()
	//defer ticker.Stop()
	for {
		select {
		case msg := <-fs.callback:
			meta, ok := msg.(*types.BitsFlow)
			if ok {
				//if meta.Request() > 0 || (params.IsGood(meta.InfoHash()) && fs.config.Mode != params.LAZY) {
				//	if meta.Request() > 0 || params.IsGood(meta.InfoHash()) {
				fs.download(context.Background(), meta.InfoHash(), meta.Request())
			}
		//	} else {
		//		fs.ttlchan <- msg
		//	}
		//case <-ttl.C:
		//	if len(fs.ttlchan) > 0 {
		//		msg := <-fs.ttlchan
		//		meta := msg.(*types.BitsFlow)
		//		fs.download(context.Background(), meta.InfoHash(), meta.Request())
		//	}
		//	ttl.Reset(3 * time.Second)
		/*case <-ticker.C:
		log.Info("Bitsflow status", "neighbors", fs.Neighbors(), "current", fs.monitor.CurrentNumber(), "rev", fs.received.Load(), "sent", fs.sent.Load(), "in", fs.in.Load(), "out", fs.out.Load(), "tunnel", fs.tunnel.Len(), "history", fs.history.Cardinality(), "retry", fs.retry.Load())
		fs.wakeup(context.Background(), fs.sampling())
		ticker.Reset(60 * time.Second)*/
		case <-fs.closeAll:
			log.Info("Bitsflow listener stop")
			return
		}
	}
}

/*func (fs *TorrentFS) Records() map[string]uint64 {
	if progs, err := fs.monitor.DB().InitTorrents(); err == nil {
		return progs
	}

	return nil
}*/

// Protocols implements the node.Service interface.
func (fs *TorrentFS) Protocols() []p2p.Protocol { return []p2p.Protocol{fs.protocol} }

// APIs implements the node.Service interface.
func (fs *TorrentFS) APIs() []rpc.API {
	return []rpc.API{
		{
			Namespace: params.ProtocolName,
			Version:   params.ProtocolVersionStr,
			Service:   NewPublicTorrentAPI(fs),
			Public:    false,
		},
	}
}

func (fs *TorrentFS) Version() uint {
	return fs.protocol.Version
}

// Start starts the data collection thread and the listening server of the dashboard.
// Implements the node.Service interface.
func (fs *TorrentFS) Start() error {
	if fs == nil {
		return errors.New("TorrentFS is nil")
	}

	log.Info("Starting FS server...")

	if fs.monitor == nil {
		return errors.New("storage monitor is not initialized")
	}

	if fs.handler == nil {
		return errors.New("handler is not initialized")
	}

	if err := fs.handler.Start(); err != nil {
		log.Error("Failed to start handler", "err", err)
		return fmt.Errorf("start handler: %w", err)
	}

	if err := fs.monitor.Start(); err != nil {
		log.Error("Failed to start monitor", "err", err)
		return fmt.Errorf("start monitor: %w", err)
	}

	fs.init()

	log.Info("FS server started successfully")
	return nil
}

func (fs *TorrentFS) init() {
	fs.initOnce.Do(func() {
		inst.wg.Add(1)
		go inst.listen()

		if fs.config.Mode != params.LAZY {
			checkpoint := fs.monitor.DB().GetRoot(395964)
			log.Info("Checkpoint loaded")
			if checkpoint == nil {
				for k, ok := range params.ColaFiles {
					if ok {
						fs.bitsflow(context.Background(), k, 1024*1024*1024)
					}
				}
			}
		}

		log.Info("Init finished")
	})
}

func (fs *TorrentFS) bitsflow(ctx context.Context, ih string, size uint64) error {
	select {
	case fs.callback <- types.NewBitsFlow(ih, size):
		return nil

	case <-ctx.Done():
		err := ctx.Err()
		log.Warn("bitsflow canceled by context", "ih", ih, "err", err)
		return err

	case <-fs.closeAll:
		log.Info("bitsflow aborted: fs is closing", "ih", ih)
		return errors.New("fs closed")
	}
}

// Stop stops the data collection thread and the connection listener of the dashboard.
// Implements the node.Service interface.
/*func (fs *TorrentFS) Stop() error {
	if fs == nil {
		log.Info("Cortex fs engine is already stopped")
		return errors.New("fs has been stopped")
	}

	fs.stopOnce.Do(func() {
		log.Info("Fs client listener synchronizing closing ... ...")
		if fs.handler != nil {
			fs.handler.Close()
		}

		if fs.monitor != nil {
			log.Info("Monior stopping ... ...")
			fs.monitor.Stop()
		}

		close(fs.closeAll)
		fs.wg.Wait()

		if len(fs.peers) > 0 {
			for _, p := range fs.peers {
				p.stop()
			}
		}

		if fs.net != nil {
			fs.net.Stop()
		}

		if fs.tunnel != nil {
			fs.tunnel.Drain()
		}

		log.Info("Cortex fs engine stopped")
	})

	return nil
}*/

var ErrAlreadyStopped = errors.New("torrent fs already stopped")

func (fs *TorrentFS) Stop() error {
	if fs == nil {
		return ErrAlreadyStopped
	}

	var stopErr error
	fs.stopOnce.Do(func() {
		log.Info("Stopping Cortex FS engine...")

		if fs.monitor != nil {
			log.Info("Stopping monitor...")
			if err := fs.monitor.Stop(); err != nil {
				log.Warn("Monitor stop error", "err", err)
			}
		}

		if fs.net != nil {
			log.Info("Stopping network...")
			fs.net.Stop()
		}

		if fs.tunnel != nil {
			log.Info("Draining tunnel...")
			fs.tunnel.Drain()
		}

		if len(fs.peers) > 0 {
			log.Info("Stopping peers...", "count", len(fs.peers))
			for _, p := range fs.peers {
				if err := p.stop(); err != nil {
					log.Warn("Peer stop error", "peer", p.ID(), "err", err)
				}
			}
		}

		if fs.handler != nil {
			log.Info("Closing handler...")
			if err := fs.handler.Close(); err != nil {
				log.Warn("Handler close error", "err", err)
			}
		}

		close(fs.closeAll)
		fs.wg.Wait()

		log.Info("Cortex FS engine stopped")
	})

	return stopErr
}
