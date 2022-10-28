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
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/CortexFoundation/CortexTheseus/p2p/enode"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"github.com/CortexFoundation/torrentfs/backend"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/anacrolix/torrent/bencode"
	"github.com/anacrolix/torrent/metainfo"
	mapset "github.com/deckarep/golang-set"
	lru "github.com/hashicorp/golang-lru"
	cp "github.com/otiai10/copy"

	"github.com/ucwong/go-ttlmap"
)

const (
	SEED_SIG uint64 = 0
)

// TorrentFS contains the torrent file system internals.
type TorrentFS struct {
	protocol p2p.Protocol // Protocol description and parameters
	config   *params.Config
	monitor  *Monitor

	peerMu sync.RWMutex     // Mutex to sync the active peer set
	peers  map[string]*Peer // Set of currently active peers

	//queryChan chan Query

	nasCache *lru.Cache
	//queryCache *lru.Cache
	nasCounter uint64

	received uint64
	sent     uint64

	// global file hash & score
	scoreTable map[string]int

	seedingNotify chan string
	closeAll      chan struct{}
	wg            sync.WaitGroup
	once          sync.Once
	worm          mapset.Set

	msg *ttlmap.Map
}

func (t *TorrentFS) storage() *backend.TorrentManager {
	return t.monitor.dl
}

func (t *TorrentFS) chain() *backend.ChainDB {
	return t.monitor.fs
}

var inst *TorrentFS = nil

func GetStorage() CortexStorage {
	mut.RLock()
	defer mut.RUnlock()

	if inst == nil {
		//inst, _ = New(&DefaultConfig, true, false, false)
		log.Warn("Storage instance get failed, should new it first")
	}
	return inst //GetTorrentInstance()
}

var mut sync.RWMutex

// New creates a new torrentfs instance with the given configuration.
func New(config *params.Config, cache, compress, listen bool) (*TorrentFS, error) {
	mut.Lock()
	defer mut.Unlock()
	if inst != nil {
		log.Warn("Storage has been already inited", "storage", inst, "config", config, "cache", cache, "compress", compress, "listen", listen)
		return inst, nil
	}

	db, fsErr := backend.NewChainDB(config)
	if fsErr != nil {
		log.Error("file storage failed", "err", fsErr)
		return nil, fsErr
	}
	log.Info("File storage initialized")

	handler, err := backend.NewTorrentManager(config, db.ID(), cache, compress, nil)
	if err != nil || handler == nil {
		log.Error("fs manager failed")
		return nil, errors.New("fs download manager initialise failed")
	}
	log.Info("Fs manager initialized")

	monitor, moErr := NewMonitor(config, cache, compress, listen, db, handler)
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

	inst.nasCache, _ = lru.New(32)
	//inst.queryCache, _ = lru.New(25)

	inst.scoreTable = make(map[string]int)
	inst.seedingNotify = make(chan string, 32)

	inst.worm = mapset.NewSet()

	inst.protocol = p2p.Protocol{
		Name:    params.ProtocolName,
		Version: uint(params.ProtocolVersion),
		Length:  params.NumberOfMessageCodes,
		Run:     inst.HandlePeer,
		NodeInfo: func() interface{} {
			return map[string]interface{}{
				"version": params.ProtocolVersion,
				"status": map[string]interface{}{
					"dht":            !config.DisableDHT,
					"tcp":            !config.DisableTCP,
					"utp":            !config.DisableUTP,
					"port":           inst.LocalPort(),
					"root":           inst.chain().Root().Hex(),
					"files":          inst.Congress(),
					"active":         inst.Candidate(),
					"nominee":        inst.Nominee(),
					"leafs":          len(inst.chain().Blocks()),
					"number":         monitor.currentNumber,
					"maxMessageSize": inst.MaxMessageSize(),
					//					"listen":         monitor.listen,
					"metrics":    inst.NasCounter(),
					"neighbours": len(inst.peers),
					"received":   inst.received,
					"sent":       inst.sent,
				},
				"score": inst.scoreTable,
			}
		},
		PeerInfo: func(id enode.ID) interface{} {
			inst.peerMu.RLock()
			defer inst.peerMu.RUnlock()
			if p := inst.peers[fmt.Sprintf("%x", id[:8])]; p != nil {
				if p.Info() != nil {
					return map[string]interface{}{
						"version": p.version,
						"listen":  p.Info().Listen,
						"root":    p.Info().Root.Hex(),
						"files":   p.Info().Files,
						"leafs":   p.Info().Leafs,
					}
				} else {

					return map[string]interface{}{
						"version": p.version,
					}
				}
			}
			return nil
		},
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

	inst.msg = ttlmap.New(options)

	inst.closeAll = make(chan struct{})

	//inst.wg.Add(1)
	//go inst.listen()
	//inst.wg.Add(1)
	//go inst.process()

	return inst, nil
}

func (tfs *TorrentFS) process() {
	defer tfs.wg.Done()
	for {
		select {
		case k := <-tfs.seedingNotify:
			tfs.query(k, 0)
		case <-tfs.closeAll:
			return
		}
	}
}

func (tfs *TorrentFS) listen() {
	defer tfs.wg.Done()
	for {
		select {
		case s := <-tfs.seedingNotify:
			tfs.notify(s)
		case <-tfs.closeAll:
			return
		}
	}
}

func (tfs *TorrentFS) MaxMessageSize() uint32 {
	return params.DefaultMaxMessageSize
}

func (tfs *TorrentFS) find(ih string) (*Peer, error) {
	for s, p := range tfs.peers {
		if p.seeding.Contains(ih) {
			// TODO
			log.Debug("Seed found !!!", "from", s, "ih", ih)
			return p, nil
		}
	}

	log.Debug("Seed not found !!!", "neighbors", len(tfs.peers), "ih", ih)
	return nil, nil
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

		log.Debug("Nas "+params.ProtocolVersionStr+" package", "size", packet.Size, "code", packet.Code)
		tfs.received++

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

				if info.Size > 0 {
					if progress, e := tfs.progress(info.Hash); e == nil {
						log.Debug("Nas msg received", "ih", info.Hash, "size", common.StorageSize(float64(info.Size)), "local", common.StorageSize(float64(progress)), "pid", p.id, "queue", tfs.msg.Len(), "peers", len(tfs.peers))
						if err := tfs.storage().Search(context.Background(), info.Hash, progress); err != nil {
							log.Error("Nas "+params.ProtocolVersionStr+" error", "err", err)
							return err
						}
					}
					tfs.nasCounter++
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
		packet.Discard()
	}
}

func (tfs *TorrentFS) progress(ih string) (uint64, error) {
	return tfs.chain().GetTorrentProgress(ih)
}

func (tfs *TorrentFS) score(ih string) bool {
	if !common.IsHexAddress(ih) {
		return false
	}

	if _, ok := tfs.scoreTable[ih]; !ok {
		tfs.scoreTable[ih] = 1
	} else {
		tfs.scoreTable[ih]++
	}

	return true
}

// Protocols implements the node.Service interface.
func (tfs *TorrentFS) Protocols() []p2p.Protocol { return []p2p.Protocol{tfs.protocol} }

// APIs implements the node.Service interface.
func (tfs *TorrentFS) APIs() []rpc.API {
	return []rpc.API{
		{
			Namespace: params.ProtocolName,
			Version:   params.ProtocolVersionStr,
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
func (tfs *TorrentFS) Start(server *p2p.Server) (err error) {
	if tfs == nil || tfs.monitor == nil {
		log.Warn("Storage fs init failed", "fs", tfs)
		return
	}
	log.Info("Started nas", "config", tfs, "mode", tfs.config.Mode, "version", params.ProtocolVersion, "queue", tfs.msg.Len(), "peers", len(tfs.peers))

	err = tfs.monitor.Start()
	if err != nil {
		return
	}

	if tfs.config.Mode != params.LAZY {
		//torrents, _ := tfs.chain().initTorrents()
		checkpoint := tfs.chain().GetRoot(395964)
		//if len(torrents) == 0 {
		if checkpoint == nil {
			for k, ok := range params.GoodFiles {
				if ok {
					if err := tfs.storage().Search(context.Background(), k, 0); err != nil {
						return err
					}

					tfs.query(k, 1024*1024*1024)
					//tfs.seedingNotify <- k
				}
			}
		}
	}
	return
}

// Stop stops the data collection thread and the connection listener of the dashboard.
// Implements the node.Service interface.
func (tfs *TorrentFS) Stop() error {
	if tfs == nil || tfs.monitor == nil {
		log.Info("Cortex fs engine is already stopped")
		return nil
	}

	tfs.once.Do(func() {
		close(tfs.closeAll)
	})

	tfs.wg.Wait()

	// Wait until every goroutine terminates.
	tfs.monitor.lock.Lock()
	tfs.monitor.stop()
	tfs.monitor.lock.Unlock()

	if tfs.nasCache != nil {
		tfs.nasCache.Purge()
	}

	if tfs.msg != nil {
		tfs.msg.Drain()
	}

	//if tfs.queryCache != nil {
	//	tfs.queryCache.Purge()
	//}
	log.Info("Cortex fs engine stopped")
	return nil
}

func (fs *TorrentFS) query(ih string, rawSize uint64) bool {
	if !common.IsHexAddress(ih) {
		return false
	}

	if _, err := fs.msg.Get(ih); err == nil {
		return false
	}

	if rawSize > 0 {
		log.Debug("Query added", "ih", ih, "size", rawSize)
		//fs.nasCache.Add(ih, rawSize)
		fs.msg.Set(ih, ttlmap.NewItem(rawSize, ttlmap.WithTTL(90*time.Second)), nil)
	} else {
		return false
	}

	return true
}

func (fs *TorrentFS) notify(infohash string) bool {
	if !common.IsHexAddress(infohash) {
		return false
	}

	fs.nasCache.Add(infohash, SEED_SIG)

	return true
}

// Available is used to check the file status
func (fs *TorrentFS) localCheck(ctx context.Context, infohash string, rawSize uint64) (bool, error) {
	ret, f, cost, err := fs.storage().Available(infohash, rawSize)

	//if fs.config.Mode == params.LAZY {
	switch {
	case errors.Is(err, backend.ErrInactiveTorrent):
		if progress, e := fs.progress(infohash); e == nil {
			//fs.seedingNotify <- infohash
			fs.wg.Add(1)
			go func() {
				defer fs.wg.Done()
				s := fs.query(infohash, progress)
				if s {
					log.Debug("Nas "+params.ProtocolVersionStr+", restarting", "ih", infohash, "request", common.StorageSize(float64(progress)), "queue", fs.msg.Len(), "peers", len(fs.peers))
				}
			}()
			if e := fs.storage().Search(ctx, infohash, progress); e == nil {
				log.Debug("Torrent wake up", "ih", infohash, "progress", progress, "available", ret, "raw", rawSize, "err", err)
			}
		} else {
			log.Warn("Try to read unregister file", "ih", infohash, "size", common.StorageSize(float64(rawSize)), "err", e)
		}
	case errors.Is(err, backend.ErrUnfinished) || errors.Is(err, backend.ErrTorrentNotFound):
		if progress, e := fs.progress(infohash); e == nil {
			var speed float64
			if cost > 0 {
				t := float64(cost) / (1000 * 1000 * 1000)
				speed = float64(f) / t
			}
			//fs.seedingNotify <- infohash
			fs.wg.Add(1)
			go func() {
				defer fs.wg.Done()
				s := fs.query(infohash, progress)
				if s {
					log.Debug("Nas "+params.ProtocolVersionStr+" query", "ih", infohash, "raw", common.StorageSize(float64(rawSize)), "finish", f, "cost", common.PrettyDuration(cost), "speed", common.StorageSize(speed), "peers", len(fs.peers), "cache", fs.nasCache.Len(), "err", err, "queue", fs.msg.Len(), "peers", len(fs.peers))
				}
			}()

			log.Debug("Torrent sync downloading", "ih", infohash, "available", ret, "raw", rawSize, "finish", f, "err", err)
		}
	}
	//}
	return ret, err
}

func (fs *TorrentFS) GetFileWithSize(ctx context.Context, infohash string, rawSize uint64, subpath string) ([]byte, error) {
	log.Debug("Get file with size", "ih", infohash, "size", rawSize, "path", subpath)
	//fs.wg.Add(1)
	//go func() {
	///	defer fs.wg.Done()
	//if ok, err := fs.localCheck(ctx, infohash, rawSize); err != nil || !ok {
	//	return nil, err
	//}
	//}()

	/*if !fs.worm.Contains(infohash) {
		timer := time.NewTicker(1 * time.Second)
		defer timer.Stop()
		select {
		case <-timer.C:
		case <-ctx.Done():
		}
	} else {
		fs.worm.Add(infohash)
	}*/

	if ret, _, err := fs.storage().GetFile(infohash, subpath); err != nil {
		//log.Warn("Get file failed", "ih", infohash, "size", rawSize, "path", subpath, "err", err)
		if ok, err := fs.localCheck(ctx, infohash, rawSize); err != nil || !ok {
			log.Debug("Get file failed", "ih", infohash, "size", rawSize, "path", subpath, "err", err)
			return nil, err
		}
		return nil, err
	} else {
		log.Debug("Get File directly", "ih", infohash, "size", rawSize, "path", subpath, "ret", len(ret))
		return ret, nil
	}
}

// Seeding Local File, validate folder, seeding and
// load files, default mode is copyMode, linkMode
// will limit user's operations for original files
func (fs *TorrentFS) SeedingLocal(ctx context.Context, filePath string, isLinkMode bool) (infoHash string, err error) {
	// 1. check folder exist
	if _, err = os.Stat(filePath); err != nil {
		return
	}

	// 2. check subfile data exist and not empty:
	// recursively iterate until meet file not empty
	var iterateForValidFile func(basePath string, dataInfo os.FileInfo) bool
	iterateForValidFile = func(basePath string, dataInfo os.FileInfo) bool {
		filePath := filepath.Join(basePath, dataInfo.Name())
		if dataInfo.IsDir() {
			dirFp, _ := os.Open(filePath)
			if fInfos, err := dirFp.Readdir(0); err != nil {
				log.Error("Read dir failed", "filePath", filePath, "err", err)
				return false
			} else {
				for _, v := range fInfos {
					// return as soon as possible if meet 'true', else continue
					if iterateForValidFile(filePath, v) {
						return true
					}
				}
			}
		} else if dataInfo.Size() > 0 {
			return true
		}
		return false
	}

	var (
		dataInfo os.FileInfo
		fileMode bool = false
	)
	dataPath := filepath.Join(filePath, "data")
	if dataInfo, err = os.Stat(dataPath); err != nil {
		dataPath = filepath.Join(filePath, "")
		if dataInfo, err = os.Stat(dataPath); err != nil {
			log.Error("Load data failed", "dataPath", dataPath)
			return
		}
		fileMode = true
	}
	validFlag := iterateForValidFile(filePath, dataInfo)
	if !validFlag {
		err = errors.New("SeedingLocal: Empty Seeding Data!")
		log.Error("SeedingLocal", "check", err.Error(), "path", dataPath, "name", dataInfo.Name(), "fileMode", fileMode)
		return
	}

	// 3. generate torrent file, rewrite if exists
	mi := metainfo.MetaInfo{
		AnnounceList: [][]string{params.MainnetTrackers},
	}
	mi.SetDefaults()
	info := metainfo.Info{PieceLength: 256 * 1024}
	if err = info.BuildFromFilePath(dataPath); err != nil {
		return
	}
	if mi.InfoBytes, err = bencode.Marshal(info); err != nil {
		return
	}

	torrentPath := filepath.Join(filePath, "torrent")
	if fileMode {
		torrentPath = filepath.Join("", "torrent")
	}

	var fileTorrent *os.File
	fileTorrent, err = os.OpenFile(torrentPath, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	if err = mi.Write(fileTorrent); err != nil {
		return
	}

	// 4. copy or link, will not cover if dst exist!
	ih := common.Address(mi.HashInfoBytes())
	log.Info("Local file Seeding", "ih", ih.Hex(), "path", dataPath)
	infoHash = strings.TrimPrefix(strings.ToLower(ih.Hex()), common.Prefix)
	linkDst := filepath.Join(fs.storage().TmpDataDir, infoHash)
	if !isLinkMode {
		if !fileMode {
			err = cp.Copy(filePath, linkDst)
		} else {
			err = os.MkdirAll(filepath.Dir(linkDst), 0777) //os.FileMode(os.ModePerm))
			if err != nil {
				log.Error("Mkdir failed", "path", linkDst)
				return
			}

			err = cp.Copy(filePath, filepath.Join(linkDst, dataInfo.Name()))
			if err != nil {
				log.Error("Mkdir failed", "filePath", filePath, "path", linkDst)
				return
			}
			log.Info("Torrent copy", "torrentPath", torrentPath, "linkDst", linkDst)
			err = cp.Copy(torrentPath, filepath.Join(linkDst, "torrent"))
			if err != nil {
				log.Error("Mkdir failed", "torrentPath", torrentPath, "path", linkDst)
				return
			}

		}
	} else {

		if fileMode {
			//TODO
			return
		}
		// check if symbol link exist
		if _, err = os.Stat(linkDst); err == nil {
			// choice-1: original symbol link exists, cover it. (passed)

			// choice-2: original symbol link exists, return error
			err = os.ErrExist
		} else {
			// create symbol link
			var absOriFilePath string
			if absOriFilePath, err = filepath.Abs(filePath); err == nil {
				err = os.Symlink(absOriFilePath, linkDst)
			}
		}
	}

	// 5. seeding
	if err == nil || err == os.ErrExist {
		log.Debug("SeedingLocal", "dest", linkDst, "err", err)
		err = fs.storage().Search(context.Background(), ih.Hex(), 0)
		if err == nil {
			fs.storage().AddLocalSeedFile(infoHash)
		}
	}

	return
}

// PauseSeeding Local File
func (fs *TorrentFS) PauseLocalSeed(ctx context.Context, ih string) error {
	return fs.storage().PauseLocalSeedFile(ih)
}

// ResumeSeeding Local File
func (fs *TorrentFS) ResumeLocalSeed(ctx context.Context, ih string) error {
	return fs.storage().ResumeLocalSeedFile(ih)
}

// List All Torrents Status (read-only)
func (fs *TorrentFS) ListAllTorrents(ctx context.Context) map[string]map[string]int {
	return fs.storage().ListAllTorrents()
}

func (fs *TorrentFS) Tunnel(ctx context.Context, ih string) error {
	if err := fs.storage().Search(ctx, ih, 1000000000); err != nil {
		return err
	}
	return nil
}

func (fs *TorrentFS) Drop(ih string) error {
	if err := fs.storage().Drop(ih); err != nil {
		return err
	}
	return nil
}

// Download is used to download file with request
func (fs *TorrentFS) download(ctx context.Context, ih string, request uint64) error {
	ih = strings.ToLower(ih)
	update, p, err := fs.chain().SetTorrentProgress(ih, request)
	if err != nil {
		return err
	}

	//fs.find(ih)
	if update {
		//fs.seedingNotify <- ih
		fs.wg.Add(1)
		go func() {
			defer fs.wg.Done()
			s := fs.query(ih, p)
			if s {
				log.Info("Nas "+params.ProtocolVersionStr+" tunnel", "ih", ih, "request", common.StorageSize(float64(p)), "queue", fs.msg.Len(), "peers", len(fs.peers))
			}
		}()

		if err := fs.storage().Search(ctx, ih, p); err != nil {
			return err
		}
	}

	//if _, ok := fs.scoreTable[ih]; !ok {
	//	fs.scoreTable[ih] = 1
	//} else {
	//	fs.scoreTable[ih]++
	//}

	//for k, _ := range GoodFiles {
	//	status, _ := fs.Status(ctx, k)
	//	log.Info("Torrent status", "ih", k, "status", status)
	//}

	return nil
}

func (fs *TorrentFS) Download(ctx context.Context, ih string, request uint64) error {
	return fs.download(ctx, ih, request)
}

func (fs *TorrentFS) Status(ctx context.Context, ih string) (int, error) {
	if fs.storage().IsPending(ih) {
		return 1, nil
	}

	if fs.storage().IsDownloading(ih) {
		return 2, nil
	}

	if fs.storage().IsSeeding(ih) {
		return 0, nil
	}

	return 3, nil
}

func (fs *TorrentFS) LocalPort() int {
	return fs.storage().LocalPort()
}

//func (fs *TorrentFS) Simulate() {
//	fs.storage().Simulate()
//}

func (fs *TorrentFS) Congress() int {
	return fs.storage().Congress()
}

func (fs *TorrentFS) FullSeed() map[string]*backend.Torrent {
	return fs.storage().FullSeed()
}

func (fs *TorrentFS) Candidate() int {
	return fs.storage().Candidate()
}

func (fs *TorrentFS) NasCounter() uint64 {
	return fs.nasCounter
}

func (fs *TorrentFS) ScoreTabler() map[string]int {
	return fs.scoreTable
}

func (fs *TorrentFS) Nominee() int {
	return fs.storage().Nominee()
}

func (fs *TorrentFS) Envelopes() *ttlmap.Map {
	fs.peerMu.RLock()
	defer fs.peerMu.RUnlock()

	return fs.msg
}
