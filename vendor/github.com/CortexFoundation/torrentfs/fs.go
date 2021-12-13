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

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/CortexFoundation/CortexTheseus/p2p/enode"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/anacrolix/torrent/bencode"
	"github.com/anacrolix/torrent/metainfo"
	lru "github.com/hashicorp/golang-lru"
	Copy "github.com/otiai10/copy"
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
	nasCounter uint64
}

func (t *TorrentFS) storage() *TorrentManager {
	return t.monitor.dl
}

func (t *TorrentFS) chain() *ChainDB {
	return t.monitor.fs
}

var inst *TorrentFS = nil

func GetStorage() CortexStorage {
	if inst == nil {
		//inst, _ = New(&DefaultConfig, true, false, false)
		log.Warn("Storage instance get failed, should new it first")
	}
	return inst //GetTorrentInstance()
}

var mut sync.Mutex

// New creates a new torrentfs instance with the given configuration.
func New(config *Config, cache, compress, listen bool) (*TorrentFS, error) {
	mut.Lock()
	defer mut.Unlock()
	if inst != nil {
		log.Warn("Storage has been already inited", "storage", inst, "config", config, "cache", cache, "compress", compress, "listen", listen)
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
					"tcp":            !config.DisableTCP,
					"utp":            !config.DisableUTP,
					"port":           inst.LocalPort(),
					"root":           inst.chain().Root().Hex(),
					"files":          inst.Congress(),
					"active":         inst.Candidate(),
					"leafs":          len(inst.chain().Blocks()),
					"number":         monitor.currentNumber,
					"maxMessageSize": inst.MaxMessageSize(),
					//					"listen":         monitor.listen,
					"metrics": inst.NasCounter(),
				},
			}
		},
		PeerInfo: func(id enode.ID) interface{} {
			inst.peerMu.Lock()
			defer inst.peerMu.Unlock()
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
			if ProtocolVersion > 1 && tfs.config.Mode == LAZY {
				var info *Query
				if err := packet.Decode(&info); err != nil {
					log.Warn("failed to decode msg, peer will be disconnected", "peer", p.peer.ID(), "err", err)
					return errors.New("invalid msg")
				}
				if suc := tfs.queryCache.Contains(info.Hash); !suc {
					log.Info("Nas msg received", "ih", info.Hash, "size", common.StorageSize(float64(info.Size)))
					if progress, e := tfs.chain().GetTorrent(info.Hash); e == nil {
						if progress >= info.Size {
							if err := tfs.storage().Search(context.Background(), info.Hash, info.Size, nil); err != nil {
								log.Error("Nas 2.0 error", "err", err)
								return err
							}
						}
					} else {
						//TODO
						// log.Error("Local unregister file", "ih", info.Hash, "err", e)
					}
					tfs.nasCounter++
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
func (tfs *TorrentFS) Start(server *p2p.Server) (err error) {
	log.Info("Started nas v.2.0", "config", tfs, "mode", tfs.config.Mode)
	if tfs == nil || tfs.monitor == nil {
		return
	}
	err = tfs.monitor.Start()
	if err != nil {
		return
	}
	//TODO
	return
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
func (fs *TorrentFS) available(ctx context.Context, infohash string, rawSize uint64) (bool, error) {
	ret, f, cost, err := fs.storage().available(infohash, rawSize)

	if fs.config.Mode == LAZY {
		if errors.Is(err, ErrInactiveTorrent) {
			if progress, e := fs.chain().GetTorrent(infohash); e == nil {
				log.Debug("Lazy mode, restarting", "ih", infohash, "request", progress)
				if e := fs.storage().Search(ctx, infohash, progress, nil); e == nil {
					log.Debug("Torrent wake up", "ih", infohash, "progress", progress, "err", err, "available", ret, "raw", rawSize, "err", err)
				}
			} else {
				//TODO local fs may be not matched with block chain for some unexpected reasons
				log.Warn("Unregister file", "ih", infohash, "size", common.StorageSize(float64(rawSize)), "err", e)
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
	if ok, err := fs.available(ctx, infohash, rawSize); err != nil || !ok {
		return nil, err
	}

	ret, f, err := fs.storage().getFile(infohash, subpath)

	if err != nil {
		log.Warn("Not avaialble err in getFile", "err", err, "ret", ret, "ih", infohash, "progress", f)
	}

	return ret, err
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

	var dataInfo os.FileInfo
	dataPath := filepath.Join(filePath, "data")
	if dataInfo, err = os.Stat(dataPath); err != nil {
		return
	} else {
		validFlag := iterateForValidFile(filePath, dataInfo)
		if !validFlag {
			err = errors.New("SeedingLocal: Empty Seeding Data!")
			log.Error("SeedingLocal", "check", err.Error(), "path", dataPath)
			return
		}
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

	var fileTorrent *os.File
	fileTorrent, err = os.OpenFile(filepath.Join(filePath, "torrent"), os.O_CREATE|os.O_WRONLY, 0644)
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
		err = Copy.Copy(filePath, linkDst)
	} else {

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
		err = fs.storage().Search(context.Background(), ih.Hex(), 0, nil)
		if err == nil {
			fs.storage().addLocalSeedFile(infoHash)
		}
	}

	return
}

// PauseSeeding Local File
func (fs *TorrentFS) PauseLocalSeed(ctx context.Context, ih string) error {
	return fs.storage().pauseLocalSeedFile(ih)
}

// ResumeSeeding Local File
func (fs *TorrentFS) ResumeLocalSeed(ctx context.Context, ih string) error {
	return fs.storage().resumeLocalSeedFile(ih)
}

// List All Torrents Status (read-only)
func (fs *TorrentFS) ListAllTorrents(ctx context.Context) map[string]map[string]int {
	return fs.storage().listAllTorrents()
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

	//for k, _ := range GoodFiles {
	//	status, _ := fs.Status(ctx, k)
	//	log.Info("Torrent status", "ih", k, "status", status)
	//}

	return nil
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

func (fs *TorrentFS) Congress() int {
	return fs.storage().Congress()
}

func (fs *TorrentFS) Candidate() int {
	return fs.storage().Candidate()
}

func (fs *TorrentFS) NasCounter() uint64 {
	return fs.nasCounter
}
