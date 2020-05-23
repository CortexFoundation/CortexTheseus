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
	"bytes"
	"crypto/sha1"
	"errors"
	"fmt"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/torrentfs/compress"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/types"
	lru "github.com/hashicorp/golang-lru"
	"golang.org/x/time/rate"
	"io/ioutil"
	//"strconv"
	//"github.com/anacrolix/missinggo/slices"
	"github.com/bradfitz/iter"
	"github.com/edsrzf/mmap-go"
	"io"
	"math"
	//"math/rand"
	"os"
	"path"
	"path/filepath"
	//"sort"
	"sync"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/anacrolix/torrent"
	//	"net"
	xlog "github.com/anacrolix/log"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/mmap_span"
	"github.com/anacrolix/torrent/storage"
)

const (
	bucket                  = params.Bucket //it is best size is 1/3 full nodes
	group                   = params.Group
	tier                    = params.TIER
	updateTorrentChanBuffer = params.SyncBatch
	torrentChanSize         = 64

	torrentPending = iota //2
	torrentPaused
	torrentRunning
	torrentSeeding
	torrentSeedingInQueue

	block = int64(params.PER_UPLOAD_BYTES)
	loops = 30
)

type TorrentManager struct {
	client              *torrent.Client
	bytes               map[metainfo.Hash]int64
	torrents            map[metainfo.Hash]*Torrent
	seedingTorrents     map[metainfo.Hash]*Torrent
	activeTorrents      map[metainfo.Hash]*Torrent
	pendingTorrents     map[metainfo.Hash]*Torrent
	maxSeedTask         int
	maxEstablishedConns int
	//maxActiveTask       int
	trackers     [][]string
	boostFetcher *BoostDataFetcher
	DataDir      string
	TmpDataDir   string
	closeAll     chan struct{}
	//removeTorrent       chan metainfo.Hash
	updateTorrent chan interface{}
	//mu                  sync.Mutex
	lock        sync.RWMutex
	wg          sync.WaitGroup
	seedingChan chan *Torrent
	activeChan  chan *Torrent
	pendingChan chan *Torrent
	//closeOnce sync.Once
	fullSeed bool
	boost    bool
	id       uint64
	slot     int
	//bucket int

	fileLock  sync.Mutex
	fileCache *lru.Cache
	fileCh    chan struct{}
	cache     bool
	compress  bool

	metrics bool
	Updates time.Duration

	hotCache *lru.Cache
}

func (tm *TorrentManager) GetLimitation(value int64) int64 {
	return ((value + block - 1) / block) * block
}

func (tm *TorrentManager) CreateTorrent(t *torrent.Torrent, requested int64, status int, ih metainfo.Hash) *Torrent {
	tt := &Torrent{
		t,
		tm.maxEstablishedConns, 5, tm.maxEstablishedConns,
		requested,
		//int64(float64(requested) * expansionFactor),
		tm.GetLimitation(requested),
		0, 0, status,
		ih.String(),
		path.Join(tm.TmpDataDir, ih.String()),
		0, 1, 0, 0, false, true, 0,
	}
	//tm.bytes[ih] = requested
	tm.SetTorrent(ih, tt)
	//tm.pendingChan <- tt
	return tt
}

func (tm *TorrentManager) GetTorrent(ih metainfo.Hash) *Torrent {
	tm.lock.RLock()
	defer tm.lock.RUnlock()
	if torrent, ok := tm.torrents[ih]; !ok {
		return nil
	} else {
		return torrent
	}
}

func (tm *TorrentManager) SetTorrent(ih metainfo.Hash, torrent *Torrent) {
	tm.lock.Lock()
	defer tm.lock.Unlock()
	tm.torrents[ih] = torrent
	tm.pendingChan <- torrent
	log.Trace("P <- B", "ih", ih)
}

func (tm *TorrentManager) Close() error {
	close(tm.closeAll)
	tm.wg.Wait()
	tm.dropAll()
	/*tm.wg.Add(1)
	tm.closeOnce.Do(func() {
		defer tm.wg.Done()
		tm.dropAll()
	})
	tm.wg.Wait()*/
	if tm.cache {
		tm.fileCache.Purge()
	}

	tm.hotCache.Purge()
	log.Info("Fs Download Manager Closed")
	return nil
}

func (tm *TorrentManager) dropAll() {
	tm.lock.Lock()
	defer tm.lock.Unlock()

	tm.client.Close()
}

//func (tm *TorrentManager) RemoveTorrent(input metainfo.Hash) error {
//	tm.removeTorrent <- input
//	return nil
//}

func (tm *TorrentManager) UpdateTorrent(input interface{}) error {
	//go func() { tm.updateTorrent <- input }()
	tm.updateTorrent <- input
	return nil
}

//func isMagnetURI(uri string) bool {
//	return strings.HasPrefix(uri, "magnet:?xt=urn:btih:")
//}

//func GetMagnetURI(infohash metainfo.Hash) string {
//	return "magnet:?xt=urn:btih:" + infohash.String()
//}

func (tm *TorrentManager) UpdateDynamicTrackers(trackers []string) {
	tm.lock.Lock()
	defer tm.lock.Unlock()
	if len(tm.trackers) == 0 {
		tm.trackers = append(tm.trackers, trackers)
	} else if len(tm.trackers) == 1 {
		tm.trackers = append(tm.trackers, trackers)
	} else {
		tm.trackers[1] = trackers
	}

	var newTrackers [][]string = [][]string{trackers}
	for _, t := range tm.pendingTorrents {
		t.AddTrackers(newTrackers)
	}
}

func (tm *TorrentManager) buildUdpTrackers(trackers []string) (array [][]string) {
	array = make([][]string, tier)
	for i, tracker := range trackers {
		array[i%tier] = append(array[i%tier], "udp"+tracker)
	}
	/*array = make([][]string, 1)
	for _, tracker := range trackers {
		array[0] = append(array[0], "udp"+tracker)
	}*/
	return array
}

//func (tm *TorrentManager) buildHttpTrackers(trackers []string) (array [][]string) {
//	array = make([][]string, tier)
//	for i, tracker := range trackers {
//		array[i%tier] = append(array[i%tier], "http"+tracker+"/announce")
//	}
//	return array
//}

func (tm *TorrentManager) SetTrackers(trackers []string, disableTCP, boost bool) {
	tm.lock.Lock()
	defer tm.lock.Unlock()
	/*array := make([][]string, len(trackers))
	for i, tracker := range trackers {
		if disableTCP {
			array[i] = []string{"udp" + tracker}
		} else {
			array[i] = []string{"http" + tracker + "/announce"}
		}
	}*/
	tm.trackers = tm.buildUdpTrackers(trackers)
	//if !disableTCP {
	//tm.trackers = append(tm.trackers, tm.buildHttpTrackers(trackers)...)
	//}
	log.Debug("Boot trackers", "t", tm.trackers)
}

func mmapFile(name string) (mm mmap.MMap, err error) {
	f, err := os.Open(name)
	if err != nil {
		return
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return
	}
	if fi.Size() == 0 {
		return
	}
	return mmap.MapRegion(f, -1, mmap.RDONLY, mmap.COPY, 0)
}

func (tm *TorrentManager) verifyTorrent(info *metainfo.Info, root string) error {
	span := new(mmap_span.MMapSpan)
	for _, file := range info.UpvertedFiles() {
		filename := filepath.Join(append([]string{root, info.Name}, file.Path...)...)
		mm, err := mmapFile(filename)
		if err != nil {
			return err
		}
		if int64(len(mm)) != file.Length {
			return fmt.Errorf("file %q has wrong length, %d / %d", filename, int64(len(mm)), file.Length)
		}
		span.Append(mm)
	}
	for i := range iter.N(info.NumPieces()) {
		p := info.Piece(i)
		hash := sha1.New()
		_, err := io.Copy(hash, io.NewSectionReader(span, p.Offset(), p.Length()))
		if err != nil {
			return err
		}
		good := bytes.Equal(hash.Sum(nil), p.Hash().Bytes())
		if !good {
			return fmt.Errorf("hash mismatch at piece %d", i)
		}
	}
	return nil
}

func (tm *TorrentManager) AddTorrent(filePath string, BytesRequested int64) *Torrent {
	if _, err := os.Stat(filePath); err != nil {
		return nil
	}
	mi, err := metainfo.LoadFromFile(filePath)
	if err != nil {
		log.Error("Error while adding torrent", "Err", err)
		return nil
	}
	spec := torrent.TorrentSpecFromMetaInfo(mi)
	ih := spec.InfoHash
	//log.Info("Get seed from local file", "InfoHash", ih.HexString())

	if t := tm.GetTorrent(ih); t != nil {
		log.Trace("Seed was already existed. Skip", "ih", ih.HexString())
		return t
	}
	TmpDir := path.Join(tm.TmpDataDir, ih.HexString())
	ExistDir := path.Join(tm.DataDir, ih.HexString())

	useExistDir := false
	if _, err := os.Stat(ExistDir); err == nil {
		log.Debug("Seeding from existing file.", "ih", ih.HexString())
		info, err := mi.UnmarshalInfo()
		if err != nil {
			log.Error("error unmarshalling info: ", "info", err)
		}
		if err := tm.verifyTorrent(&info, ExistDir); err != nil {
			log.Warn("Seed failed verification:", "err", err)
		} else {
			useExistDir = true
		}
	}

	if useExistDir {
		log.Trace("existing dir", "dir", ExistDir)
		spec.Storage = storage.NewFile(ExistDir)
		//for _, tracker := range tm.trackers {
		//	spec.Trackers = append(spec.Trackers, tracker)
		//}
		//spec.Trackers = tm.trackers
		spec.Trackers = nil
		//spec.Trackers = append(spec.Trackers, tm.trackers...)
		if t, _, err := tm.client.AddTorrentSpec(spec); err == nil {
			//var ss []string
			//slices.MakeInto(&ss, mi.Nodes)
			//tm.client.AddDHTNodes(ss)
			torrent := tm.CreateTorrent(t, BytesRequested, torrentPending, ih)
			return torrent
		} else {
			log.Warn("Create error")
		}
	} else {
		spec.Storage = storage.NewFile(TmpDir)
		/*for _, tracker := range tm.trackers {
			spec.Trackers = append(spec.Trackers, tracker)
		}*/
		//spec.Trackers = tm.trackers
		spec.Trackers = nil
		//spec.Trackers = append(spec.Trackers, tm.trackers...)
		if t, _, err := tm.client.AddTorrentSpec(spec); err == nil {
			//var ss []string
			//slices.MakeInto(&ss, mi.Nodes)
			//tm.client.AddDHTNodes(ss)
			torrent := tm.CreateTorrent(t, BytesRequested, torrentPending, ih)
			return torrent
		} else {
			log.Warn("Create error ... ")
		}
	}
	return nil
}

func (tm *TorrentManager) AddInfoHash(ih metainfo.Hash, BytesRequested int64) *Torrent {
	if t := tm.GetTorrent(ih); t != nil {
		return t
	}

	dataPath := path.Join(tm.TmpDataDir, ih.HexString())
	torrentPath := path.Join(tm.TmpDataDir, ih.HexString(), "torrent")
	seedTorrentPath := path.Join(tm.DataDir, ih.HexString(), "torrent")

	if _, err := os.Stat(seedTorrentPath); err == nil {
		return tm.AddTorrent(seedTorrentPath, BytesRequested)
	} else if _, err := os.Stat(torrentPath); err == nil {
		return tm.AddTorrent(torrentPath, BytesRequested)
	}

	log.Trace("Get file from infohash", "ih", ih.HexString())

	spec := &torrent.TorrentSpec{
		Trackers: [][]string{}, //tm.trackers, //[][]string{},
		//DisplayName: ih.String(),
		InfoHash: ih,
		Storage:  storage.NewFile(dataPath),
	}

	//for _, tracker := range tm.trackers {
	//	spec.Trackers = append(spec.Trackers, tracker)
	//}
	//spec.Trackers = tm.trackers
	//spec.Trackers = append(spec.Trackers, tm.trackers...)
	//log.Info("Torrent specific info", "spec", spec)

	t, _, err := tm.client.AddTorrentSpec(spec)
	if err != nil {
		return nil
	}
	/*go func() {
		<-t.GotInfo()
		t.VerifyData()
	}()*/
	tt := tm.CreateTorrent(t, BytesRequested, torrentPending, ih)
	//tm.mu.Unlock()
	//log.Info("Torrent is waiting for gotInfo", "InfoHash", ih.HexString())
	return tt
}

// UpdateInfoHash ...
func (tm *TorrentManager) UpdateInfoHash(ih metainfo.Hash, BytesRequested int64) {
	log.Debug("Update seed", "ih", ih, "bytes", BytesRequested)
	tm.lock.Lock()
	defer tm.lock.Unlock()
	if t, ok := tm.bytes[ih]; !ok || t < BytesRequested {
		tm.bytes[ih] = BytesRequested
	}
	/*if t := tm.GetTorrent(ih); t != nil {
		if BytesRequested < t.bytesRequested {
			return
		}
		t.bytesRequested = BytesRequested
		if t.bytesRequested > t.bytesLimitation {
			t.bytesLimitation = GetLimitation(BytesRequested)
		}
	}*/
}

// DropInfoHash ...
/*func (tm *TorrentManager) DropInfoHash(ih metainfo.Hash) bool {
	if t := tm.GetTorrent(ih); t != nil {
		t.Torrent.Drop()
		tm.lock.Lock()
		delete(tm.torrents, ih)
		tm.lock.Unlock()
		return true
	}
	return false
}*/

//var CurrentTorrentManager *TorrentManager = nil

// NewTorrentManager ...
func NewTorrentManager(config *Config, fsid uint64, cache, compress bool) (*TorrentManager, error) {
	//    log.Info("config",
	//      "port", config.Port,
	//      "datadir", config.DataDir,
	//      "rpcuri", config.RpcURI,
	//      "ipcuri", config.IpcPath,
	//      "boostnodes", config.BoostNodes,
	//      "trackers", config.DefaultTrackers,
	//      "syncmode", config.SyncMode,
	//      "max_seedingnum", config.MaxSeedingNum,
	//      "max_activenum", config.MaxActiveNum,
	//    )
	cfg := torrent.NewDefaultClientConfig()
	cfg.DisableUTP = config.DisableUTP
	cfg.NoDHT = config.DisableDHT
	cfg.DisableTCP = config.DisableTCP

	//cfg.HeaderObfuscationPolicy.Preferred = true
	//cfg.HeaderObfuscationPolicy.RequirePreferred = true

	cfg.DataDir = config.DataDir
	if config.UploadRate > 0 {
		cfg.UploadRateLimiter = rate.NewLimiter(rate.Limit(config.UploadRate), 256<<10)
	}
	if config.DownloadRate > 0 {
		cfg.DownloadRateLimiter = rate.NewLimiter(rate.Limit(config.DownloadRate), 1<<20)
	}
	//cfg.DisableEncryption = true
	//listenAddr := &net.TCPAddr{}
	//log.Info("Torrent client listening on", "addr", listenAddr)
	//cfg.SetListenAddr(listenAddr.String())
	//cfg.HTTPUserAgent = "Cortex"
	cfg.Seed = true

	cfg.EstablishedConnsPerTorrent = 25 //len(config.DefaultTrackers)
	cfg.HalfOpenConnsPerTorrent = 25

	cfg.ListenPort = config.Port
	if config.Quiet {
		cfg.Logger = xlog.Discard
	}
	//cfg.Debug = true
	cfg.DropDuplicatePeerIds = true
	//cfg.ListenHost = torrent.LoopbackListenHost
	//cfg.DhtStartingNodes = dht.GlobalBootstrapAddrs //func() ([]dht.Addr, error) { return nil, nil }
	//log.Info("Torrent client configuration", "config", cfg)
	cl, err := torrent.NewClient(cfg)
	if err != nil {
		log.Error("Error while create torrent client", "err", err)
		return nil, err
	}

	tmpFilePath := path.Join(config.DataDir, defaultTmpFilePath)
	/*if _, err := os.Stat(tmpFilePath); err == nil {
		err := os.Remove(tmpFilePath)
		if err != nil {
			log.Warn("Purge the current file path failed", "path", tmpFilePath, "err", err)
		}
	}*/

	if _, err := os.Stat(tmpFilePath); err != nil {
		err = os.MkdirAll(filepath.Dir(tmpFilePath), 0750) //os.FileMode(os.ModePerm))
		if err != nil {
			log.Error("Mkdir failed", "path", tmpFilePath)
			return nil, err
		}
	}

	TorrentManager := &TorrentManager{
		client:          cl,
		torrents:        make(map[metainfo.Hash]*Torrent),
		pendingTorrents: make(map[metainfo.Hash]*Torrent),
		seedingTorrents: make(map[metainfo.Hash]*Torrent),
		activeTorrents:  make(map[metainfo.Hash]*Torrent),
		bytes:           make(map[metainfo.Hash]int64),
		maxSeedTask:     config.MaxSeedingNum,
		//maxActiveTask:       config.MaxActiveNum,
		maxEstablishedConns: cfg.EstablishedConnsPerTorrent,
		DataDir:             config.DataDir,
		TmpDataDir:          tmpFilePath,
		boostFetcher:        NewBoostDataFetcher(config.BoostNodes),
		closeAll:            make(chan struct{}),
		//removeTorrent:       make(chan metainfo.Hash, removeTorrentChanBuffer),
		updateTorrent: make(chan interface{}, updateTorrentChanBuffer),
		seedingChan:   make(chan *Torrent, torrentChanSize),
		activeChan:    make(chan *Torrent, torrentChanSize),
		pendingChan:   make(chan *Torrent, torrentChanSize),
		//updateTorrent:       make(chan interface{}),
		fullSeed: config.FullSeed,
		//boost:    config.Boost,
		id: fsid,
		//bucket:1024
		slot: int(fsid % bucket),
	}

	TorrentManager.fileCache, _ = lru.New(8)
	TorrentManager.fileCh = make(chan struct{}, 4)
	TorrentManager.compress = compress
	TorrentManager.cache = cache

	TorrentManager.metrics = config.Metrics

	TorrentManager.hotCache, _ = lru.New(32)

	if len(config.DefaultTrackers) > 0 {
		log.Debug("Tracker list", "trackers", config.DefaultTrackers)
		TorrentManager.SetTrackers(config.DefaultTrackers, config.DisableTCP, config.Boost)
	}
	log.Debug("Fs client initialized", "config", config)

	//CurrentTorrentManager = TorrentManager
	//cl.WaitAll()
	return TorrentManager, nil
}

func (tm *TorrentManager) Start() error {
	tm.init()

	tm.wg.Add(1)
	go tm.mainLoop()
	tm.wg.Add(1)
	go tm.pendingTorrentLoop()
	tm.wg.Add(1)
	go tm.activeTorrentLoop()
	tm.wg.Add(1)
	go tm.seedingTorrentLoop()

	return nil
}

func (tm *TorrentManager) seedingTorrentLoop() {
	defer tm.wg.Done()
	//timer := time.NewTimer(time.Second * queryTimeInterval * 60)
	//defer timer.Stop()
	for {
		select {
		case t := <-tm.seedingChan:
			tm.seedingTorrents[t.Torrent.InfoHash()] = t
			t.Seed()
			if len(tm.seedingTorrents) > params.LimitSeeding {
				tm.dropSeeding(tm.slot)
			} else if len(tm.seedingTorrents) > tm.maxSeedTask {
				tm.maxSeedTask++
				tm.graceSeeding(tm.slot)
			}
			//	case <- timer.C:
			//for _, t := range tm.seedingTorrents {
			//t.SetMaxEstablishedConns(0)
			//t.SetMaxEstablishedConns(1)
			//t.Torrent.Drop()
			//}
			//log.Info("Seeding status refresh", "len", len(tm.seedingTorrents))
			//		timer.Reset(time.Second * queryTimeInterval * 60)
		case <-tm.closeAll:
			log.Info("Seeding loop closed")
			return
		}
	}
}

/*func (tm *TorrentManager) Stop() error {
	close(tm.closeAll)
	tm.wg.Wait()
	return nil
}*/
func (tm *TorrentManager) init() {
	for k, ok := range GoodFiles {
		if ok {
			tm.SearchAndDownload(k, 0)
		}
	}
}

func (tm *TorrentManager) SearchAndDownload(hex string, request int64) {
	hash := metainfo.NewHashFromHex(hex)
	if t := tm.AddInfoHash(hash, request); t != nil {
		if request > 0 {
			tm.UpdateInfoHash(hash, request)
		}
	}
}

func (tm *TorrentManager) mainLoop() {
	defer tm.wg.Done()
	for {
		select {
		case msg := <-tm.updateTorrent:
			meta := msg.(types.FlowControlMeta)
			if _, ok := BadFiles[meta.InfoHash.HexString()]; ok {
				continue
			}

			if meta.IsCreate {
				counter := 0
				for {
					if t := tm.AddInfoHash(meta.InfoHash, int64(meta.BytesRequested)); t != nil {
						log.Debug("Seed [create] success", "ih", meta.InfoHash, "request", meta.BytesRequested)
						if int64(meta.BytesRequested) > 0 {
							tm.UpdateInfoHash(meta.InfoHash, int64(meta.BytesRequested))
						}
						break
					} else {
						if counter > 10 {
							panic("Fail adding file for 10 times")
						}
						log.Error("Seed [create] failed", "ih", meta.InfoHash, "request", meta.BytesRequested, "counter", counter)
						counter++
					}
				}
			} else {
				log.Debug("Seed [update] success", "ih", meta.InfoHash, "request", meta.BytesRequested)
				tm.UpdateInfoHash(meta.InfoHash, int64(meta.BytesRequested))
			}
		case <-tm.closeAll:
			return
		}
	}
}

//type ActiveTorrentList []*Torrent

//func (s ActiveTorrentList) Len() int      { return len(s) }
//func (s ActiveTorrentList) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
//func (s ActiveTorrentList) Less(i, j int) bool {
//	return s[i].BytesLeft() > s[j].BytesLeft() || (s[i].BytesLeft() == s[j].BytesLeft() && s[i].weight > s[j].weight)
//}

//type seedingTorrentList []*Torrent

//func (s seedingTorrentList) Len() int           { return len(s) }
//func (s seedingTorrentList) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
//func (s seedingTorrentList) Less(i, j int) bool { return s[i].weight > s[j].weight }

func (tm *TorrentManager) pendingTorrentLoop() {
	defer tm.wg.Done()
	timer := time.NewTimer(time.Second * queryTimeInterval)
	defer timer.Stop()
	for {
		select {
		case t := <-tm.pendingChan:
			tm.pendingTorrents[t.Torrent.InfoHash()] = t
			//if len(tm.pendingTorrents) == 1 {
			//	ok := timer.Reset(time.Millisecond * 0)
			//	log.Trace("P -> [ON]", "ok", ok)
			//}
		case <-timer.C:
			for ih, t := range tm.pendingTorrents {
				//		ih := t.Torrent.InfoHash()
				if _, ok := BadFiles[ih.String()]; ok {
					continue
				}
				t.loop += 1
				if t.Torrent.Info() != nil {
					if t.start == 0 {
						if t.isBoosting {
							log.Trace("A <- P (BOOST)", "ih", ih, "boost", t.isBoosting)
							t.isBoosting = false
						} else {
							log.Trace("A <- P (UDP)", "ih", ih, "boost", t.isBoosting)
						}
						t.AddTrackers(tm.trackers)
						t.start = mclock.Now()
					}

					if err := t.WriteTorrent(); err == nil {
						if len(tm.activeChan) < cap(tm.activeChan) {
							delete(tm.pendingTorrents, ih)
							t.loop = 0
							/*if t.start == 0 {
								log.Info("A <- P (UDP)", "hash", ih, "pieces", t.Torrent.NumPieces())
								t.AddTrackers(tm.trackers)
								t.start = mclock.Now()
							} else {
								log.Info("A <- P", "hash", ih, "pieces", t.Torrent.NumPieces(), "elapsed", time.Duration(mclock.Now())-time.Duration(t.start))
							}*/
							//t.start = mclock.Now()
							tm.activeChan <- t
						}
					}
				} else if t.loop > torrentWaitingTime/queryTimeInterval || (t.start == 0 && tm.boost && tm.bytes[ih] > 0) {
					if !t.isBoosting {
						t.loop = 0
						t.isBoosting = true
						//go func(t *Torrent) {
						//defer t.BoostOff()
						//log.Info("Try to boost seed", "hash", t.infohash)
						if data, err := tm.boostFetcher.GetTorrent(ih.String()); err == nil {
							if t.Torrent.Info() != nil {
								//log.Warn("Seed already exist", "hash", t.infohash)
								t.BoostOff()
								//return
								continue
							}
							//t.Torrent.Drop()
							if err := t.ReloadTorrent(data, tm); err == nil {
								tm.lock.Lock()
								tm.torrents[ih] = t
								tm.lock.Unlock()
							} else {
								t.BoostOff()
							}

							/*bytesRequested := t.bytesRequested
							tm.UpdateTorrent(FlowControlMeta{
								InfoHash:       ih,
								BytesRequested: uint64(bytesRequested),
								IsCreate:       true,
							})*/
						} else {
							log.Warn("Boost failed", "ih", ih.String(), "err", err)
							//boost failed , use the normal way
							if t.start == 0 && (tm.bytes[ih] > 0 || tm.fullSeed || t.loop > 600) { //|| len(tm.pendingTorrents) == 1) {
								t.AddTrackers(tm.trackers)
								t.start = mclock.Now()
							}
							t.BoostOff()
						}
						//t.BoostOff()
						//}(t)
					}
				} else {
					//if (tm.bytes[ih] > 0 && t.start == 0) || (t.start == 0 && t.loop > 60) {
					//if (tm.bytes[ih] > 0 && t.start == 0) || (t.start == 0 && tm.fullSeed) || (t.start == 0 && t.loop > 1800) {
					if _, ok := GoodFiles[t.InfoHash()]; t.start == 0 && (ok || tm.bytes[ih] > 0 || tm.fullSeed || t.loop > 600) {
						if ok {
							log.Debug("Good file found in pending", "ih", common.HexToHash(ih.String()))
						}
						t.AddTrackers(tm.trackers)
						t.start = mclock.Now()
					}
				}
			}
			//if len(tm.pendingTorrents) > 0 {
			timer.Reset(time.Second * queryTimeInterval)
			//} else {
			//	log.Trace("P -> [OFF]")
			//}
		case <-tm.closeAll:
			log.Info("Pending seed loop closed")
			return
		}
	}
}

func (tm *TorrentManager) activeTorrentLoop() {
	defer tm.wg.Done()
	timer := time.NewTimer(time.Second * queryTimeInterval)
	defer timer.Stop()
	var total_size, current_size, log_counter, counter uint64
	var active_paused, active_wait, active_boost, active_running int
	for {
		counter++
		select {
		case t := <-tm.activeChan:
			tm.activeTorrents[t.Torrent.InfoHash()] = t
			//if len(tm.activeTorrents) == 1 {
			//	ok := timer.Reset(time.Millisecond * 0)
			//	log.Trace("A -> [ON]", "ok", ok)
			//}
		case <-timer.C:
			//for _, t := range tm.torrents {
			//	t.weight = 1 + int(t.cited*10/maxCited)
			//}
			log_counter++
			//var active_paused, active_wait, active_boost, active_running int
			//var activeTorrents []*Torrent

			for ih, t := range tm.activeTorrents {
				//ih := t.Torrent.InfoHash()
				BytesRequested := int64(0)
				if _, ok := GoodFiles[t.InfoHash()]; ok {
					if t.Length() != t.bytesRequested || !t.fast {
						BytesRequested = t.Length()
						t.fast = true
						log.Debug("Good file found", "hash", common.HexToHash(ih.String()), "size", common.StorageSize(BytesRequested), "request", common.StorageSize(t.bytesRequested), "len", common.StorageSize(t.Length()), "limit", common.StorageSize(t.bytesLimitation))
					}
				} else {
					tm.lock.RLock()
					if tm.fullSeed {
						if tm.bytes[ih] >= t.Length() {
							BytesRequested = tm.bytes[ih]
							t.fast = true
						} else {
							if t.bytesRequested <= t.BytesCompleted()+block/2 {
								BytesRequested = int64(math.Min(float64(t.Length()), float64(t.bytesRequested+block)))
								t.fast = false
							}
						}
					} else {
						if tm.bytes[ih] >= t.Length() {
							BytesRequested = tm.bytes[ih]
							t.fast = true
						} else {
							if t.bytesRequested <= t.BytesCompleted()+block/2 {
								BytesRequested = int64(math.Min(float64(tm.bytes[ih]), float64(t.bytesRequested+block)))
								t.fast = false
							}
						}
					}
					tm.lock.RUnlock()
				}

				if t.bytesRequested < BytesRequested {
					t.bytesRequested = BytesRequested
					t.bytesLimitation = tm.GetLimitation(BytesRequested)
				}

				if t.bytesRequested == 0 {
					active_wait += 1
					if log_counter%60 == 0 {
						log.Debug("[Waiting]", "ih", ih.String(), "complete", common.StorageSize(t.bytesCompleted), "req", common.StorageSize(t.bytesRequested), "quota", common.StorageSize(t.bytesRequested), "limit", common.StorageSize(t.bytesLimitation), "total", common.StorageSize(t.BytesMissing()), "seg", len(t.Torrent.PieceStateRuns()), "peers", t.currentConns, "max", t.Torrent.NumPieces())
					}
					continue
				}

				if t.BytesCompleted() > t.bytesCompleted {
					total_size += uint64(t.BytesCompleted() - t.bytesCompleted)
					current_size += uint64(t.BytesCompleted() - t.bytesCompleted)
				}

				t.bytesCompleted = t.BytesCompleted()
				t.bytesMissing = t.BytesMissing()

				if t.Finished() {
					tm.lock.Lock()
					if _, err := os.Stat(path.Join(tm.DataDir, ih.String())); err == nil {
						if len(tm.seedingChan) < cap(tm.seedingChan) {
							log.Debug("Path exist", "ih", ih, "path", path.Join(tm.DataDir, ih.String()))
							delete(tm.activeTorrents, ih)
							log.Trace("S <- A", "ih", ih) //, "elapsed", time.Duration(mclock.Now())-time.Duration(t.start))
							//t.start = mclock.Now()
							tm.seedingChan <- t
						}
					} else {
						err := os.Symlink(
							path.Join(defaultTmpFilePath, ih.String()),
							path.Join(tm.DataDir, ih.String()),
						)
						if err != nil {
							err = os.Remove(
								path.Join(tm.DataDir, ih.String()),
							)
							if err == nil {
								log.Debug("Fix path success", "ih", ih, "size", t.bytesCompleted, "miss", t.bytesMissing, "loop", log_counter)
							}
						} else {
							if len(tm.seedingChan) < cap(tm.seedingChan) {
								delete(tm.activeTorrents, ih)
								log.Trace("S <- A", "ih", ih) //, "elapsed", time.Duration(mclock.Now())-time.Duration(t.start))
								//t.start = mclock.Now()
								tm.seedingChan <- t
							}
						}
					}

					tm.lock.Unlock()
					continue
				}

				if t.bytesCompleted >= t.bytesLimitation {
					t.Pause()
					active_paused += 1
					if log_counter%45 == 0 {
						bar := ProgressBar(t.bytesCompleted, t.Torrent.Length(), "[Paused]")
						log.Info(bar, "hash", common.HexToHash(ih.String()), "complete", common.StorageSize(t.bytesCompleted), "req", common.StorageSize(t.bytesRequested), "limit", common.StorageSize(t.bytesLimitation), "total", common.StorageSize(t.bytesMissing+t.bytesCompleted), "prog", math.Min(float64(t.bytesCompleted), float64(t.bytesRequested))/float64(t.bytesCompleted+t.bytesMissing), "seg", len(t.Torrent.PieceStateRuns()), "peers", t.currentConns, "max", t.Torrent.NumPieces())
					}
					continue
				} else if t.bytesRequested >= t.bytesCompleted+t.bytesMissing {
					t.loop += 1
					if t.loop > downloadWaitingTime/queryTimeInterval && t.bytesCompleted*2 < t.bytesRequested {
						t.loop = 0
						if t.isBoosting {
							continue
						}
						t.Pause()
						t.isBoosting = true
						tm.wg.Add(1)
						go func(t *Torrent) {
							defer tm.wg.Done()
							defer t.BoostOff()
							filepaths := []string{}
							filedatas := [][]byte{}
							for _, file := range t.Files() {
								if file.BytesCompleted() > 0 {
									continue
								}
								subpath := file.Path()
								if data, err := tm.boostFetcher.GetFile(ih.String(), subpath); err == nil {
									filedatas = append(filedatas, data)
									filepaths = append(filepaths, subpath)
								} else {
									return
								}
							}
							t.Torrent.Drop()
							t.ReloadFile(filepaths, filedatas, tm)
						}(t)
						active_boost += 1
						if log_counter%30 == 0 {
							log.Debug("[Boosting]", "hash", ih.String(), "complete", common.StorageSize(t.bytesCompleted), "quota", common.StorageSize(t.bytesRequested), "total", common.StorageSize(t.bytesMissing+t.bytesCompleted), "prog", math.Min(float64(t.bytesCompleted), float64(t.bytesRequested))/float64(t.bytesCompleted+t.bytesMissing), "seg", len(t.Torrent.PieceStateRuns()), "max", t.Torrent.NumPieces(), "status", t.status, "boost", t.isBoosting)
						}
						continue
					}
				}

				if log_counter%60 == 0 && t.bytesCompleted > 0 {
					bar := ProgressBar(t.bytesCompleted, t.Torrent.Length(), "")
					elapsed := time.Duration(mclock.Now()) - time.Duration(t.start)
					log.Info( /*"[Downloading]" + */ bar, "hash", common.HexToHash(ih.String()), "complete", common.StorageSize(t.bytesCompleted) /*"req", common.StorageSize(t.bytesRequested),*/, "limit", common.StorageSize(t.bytesLimitation), "total", common.StorageSize(t.Torrent.Length()) /*"prog", math.Min(float64(t.bytesCompleted), float64(t.bytesRequested))/float64(t.bytesCompleted+t.bytesMissing),*/, "seg", len(t.Torrent.PieceStateRuns()), "peers", t.currentConns, "max", t.Torrent.NumPieces(), "speed", common.StorageSize(float64(t.bytesCompleted*1000*1000*1000)/float64(elapsed)).String()+"/s", "elapsed", common.PrettyDuration(elapsed))
				}

				if t.bytesCompleted < t.bytesLimitation && !t.isBoosting {
					//activeTorrents = append(activeTorrents, t)
					t.Run(tm.slot)
					active_running += 1
				}
			}

			/*if len(activeTorrents) <= tm.maxActiveTask {
				for _, t := range activeTorrents {
					t.Run(tm.slot)
					active_running += 1
				}
			} else {
				sort.Slice(activeTorrents, func(i, j int) bool {
					return activeTorrents[i].BytesLeft() > activeTorrents[j].BytesLeft() || (activeTorrents[i].BytesLeft() == activeTorrents[j].BytesLeft() && activeTorrents[i].weight > activeTorrents[j].weight)
				})
				for i := 0; i < tm.maxActiveTask; i++ {
					activeTorrents[i].Run(tm.slot)
					active_running += 1
				}
				for i := tm.maxActiveTask; i < len(activeTorrents); i++ {
					if activeTorrents[i].bytesRequested > activeTorrents[i].bytesCompleted {
						activeTorrents[i].Run(tm.slot)
						active_running += 1
					} else {
						activeTorrents[i].Pause()
						active_paused += 1
					}
				}
			}*/

			if counter >= 5*loops && (len(tm.pendingTorrents) > 0 || active_running > 0 || active_paused > 0 || counter >= 10*loops) {
				//for _, ttt := range tm.client.Torrents() {
				//	all += len(ttt.KnownSwarm())
				//}
				log.Info("Fs status", "pending", len(tm.pendingTorrents) /* "active", len(tm.activeTorrents),*/, "waiting", active_wait, "downloading", active_running, "paused", active_paused /*"boost", active_boost,*/, "seeding", len(tm.seedingTorrents), "size", common.StorageSize(total_size), "speed_a", common.StorageSize(total_size/log_counter*queryTimeInterval).String()+"/s", "speed_b", common.StorageSize(current_size/counter*queryTimeInterval).String()+"/s" /*"channel", len(tm.updateTorrent)+len(tm.seedingChan)+len(tm.pendingChan)+len(tm.activeChan),*/, "slot", tm.slot, "metrics", common.PrettyDuration(tm.Updates), "hot", tm.hotCache.Len())
				/*tmp := make(map[common.Hash]int)
				sum := 0
				for _, ttt := range tm.client.Torrents() {
					for _, p := range ttt.KnownSwarm() {
						if common.BytesToHash(p.Id[:]) == common.EmptyHash {
							continue
						}
						k := common.BytesToHash(append(p.Id[:], p.IP[:]...))
						if v, ok := tmp[k]; !ok {
							log.Debug("Active peer status", "hash", ttt.InfoHash(), "id", common.BytesToHash(p.Id[:]), "k", k, "ip", p.IP.String(), "port", p.Port, "source", p.Source, "encrypt", p.SupportsEncryption, "flag", p.PexPeerFlags, "buk", len(tmp), "active", sum, "total", len(ttt.KnownSwarm()))
							tmp[k] = 1
						} else {
							tmp[k] = v + 1
						}
						sum += tmp[k]
					}
				}

				//for k, v := range tmp {
				//	log.Trace("Storage peers statics", "k", k, "v", v)
				//}

				for _, ip := range tm.client.BadPeerIPs() {
					log.Warn("Bad peer", "ip", ip)
				}*/

				counter = 0
				current_size = 0
			}
			active_paused, active_wait, active_boost, active_running = 0, 0, 0, 0
			//if len(tm.activeTorrents) > 0 {
			timer.Reset(time.Second * queryTimeInterval)
			//} else {
			//	log.Trace("A -> [OFF]")
			//}
		case <-tm.closeAll:
			log.Info("Active seed loop closed")
			return
		}
	}
}

func (tm *TorrentManager) dropSeeding(slot int) error {
	g := int(math.Min(float64(group), float64(tm.maxSeedTask)))
	s := slot % g
	i := 0
	for ih, t := range tm.seedingTorrents {
		if i%group == s {
			if t.currentConns <= 1 {
				continue
			}
			if tm.hotCache.Contains(ih) {
				log.Warn("Encounter active torrent", "ih", ih, "index", i, "group", s, "slot", slot, "len", len(tm.seedingTorrents), "max", tm.maxSeedTask, "peers", t.currentConns, "cited", t.cited)
				continue
			}
			t.currentConns = 1
			t.Torrent.SetMaxEstablishedConns(t.currentConns)
			log.Warn("Drop seeding invoke", "ih", ih, "index", i, "group", s, "slot", slot, "len", len(tm.seedingTorrents), "max", tm.maxSeedTask, "peers", t.currentConns, "cited", t.cited)
		}
		i++
	}
	return nil
}

func (tm *TorrentManager) graceSeeding(slot int) error {
	g := int(math.Min(float64(group), float64(tm.maxSeedTask)))
	s := slot % g
	i := 0
	for ih, t := range tm.seedingTorrents {
		if i%group == s {
			if t.currentConns <= t.minEstablishedConns {
				continue
			}
			if tm.hotCache.Contains(ih) {
				log.Warn("Encounter active torrent", "ih", ih, "index", i, "group", s, "slot", slot, "len", len(tm.seedingTorrents), "max", tm.maxSeedTask, "peers", t.currentConns, "cited", t.cited)
				continue
			}
			t.currentConns = t.minEstablishedConns
			t.Torrent.SetMaxEstablishedConns(t.currentConns)
			log.Warn("Grace seeding invoke", "ih", ih, "index", i, "group", s, "slot", slot, "len", len(tm.seedingTorrents), "max", tm.maxSeedTask, "peers", t.currentConns, "cited", t.cited)
		}
		i++
	}
	return nil
}

func (fs *TorrentManager) Available(infohash string, rawSize int64) (bool, error) {
	if fs.metrics {
		defer func(start time.Time) { fs.Updates += time.Since(start) }(time.Now())
	}
	ih := metainfo.NewHashFromHex(infohash)
	if torrent := fs.GetTorrent(ih); torrent == nil {
		return false, errors.New("file not exist")
	} else {
		if !torrent.IsAvailable() {
			return false, errors.New("download not completed")
		}
		return torrent.BytesCompleted() <= rawSize, nil
	}
}

func (fs *TorrentManager) GetFile(infohash, subpath string) ([]byte, error) {
	if fs.metrics {
		defer func(start time.Time) { fs.Updates += time.Since(start) }(time.Now())
	}
	ih := metainfo.NewHashFromHex(infohash)
	//tm := fs.monitor.dl
	if torrent := fs.GetTorrent(ih); torrent == nil {
		log.Debug("Torrent not found", "hash", infohash)
		return nil, errors.New("file not exist")
	} else {
		if !torrent.IsAvailable() {
			log.Error("Read unavailable file", "hash", infohash, "subpath", subpath)
			return nil, errors.New("download not completed")
		}

		fs.hotCache.Add(ih, true)
		if torrent.currentConns < fs.maxEstablishedConns {
			torrent.currentConns = fs.maxEstablishedConns
			torrent.SetMaxEstablishedConns(torrent.currentConns)
			log.Info("Torrent active", "ih", ih, "peers", torrent.currentConns)
		}

		fs.fileCh <- struct{}{}
		defer fs.release()
		var key = infohash + subpath
		if fs.cache {
			if cache, ok := fs.fileCache.Get(key); ok {
				if c, err := fs.unzip(cache.([]byte)); err != nil {
					return nil, err
				} else {
					if fs.compress {
						log.Info("File cache", "hash", infohash, "path", subpath, "size", fs.fileCache.Len(), "compress", len(cache.([]byte)), "origin", len(c), "compress", fs.compress)
					}
					return c, nil
				}
			}
		}

		fs.fileLock.Lock()
		defer fs.fileLock.Unlock()

		data, err := ioutil.ReadFile(path.Join(fs.DataDir, infohash, subpath))

		//data final verification
		for _, file := range torrent.Files() {
			log.Debug("File path info", "path", file.Path(), "subpath", subpath)
			if file.Path() == subpath[1:] {
				if int64(len(data)) != file.Length() {
					log.Error("Read file not completed", "hash", infohash, "len", len(data), "total", file.Path())
					return nil, errors.New("not a complete file")
				} else {
					log.Debug("Read data success", "hash", infohash, "size", len(data), "path", file.Path())
					if c, err := fs.zip(data); err != nil {
						log.Warn("Compress data failed", "hash", infohash, "err", err)
					} else {
						if fs.cache {
							fs.fileCache.Add(key, c)
						}
					}
				}
				break
			}
		}

		return data, err
	}
}

func (fs *TorrentManager) release() {
	<-fs.fileCh
}

func (fs *TorrentManager) unzip(data []byte) ([]byte, error) {
	if fs.compress {
		return compress.UnzipData(data)
	} else {
		return data, nil
	}
}

func (fs *TorrentManager) zip(data []byte) ([]byte, error) {
	if fs.compress {
		return compress.ZipData(data)
	} else {
		return data, nil
	}
}

func (fs *TorrentManager) Metrics() time.Duration {
	return fs.Updates
}
