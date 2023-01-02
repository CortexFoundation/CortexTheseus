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

package backend

import (
	"bytes"
	"context"
	"crypto/sha1"
	"errors"
	"fmt"
	"io"
	//"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	//"strconv"
	"math"
	"runtime"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/metrics"
	"github.com/CortexFoundation/torrentfs/compress"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/types"
	"github.com/CortexFoundation/torrentfs/wormhole"

	//"github.com/allegro/bigcache/v3"
	"github.com/bradfitz/iter"
	"github.com/edsrzf/mmap-go"
	//lru "github.com/hashicorp/golang-lru"
	//"golang.org/x/time/rate"

	//xlog "github.com/anacrolix/log"
	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/bencode"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/mmap_span"
	pp "github.com/anacrolix/torrent/peer_protocol"
	"github.com/anacrolix/torrent/storage"
	"github.com/ucwong/golang-kv"
)

const (
	bucket          = params.Bucket //it is best size is 1/3 full nodes
	group           = params.Group
	taskChanBuffer  = params.SyncBatch
	torrentChanSize = 64

	block = int64(params.PER_UPLOAD_BYTES)
	loops = 30

	torrentTypeOnChain = 0
	torrentTypeLocal   = 1

	TORRENT = "torrent"
)

var (
	server         bool = false
	worm           bool = false
	getfileMeter        = metrics.NewRegisteredMeter("torrent/getfile/call", nil)
	availableMeter      = metrics.NewRegisteredMeter("torrent/available/call", nil)
	diskReadMeter       = metrics.NewRegisteredMeter("torrent/disk/read", nil)

	downloadMeter = metrics.NewRegisteredMeter("torrent/download/call", nil)
	updateMeter   = metrics.NewRegisteredMeter("torrent/update/call", nil)

	memcacheHitMeter  = metrics.NewRegisteredMeter("torrent/memcache/hit", nil)
	memcacheReadMeter = metrics.NewRegisteredMeter("torrent/memcache/read", nil)

	memcacheMissMeter  = metrics.NewRegisteredMeter("torrent/memcache/miss", nil)
	memcacheWriteMeter = metrics.NewRegisteredMeter("torrent/memcache/write", nil)
)

type TorrentManager struct {
	client *torrent.Client
	//bytes               map[metainfo.Hash]int64
	torrents        map[string]*Torrent
	seedingTorrents map[string]*Torrent
	activeTorrents  map[string]*Torrent
	pendingTorrents map[string]*Torrent
	//maxSeedTask         int
	maxEstablishedConns int
	trackers            [][]string
	globalTrackers      [][]string
	//boostFetcher        *BoostDataFetcher
	DataDir           string
	TmpDataDir        string
	closeAll          chan struct{}
	taskChan          chan any
	lock              sync.RWMutex
	wg                sync.WaitGroup
	seedingChan       chan *Torrent
	activeChan        chan *Torrent
	pendingChan       chan *Torrent
	pendingRemoveChan chan string
	droppingChan      chan string
	mode              string
	//boost               bool
	id   uint64
	slot int

	//fileLock sync.RWMutex
	//fileCache *bigcache.BigCache
	cache    bool
	compress bool

	metrics bool
	Updates time.Duration

	//hotCache *lru.Cache

	// For manage torrents Seeding by SeedingLocal(), true/false means seeding/pause
	localSeedLock  sync.RWMutex
	localSeedFiles map[string]bool

	//initCh   chan struct{}
	//simulate bool
	//good     uint64

	startOnce     sync.Once
	seedingNotify chan string

	badger kv.Bucket
}

// can only call by fs.go: 'SeedingLocal()'
func (tm *TorrentManager) AddLocalSeedFile(ih string) bool {
	if !common.IsHexAddress(ih) {
		return false
	}
	ih = strings.TrimPrefix(strings.ToLower(ih), common.Prefix)

	if _, ok := params.GoodFiles[ih]; ok {
		return false
	}

	tm.localSeedLock.Lock()
	tm.localSeedFiles[ih] = true
	tm.localSeedLock.Unlock()

	return true
}

// only files in map:localSeedFile can be paused!
func (tm *TorrentManager) PauseLocalSeedFile(ih string) error {
	if !common.IsHexAddress(ih) {
		return errors.New("invalid infohash format")
	}
	ih = strings.TrimPrefix(strings.ToLower(ih), common.Prefix)

	tm.localSeedLock.Lock()
	defer tm.localSeedLock.Unlock()

	if valid, ok := tm.localSeedFiles[ih]; !ok {
		return errors.New(fmt.Sprintf("Not Local Seeding File<%s>", ih))
	} else if _, ok := params.GoodFiles[ih]; ok {
		return errors.New(fmt.Sprintf("Cannot Pause On-Chain GoodFile<%s>", ih))
	} else if !valid {
		return errors.New(fmt.Sprintf("Local Seeding File Is Not Seeding<%s>", ih))
	}

	if t := tm.getTorrent(ih); t != nil {
		log.Debug("TorrentFS", "from seed to pause", "ok")
		t.Pause()
		tm.localSeedFiles[ih] = !t.Paused()
	}

	return nil
}

// only files in map:localSeedFile can be resumed!
func (tm *TorrentManager) ResumeLocalSeedFile(ih string) error {
	if !common.IsHexAddress(ih) {
		return errors.New("invalid infohash format")
	}
	ih = strings.TrimPrefix(strings.ToLower(ih), common.Prefix)

	tm.localSeedLock.Lock()
	defer tm.localSeedLock.Unlock()

	if valid, ok := tm.localSeedFiles[ih]; !ok {
		return errors.New(fmt.Sprintf("Not Local Seeding File<%s>", ih))
	} else if _, ok := params.GoodFiles[ih]; ok {
		return errors.New(fmt.Sprintf("Cannot Operate On-Chain GoodFile<%s>", ih))
	} else if valid {
		return errors.New(fmt.Sprintf("Local Seeding File Is Already Seeding<%s>", ih))
	}

	if t := tm.getTorrent(ih); t != nil {
		resumeFlag := t.Seed()
		log.Debug("TorrentFS", "from pause to seed", resumeFlag)
		tm.localSeedFiles[ih] = resumeFlag
	}

	return nil
}

// divide localSeed/on-chain Files
// return status of torrents
func (tm *TorrentManager) ListAllTorrents() map[string]map[string]int {
	tm.lock.RLock()
	tm.localSeedLock.RLock()
	defer tm.lock.RUnlock()
	defer tm.localSeedLock.RUnlock()

	tts := make(map[string]map[string]int)
	for ih, tt := range tm.torrents {
		tType := torrentTypeOnChain
		if _, ok := tm.localSeedFiles[ih]; ok {
			tType = torrentTypeLocal
		}
		tts[ih] = map[string]int{
			"status": tt.status,
			"type":   tType,
		}
	}

	return tts
}

func (tm *TorrentManager) getLimitation(value int64) int64 {
	return ((value + block - 1) / block) * block
}

func (tm *TorrentManager) blockCaculate(value int64) int64 {
	return ((value + block - 1) / block)
}

func (tm *TorrentManager) register(t *torrent.Torrent, requested int64, status int, ih string) *Torrent {
	tt := &Torrent{
		Torrent:             t,
		maxEstablishedConns: tm.maxEstablishedConns,
		minEstablishedConns: 1,
		currentConns:        tm.maxEstablishedConns,
		bytesRequested:      requested,
		bytesLimitation:     tm.getLimitation(requested),
		bytesCompleted:      0,
		//bytesMissing:        0,
		status:   status,
		infohash: ih,
		filepath: filepath.Join(tm.TmpDataDir, ih),
		cited:    0,
		//weight:     1,
		//loop:       0,
		maxPieces: 0,
		//isBoosting: false,
		fast:  false,
		start: 0,
	}

	tm.setTorrent(ih, tt)

	tm.pendingChan <- tt
	return tt
}

func (tm *TorrentManager) getTorrent(ih string) *Torrent {
	tm.lock.RLock()
	defer tm.lock.RUnlock()
	if torrent, ok := tm.torrents[ih]; ok {
		return torrent
	}
	return nil
}

func (tm *TorrentManager) setTorrent(ih string, t *Torrent) {
	tm.lock.Lock()
	defer tm.lock.Unlock()

	tm.torrents[ih] = t
}

func (tm *TorrentManager) removeTorrent(ih string) {
	tm.lock.Lock()
	defer tm.lock.Unlock()

	delete(tm.torrents, ih)
}

func (tm *TorrentManager) Close() error {
	tm.client.Close()
	tm.client.WaitAll()
	close(tm.closeAll)
	tm.wg.Wait()
	//if tm.fileCache != nil {
	//	tm.fileCache.Reset()
	//}

	if tm.badger != nil {
		tm.badger.Close()
	}
	//if tm.hotCache != nil {
	//	tm.hotCache.Purge()
	//}
	log.Info("Fs Download Manager Closed")
	return nil
}

/*func (tm *TorrentManager) buildUdpTrackers(trackers []string) (array [][]string) {
	array = make([][]string, 1)
	for _, tracker := range trackers {
		array[0] = append(array[0], "udp"+tracker)
	}
	return array
}*/

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
	span.InitIndex()
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

// func (tm *TorrentManager) loadSpec(ih metainfo.Hash, filePath string, BytesRequested int64) *torrent.TorrentSpec {
func (tm *TorrentManager) loadSpec(ih string, filePath string) *torrent.TorrentSpec {
	if _, err := os.Stat(filePath); err != nil {
		return nil
	}
	mi, err := metainfo.LoadFromFile(filePath)
	if err != nil {
		log.Error("Error while adding torrent", "Err", err)
		return nil
	}

	spec := torrent.TorrentSpecFromMetaInfo(mi)

	if ih != spec.InfoHash.HexString() {
		log.Warn("Info hash mismatch", "ih", ih, "new", spec.InfoHash)
		return nil
	}

	TmpDir := filepath.Join(tm.TmpDataDir, ih)
	ExistDir := filepath.Join(tm.DataDir, ih)

	var useExistDir bool
	if _, err := os.Stat(ExistDir); err == nil {
		log.Debug("Seeding from existing file.", "ih", ih)
		info, err := mi.UnmarshalInfo()
		if err != nil {
			log.Error("error unmarshalling info: ", "info", err)
			return nil
		}

		if err := tm.verifyTorrent(&info, ExistDir); err == nil {
			useExistDir = true
		}
	}

	if useExistDir {
		spec.Storage = storage.NewFile(ExistDir)
	} else {
		spec.Storage = storage.NewFile(TmpDir)
	}
	spec.Trackers = nil

	return spec
}

func (tm *TorrentManager) addInfoHash(ih string, bytesRequested int64) *Torrent {
	if t := tm.getTorrent(ih); t != nil {
		tm.updateInfoHash(t, bytesRequested)
		return t
	}

	if bytesRequested < 0 {
		return nil
	}

	if !server && worm {
		tm.wg.Add(1)
		go func() {
			defer tm.wg.Done()
			err := wormhole.Tunnel(ih)
			if err != nil {
				log.Error("Wormhole error", "err", err)
			}
		}()
	}

	var (
		spec *torrent.TorrentSpec
		v    []byte
	)

	if tm.badger != nil {
		if v = tm.badger.Get([]byte(ih)); v == nil {
			seedTorrentPath := filepath.Join(tm.DataDir, ih, TORRENT)
			if _, err := os.Stat(seedTorrentPath); err == nil {
				spec = tm.loadSpec(ih, seedTorrentPath)
			}

			if spec == nil {
				tmpTorrentPath := filepath.Join(tm.TmpDataDir, ih, TORRENT)
				if _, err := os.Stat(tmpTorrentPath); err == nil {
					spec = tm.loadSpec(ih, tmpTorrentPath)
				}
			}
		}
	}

	if spec == nil {
		tmpDataPath := filepath.Join(tm.TmpDataDir, ih)

		if _, err := os.Stat(tmpDataPath); err != nil {
			if err := os.MkdirAll(tmpDataPath, 0777); err != nil {
				log.Warn("torrent path create failed", "err", err)
				return nil
			}
		}

		spec = &torrent.TorrentSpec{
			InfoHash:  metainfo.NewHashFromHex(ih),
			Storage:   storage.NewFile(tmpDataPath),
			InfoBytes: v,
		}
	}

	if t, n, err := tm.client.AddTorrentSpec(spec); err == nil {
		if !n {
			log.Warn("Try to add a dupliated torrent", "ih", ih)
		}

		t.AddTrackers(tm.trackers)

		if tm.globalTrackers != nil {
			t.AddTrackers(tm.globalTrackers)
		}

		if t.Info() == nil && tm.badger != nil {
			if v := tm.badger.Get([]byte(ih)); v != nil {
				t.SetInfoBytes(v)
			}
		}

		return tm.register(t, bytesRequested, torrentPending, ih)
	}

	return nil
}

func (tm *TorrentManager) updateGlobalTrackers() {
	tm.lock.Lock()
	defer tm.lock.Unlock()
	if global := wormhole.BestTrackers(); len(global) > 0 {
		tm.globalTrackers = [][]string{global}
		log.Info("Global trackers update", "size", len(global), "cap", wormhole.CAP)
	}
}

func (tm *TorrentManager) GlobalTrackers() [][]string {
	tm.lock.RLock()
	defer tm.lock.RUnlock()

	return tm.globalTrackers
}

func (tm *TorrentManager) updateInfoHash(t *Torrent, bytesRequested int64) {
	t.lock.Lock()
	defer t.lock.Unlock()
	if t.bytesRequested < bytesRequested {
		t.bytesRequested = bytesRequested
		t.bytesLimitation = tm.getLimitation(bytesRequested)
	} else {
		atomic.AddInt64(&t.cited, 1)
	}
	updateMeter.Mark(1)
}

func NewTorrentManager(config *params.Config, fsid uint64, cache, compress bool, notify chan string) (*TorrentManager, error) {
	server = config.Server
	worm = config.Wormhole

	cfg := torrent.NewDefaultClientConfig()
	cfg.DisableUTP = config.DisableUTP
	cfg.NoDHT = config.DisableDHT
	cfg.DisableTCP = config.DisableTCP
	cfg.DisableIPv6 = config.DisableIPv6

	cfg.MinPeerExtensions.SetBit(pp.ExtensionBitFast, true)
	//cfg.DisableWebtorrent = false
	//cfg.DisablePEX = false
	//cfg.DisableWebseeds = false
	cfg.DisableIPv4 = false
	cfg.DisableAcceptRateLimiting = true
	//cfg.DisableWebtorrent = false
	//cfg.HeaderObfuscationPolicy.Preferred = true
	//cfg.HeaderObfuscationPolicy.RequirePreferred = true

	cfg.DataDir = config.DataDir
	//cfg.DisableEncryption = true
	//cfg.HTTPUserAgent = "Cortex"
	cfg.Seed = true
	//cfg.Debug=true

	cfg.EstablishedConnsPerTorrent = int(math.Min(float64(runtime.NumCPU()*2), float64(50))) //4 //len(config.DefaultTrackers)
	cfg.HalfOpenConnsPerTorrent = cfg.EstablishedConnsPerTorrent / 2

	cfg.ListenPort = config.Port
	if config.Quiet {
		//cfg.Logger = xlog.Discard
	}
	//cfg.Debug = true
	cfg.DropDuplicatePeerIds = true
	cfg.Bep20 = params.ClientVersion //"-COLA01-"
	//id := strconv.FormatUint(fsid, 16)[0:14]
	//cfg.PeerID = "cortex" + id
	//cfg.ListenHost = torrent.LoopbackListenHost
	//cfg.DhtStartingNodes = dht.GlobalBootstrapAddrs //func() ([]dht.Addr, error) { return nil, nil }
	cl, err := torrent.NewClient(cfg)
	if err != nil {
		log.Error("Error while create torrent client", "err", err)
		return nil, err
	}

	log.Info("Listening local", "port", cl.LocalPort())

	tmpFilePath := filepath.Join(config.DataDir, params.DefaultTmpPath)

	if _, err := os.Stat(tmpFilePath); err != nil {
		err = os.MkdirAll(filepath.Dir(tmpFilePath), 0777) //os.FileMode(os.ModePerm))
		if err != nil {
			log.Error("Mkdir failed", "path", tmpFilePath)
			return nil, err
		}
	}

	torrentManager := &TorrentManager{
		client:          cl,
		torrents:        make(map[string]*Torrent),
		pendingTorrents: make(map[string]*Torrent),
		seedingTorrents: make(map[string]*Torrent),
		activeTorrents:  make(map[string]*Torrent),
		//bytes:               make(map[metainfo.Hash]int64),
		//maxSeedTask:         config.MaxSeedingNum,
		maxEstablishedConns: cfg.EstablishedConnsPerTorrent,
		DataDir:             config.DataDir,
		TmpDataDir:          tmpFilePath,
		//boostFetcher:        NewBoostDataFetcher(config.BoostNodes),
		closeAll: make(chan struct{}),
		//initCh:              make(chan struct{}),
		//simulate:          false,
		taskChan:          make(chan any, taskChanBuffer),
		seedingChan:       make(chan *Torrent, torrentChanSize),
		activeChan:        make(chan *Torrent, torrentChanSize),
		pendingChan:       make(chan *Torrent, torrentChanSize),
		pendingRemoveChan: make(chan string, torrentChanSize),
		droppingChan:      make(chan string, 1),
		mode:              config.Mode,
		//boost:             config.Boost,
		id:             fsid,
		slot:           int(fsid % bucket),
		localSeedFiles: make(map[string]bool),
		seedingNotify:  notify,
		badger:         kv.Badger(config.DataDir),
	}

	if cache {
		/*conf := bigcache.Config{
			Shards:             1024,
			LifeWindow:         600 * time.Second,
			CleanWindow:        1 * time.Second,
			MaxEntriesInWindow: 1000 * 10 * 60,
			MaxEntrySize:       512,
			StatsEnabled:       true,
			Verbose:            true,
			HardMaxCacheSize:   512, //MB
		}*/

		//	torrentManager.fileCache, err = bigcache.NewBigCache(conf)
		//	if err != nil {
		//		log.Error("File system cache initialized failed", "err", err)
		//	} else {
		//		torrentManager.cache = cache
		//		torrentManager.compress = compress
		//	}
	}

	torrentManager.metrics = config.Metrics

	//hotSize := config.MaxSeedingNum/64 + 1
	//torrentManager.hotCache, _ = lru.New(hotSize)
	//log.Info("Hot cache created", "size", hotSize)

	if len(config.DefaultTrackers) > 0 {
		log.Debug("Tracker list", "trackers", config.DefaultTrackers)
		torrentManager.trackers = [][]string{config.DefaultTrackers}
	}

	torrentManager.updateGlobalTrackers()
	//if global, err := wormhole.BestTrackers(); global != nil && err == nil {
	//	torrentManager.globalTrackers = [][]string{global}
	//}

	log.Debug("Fs client initialized", "config", config, "trackers", torrentManager.trackers)

	return torrentManager, nil
}

func (tm *TorrentManager) Start() (err error) {
	tm.startOnce.Do(func() {
		tm.wg.Add(1)
		go tm.droppingLoop()
		tm.wg.Add(1)
		go tm.seedingLoop()
		tm.wg.Add(1)
		go tm.activeLoop()
		tm.wg.Add(1)
		go tm.pendingLoop()

		tm.wg.Add(1)
		go tm.mainLoop()

		//err = tm.init()
	})

	return
}

func (tm *TorrentManager) prepare() bool {
	return true
}

func (tm *TorrentManager) init() error {
	log.Debug("Chain files init", "files", len(params.GoodFiles))

	//if tm.mode == params.DEV || tm.mode == params.LAZY {
	//	tm.Simulate()
	//}

	//if !tm.simulate {
	/*for k, ok := range GoodFiles {
		if ok {
			if err := tm.Search(context.Background(), k, 0); err == nil {
				tm.good++
				//atomic.AddUint64(&tm.good, 1)
			} else {
				log.Info("Fs init failed", "err", err)
				return err
			}
		}
	}*/
	/*select {
	case <-tm.initCh:
		log.Info("Chain files sync init OK !!!", "seeding", len(tm.seedingTorrents), "pending", len(tm.pendingTorrents), "active", len(tm.activeTorrents), "good", len(GoodFiles), "active", tm.good)
	case <-tm.closeAll:
		log.Info("Init files closed")
	}*/
	//}

	log.Debug("Chain files OK !!!")
	return nil
}

/*func (tm *TorrentManager) Simulate() {
	tm.lock.Lock()
	defer tm.lock.Unlock()

	if !tm.simulate {
		tm.simulate = true
	}
	log.Info("Do simulate")
}*/

// Search and donwload files from torrent
func (tm *TorrentManager) Search(ctx context.Context, hex string, request uint64) error {
	if !common.IsHexAddress(hex) {
		return errors.New("invalid infohash format")
	}

	hex = strings.TrimPrefix(strings.ToLower(hex), common.Prefix)

	if params.IsBad(hex) {
		return nil
	}

	if request == 0x7fffffffffffffff {
		// TODO 0x7fffffffffffffff local downloading file
		// GoodFiles[hex] = false
	}

	//if tm.mode == params.FULL {
	//if request == 0 {
	//	log.Warn("Prepare mode", "ih", hex)
	//	request = uint64(block)
	//}
	//}

	downloadMeter.Mark(1)

	return tm.commit(ctx, hex, request)
}

func (tm *TorrentManager) commit(ctx context.Context, hex string, request uint64) error {
	//log.Debug("Commit task", "ih", hex, "request", request, "ch", ch)

	/*if !server {
	        tm.wg.Add(1)
	        go func() {
	                defer tm.wg.Done()
	                err := wormhole.Tunnel(hex)
	                if err != nil {
	                        log.Error("Wormhole error", "err", err)
	                }
	        }()
	}*/

	task := types.NewBitsFlow(hex, request) /*types.BitsFlow{
		infohash: hex,
		request:  request,
	}*/
	select {
	case tm.taskChan <- task:
	case <-ctx.Done():
		return ctx.Err()
	}

	return nil
}

func (tm *TorrentManager) mainLoop() {
	defer tm.wg.Done()
	var bytes int64
	for {
		select {
		case msg := <-tm.taskChan:
			meta := msg.(*types.BitsFlow)
			if params.IsBad(meta.InfoHash()) {
				continue
			}

			bytes = int64(meta.Request())
			if bytes == 0 {
				bytes = block
			}

			if t := tm.addInfoHash(meta.InfoHash(), bytes); t == nil {
				log.Error("Seed [create] failed", "ih", meta.InfoHash(), "request", bytes)
			}
		case <-tm.closeAll:
			return
		}
	}
}

func (tm *TorrentManager) pendingLoop() {
	defer tm.wg.Done()
	//timer := time.NewTimer(time.Second * 60)
	//defer timer.Stop()
	//ctx, cancel := context.WithCancel(context.Background())
	//defer cancel()
	for {
		select {
		case t := <-tm.pendingChan:
			tm.pendingTorrents[t.infohash] = t
			tm.wg.Add(1)
			go func() {
				defer tm.wg.Done()
				if t.start == 0 {
					t.start = mclock.Now()
				}
				ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
				defer cancel()
				select {
				case <-t.GotInfo():
					elapsed := time.Duration(mclock.Now()) - time.Duration(t.start)
					log.Debug("Imported new seed", "ih", t.infohash, "elapsed", common.PrettyDuration(elapsed))
					if b, err := bencode.Marshal(t.Torrent.Info()); err == nil {
						log.Debug("Record full torrent in history", "ih", t.infohash, "info", len(b))
						if tm.badger != nil && tm.badger.Get([]byte(t.infohash)) == nil {
							tm.badger.Set([]byte(t.infohash), b)
						}
					} else {
						log.Error("Meta info marshal failed", "ih", t.infohash, "err", err)
						tm.Drop(t.infohash)
						return
					}

					if err := t.WriteTorrent(); err == nil {
						if params.IsGood(t.infohash) || tm.mode == params.FULL {
							t.lock.Lock()
							t.bytesRequested = t.Length()
							t.bytesLimitation = tm.getLimitation(t.Length())
							t.lock.Unlock()
						}
						tm.activeChan <- t
						tm.pendingRemoveChan <- t.infohash
					} else {
						log.Error("Write torrent info to file failed", "ih", t.infohash, "err", err)
						tm.Drop(t.infohash)
					}
				case <-t.Closed():
				case <-tm.closeAll:
				case <-ctx.Done():
					tm.Drop(t.infohash)
				}
			}()
		case i := <-tm.pendingRemoveChan:
			delete(tm.pendingTorrents, i)
		//case <-timer.C:
		//	for ih, t := range tm.pendingTorrents {
		//		elapsed := time.Duration(mclock.Now()) - time.Duration(t.start)
		//		log.Info("Pending seed", "ih", ih, "elapsed", common.PrettyDuration(elapsed))
		//	}
		//	timer.Reset(time.Second * 60)
		case <-tm.closeAll:
			log.Info("Pending seed loop closed")
			return
		}
	}
}

func (tm *TorrentManager) finish(ih string, t *Torrent) {
	//t.lock.Lock()
	//defer t.lock.Unlock()
	if _, err := os.Stat(filepath.Join(tm.DataDir, ih)); err == nil {
		tm.seedingChan <- t
		delete(tm.activeTorrents, ih)
	} else {
		if err := os.Symlink(
			filepath.Join(params.DefaultTmpPath, ih),
			filepath.Join(tm.DataDir, ih),
		); err == nil {
			tm.seedingChan <- t
			delete(tm.activeTorrents, ih)
		}
	}
}

func (tm *TorrentManager) activeLoop() {
	defer tm.wg.Done()
	timer := time.NewTicker(time.Second * params.QueryTimeInterval)
	defer timer.Stop()
	var total_size, current_size, log_counter, counter uint64 = 0, 0, 1, 1
	for {
		select {
		case t := <-tm.activeChan:
			t.status = torrentRunning
			tm.activeTorrents[t.infohash] = t
			n := tm.blockCaculate(t.Torrent.BytesCompleted() + t.Torrent.BytesMissing())
			if n < 10 {
				n = 10
			}
			tm.wg.Add(1)
			go func(i string, n int64) {
				defer tm.wg.Done()
				timer := time.NewTicker(time.Duration(n) * time.Minute)
				defer timer.Stop()
				for {
					select {
					case <-timer.C:
						if t := tm.getTorrent(i); t != nil { //&& t.Ready() {
							if t.cited <= 0 {
								tm.Drop(i)
								return
							} else {
								atomic.AddInt64(&t.cited, -1)
								log.Info("Seed cited has been decreased", "ih", i, "cited", t.cited, "n", n, "status", t.status)
							}
						} else {
							log.Warn("Seed broken", "ih", i, "cited", t.cited, "n", n, "status", t.status)
							return
						}
					case <-tm.closeAll:
						return
					}
				}
			}(t.infohash, n)
		case <-timer.C:
			counter++
			log_counter++

			for ih, t := range tm.activeTorrents {
				if t.BytesCompleted() > t.bytesCompleted {
					total_size += uint64(t.BytesCompleted() - t.bytesCompleted)
					current_size += uint64(t.BytesCompleted() - t.bytesCompleted)

					t.bytesCompleted = t.BytesCompleted()
				}

				if t.BytesMissing() == 0 {
					tm.finish(ih, t)
					continue
				}

				if log_counter%60 == 0 {
					elapsed := time.Duration(mclock.Now()) - time.Duration(t.start)
					log.Debug(ProgressBar(t.bytesCompleted, t.Torrent.Length(), ""), "ih", ih, "complete", common.StorageSize(t.bytesCompleted), "limit", common.StorageSize(t.bytesLimitation), "total", common.StorageSize(t.Torrent.Length()), "want", t.maxPieces, "max", t.Torrent.NumPieces(), "speed", common.StorageSize(float64(t.bytesCompleted*1000*1000*1000)/float64(elapsed)).String()+"/s", "elapsed", common.PrettyDuration(elapsed))
				}

				if t.bytesCompleted < t.bytesLimitation { //&& !t.isBoosting {
					t.Run(tm.slot)
				}
			}

			if counter >= 2*loops {
				log.Info("Fs status", "pending", len(tm.pendingTorrents), "downloading", len(tm.activeTorrents), "seeding", len(tm.seedingTorrents), "size", common.StorageSize(total_size), "speed_a", common.StorageSize(total_size/log_counter*params.QueryTimeInterval).String()+"/s", "speed_b", common.StorageSize(current_size/counter*params.QueryTimeInterval).String()+"/s", "metrics", common.PrettyDuration(tm.Updates))
				counter = 1
				current_size = 0
			}

			if log_counter%(3600*24) == 0 {
				tm.wg.Add(1)
				go func() {
					defer tm.wg.Done()
					tm.updateGlobalTrackers()
				}()
			}

			//timer.Reset(time.Second * queryTimeInterval)
		case <-tm.closeAll:
			log.Info("Active seed loop closed")
			return
		}
	}
}

func (tm *TorrentManager) seedingLoop() {
	defer tm.wg.Done()
	for {
		select {
		case t := <-tm.seedingChan:
			tm.seedingTorrents[t.infohash] = t

			s := t.Seed()

			if s {
				//tm.hotCache.Add(t.infohash, true)
				//if len(tm.seedingTorrents) > params.LimitSeeding {
				//tm.dropSeeding(tm.slot)
				//} else if len(tm.seedingTorrents) > tm.maxSeedTask {
				//	tm.maxSeedTask++
				//tm.graceSeeding(tm.slot)
				//}

				if tm.seedingNotify != nil {
					tm.wg.Add(1)
					go func() {
						defer tm.wg.Done()
						tm.seedingNotify <- t.InfoHash()
					}()
				}
			}
		case <-tm.closeAll:
			log.Info("Seeding loop closed")
			return
		}
	}
}

func (tm *TorrentManager) droppingLoop() {
	defer tm.wg.Done()
	for {
		select {
		case ih := <-tm.droppingChan:
			if t := tm.getTorrent(ih); t != nil { //&& t.Ready() {
				t.Torrent.Drop()
				if t.status == torrentPending {
					delete(tm.pendingTorrents, ih)
				}
				if t.status == torrentRunning || t.status == torrentPaused {
					delete(tm.activeTorrents, ih)
				}

				if t.status == torrentSeeding {
					delete(tm.seedingTorrents, ih)
				}

				tm.removeTorrent(ih)

				log.Info("Seed has been dropped", "ih", ih, "cited", t.cited, "status", t.status)
			} else {
				log.Warn("Drop seed not found", "ih", ih)
			}
		case <-tm.closeAll:
			log.Info("Dropping loop closed")
			return
		}
	}
}

func (tm *TorrentManager) Drop(ih string) error {
	tm.droppingChan <- ih
	return nil
}

/*
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
					log.Debug("Encounter active torrent", "ih", ih, "index", i, "group", s, "slot", slot, "len", len(tm.seedingTorrents), "max", tm.maxSeedTask, "peers", t.currentConns)
					continue
				}

				if tm.mode == params.LAZY {
					t.setCurrentConns(1)
					log.Debug("Lazy mode dropped", "ih", ih, "seeding", len(tm.seedingTorrents), "torrents", len(tm.torrents), "max", tm.maxSeedTask, "peers", t.currentConns)
				} else {
					t.setCurrentConns(2)
				}
				t.Torrent.SetMaxEstablishedConns(t.currentConns)
				log.Debug("Drop seeding invoke", "ih", ih, "index", i, "group", s, "slot", slot, "len", len(tm.seedingTorrents), "max", tm.maxSeedTask, "peers", t.currentConns)
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
					log.Debug("Encounter active torrent", "ih", ih, "index", i, "group", s, "slot", slot, "len", len(tm.seedingTorrents), "max", tm.maxSeedTask, "peers", t.currentConns)
					continue
				}
				if tm.mode == params.LAZY {
					t.setCurrentConns(1)
				} else {
					t.setCurrentConns(t.minEstablishedConns)
				}
				t.Torrent.SetMaxEstablishedConns(t.currentConns)
				log.Debug("Grace seeding invoke", "ih", ih, "index", i, "group", s, "slot", slot, "len", len(tm.seedingTorrents), "max", tm.maxSeedTask, "peers", t.currentConns)
			}
			i++
		}
		return nil
	}
*/
func (tm *TorrentManager) Available(ih string, rawSize uint64) (bool, uint64, mclock.AbsTime, error) {
	availableMeter.Mark(1)
	if rawSize <= 0 {
		return false, 0, 0, errors.New("raw size is zero or negative")
	}

	if !common.IsHexAddress(ih) {
		return false, 0, 0, errors.New("invalid infohash format")
	}

	ih = strings.TrimPrefix(strings.ToLower(ih), common.Prefix)

	if t := tm.getTorrent(ih); t == nil {
		return false, 0, 0, ErrInactiveTorrent
	} else {
		if !t.Ready() {
			//if torrent.ch != nil {
			//	<-torrent.ch
			//	if torrent.Ready() {
			//		return torrent.BytesCompleted() <= int64(rawSize), nil
			//	}
			//}
			if t.Torrent.Info() == nil {
				return false, uint64(t.BytesCompleted()), 0, ErrTorrentNotFound
			}
			return false, uint64(t.BytesCompleted()), mclock.Now() - t.start, ErrUnfinished
		}

		// TODO
		ok := t.BytesCompleted() <= int64(rawSize)
		//if t.currentConns <= 1 && ok {
		//	t.currentConns = tm.maxEstablishedConns
		//	t.Torrent.SetMaxEstablishedConns(tm.maxEstablishedConns)
		//}

		return ok, uint64(t.BytesCompleted()), mclock.Now() - t.start, nil
	}
}

func (tm *TorrentManager) GetFile(infohash, subpath string) ([]byte, uint64, error) {
	getfileMeter.Mark(1)
	if tm.metrics {
		defer func(start time.Time) { tm.Updates += time.Since(start) }(time.Now())
	}

	if !common.IsHexAddress(infohash) {
		return nil, 0, errors.New("invalid infohash format")
	}
	infohash = strings.TrimPrefix(strings.ToLower(infohash), common.Prefix)

	//if t := tm.getTorrent(infohash); t == nil {
	//	return nil, 0, ErrInactiveTorrent
	//} else {

	subpath = strings.TrimPrefix(subpath, "/")
	subpath = strings.TrimSuffix(subpath, "/")

	//	if !t.Ready() {
	//		log.Error("Read unavailable file", "hash", infohash, "subpath", subpath)
	//		return nil, uint64(t.BytesCompleted()), ErrUnfinished
	//	}

	//tm.hotCache.Add(infohash, true)
	//if t.currentConns < tm.maxEstablishedConns {
	//t.setCurrentConns(tm.maxEstablishedConns)
	//t.Torrent.SetMaxEstablishedConns(t.currentConns)
	//	log.Debug("Torrent active", "ih", infohash, "peers", t.currentConns)
	//}

	var key = filepath.Join(infohash, subpath)
	/*if tm.fileCache != nil {
		if cache, err := tm.fileCache.Get(key); err == nil {
			memcacheHitMeter.Mark(1)
			memcacheReadMeter.Mark(int64(len(cache)))
			if c, err := tm.unzip(cache); err != nil {
				return nil, 0, err
			} else {
				if tm.compress {
					log.Info("File cache", "hash", infohash, "path", subpath, "size", tm.fileCache.Len(), "compress", len(cache), "origin", len(c), "compress", tm.compress)
				}
				//return c, uint64(t.BytesCompleted()), nil
				return c, 0, nil
			}
		}
	}*/

	//if t := tm.getTorrent(infohash); t != nil {
	//	t.lock.RLock()
	//	defer t.lock.RUnlock()
	//}
	diskReadMeter.Mark(1)

	log.Debug("Get File", "dir", tm.DataDir, "key", key)

	data, err := os.ReadFile(filepath.Join(tm.DataDir, key))

	//data final verification
	/*for _, file := range t.Files() {
		if file.Path() == subpath {
			log.Debug("File location info", "ih", infohash, "path", file.Path(), "key", key)
			if int64(len(data)) != file.Length() {
				log.Error("Read file not completed", "hash", infohash, "len", len(data), "total", file.Path())
				return nil, 0, errors.New("not a complete file")
			} else {
				log.Debug("Read data success", "hash", infohash, "size", len(data), "path", file.Path())
				if c, err := tm.zip(data); err != nil {
					log.Warn("Compress data failed", "hash", infohash, "err", err)
				} else {
					if tm.fileCache != nil {
						tm.fileCache.Set(key, c)
						memcacheMissMeter.Mark(1)
						memcacheWriteMeter.Mark(int64(len(c)))
					}
				}
			}
			break
		}
	}*/

	//return data, uint64(t.BytesCompleted()), err
	return data, 0, err
	//}
}

func (tm *TorrentManager) unzip(data []byte) ([]byte, error) {
	if tm.compress {
		return compress.UnzipData(data)
	}
	return data, nil
}

func (tm *TorrentManager) zip(data []byte) ([]byte, error) {
	if tm.compress {
		return compress.ZipData(data)
	}
	return data, nil
}

func (tm *TorrentManager) Metrics() time.Duration {
	return tm.Updates
}

func (tm *TorrentManager) LocalPort() int {
	return tm.client.LocalPort()
}

func (tm *TorrentManager) Congress() int {
	return len(tm.seedingTorrents)
}

func (tm *TorrentManager) FullSeed() map[string]*Torrent {
	return tm.seedingTorrents
}

func (tm *TorrentManager) Candidate() int {
	return len(tm.activeTorrents)
}

func (tm *TorrentManager) Nominee() int {
	return len(tm.pendingTorrents)
}

func (tm *TorrentManager) IsPending(ih string) bool {
	return tm.pendingTorrents[ih] != nil
}

func (tm *TorrentManager) IsDownloading(ih string) bool {
	return tm.activeTorrents[ih] != nil
}

func (tm *TorrentManager) IsSeeding(ih string) bool {
	return tm.seedingTorrents[ih] != nil
}
