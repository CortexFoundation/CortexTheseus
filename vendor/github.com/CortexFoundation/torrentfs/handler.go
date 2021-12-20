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
	"context"
	"crypto/sha1"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/metrics"
	"github.com/CortexFoundation/torrentfs/compress"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/types"

	"github.com/allegro/bigcache/v3"
	"github.com/bradfitz/iter"
	"github.com/edsrzf/mmap-go"
	lru "github.com/hashicorp/golang-lru"
	"golang.org/x/time/rate"

	xlog "github.com/anacrolix/log"
	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/mmap_span"
	"github.com/anacrolix/torrent/storage"
)

const (
	bucket          = params.Bucket //it is best size is 1/3 full nodes
	group           = params.Group
	taskChanBuffer  = params.SyncBatch
	torrentChanSize = 64

	torrentPending = iota //2
	torrentPaused
	torrentRunning
	torrentSeeding

	block = int64(params.PER_UPLOAD_BYTES)
	loops = 30

	torrentTypeOnChain = 0
	torrentTypeLocal   = 1
)

var (
	getfileMeter   = metrics.NewRegisteredMeter("torrent/getfile/call", nil)
	availableMeter = metrics.NewRegisteredMeter("torrent/available/call", nil)
	diskReadMeter  = metrics.NewRegisteredMeter("torrent/disk/read", nil)

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
	torrents            map[string]*Torrent
	seedingTorrents     map[string]*Torrent
	activeTorrents      map[string]*Torrent
	pendingTorrents     map[string]*Torrent
	maxSeedTask         int
	maxEstablishedConns int
	trackers            [][]string
	boostFetcher        *BoostDataFetcher
	DataDir             string
	TmpDataDir          string
	closeAll            chan struct{}
	taskChan            chan interface{}
	lock                sync.RWMutex
	wg                  sync.WaitGroup
	seedingChan         chan *Torrent
	activeChan          chan *Torrent
	pendingChan         chan *Torrent
	mode                string
	boost               bool
	id                  uint64
	slot                int

	fileLock  sync.Mutex
	fileCache *bigcache.BigCache
	cache     bool
	compress  bool

	metrics bool
	Updates time.Duration

	hotCache *lru.Cache

	// For manage torrents Seeding by SeedingLocal(), true/false means seeding/pause
	localSeedLock  sync.RWMutex
	localSeedFiles map[string]bool

	initCh   chan struct{}
	simulate bool
	good     int

	startOnce sync.Once
}

// can only call by fs.go: 'SeedingLocal()'
func (tm *TorrentManager) addLocalSeedFile(ih string) bool {
	if !common.IsHexAddress(ih) {
		return false
	}
	ih = strings.TrimPrefix(strings.ToLower(ih), common.Prefix)

	if _, ok := GoodFiles[ih]; ok {
		return false
	}

	tm.localSeedLock.Lock()
	tm.localSeedFiles[ih] = true
	tm.localSeedLock.Unlock()

	return true
}

// only files in map:localSeedFile can be paused!
func (tm *TorrentManager) pauseLocalSeedFile(ih string) error {
	if !common.IsHexAddress(ih) {
		return errors.New("invalid infohash format")
	}
	ih = strings.TrimPrefix(strings.ToLower(ih), common.Prefix)

	tm.localSeedLock.Lock()
	defer tm.localSeedLock.Unlock()
	if valid, ok := tm.localSeedFiles[ih]; !ok {
		return errors.New(fmt.Sprintf("Not Local Seeding File<%s>", ih))
	} else if _, ok := GoodFiles[ih]; ok {
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
func (tm *TorrentManager) resumeLocalSeedFile(ih string) error {
	if !common.IsHexAddress(ih) {
		return errors.New("invalid infohash format")
	}
	ih = strings.TrimPrefix(strings.ToLower(ih), common.Prefix)

	tm.localSeedLock.Lock()
	defer tm.localSeedLock.Unlock()

	if valid, ok := tm.localSeedFiles[ih]; !ok {
		return errors.New(fmt.Sprintf("Not Local Seeding File<%s>", ih))
	} else if _, ok := GoodFiles[ih]; ok {
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
func (tm *TorrentManager) listAllTorrents() map[string]map[string]int {
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

func (tm *TorrentManager) register(t *torrent.Torrent, requested int64, status int, ih string, ch chan bool) *Torrent {
	tt := &Torrent{
		Torrent:             t,
		maxEstablishedConns: tm.maxEstablishedConns,
		minEstablishedConns: 5,
		currentConns:        tm.maxEstablishedConns,
		bytesRequested:      requested,
		bytesLimitation:     tm.getLimitation(requested),
		bytesCompleted:      0,
		bytesMissing:        0,
		status:              status,
		infohash:            ih,
		filepath:            filepath.Join(tm.TmpDataDir, ih),
		cited:               0,
		weight:              1,
		loop:                0,
		maxPieces:           0,
		isBoosting:          false,
		fast:                true,
		start:               0,
		ch:                  ch,
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
	tm.torrents[ih] = t
	tm.lock.Unlock()
}

func (tm *TorrentManager) Close() error {
	close(tm.closeAll)
	tm.wg.Wait()
	tm.dropAll()
	if tm.fileCache != nil {
		tm.fileCache.Reset()
	}

	tm.hotCache.Purge()
	log.Info("Fs Download Manager Closed")
	return nil
}

func (tm *TorrentManager) dropAll() {
	tm.lock.Lock()
	defer tm.lock.Unlock()
	for _, t := range tm.torrents {
		if t.ch != nil {
			close(t.ch)
		}
	}

	tm.client.Close()
}

func (tm *TorrentManager) commit(ctx context.Context, hex string, request uint64, ch chan bool) error {
	log.Debug("Commit task", "ih", hex, "request", request, "ch", ch)
	task := types.FlowControlMeta{
		InfoHash:       hex,
		BytesRequested: request,
		Ch:             ch,
	}
	select {
	case tm.taskChan <- task:
	case <-ctx.Done():
		return ctx.Err()
	}

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

//func (tm *TorrentManager) loadSpec(ih metainfo.Hash, filePath string, BytesRequested int64) *torrent.TorrentSpec {
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

	useExistDir := false
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

func (tm *TorrentManager) addInfoHash(ih string, BytesRequested int64, ch chan bool) *Torrent {
	log.Debug("Add seed", "ih", ih, "bytes", BytesRequested, "ch", ch)
	if t := tm.getTorrent(ih); t != nil {
		//update
		tm.updateInfoHash(t, BytesRequested)
		return t
	}

	if BytesRequested < 0 {
		return nil
	}

	tmpTorrentPath := filepath.Join(tm.TmpDataDir, ih, "torrent")
	seedTorrentPath := filepath.Join(tm.DataDir, ih, "torrent")

	var spec *torrent.TorrentSpec

	if _, err := os.Stat(seedTorrentPath); err == nil {
		spec = tm.loadSpec(ih, seedTorrentPath)
	} else if _, err := os.Stat(tmpTorrentPath); err == nil {
		spec = tm.loadSpec(ih, tmpTorrentPath)
	}

	/*if _, err := os.Stat(filepath.Join(tm.TmpDataDir, ih, ".torrent.db")); err != nil {
		os.Symlink(
			filepath.Join(tm.DataDir, ".torrent.db"),
			filepath.Join(tm.TmpDataDir, ih, ".torrent.db"),
		)
	}
	if _, err := os.Stat(filepath.Join(tm.TmpDataDir, ih, ".torrent.bolt.db")); err != nil {
		os.Symlink(
			filepath.Join(tm.DataDir, ".torrent.bolt.db"),
			filepath.Join(tm.TmpDataDir, ih, ".torrent.bolt.db"),
		)
	}*/

	//if spec == nil {
	/*if tm.boost {
		if data, err := tm.boostFetcher.FetchTorrent(ih.String()); err == nil {
			buf := bytes.NewBuffer(data)
			mi, err := metainfo.Load(buf)

			if err == nil {
				spec = torrent.TorrentSpecFromMetaInfo(mi)
				tmpDataPath := filepath.Join(tm.TmpDataDir, ih.HexString())
				spec.Storage = storage.NewFile(tmpDataPath)

				if t, _, err := tm.client.AddTorrentSpec(spec); err == nil {
					for _, file := range t.Files() {
						subpath := file.Path()
						if data, err := tm.boostFetcher.FetchFile(ih.String(), subpath); err == nil {
							log.Debug("Boost download file", "ih", ih, "path", subpath)

							p := filepath.Join(tmpDataPath, subpath)
							if subpath != "data" {
								dir := filepath.Join(tmpDataPath, "data")
								log.Trace("mkdir", "tmpDataPath", tmpDataPath, "subpath", subpath, "dir", dir)
								if err := os.MkdirAll(dir, 0750); err != nil {
									log.Error("mkdir", "err", err)
									continue
								}
							}
							f, e := os.OpenFile(p, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
							if e != nil {
								log.Error("openfile failed", "err", e, "filePath", p)
								continue
							}
							defer f.Close()
							if _, err := f.Write(data); err != nil {
								log.Error("write data failed", "data", len(data), "err", err, "path", p)
							}
						} else {
							log.Warn("Boost download file failed", "ih", ih, "path", subpath)
						}
					}

					return tm.register(t, BytesRequested, torrentPending, ih, ch)
				}
			}

		}
	}*/

	if spec == nil {
		tmpDataPath := filepath.Join(tm.TmpDataDir, ih)

		//if _, err := os.Stat(filepath.Join(tmpDataPath, ".torrent.db")); err != nil {
		if _, err := os.Stat(tmpDataPath); err != nil {
			if err := os.MkdirAll(tmpDataPath, 0600); err != nil {
				log.Warn("torrent path create failed", "err", err)
				return nil
			}
		}

		//if _, err := os.Stat(filepath.Join(tmpDataPath, ".torrent.db")); err != nil {
		//	os.Symlink(
		//		filepath.Join(tm.DataDir, ".torrent.db"),
		//		filepath.Join(tmpDataPath, ".torrent.db"),
		//	)
		//}

		//if _, err := os.Stat(filepath.Join(tmpDataPath, ".torrent.bolt.db")); err != nil {
		//	os.Symlink(
		//		filepath.Join(tm.DataDir, ".torrent.bolt.db"),
		//		filepath.Join(tmpDataPath, ".torrent.bolt.db"),
		//	)
		//}

		spec = &torrent.TorrentSpec{
			InfoHash: metainfo.NewHashFromHex(ih),
			Storage:  storage.NewFile(tmpDataPath),
			//Storage: storage.NewFileWithCompletion(tmpDataPath, storage.NewMapPieceCompletion()),
		}
	}
	//}

	if t, _, err := tm.client.AddTorrentSpec(spec); err == nil {
		return tm.register(t, BytesRequested, torrentPending, ih, ch)
	}

	return nil
}

func (tm *TorrentManager) updateInfoHash(t *Torrent, BytesRequested int64) {
	tm.lock.Lock()
	defer tm.lock.Unlock()
	if t.bytesRequested < BytesRequested {
		t.bytesRequested = BytesRequested
		t.bytesLimitation = tm.getLimitation(BytesRequested)
	}
	updateMeter.Mark(1)
}

func NewTorrentManager(config *Config, fsid uint64, cache, compress bool) (*TorrentManager, error) {
	cfg := torrent.NewDefaultClientConfig()
	cfg.DisableUTP = config.DisableUTP
	cfg.NoDHT = config.DisableDHT
	cfg.DisableTCP = config.DisableTCP
	cfg.DisableIPv6 = config.DisableIPv6

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
	cl, err := torrent.NewClient(cfg)
	if err != nil {
		log.Error("Error while create torrent client", "err", err)
		return nil, err
	}

	log.Info("Listening local", "port", cl.LocalPort())

	tmpFilePath := filepath.Join(config.DataDir, defaultTmpPath)

	if _, err := os.Stat(tmpFilePath); err != nil {
		err = os.MkdirAll(filepath.Dir(tmpFilePath), 0600) //os.FileMode(os.ModePerm))
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
		maxSeedTask:         config.MaxSeedingNum,
		maxEstablishedConns: cfg.EstablishedConnsPerTorrent,
		DataDir:             config.DataDir,
		TmpDataDir:          tmpFilePath,
		boostFetcher:        NewBoostDataFetcher(config.BoostNodes),
		closeAll:            make(chan struct{}),
		initCh:              make(chan struct{}),
		simulate:            false,
		taskChan:            make(chan interface{}, taskChanBuffer),
		seedingChan:         make(chan *Torrent, torrentChanSize),
		activeChan:          make(chan *Torrent, torrentChanSize),
		pendingChan:         make(chan *Torrent, torrentChanSize),
		mode:                config.Mode,
		boost:               config.Boost,
		id:                  fsid,
		slot:                int(fsid % bucket),
		localSeedFiles:      make(map[string]bool),
	}

	if cache {
		conf := bigcache.Config{
			Shards:             1024,
			LifeWindow:         600 * time.Second,
			CleanWindow:        1 * time.Second,
			MaxEntriesInWindow: 1000 * 10 * 60,
			MaxEntrySize:       512,
			StatsEnabled:       true,
			Verbose:            true,
			HardMaxCacheSize:   512, //MB
		}

		torrentManager.fileCache, err = bigcache.NewBigCache(conf)
		if err != nil {
			log.Error("File system cache initialized failed", "err", err)
		} else {
			torrentManager.cache = cache
			torrentManager.compress = compress
		}
	}

	torrentManager.metrics = config.Metrics

	hotSize := config.MaxSeedingNum/64 + 1
	torrentManager.hotCache, _ = lru.New(hotSize)
	log.Info("Hot cache created", "size", hotSize)

	if len(config.DefaultTrackers) > 0 {
		log.Debug("Tracker list", "trackers", config.DefaultTrackers)
		torrentManager.trackers = [][]string{config.DefaultTrackers}
	}
	log.Debug("Fs client initialized", "config", config)

	return torrentManager, nil
}

func (tm *TorrentManager) Start() error {
	tm.startOnce.Do(func() {
		tm.wg.Add(1)
		go tm.seedingLoop()
		tm.wg.Add(1)
		go tm.activeLoop()
		tm.wg.Add(1)
		go tm.pendingLoop()

		tm.wg.Add(1)
		go tm.mainLoop()
	})

	return tm.init()
}

func (tm *TorrentManager) prepare() bool {
	return true
}

func (tm *TorrentManager) seedingLoop() {
	defer tm.wg.Done()
	for {
		select {
		case t := <-tm.seedingChan:
			tm.seedingTorrents[t.Torrent.InfoHash().HexString()] = t

			s := t.Seed()
			if t.ch != nil {
				log.Warn("Torrent seeding ready for lazy", "ih", t.InfoHash())
				go func() {
					t.ch <- s
				}()
			}

			if !tm.simulate {
				if len(tm.seedingTorrents) == tm.good {
					close(tm.initCh)
				}
			}

			if s {
				//if active, ok := GoodFiles[t.InfoHash()]; tm.cache && ok && active {
				//	for _, file := range t.Files() {
				//		log.Trace("Precache file", "ih", t.InfoHash(), "ok", ok, "active", active)
				//		go tm.getFile(t.InfoHash(), file.Path())
				//	}
				//}
				tm.hotCache.Add(t.Torrent.InfoHash(), true)
				if len(tm.seedingTorrents) > params.LimitSeeding {
					tm.dropSeeding(tm.slot)
				} else if len(tm.seedingTorrents) > tm.maxSeedTask {
					tm.maxSeedTask++
					tm.graceSeeding(tm.slot)
				}
			}
		case <-tm.closeAll:
			log.Info("Seeding loop closed")
			return
		}
	}
}

func (tm *TorrentManager) init() error {
	log.Debug("Chain files init", "files", len(GoodFiles))

	if tm.mode == LAZY {
		tm.Simulate()
	}

	for k, ok := range GoodFiles {
		if ok {
			if err := tm.Search(context.Background(), k, 0, nil); err == nil {
				tm.good++
			} else {
				log.Info("Fs init failed", "err", err)
				return err
			}
		}
	}

	if !tm.simulate {
		select {
		case <-tm.initCh:
			log.Info("Chain files sync init OK !!!", "seeding", len(tm.seedingTorrents), "pending", len(tm.pendingTorrents), "active", len(tm.activeTorrents), "good", len(GoodFiles), "active", tm.good)
		case <-tm.closeAll:
			log.Info("Init files closed")
		}
	}

	log.Debug("Chain files OK !!!")
	return nil
}

func (tm *TorrentManager) Simulate() {
	tm.lock.Lock()
	defer tm.lock.Unlock()

	if !tm.simulate {
		tm.simulate = true
	}
}

//Search and donwload files from torrent
func (tm *TorrentManager) Search(ctx context.Context, hex string, request uint64, ch chan bool) error {
	if !common.IsHexAddress(hex) {
		return errors.New("invalid infohash format")
	}

	if request < 0 {
		return errors.New("request can't be negative")
	}

	hex = strings.TrimPrefix(strings.ToLower(hex), common.Prefix)
	//if _, ok := BadFiles[hex]; ok {
	//	return nil
	//}
	if IsBad(hex) {
		return nil
	}

	//if tm.mode == FULL {
	//if request == 0 {
	//	log.Warn("Prepare mode", "ih", hex)
	//	request = uint64(block)
	//}
	//}

	downloadMeter.Mark(1)

	return tm.commit(ctx, hex, request, ch)
}

func (tm *TorrentManager) mainLoop() {
	defer tm.wg.Done()
	for {
		select {
		case msg := <-tm.taskChan:
			meta := msg.(types.FlowControlMeta)
			//if _, ok := BadFiles[meta.InfoHash.HexString()]; ok {
			//	continue
			//}
			if IsBad(meta.InfoHash) {
				continue
			}

			bytes := int64(meta.BytesRequested)
			if bytes == 0 {
				//preloading
				bytes = block
			}

			//tm.addInfoHash(meta.InfoHash, bytes, meta.Ch)

			if t := tm.addInfoHash(meta.InfoHash, bytes, meta.Ch); t == nil {
				log.Error("Seed [create] failed", "ih", meta.InfoHash, "request", bytes)
				continue
			} else {
				//if bytes > 0 {
				//	tm.updateInfoHash(t, bytes)
				t.lock.Lock()
				if t.currentConns <= 1 {
					t.setCurrentConns(tm.maxEstablishedConns)
					t.Torrent.SetMaxEstablishedConns(tm.maxEstablishedConns)
					log.Warn("Active torrent", "ih", meta.InfoHash, "bytes", bytes)
				}
				t.lock.Unlock()
				//}
			}
			//time.Sleep(time.Second)
		case <-tm.closeAll:
			return
		}
	}
}

func (tm *TorrentManager) pendingLoop() {
	defer tm.wg.Done()
	timer := time.NewTimer(time.Second * queryTimeInterval)
	defer timer.Stop()
	for {
		select {
		case t := <-tm.pendingChan:
			tm.pendingTorrents[t.Torrent.InfoHash().HexString()] = t
			timer.Reset(0)
		case <-timer.C:
			for ih, t := range tm.pendingTorrents {
				if _, ok := BadFiles[ih]; ok {
					continue
				}
				t.loop++
				if tm.boost {
					//TODO
				}
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
					t.lock.Lock()
					if err := t.WriteTorrent(); err == nil {
						if len(tm.activeChan) < cap(tm.activeChan) {
							delete(tm.pendingTorrents, ih)
							t.loop = 0
							tm.activeChan <- t
						}
					}
					t.lock.Unlock()
					/*} else if tm.boost && (t.loop > torrentWaitingTime/queryTimeInterval || (t.start == 0 && t.bytesRequested > 0)) {
						if !t.isBoosting {
							log.Info("Boost download seed", "ih", ih)
							t.loop = 0
							t.isBoosting = true
							if data, err := tm.boostFetcher.FetchTorrent(ih.String()); err == nil {
								if t.Torrent.Info() != nil {
									t.BoostOff()
									continue
								}
								if err := t.ReloadTorrent(data, tm); err == nil {
									tm.setTorrent(ih, t)
								} else {
									t.BoostOff()
								}

							} else {
								log.Debug("Boost failed", "ih", ih.String(), "err", err)
								if t.start == 0 && (t.bytesRequested > 0 || tm.mode == FULL || t.loop > 600) { //|| len(tm.pendingTorrents) == 1) {
									t.AddTrackers(tm.trackers)
									t.start = mclock.Now()
								}
								t.BoostOff()
							}
						}
					} else if tm.boost && (t.loop > 60 || (t.start == 0 && t.bytesRequested > 0)) {
						log.Trace("Boost seed", "ih", ih, "loop", t.loop, "request", t.bytesRequested, "start", t.start)
						t.loop = 0
						t.start = mclock.Now()
						if data, err := tm.boostFetcher.FetchTorrent(ih); err == nil {
							if t.Torrent.Info() != nil {
								continue
							}
							if err := t.ReloadTorrent(data, tm); err == nil {
								tm.setTorrent(ih, t)
							}
						}*/
				} else {
					if _, ok := GoodFiles[t.InfoHash()]; t.start == 0 && (ok || t.bytesRequested > 0 || tm.mode == FULL || t.loop > 600) {
						if ok {
							log.Debug("Good file found in pending", "ih", common.HexToHash(ih))
						}
						t.AddTrackers(tm.trackers)
						t.start = mclock.Now()
					} else {
						if t.loop > 600 {
							//TODO
						}
					}
				}
			}
			timer.Reset(time.Second * queryTimeInterval)
		case <-tm.closeAll:
			log.Info("Pending seed loop closed")
			return
		}
	}
}

func (tm *TorrentManager) activeLoop() {
	defer tm.wg.Done()
	timer := time.NewTimer(time.Second * queryTimeInterval)
	defer timer.Stop()
	var total_size, current_size, log_counter, counter, actual_counter uint64 = 0, 0, 1, 1, 1
	var active_paused, active_wait, active_running int
	for {
		//counter++
		select {
		case t := <-tm.activeChan:
			tm.activeTorrents[t.Torrent.InfoHash().HexString()] = t
			timer.Reset(0)
		case <-timer.C:
			counter++
			log_counter++

			for ih, t := range tm.activeTorrents {
				//BytesRequested := int64(0)
				//if _, ok := GoodFiles[t.InfoHash()]; ok {
				if IsGood(t.InfoHash()) {
					tm.lock.Lock()
					t.bytesRequested = t.Length()
					t.bytesLimitation = tm.getLimitation(t.bytesRequested)
					tm.lock.Unlock()
					t.fast = true
				} else {
					if tm.mode == FULL {
						if t.bytesRequested >= t.Length() {
							t.fast = true
						} else {
							if t.bytesRequested <= t.BytesCompleted()+block/2 {
								tm.lock.Lock()
								t.bytesRequested = int64(math.Min(float64(t.Length()), float64(t.bytesRequested+block)))
								t.bytesLimitation = tm.getLimitation(t.bytesRequested)
								tm.lock.Unlock()
								t.fast = false
							}
						}
					} else {
						if t.bytesRequested >= t.Length() {
							t.fast = true
						} else {
							if t.bytesRequested <= t.BytesCompleted()+block/2 {
								t.fast = false
							}
						}
					}
				}

				if t.bytesRequested == 0 {
					active_wait++
					if log_counter%60 == 0 {
						log.Debug("[Waiting]", "ih", ih, "complete", common.StorageSize(t.bytesCompleted), "req", common.StorageSize(t.bytesRequested), "quota", common.StorageSize(t.bytesRequested), "limit", common.StorageSize(t.bytesLimitation), "total", common.StorageSize(t.BytesMissing()), "seg", len(t.Torrent.PieceStateRuns()), "peers", t.currentConns, "max", t.Torrent.NumPieces())
					}
					continue
				}

				if t.BytesCompleted() > t.bytesCompleted {
					total_size += uint64(t.BytesCompleted() - t.bytesCompleted)
					current_size += uint64(t.BytesCompleted() - t.bytesCompleted)
					//actual_size +=  uint64(t.BytesCompleted() - t.bytesCompleted)
					actual_counter++
				}

				t.bytesCompleted = t.BytesCompleted()
				t.bytesMissing = t.BytesMissing()

				if t.Finished() {
					//tm.lock.Lock()
					if _, err := os.Stat(filepath.Join(tm.DataDir, ih)); err == nil {
						if len(tm.seedingChan) < cap(tm.seedingChan) {
							log.Debug("Path exist", "ih", ih, "path", filepath.Join(tm.DataDir, ih))
							delete(tm.activeTorrents, ih)
							log.Trace("S <- A", "ih", ih) //, "elapsed", time.Duration(mclock.Now())-time.Duration(t.start))
							tm.seedingChan <- t
						}
					} else {
						err := os.Symlink(
							filepath.Join(defaultTmpPath, ih),
							filepath.Join(tm.DataDir, ih),
						)
						if err != nil {
							err = os.Remove(
								filepath.Join(tm.DataDir, ih),
							)
							if err == nil {
								log.Debug("Fix path success", "ih", ih, "size", t.bytesCompleted, "miss", t.bytesMissing, "loop", log_counter)
							}
						} else {
							if len(tm.seedingChan) < cap(tm.seedingChan) {
								delete(tm.activeTorrents, ih)
								log.Trace("S <- A", "ih", ih) //, "elapsed", time.Duration(mclock.Now())-time.Duration(t.start))
								tm.seedingChan <- t
							}
						}
					}

					//tm.lock.Unlock()
					continue
				}

				if t.bytesCompleted >= t.bytesLimitation {
					t.Pause()
					active_paused++
					if log_counter%300 == 0 && active_paused < 11 {
						bar := ProgressBar(t.bytesCompleted, t.Torrent.Length(), "[Paused]")
						log.Debug(bar, "ih", ih, "complete", common.StorageSize(t.bytesCompleted), "req", common.StorageSize(t.bytesRequested), "limit", common.StorageSize(t.bytesLimitation), "total", common.StorageSize(t.bytesMissing+t.bytesCompleted), "seg", len(t.Torrent.PieceStateRuns()), "peers", t.currentConns, "max", t.Torrent.NumPieces())
					}
					continue
				} /*else if t.bytesRequested >= t.bytesCompleted+t.bytesMissing {
					t.loop++
					if tm.boost && t.bytesCompleted == 0 { //t.loop > downloadWaitingTime/queryTimeInterval && t.bytesCompleted*2 < t.bytesRequested {
						log.Info("Boost status", "request", t.bytesRequested, "complete", t.bytesCompleted, "missing", t.bytesMissing)
						t.loop = 0
						if t.isBoosting {
							continue
						}
						t.Pause()
						t.isBoosting = true
						//tm.wg.Add(1)
						filepaths := []string{}
						filedatas := [][]byte{}
						for _, file := range t.Files() {
							if file.BytesCompleted() > 0 {
								continue
							}
							subpath := file.Path()
							if data, err := tm.boostFetcher.FetchFile(ih.String(), subpath); err == nil {
								log.Info("Boost download file", "ih", ih, "path", subpath)
								filedatas = append(filedatas, data)
								filepaths = append(filepaths, subpath)
							} else {
								log.Warn("Boost download file failed", "ih", ih, "path", subpath)
								//return
							}
						}
						t.Torrent.Drop()
						t.ReloadFile(filepaths, filedatas, tm)
						t.BoostOff()
						continue
					}
				}*/

				if log_counter%60 == 0 && t.bytesCompleted > 0 {
					bar := ProgressBar(t.bytesCompleted, t.Torrent.Length(), "")
					elapsed := time.Duration(mclock.Now()) - time.Duration(t.start)
					log.Info(bar, "ih", ih, "complete", common.StorageSize(t.bytesCompleted), "limit", common.StorageSize(t.bytesLimitation), "total", common.StorageSize(t.Torrent.Length()), "seg", len(t.Torrent.PieceStateRuns()), "peers", t.currentConns, "max", t.Torrent.NumPieces(), "speed", common.StorageSize(float64(t.bytesCompleted*1000*1000*1000)/float64(elapsed)).String()+"/s", "elapsed", common.PrettyDuration(elapsed))
				}

				if t.bytesCompleted < t.bytesLimitation && !t.isBoosting {
					go t.Run(tm.slot)
					active_running++
				}
			}

			if counter >= 5*loops {
				if len(tm.seedingTorrents) < len(GoodFiles) && tm.mode != LAZY {
					log.Warn(ProgressBar(int64(len(tm.seedingTorrents)), int64(len(GoodFiles)), "Network scanning"), "mode", tm.mode, "ports", "40401,5008,40404", "seeding", len(tm.seedingTorrents), "expected", len(GoodFiles))
				} else {
					if tm.cache {
						log.Info("Fs status", "pending", len(tm.pendingTorrents), "waiting", active_wait, "downloading", active_running, "paused", active_paused, "seeding", len(tm.seedingTorrents), "size", common.StorageSize(total_size), "speed", common.StorageSize(total_size/actual_counter*queryTimeInterval).String()+"/s", "speed_a", common.StorageSize(total_size/log_counter*queryTimeInterval).String()+"/s", "speed_b", common.StorageSize(current_size/counter*queryTimeInterval).String()+"/s", "metrics", common.PrettyDuration(tm.Updates), "hot", tm.hotCache.Len(), "stats", tm.fileCache.Stats(), "len", tm.fileCache.Len(), "capacity", common.StorageSize(tm.fileCache.Capacity()).String())
					} else {
						log.Info("Fs status", "pending", len(tm.pendingTorrents), "waiting", active_wait, "downloading", active_running, "paused", active_paused, "seeding", len(tm.seedingTorrents), "size", common.StorageSize(total_size), "speed", common.StorageSize(total_size/actual_counter*queryTimeInterval).String()+"/s", "speed_a", common.StorageSize(total_size/log_counter*queryTimeInterval).String()+"/s", "speed_b", common.StorageSize(current_size/counter*queryTimeInterval).String()+"/s", "metrics", common.PrettyDuration(tm.Updates), "hot", tm.hotCache.Len())
					}
				}
				counter = 1
				current_size = 0
			}
			active_paused, active_wait, active_running = 0, 0, 0
			timer.Reset(time.Second * queryTimeInterval)
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
				log.Debug("Encounter active torrent", "ih", ih, "index", i, "group", s, "slot", slot, "len", len(tm.seedingTorrents), "max", tm.maxSeedTask, "peers", t.currentConns, "cited", t.cited)
				continue
			}

			if tm.mode == LAZY {
				t.setCurrentConns(1)
				log.Debug("Lazy mode dropped", "ih", ih, "seeding", len(tm.seedingTorrents), "torrents", len(tm.torrents), "max", tm.maxSeedTask, "peers", t.currentConns, "cited", t.cited)
			} else {
				t.setCurrentConns(2)
			}
			t.Torrent.SetMaxEstablishedConns(t.currentConns)
			log.Debug("Drop seeding invoke", "ih", ih, "index", i, "group", s, "slot", slot, "len", len(tm.seedingTorrents), "max", tm.maxSeedTask, "peers", t.currentConns, "cited", t.cited)
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
				log.Debug("Encounter active torrent", "ih", ih, "index", i, "group", s, "slot", slot, "len", len(tm.seedingTorrents), "max", tm.maxSeedTask, "peers", t.currentConns, "cited", t.cited)
				continue
			}
			if tm.mode == LAZY {
				t.setCurrentConns(1)
			} else {
				t.setCurrentConns(t.minEstablishedConns)
			}
			t.Torrent.SetMaxEstablishedConns(t.currentConns)
			log.Debug("Grace seeding invoke", "ih", ih, "index", i, "group", s, "slot", slot, "len", len(tm.seedingTorrents), "max", tm.maxSeedTask, "peers", t.currentConns, "cited", t.cited)
		}
		i++
	}
	return nil
}

func (tm *TorrentManager) available(ih string, rawSize uint64) (bool, uint64, mclock.AbsTime, error) {
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
			if t.start == 0 {
				return false, uint64(t.BytesCompleted()), 0, ErrUnfinished
			}
			return false, uint64(t.BytesCompleted()), mclock.Now() - t.start, ErrUnfinished
		}

		ok := t.BytesCompleted() <= int64(rawSize)
		//if t.currentConns <= 1 && ok {
		//	t.currentConns = tm.maxEstablishedConns
		//	t.Torrent.SetMaxEstablishedConns(tm.maxEstablishedConns)
		//}

		return ok, uint64(t.BytesCompleted()), mclock.Now() - t.start, nil
	}
}

func (tm *TorrentManager) getFile(infohash, subpath string) ([]byte, uint64, error) {
	getfileMeter.Mark(1)
	if tm.metrics {
		defer func(start time.Time) { tm.Updates += time.Since(start) }(time.Now())
	}

	if !common.IsHexAddress(infohash) {
		return nil, 0, errors.New("invalid infohash format")
	}
	infohash = strings.TrimPrefix(strings.ToLower(infohash), common.Prefix)

	if t := tm.getTorrent(infohash); t == nil {
		return nil, 0, ErrInactiveTorrent
	} else {

		subpath = strings.TrimPrefix(subpath, "/")
		subpath = strings.TrimSuffix(subpath, "/")

		if !t.Ready() {
			log.Error("Read unavailable file", "hash", infohash, "subpath", subpath)
			return nil, uint64(t.BytesCompleted()), ErrUnfinished
		}

		tm.hotCache.Add(infohash, true)
		if t.currentConns < tm.maxEstablishedConns {
			t.setCurrentConns(tm.maxEstablishedConns)
			t.Torrent.SetMaxEstablishedConns(t.currentConns)
			log.Debug("Torrent active", "ih", infohash, "peers", t.currentConns)
		}

		var key = filepath.Join(infohash, subpath)
		if tm.fileCache != nil {
			if cache, err := tm.fileCache.Get(key); err == nil {
				memcacheHitMeter.Mark(1)
				memcacheReadMeter.Mark(int64(len(cache)))
				if c, err := tm.unzip(cache); err != nil {
					return nil, 0, err
				} else {
					if tm.compress {
						log.Info("File cache", "hash", infohash, "path", subpath, "size", tm.fileCache.Len(), "compress", len(cache), "origin", len(c), "compress", tm.compress)
					}
					return c, uint64(t.BytesCompleted()), nil
				}
			}
		}

		tm.fileLock.Lock()
		defer tm.fileLock.Unlock()
		diskReadMeter.Mark(1)
		data, err := os.ReadFile(filepath.Join(tm.DataDir, key))

		//data final verification
		for _, file := range t.Files() {
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
		}

		return data, uint64(t.BytesCompleted()), err
	}
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
