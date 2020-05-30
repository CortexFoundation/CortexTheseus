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
	"github.com/allegro/bigcache/v2"
	"github.com/bradfitz/iter"
	"github.com/edsrzf/mmap-go"
	lru "github.com/hashicorp/golang-lru"
	"golang.org/x/time/rate"
	"io"
	"io/ioutil"
	"math"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	xlog "github.com/anacrolix/log"
	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/mmap_span"
	"github.com/anacrolix/torrent/storage"
)

const (
	bucket                  = params.Bucket //it is best size is 1/3 full nodes
	group                   = params.Group
	updateTorrentChanBuffer = params.SyncBatch
	torrentChanSize         = 64

	torrentPending = iota //2
	torrentPaused
	torrentRunning
	torrentSeeding

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
	trackers            [][]string
	boostFetcher        *BoostDataFetcher
	DataDir             string
	TmpDataDir          string
	closeAll            chan struct{}
	updateTorrent       chan interface{}
	lock                sync.RWMutex
	wg                  sync.WaitGroup
	seedingChan         chan *Torrent
	activeChan          chan *Torrent
	pendingChan         chan *Torrent
	fullSeed            bool
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
}

func (tm *TorrentManager) getLimitation(value int64) int64 {
	return ((value + block - 1) / block) * block
}

func (tm *TorrentManager) register(t *torrent.Torrent, requested int64, status int, ih metainfo.Hash) *Torrent {
	tt := &Torrent{
		t,
		tm.maxEstablishedConns, 5, tm.maxEstablishedConns,
		requested,
		tm.getLimitation(requested),
		0, 0, status,
		ih.String(),
		path.Join(tm.TmpDataDir, ih.String()),
		0, 1, 0, 0, false, true, 0,
	}
	tm.lock.Lock()
	tm.torrents[ih] = tt
	tm.lock.Unlock()

	tm.pendingChan <- tt
	return tt
}

func (tm *TorrentManager) getTorrent(ih metainfo.Hash) *Torrent {
	tm.lock.RLock()
	defer tm.lock.RUnlock()
	if torrent, ok := tm.torrents[ih]; !ok {
		return nil
	} else {
		return torrent
	}
}

func (tm *TorrentManager) Close() error {
	close(tm.closeAll)
	tm.wg.Wait()
	tm.dropAll()
	if tm.cache {
		tm.fileCache.Reset()
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

func (tm *TorrentManager) UpdateTorrent(input interface{}) error {
	tm.updateTorrent <- input
	return nil
}

func (tm *TorrentManager) buildUdpTrackers(trackers []string) (array [][]string) {
	array = make([][]string, 1)
	for _, tracker := range trackers {
		array[0] = append(array[0], "udp"+tracker)
	}
	return array
}

func (tm *TorrentManager) setTrackers(trackers []string, disableTCP, boost bool) {
	tm.lock.Lock()
	defer tm.lock.Unlock()
	tm.trackers = tm.buildUdpTrackers(trackers)
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

func (tm *TorrentManager) loadSpec(ih metainfo.Hash, filePath string, BytesRequested int64) *torrent.TorrentSpec {
	if _, err := os.Stat(filePath); err != nil {
		return nil
	}
	mi, err := metainfo.LoadFromFile(filePath)
	if err != nil {
		log.Error("Error while adding torrent", "Err", err)
		return nil
	}

	TmpDir := path.Join(tm.TmpDataDir, ih.HexString())
	ExistDir := path.Join(tm.DataDir, ih.HexString())

	useExistDir := false
	if _, err := os.Stat(ExistDir); err == nil {
		log.Debug("Seeding from existing file.", "ih", ih.HexString())
		info, err := mi.UnmarshalInfo()
		if err != nil {
			log.Error("error unmarshalling info: ", "info", err)
			return nil
		}

		if err := tm.verifyTorrent(&info, ExistDir); err == nil {
			useExistDir = true
		}
	}

	spec := torrent.TorrentSpecFromMetaInfo(mi)

	if ih != spec.InfoHash {
		log.Warn("Info hash mismatch", "ih", ih.HexString(), "new", spec.InfoHash.HexString())
		return nil
	}

	if useExistDir {
		spec.Storage = storage.NewFile(ExistDir)
	} else {
		spec.Storage = storage.NewFile(TmpDir)
	}

	spec.Trackers = nil

	return spec
}

func (tm *TorrentManager) addInfoHash(ih metainfo.Hash, BytesRequested int64) *Torrent {
	if t := tm.getTorrent(ih); t != nil {
		return t
	}

	tmpTorrentPath := path.Join(tm.TmpDataDir, ih.HexString(), "torrent")
	seedTorrentPath := path.Join(tm.DataDir, ih.HexString(), "torrent")

	var spec *torrent.TorrentSpec

	if _, err := os.Stat(seedTorrentPath); err == nil {
		spec = tm.loadSpec(ih, seedTorrentPath, BytesRequested)
	} else if _, err := os.Stat(tmpTorrentPath); err == nil {
		spec = tm.loadSpec(ih, tmpTorrentPath, BytesRequested)
	}

	if spec == nil {
		tmpDataPath := path.Join(tm.TmpDataDir, ih.HexString())
		spec = &torrent.TorrentSpec{
			Trackers: [][]string{}, //tm.trackers, //[][]string{},
			InfoHash: ih,
			Storage:  storage.NewFile(tmpDataPath),
		}
	}

	if t, _, err := tm.client.AddTorrentSpec(spec); err == nil {
		return tm.register(t, BytesRequested, torrentPending, ih)
	}

	return nil
}

func (tm *TorrentManager) updateInfoHash(ih metainfo.Hash, BytesRequested int64) {
	log.Debug("Update seed", "ih", ih, "bytes", BytesRequested)
	tm.lock.Lock()
	defer tm.lock.Unlock()
	if t, ok := tm.bytes[ih]; !ok || t < BytesRequested {
		tm.bytes[ih] = BytesRequested
	}
}

func NewTorrentManager(config *Config, fsid uint64, cache, compress bool) (*TorrentManager, error) {
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

	tmpFilePath := path.Join(config.DataDir, defaultTmpFilePath)

	if _, err := os.Stat(tmpFilePath); err != nil {
		err = os.MkdirAll(filepath.Dir(tmpFilePath), 0750) //os.FileMode(os.ModePerm))
		if err != nil {
			log.Error("Mkdir failed", "path", tmpFilePath)
			return nil, err
		}
	}

	torrentManager := &TorrentManager{
		client:              cl,
		torrents:            make(map[metainfo.Hash]*Torrent),
		pendingTorrents:     make(map[metainfo.Hash]*Torrent),
		seedingTorrents:     make(map[metainfo.Hash]*Torrent),
		activeTorrents:      make(map[metainfo.Hash]*Torrent),
		bytes:               make(map[metainfo.Hash]int64),
		maxSeedTask:         config.MaxSeedingNum,
		maxEstablishedConns: cfg.EstablishedConnsPerTorrent,
		DataDir:             config.DataDir,
		TmpDataDir:          tmpFilePath,
		boostFetcher:        NewBoostDataFetcher(config.BoostNodes),
		closeAll:            make(chan struct{}),
		updateTorrent:       make(chan interface{}, updateTorrentChanBuffer),
		seedingChan:         make(chan *Torrent, torrentChanSize),
		activeChan:          make(chan *Torrent, torrentChanSize),
		pendingChan:         make(chan *Torrent, torrentChanSize),
		fullSeed:            config.FullSeed,
		id:                  fsid,
		slot:                int(fsid % bucket),
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
			HardMaxCacheSize:   2048, //MB
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

	torrentManager.hotCache, _ = lru.New(32)

	if len(config.DefaultTrackers) > 0 {
		log.Debug("Tracker list", "trackers", config.DefaultTrackers)
		torrentManager.setTrackers(config.DefaultTrackers, config.DisableTCP, config.Boost)
	}
	log.Debug("Fs client initialized", "config", config)

	return torrentManager, nil
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
	for {
		select {
		case t := <-tm.seedingChan:
			tm.seedingTorrents[t.Torrent.InfoHash()] = t
			if t.Seed() {
				if active, ok := GoodFiles[t.InfoHash()]; tm.cache && ok && active {
					for _, file := range t.Files() {
						log.Trace("Precache file", "ih", t.InfoHash(), "ok", ok, "active", active)
						go tm.GetFile(t.InfoHash(), file.Path())
					}
				}

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

func (tm *TorrentManager) init() {
	log.Info("Chain files init", "files", len(GoodFiles))

	for k, _ := range GoodFiles {
		tm.searchAndDownload(k, 0)
	}

	log.Info("Chain files OK !!!")
}

func (tm *TorrentManager) searchAndDownload(hex string, request int64) {
	hash := metainfo.NewHashFromHex(hex)
	if t := tm.addInfoHash(hash, request); t != nil {
		if request > 0 {
			tm.updateInfoHash(hash, request)
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
					if t := tm.addInfoHash(meta.InfoHash, int64(meta.BytesRequested)); t != nil {
						log.Debug("Seed [create] success", "ih", meta.InfoHash, "request", meta.BytesRequested)
						if int64(meta.BytesRequested) > 0 {
							tm.updateInfoHash(meta.InfoHash, int64(meta.BytesRequested))
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
				tm.updateInfoHash(meta.InfoHash, int64(meta.BytesRequested))
			}
		case <-tm.closeAll:
			return
		}
	}
}

func (tm *TorrentManager) pendingTorrentLoop() {
	defer tm.wg.Done()
	timer := time.NewTimer(time.Second * queryTimeInterval)
	defer timer.Stop()
	for {
		select {
		case t := <-tm.pendingChan:
			tm.pendingTorrents[t.Torrent.InfoHash()] = t
		case <-timer.C:
			for ih, t := range tm.pendingTorrents {
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
							tm.activeChan <- t
						}
					}
				} else if t.loop > torrentWaitingTime/queryTimeInterval || (t.start == 0 && tm.boost && tm.bytes[ih] > 0) {
					if !t.isBoosting {
						t.loop = 0
						t.isBoosting = true
						if data, err := tm.boostFetcher.FetchTorrent(ih.String()); err == nil {
							if t.Torrent.Info() != nil {
								t.BoostOff()
								continue
							}
							if err := t.ReloadTorrent(data, tm); err == nil {
								tm.lock.Lock()
								tm.torrents[ih] = t
								tm.lock.Unlock()
							} else {
								t.BoostOff()
							}

						} else {
							log.Warn("Boost failed", "ih", ih.String(), "err", err)
							if t.start == 0 && (tm.bytes[ih] > 0 || tm.fullSeed || t.loop > 600) { //|| len(tm.pendingTorrents) == 1) {
								t.AddTrackers(tm.trackers)
								t.start = mclock.Now()
							}
							t.BoostOff()
						}
					}
				} else {
					if _, ok := GoodFiles[t.InfoHash()]; t.start == 0 && (ok || tm.bytes[ih] > 0 || tm.fullSeed || t.loop > 600) {
						if ok {
							log.Debug("Good file found in pending", "ih", common.HexToHash(ih.String()))
						}
						t.AddTrackers(tm.trackers)
						t.start = mclock.Now()
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
		case <-timer.C:
			log_counter++

			for ih, t := range tm.activeTorrents {
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
					t.bytesLimitation = tm.getLimitation(BytesRequested)
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
								if data, err := tm.boostFetcher.FetchFile(ih.String(), subpath); err == nil {
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
					log.Info(bar, "hash", common.HexToHash(ih.String()), "complete", common.StorageSize(t.bytesCompleted), "limit", common.StorageSize(t.bytesLimitation), "total", common.StorageSize(t.Torrent.Length()), "seg", len(t.Torrent.PieceStateRuns()), "peers", t.currentConns, "max", t.Torrent.NumPieces(), "speed", common.StorageSize(float64(t.bytesCompleted*1000*1000*1000)/float64(elapsed)).String()+"/s", "elapsed", common.PrettyDuration(elapsed))
				}

				if t.bytesCompleted < t.bytesLimitation && !t.isBoosting {
					t.Run(tm.slot)
					active_running += 1
				}
			}

			if counter >= 5*loops {
				if tm.cache {
					log.Info("Fs status", "pending", len(tm.pendingTorrents), "waiting", active_wait, "downloading", active_running, "paused", active_paused, "seeding", len(tm.seedingTorrents), "size", common.StorageSize(total_size), "speed_a", common.StorageSize(total_size/log_counter*queryTimeInterval).String()+"/s", "speed_b", common.StorageSize(current_size/counter*queryTimeInterval).String()+"/s", "slot", tm.slot, "metrics", common.PrettyDuration(tm.Updates), "hot", tm.hotCache.Len(), "stats", tm.fileCache.Stats(), "len", tm.fileCache.Len(), "capacity", common.StorageSize(tm.fileCache.Capacity()).String())
				} else {
					log.Info("Fs status", "pending", len(tm.pendingTorrents), "waiting", active_wait, "downloading", active_running, "paused", active_paused, "seeding", len(tm.seedingTorrents), "size", common.StorageSize(total_size), "speed_a", common.StorageSize(total_size/log_counter*queryTimeInterval).String()+"/s", "speed_b", common.StorageSize(current_size/counter*queryTimeInterval).String()+"/s", "slot", tm.slot, "metrics", common.PrettyDuration(tm.Updates), "hot", tm.hotCache.Len())
				}
				counter = 0
				current_size = 0
			}
			active_paused, active_wait, active_boost, active_running = 0, 0, 0, 0
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
	//if fs.metrics {
	//	defer func(start time.Time) { fs.Updates += time.Since(start) }(time.Now())
	//}

	if rawSize <= 0 {
		return false, errors.New("raw size is zero or negative")
	}

	ih := metainfo.NewHashFromHex(infohash)
	if torrent := fs.getTorrent(ih); torrent == nil {
		return false, errors.New("file not exist")
	} else {
		if !torrent.Ready() {
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
	if torrent := fs.getTorrent(ih); torrent == nil {
		log.Debug("Torrent not found", "hash", infohash)
		return nil, errors.New("file not exist")
	} else {

		subpath = strings.TrimPrefix(subpath, "/")
		subpath = strings.TrimSuffix(subpath, "/")

		if !torrent.Ready() {
			log.Error("Read unavailable file", "hash", infohash, "subpath", subpath)
			return nil, errors.New("download not completed")
		}

		fs.hotCache.Add(ih, true)
		if torrent.currentConns < fs.maxEstablishedConns {
			torrent.currentConns = fs.maxEstablishedConns
			torrent.SetMaxEstablishedConns(torrent.currentConns)
			log.Info("Torrent active", "ih", ih, "peers", torrent.currentConns)
		}

		var key = infohash + "/" + subpath
		if fs.cache {
			if cache, err := fs.fileCache.Get(key); err == nil {
				if c, err := fs.unzip(cache); err != nil {
					return nil, err
				} else {
					if fs.compress {
						log.Info("File cache", "hash", infohash, "path", subpath, "size", fs.fileCache.Len(), "compress", len(cache), "origin", len(c), "compress", fs.compress)
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
			if file.Path() == subpath {
				log.Debug("File location info", "ih", infohash, "path", file.Path(), "key", key)
				if int64(len(data)) != file.Length() {
					log.Error("Read file not completed", "hash", infohash, "len", len(data), "total", file.Path())
					return nil, errors.New("not a complete file")
				} else {
					log.Debug("Read data success", "hash", infohash, "size", len(data), "path", file.Path())
					if c, err := fs.zip(data); err != nil {
						log.Warn("Compress data failed", "hash", infohash, "err", err)
					} else {
						if fs.cache {
							fs.fileCache.Set(key, c)
						}
					}
				}
				break
			}
		}

		return data, err
	}
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
