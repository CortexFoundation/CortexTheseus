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
	"math"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/torrentfs/backend/caffe"
	"github.com/CortexFoundation/torrentfs/backend/job"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/tool"
	"github.com/CortexFoundation/torrentfs/types"
	"github.com/CortexFoundation/wormhole"
	"github.com/ucwong/shard"

	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/analysis"
	"github.com/anacrolix/torrent/bencode"
	"github.com/anacrolix/torrent/iplist"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/mmap_span"
	pp "github.com/anacrolix/torrent/peer_protocol"
	"github.com/anacrolix/torrent/storage"
	"github.com/bradfitz/iter"
	"github.com/edsrzf/mmap-go"
	"github.com/ucwong/golang-kv"

	"github.com/anacrolix/dht/v2"
	"github.com/anacrolix/dht/v2/int160"
	peer_store "github.com/anacrolix/dht/v2/peer-store"

	"github.com/ucwong/filecache"
)

type TorrentManager struct {
	client *torrent.Client
	//bytes               map[metainfo.Hash]int64
	//torrents        map[string]*Torrent
	torrents *shard.Map[*caffe.Torrent]
	//seedingTorrents map[string]*Torrent
	//seedingTorrents *shard.Map[*Torrent]
	//activeTorrents  map[string]*Torrent
	//activeTorrents *shard.Map[*Torrent]
	//pendingTorrents *shard.Map[*Torrent]
	//maxSeedTask         int
	//maxEstablishedConns int
	trackers       [][]string
	globalTrackers [][]string
	//boostFetcher        *BoostDataFetcher
	DataDir    string
	TmpDataDir string
	closeAll   chan struct{}
	taskChan   chan any
	lock       sync.RWMutex
	//pending_lock sync.RWMutex
	//active_lock  sync.RWMutex
	//seeding_lock sync.RWMutex
	wg          sync.WaitGroup
	seedingChan chan *caffe.Torrent
	activeChan  chan *caffe.Torrent
	pendingChan chan *caffe.Torrent
	//pendingRemoveChan chan string
	droppingChan chan string
	mode         string
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

	startOnce sync.Once
	closeOnce sync.Once
	//seedingNotify chan string

	kvdb kv.Bucket

	//colaList mapset.Set[string]

	fc *filecache.FileCache

	//seconds  atomic.Uint64
	recovery atomic.Uint32
	seeds    atomic.Int32
	pends    atomic.Int32
	actives  atomic.Int32
	stops    atomic.Int32

	//worms []Worm

	worm *wormhole.Wormhole
}

func (tm *TorrentManager) getLimitation(value int64) int64 {
	return ((value + block - 1) / block) * block
}

func (tm *TorrentManager) blockCaculate(value int64) int64 {
	return ((value + block - 1) / block)
}

func (tm *TorrentManager) register(t *torrent.Torrent, requested int64, ih string, spec *torrent.TorrentSpec) *caffe.Torrent {
	/*tt := &Torrent{
		Torrent: t,
		//maxEstablishedConns: tm.maxEstablishedConns,
		//minEstablishedConns: 1,
		//currentConns:        tm.maxEstablishedConns,
		bytesRequested: requested,
		//bytesLimitation: tm.getLimitation(requested),
		//bytesCompleted: 0,
		//bytesMissing:        0,
		status:   status,
		infohash: ih,
		filepath: filepath.Join(tm.TmpDataDir, ih),
		cited:    0,
		//weight:     1,
		//loop:       0,
		maxPieces: 0,
		//isBoosting: false,
		//fast:  false,
		start: mclock.Now(),
	}*/

	tt := caffe.NewTorrent(t, requested, ih, filepath.Join(tm.TmpDataDir, ih), tm.slot, spec)
	tm.setTorrent(tt)

	tm.Pending(tt)
	return tt
}

func (tm *TorrentManager) getTorrent(ih string) *caffe.Torrent {
	/*tm.lock.RLock()
	defer tm.lock.RUnlock()
	if torrent, ok := tm.torrents[ih]; ok {
		return torrent
	}
	return nil*/

	if torrent, ok := tm.torrents.Get(ih); ok {
		return torrent
	}
	return nil
}

func (tm *TorrentManager) setTorrent(t *caffe.Torrent) {
	/*tm.lock.Lock()
	defer tm.lock.Unlock()

	tm.torrents[ih] = t*/
	tm.torrents.Set(t.InfoHash(), t)
}

func (tm *TorrentManager) dropTorrent(t *caffe.Torrent) {
	//tm.lock.Lock()
	//defer tm.lock.Unlock()

	defer func() {
		t.Stop()
		tm.stops.Add(1)
	}()

	switch t.Status() {
	case caffe.TorrentPending:
	case caffe.TorrentRunning:
		tm.actives.Add(-1)
	case caffe.TorrentPaused:
	case caffe.TorrentSeeding:
		tm.seeds.Add(-1)
	//case torrentStopping:
	default:
		log.Warn("Unknown status", "ih", t.InfoHash(), "status", t.Status())
	}

	/*if t.Status() == torrentPending {
		//tm.pending_lock.Lock()
		//delete(tm.pendingTorrents, t.InfoHash())
		//tm.pending_lock.Unlock()
		//tm.pendingTorrents.Delete(t.InfoHash())
	} else if t.Status() == torrentRunning || t.Status() == torrentPaused {
		//tm.active_lock.Lock()
		//delete(tm.activeTorrents, t.InfoHash())
		//tm.active_lock.Unlock()
		//tm.activeTorrents.Delete(t.InfoHash())
		tm.actives.Add(-1)
	} else if t.Status() == torrentSeeding {
		//tm.seeding_lock.Lock()
		//delete(tm.seedingTorrents, t.InfoHash())
		//tm.seeding_lock.Unlock()
		//tm.seedingTorrents.Delete(t.InfoHash())
		tm.seeds.Add(-1)
	} else {
		log.Warn("Unknown status", "ih", t.InfoHash(), "status", t.Status())
	}*/

	//delete(tm.torrents, t.InfoHash())
	//tm.torrents.Delete(t.InfoHash())
}

func (tm *TorrentManager) Close() error {
	tm.lock.Lock()
	defer tm.lock.Unlock()

	tm.closeOnce.Do(func() {
		log.Info("Current running torrents", "size", tm.torrents.Len())
		//tm.torrents.Range(func(_ string, t *Torrent) bool {
		//	t.Close()
		//	return true
		//})

		tm.client.Close()
		tm.client.WaitAll()

		close(tm.closeAll)
		tm.wg.Wait()

		if tm.kvdb != nil {
			log.Info("Nas engine close", "engine", tm.kvdb.Name())
			tm.kvdb.Close()
		}

		if tm.fc != nil {
			tm.fc.Stop()
		}
		log.Info("Fs Download Manager Closed")
	})

	return nil
}

/*func (tm *TorrentManager) buildUdpTrackers(trackers []string) (array [][]string) {
	array = make([][]string, 1)
	for _, tracker := range trackers {
		array[0] = append(array[0], "udp"+tracker)
	}
	return array
}*/

/*func mmapFile(name string) (mm mmap.MMap, err error) {
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
}*/

func mmapFile(name string) (mm storage.FileMapping, err error) {
	f, err := os.Open(name)
	if err != nil {
		return
	}
	defer func() {
		if err != nil {
			f.Close()
		}
	}()
	fi, err := f.Stat()
	if err != nil {
		return
	}
	if fi.Size() == 0 {
		return
	}
	reg, err := mmap.MapRegion(f, -1, mmap.RDONLY, mmap.COPY, 0)
	if err != nil {
		return
	}
	return storage.WrapFileMapping(reg, f), nil
}

func (tm *TorrentManager) verifyTorrent(info *metainfo.Info, root string) error {
	span := new(mmap_span.MMapSpan)
	for _, file := range info.UpvertedFiles() {
		filename := filepath.Join(append([]string{root, info.Name}, file.Path...)...)
		mm, err := mmapFile(filename)
		if err != nil {
			return err
		}
		if int64(len(mm.Bytes())) != file.Length {
			return fmt.Errorf("file %q has wrong length", filename)
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

/*func (tm *TorrentManager) verifyTorrent(info *metainfo.Info, root string) error {
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

	// Use a channel to collect errors from goroutines
	errChan := make(chan error, info.NumPieces())
	var wg sync.WaitGroup
	for i := range iter.N(info.NumPieces()) {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			p := info.Piece(i)
			hash := sha1.New()
			_, err := io.CopyBuffer(hash, io.NewSectionReader(span, p.Offset(), p.Length()), make([]byte, 64*1024))
			if err != nil {
				errChan <- err
				return
			}
			good := bytes.Equal(hash.Sum(nil), p.Hash().Bytes())
			if !good {
				errChan <- fmt.Errorf("hash mismatch at piece %d", i)
				return
			}
			errChan <- nil
		}(i)
	}
	go func() {
		wg.Wait()
		close(errChan)
	}()
	for err := range errChan {
		if err != nil {
			return err
		}
	}
	return nil
}*/

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

	var (
		TmpDir   = filepath.Join(tm.TmpDataDir, ih)
		ExistDir = filepath.Join(tm.DataDir, ih)

		useExistDir bool
	)
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
		spec.Storage = storage.NewMMap(ExistDir) //storage.NewFile(ExistDir)
	} else {
		spec.Storage = storage.NewMMap(TmpDir)
	}
	spec.Trackers = nil

	return spec
}

func (tm *TorrentManager) addInfoHash(ih string, bytesRequested int64) *caffe.Torrent {
	if t := tm.getTorrent(ih); t != nil {
		tm.updateInfoHash(t, bytesRequested)
		return t
	}

	if bytesRequested < 0 {
		return nil
	}

	if !server && enableWorm {
		tm.wg.Add(1)
		go func() {
			defer tm.wg.Done()
			err := tm.worm.Tunnel(ih)
			if err != nil {
				log.Error("Wormhole error", "err", err)
			}
		}()
	}

	var (
		spec *torrent.TorrentSpec
		v    []byte
	)

	if tm.kvdb != nil {
		if v = tm.kvdb.Get([]byte(SEED_PRE + ih)); v == nil {
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
				log.Warn("nas path create failed", "err", err)
				return nil
			}
		}

		spec = &torrent.TorrentSpec{
			InfoHash:  metainfo.NewHashFromHex(ih),
			Storage:   storage.NewMMap(tmpDataPath),
			InfoBytes: v,
		}
	}

	/*if t, n, err := tm.client.AddTorrentSpec(spec); err == nil {
		if !n {
			log.Warn("Try to add a dupliated torrent", "ih", ih)
		}

		t.AddTrackers(tm.trackers)

		if tm.globalTrackers != nil {
			t.AddTrackers(tm.globalTrackers)
		}

		if t.Info() == nil && tm.kvdb != nil {
			if v := tm.kvdb.Get([]byte(SEED_PRE + ih)); v != nil {
				t.SetInfoBytes(v)
			}
		}
	} else {
		return nil
	}*/

	if t, err := tm.injectSpec(ih, spec); err != nil {
		return nil
	} else {
		return tm.register(t, bytesRequested, ih, spec)
	}
}

func (tm *TorrentManager) injectSpec(ih string, spec *torrent.TorrentSpec) (*torrent.Torrent, error) {
	if t, n, err := tm.client.AddTorrentSpec(spec); err == nil {
		if !n {
			log.Warn("Try to add a dupliated torrent", "ih", ih)
		}

		t.AddTrackers(tm.trackers)

		if tm.globalTrackers != nil {
			t.AddTrackers(tm.globalTrackers)
		}

		if t.Info() == nil && tm.kvdb != nil {
			if v := tm.kvdb.Get([]byte(SEED_PRE + ih)); v != nil {
				t.SetInfoBytes(v)
			}
		}
		return t, nil
	} else {
		return nil, err
	}
}

func (tm *TorrentManager) updateGlobalTrackers() error {
	tm.lock.Lock()
	defer tm.lock.Unlock()

	if global := tm.worm.BestTrackers(); len(global) > 0 {
		tm.globalTrackers = [][]string{global}
		log.Info("Global trackers update", "size", len(global), "cap", wormhole.CAP, "health", float32(len(global))/float32(wormhole.CAP))

		for _, url := range global {
			score, _ := tm.wormScore(url)
			log.Info("Tracker status", "url", url, "score", score)
		}
	} else {
		// TODO
		return errors.New("best trackers failed")
	}

	// TODO

	return nil
}

func (tm *TorrentManager) wormScore(url string) (score uint64, err error) {
	if tm.kvdb == nil {
		return
	}

	score = 1
	if v := tm.kvdb.Get([]byte(url)); v != nil {
		if score, err = strconv.ParseUint(string(v), 16, 64); err == nil {
			score++
			tm.kvdb.Set([]byte(url), []byte(strconv.FormatUint(score, 16)))
		}
	} else {
		tm.kvdb.Set([]byte(url), []byte(strconv.FormatUint(score, 16)))
	}

	return
}

/*func (tm *TorrentManager) updateColaList() {
	tm.lock.Lock()
	defer tm.lock.Unlock()
	tm.colaList = wormhole.ColaList()
}

func (tm *TorrentManager) ColaList() mapset.Set[string] {
	return tm.colaList
}*/

func (tm *TorrentManager) updateInfoHash(t *caffe.Torrent, bytesRequested int64) {
	if t.Status() != caffe.TorrentSeeding {
		if t.BytesRequested() < bytesRequested {
			//if bytesRequested > t.Length() {
			//	bytesRequested = t.Length()
			//}
			//t.lock.Lock()
			//t.bytesRequested = bytesRequested
			//t.bytesLimitation = tm.getLimitation(bytesRequested)
			//t.lock.Unlock()

			t.SetBytesRequested(bytesRequested)
			//t.leechCh <- struct{}{}

			//if t.Status() == torrentRunning {
			//if t.QuotaFull() {
			//t.Leech()
			//}
			//}
		}
	} else if t.Cited() < 10 {
		// call seeding t
		log.Debug("Already seeding", "ih", t.InfoHash(), "cited", t.Cited())
		t.CitedInc()
	}
	updateMeter.Mark(1)
}

func NewTorrentManager(config *params.Config, fsid uint64, cache, compress bool) (*TorrentManager, error) {
	server = config.Server
	enableWorm = config.Wormhole

	cfg := torrent.NewDefaultClientConfig()
	cfg.DisableUTP = config.DisableUTP
	cfg.NoDHT = config.DisableDHT
	cfg.DisableTCP = config.DisableTCP
	cfg.DisableIPv6 = config.DisableIPv6

	cfg.IPBlocklist = iplist.New([]iplist.Range{
		{First: net.ParseIP("10.0.0.1"), Last: net.ParseIP("10.0.0.255")}})

	if blocklist, err := iplist.MMapPackedFile("packed-blocklist"); err == nil {
		log.Info("Block list loaded")
		cfg.IPBlocklist = blocklist
	}

	cfg.MinPeerExtensions.SetBit(pp.ExtensionBitFast, true)
	//cfg.DisableWebtorrent = false
	//cfg.DisablePEX = false
	//cfg.DisableWebseeds = false
	cfg.DisableIPv4 = false
	cfg.DisableAcceptRateLimiting = true
	//cfg.DisableWebtorrent = false
	//cfg.HeaderObfuscationPolicy.Preferred = true
	//cfg.HeaderObfuscationPolicy.RequirePreferred = true

	/*fc, err := filecache.NewCache(config.DataDir)
	if err != nil {
		return nil, err
	}
	fc.SetCapacity(10 << 30)
	cfg.DefaultStorage = storage.NewResourcePieces(fc.AsResourceProvider())*/

	cfg.DefaultStorage = storage.NewMMap(config.DataDir)

	cfg.DataDir = config.DataDir
	//cfg.DisableEncryption = true
	//cfg.HTTPUserAgent = "Cortex"
	cfg.Seed = true

	cfg.EstablishedConnsPerTorrent = int(math.Min(float64(runtime.NumCPU()*2), float64(50))) //4 //len(config.DefaultTrackers)
	cfg.HalfOpenConnsPerTorrent = cfg.EstablishedConnsPerTorrent / 2

	cfg.ListenPort = config.Port
	if !config.Quiet {
		var pieceOrdering analysis.PeerUploadOrder
		pieceOrdering.Init()
		pieceOrdering.Install(&cfg.Callbacks)

		//cfg.Debug=true
	}
	cfg.DropDuplicatePeerIds = true
	cfg.Bep20 = params.ClientVersion //"-COLA01-"
	//id := strconv.FormatUint(fsid, 16)[0:14]
	//cfg.PeerID = "cortex" + id
	//cfg.ListenHost = torrent.LoopbackListenHost
	//cfg.DhtStartingNodes = dht.GlobalBootstrapAddrs //func() ([]dht.Addr, error) { return nil, nil }

	cfg.ConfigureAnacrolixDhtServer = func(cfg *dht.ServerConfig) {
		cfg.InitNodeId()
		if cfg.PeerStore == nil {
			cfg.PeerStore = &peer_store.InMemory{
				RootId: int160.FromByteArray(cfg.NodeId),
			}
		}
	}

	cl, err := torrent.NewClient(cfg)
	if err != nil {
		log.Error("Error while create nas client", "err", err)
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
		client: cl,
		//torrents:        make(map[string]*Torrent),
		torrents: shard.New[*caffe.Torrent](),
		//pendingTorrents: make(map[string]*Torrent),
		//pendingTorrents: shard.New[*Torrent](),
		//seedingTorrents: make(map[string]*Torrent),
		//seedingTorrents: shard.New[*Torrent](),
		//activeTorrents:  make(map[string]*Torrent),
		//activeTorrents: shard.New[*Torrent](),
		//bytes:               make(map[metainfo.Hash]int64),
		//maxSeedTask:         config.MaxSeedingNum,
		//maxEstablishedConns: cfg.EstablishedConnsPerTorrent,
		DataDir:    config.DataDir,
		TmpDataDir: tmpFilePath,
		//boostFetcher:        NewBoostDataFetcher(config.BoostNodes),
		closeAll: make(chan struct{}),
		//initCh:              make(chan struct{}),
		//simulate:          false,
		taskChan:    make(chan any, taskChanBuffer),
		seedingChan: make(chan *caffe.Torrent, torrentChanSize),
		activeChan:  make(chan *caffe.Torrent, torrentChanSize),
		pendingChan: make(chan *caffe.Torrent, torrentChanSize),
		//pendingRemoveChan: make(chan string, torrentChanSize),
		droppingChan: make(chan string, 1),
		mode:         config.Mode,
		//boost:             config.Boost,
		id:             fsid,
		slot:           int(fsid & (bucket - 1)),
		localSeedFiles: make(map[string]bool),
		//seedingNotify:  notify,
		//kvdb: kv.Badger(config.DataDir),
	}

	//torrentManager.seconds.Store(1)
	torrentManager.recovery.Store(0)
	torrentManager.seeds.Store(0)
	torrentManager.pends.Store(0)

	switch config.Engine {
	case "pebble":
		torrentManager.kvdb = kv.Pebble(config.DataDir)
	case "leveldb":
		torrentManager.kvdb = kv.LevelDB(config.DataDir)
	case "badger":
		torrentManager.kvdb = kv.Badger(config.DataDir)
	case "bolt":
		torrentManager.kvdb = kv.Bolt(config.DataDir)
	case "nutsdb":
		torrentManager.kvdb = kv.NutsDB(config.DataDir)
	case "rosedb":
		torrentManager.kvdb = kv.RoseDB(config.DataDir)
	default:
		panic("Invalid nas engine " + config.Engine)
	}

	log.Info("Using ["+config.Engine+"] as nas db engine", "engine", torrentManager.kvdb.Name())

	if cache {
		torrentManager.fc = filecache.NewDefaultCache()
		torrentManager.fc.MaxSize = 256 * filecache.Megabyte
		//torrentManager.fc.MaxItems = 8
		//torrentManager.fc.Every = 30
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

	torrentManager.worm = wormhole.New()
	torrentManager.updateGlobalTrackers()
	//if global, err := wormhole.BestTrackers(); global != nil && err == nil {
	//	torrentManager.globalTrackers = [][]string{global}
	//}

	//torrentManager.updateColaList()

	log.Debug("Fs client initialized", "config", config, "trackers", torrentManager.trackers)

	return torrentManager, nil
}

func (tm *TorrentManager) Start() (err error) {
	tm.startOnce.Do(func() {
		if tm.fc != nil {
			if err := tm.fc.Start(); err != nil {
				log.Error("File cache start", "err", err)
				return
			}
		}

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

		err = tm.init()
	})

	return
}

func (tm *TorrentManager) prepare() bool {
	return true
}

func (tm *TorrentManager) init() error {
	log.Info("Chain files init", "files", len(params.GoodFiles))

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

	log.Info("Chain files OK !!!")
	return nil
}

/*func (tm *TorrentManager) Simulate() {
	tm.lock.Lock()
	defer tm.lock.Unlock()

	if !tm.simulate {
		tm.simulate = true
	}
	log.Info("Do simulate")
}

func (tm *TorrentManager) commit(ctx context.Context, hex string, request uint64) error {
	select {
	case tm.taskChan <- types.NewBitsFlow(hex, request):
		// TODO
	case <-ctx.Done():
		return ctx.Err()
	case <-tm.closeAll:
	}

	return nil
}*/

func (tm *TorrentManager) Pending(t *caffe.Torrent) {
	select {
	case tm.pendingChan <- t:
	case <-tm.closeAll:
	}
}

func (tm *TorrentManager) Running(t *caffe.Torrent) {
	select {
	case tm.activeChan <- t:
	case <-tm.closeAll:
	}
}

func (tm *TorrentManager) Seeding(t *caffe.Torrent) {
	select {
	case tm.seedingChan <- t:
	case <-tm.closeAll:
	}
}

func (tm *TorrentManager) mainLoop() {
	defer tm.wg.Done()
	timer := time.NewTimer(time.Second * params.QueryTimeInterval * 3600 * 24)
	defer timer.Stop()
	for {
		select {
		case msg := <-tm.taskChan:
			meta := msg.(*types.BitsFlow)
			if params.IsBad(meta.InfoHash()) {
				continue
			}

			if t := tm.addInfoHash(meta.InfoHash(), int64(meta.Request())); t == nil {
				log.Error("Seed [create] failed", "ih", meta.InfoHash(), "request", meta.Request())
			} else {
				if t.Stopping() {
					log.Debug("Nas recovery", "ih", t.InfoHash(), "status", t.Status(), "complete", common.StorageSize(t.Torrent.BytesCompleted()))
					if tt, err := tm.injectSpec(t.InfoHash(), t.Spec()); err == nil && tt != nil {
						t.SetStatus(caffe.TorrentPending)
						t.Lock()
						//t.status = torrentPending
						t.Torrent = tt
						t.SetStart(mclock.Now())
						t.Unlock()

						tm.Pending(t)
						tm.recovery.Add(1)
						tm.stops.Add(-1)
					} else {
						log.Warn("Nas recovery failed", "ih", t.InfoHash(), "status", t.Status(), "complete", common.StorageSize(t.Torrent.BytesCompleted()), "err", err)
					}
				}
			}
		case <-timer.C:
			tm.wg.Add(1)
			go func() {
				defer tm.wg.Done()
				if err := tm.updateGlobalTrackers(); err == nil {
					timer.Reset(time.Second * params.QueryTimeInterval * 3600 * 24)
				} else {
					log.Error("No global tracker found", "err", err)
					timer.Reset(time.Second * params.QueryTimeInterval * 3600)
				}
			}()
		case <-tm.closeAll:
			return
		}
	}
}

func (tm *TorrentManager) pendingLoop() {
	defer tm.wg.Done()
	for {
		select {
		case t := <-tm.pendingChan:
			//tm.pending_lock.Lock()
			//tm.pendingTorrents[t.InfoHash()] = t
			//tm.pending_lock.Unlock()
			//tm.pendingTorrents.Set(t.InfoHash(), t)
			tm.wg.Add(1)
			tm.pends.Add(1)
			go func(t *caffe.Torrent) {
				defer func() {
					tm.wg.Done()
					tm.pends.Add(-1)
				}()
				var timeout time.Duration = 10 + time.Duration(tm.slot&9)
				if tm.mode == params.FULL {
					//timeout *= 2
				}
				ctx, cancel := context.WithTimeout(context.Background(), timeout*time.Minute)
				defer cancel()
				select {
				case <-t.Torrent.GotInfo():
					if b, err := bencode.Marshal(t.Torrent.Info()); err == nil {
						log.Debug("Record full nas in history", "ih", t.InfoHash(), "info", len(b))
						if tm.kvdb != nil && tm.kvdb.Get([]byte(SEED_PRE+t.InfoHash())) == nil {
							elapsed := time.Duration(mclock.Now()) - time.Duration(t.Birth())
							log.Debug("Imported new seed", "ih", t.InfoHash(), "request", common.StorageSize(t.Length()), "ts", common.StorageSize(len(b)), "good", params.IsGood(t.InfoHash()), "elapsed", common.PrettyDuration(elapsed))
							/*tm.wg.Add(1)
							go func(tt *caffe.Torrent, bb []byte) {
								tm.wg.Done()
								if err := tt.WriteTorrent(); err == nil {
									tm.kvdb.Set([]byte(SEED_PRE+t.InfoHash()), bb)
								}
							}(t, b)*/

							if err := t.WriteTorrent(); err == nil {
								tm.kvdb.Set([]byte(SEED_PRE+t.InfoHash()), b)
							}

							// job TODO
							valid := func(a *caffe.Torrent) bool {
								switch a.Status() {
								case caffe.TorrentPending:
									log.Info("Caffe is pending", "ih", t.InfoHash())
								case caffe.TorrentPaused:
									log.Info("Caffe is pausing", "ih", t.InfoHash())
								case caffe.TorrentRunning:
									log.Trace("Caffe is running", "ih", t.InfoHash())
								case caffe.TorrentSeeding:
									log.Info("Caffe is seeding", "ih", t.InfoHash())
									return true
								case caffe.TorrentStopping:
									log.Info("Caffe is stopping", "ih", t.InfoHash(), "complete", t.BytesCompleted(), "miss", t.BytesMissing())
									return true
								}
								return false
							}

							tm.wg.Add(1)
							go func(t *caffe.Torrent, fn func(t *caffe.Torrent) bool) {
								defer tm.wg.Done()

								j := job.New(t)
								log.Info("Job started", "ih", t.InfoHash(), "id", j.ID())
								ch := j.Completed(fn)
								defer func() {
									close(ch)
									// TODO
								}()

								//ctx, cancel := context.WithTimeout(context.Background(), timeout*time.Minute)
								//defer cancel()

								select {
								case suc := <-ch:
									if !suc {
										log.Warn("Uncompleted jobs", "ih", t.InfoHash(), "suc", suc, "job", j.ID(), "ready", common.PrettyDuration(time.Duration(j.Birth()-t.Birth())), "elapse", common.PrettyDuration(time.Duration(mclock.Now()-j.Birth())))
									} else {
										log.Info("Job has been completed", "ih", t.InfoHash(), "suc", suc, "job", j.ID(), "ready", common.PrettyDuration(time.Duration(j.Birth()-t.Birth())), "elapse", common.PrettyDuration(time.Duration(mclock.Now()-j.Birth())))
									}
								//case <-ctx.Done():
								case <-tm.closeAll:
									log.Info("Job quit", "ih", t.InfoHash(), "id", j.ID())
								}
							}(t, valid)
						}
						//t.lock.Lock()
						//t.Birth() = mclock.Now()
						//t.lock.Unlock()
					} else {
						log.Error("Meta info marshal failed", "ih", t.InfoHash(), "err", err)
						tm.Dropping(t.InfoHash())
						return
					}

					if err := t.Start(); err != nil {
						log.Error("Nas start failed", "ih", t.InfoHash(), "err", err)
						// TODO
					}

					//if err := t.WriteTorrent(); err != nil {
					//	log.Warn("Write torrent file error", "ih", t.InfoHash(), "err", err)
					//}

					if params.IsGood(t.InfoHash()) || tm.mode == params.FULL { //|| tm.colaList.Contains(t.InfoHash()) {
						//t.lock.Lock()
						//t.bytesRequested = t.Length()
						//t.bytesLimitation = tm.getLimitation(t.Length())
						//t.lock.Unlock()

						t.SetBytesRequested(t.Length())
					} else {
						if t.BytesRequested() > t.Length() {
							//t.lock.Lock()
							//t.bytesRequested = t.Length()
							//t.bytesLimitation = tm.getLimitation(t.Length())
							//t.lock.Unlock()

							t.SetBytesRequested(t.Length())
						}
					}
					//tm.pending_lock.Lock()
					//delete(tm.pendingTorrents, t.InfoHash())
					//tm.pending_lock.Unlock()
					//tm.pendingTorrents.Delete(t.InfoHash())

					//tm.activeChan <- t
					tm.Running(t)
				case <-t.Closed():
				case <-tm.closeAll:
				case <-ctx.Done():
					tm.Dropping(t.InfoHash())
				}
			}(t)
		case <-tm.closeAll:
			log.Info("Pending seed loop closed")
			return
		}
	}
}

func (tm *TorrentManager) finish(t *caffe.Torrent) {
	t.Lock()
	defer t.Unlock()

	if _, err := os.Stat(filepath.Join(tm.DataDir, t.InfoHash())); err == nil {
		//tm.activeTorrents.Delete(t.InfoHash())
		//tm.seedingChan <- t
		tm.Seeding(t)
	} else {
		if err := os.Symlink(
			filepath.Join(params.DefaultTmpPath, t.InfoHash()),
			filepath.Join(tm.DataDir, t.InfoHash()),
		); err == nil {
			//tm.activeTorrents.Delete(t.InfoHash())
			//tm.seedingChan <- t
			tm.Seeding(t)
		}
	}
}

/*func (tm *TorrentManager) dur() uint64 {
	return tm.seconds.Load()
}

func (tm *TorrentManager) cost(s uint64) {
	tm.seconds.Add(s)
}*/

func (tm *TorrentManager) activeLoop() {
	var (
		timer   = time.NewTicker(time.Second * params.QueryTimeInterval)
		timer_1 = time.NewTicker(time.Second * params.QueryTimeInterval * 60)
		//timer_2 = time.NewTicker(time.Second * params.QueryTimeInterval * 3600 * 18)
		//clean = []*Torrent{}
	)

	defer func() {
		tm.wg.Done()
		timer.Stop()
		timer_1.Stop()
		//timer_2.Stop()
	}()

	for {
		select {
		case t := <-tm.activeChan:
			//tm.active_lock.Lock()
			//tm.activeTorrents[t.InfoHash()] = t
			//tm.active_lock.Unlock()

			//tm.activeTorrents.Set(t.InfoHash(), t)

			if t.QuotaFull() { //t.Length() <= t.BytesRequested() {
				//t.Leech()
			}

			n := tm.blockCaculate(t.Torrent.Length())
			if n < 300 {
				n += 300
			}

			n += tool.Rand(300)
			if tm.mode == params.FULL {
				n *= 2
			}
			tm.actives.Add(1)
			tm.wg.Add(1)
			go func(i string, n int64) {
				defer tm.wg.Done()
				timer := time.NewTicker(time.Duration(n) * time.Second)
				defer timer.Stop()
				for {
					select {
					// TODO download operation check case
					//case <-t.leechCh:
					//	t.Leech()
					case <-timer.C:
						if t := tm.getTorrent(i); t != nil { //&& t.Ready() {
							if t.Cited() <= 0 {
								tm.Dropping(i)
								return
							} else {
								t.CitedDec()
								log.Debug("Seed cited has been decreased", "ih", i, "cited", t.Cited(), "n", n, "status", t.Status(), "elapsed", common.PrettyDuration(time.Duration(mclock.Now())-time.Duration(t.Birth())))
							}
						} else {
							return
						}
					case <-tm.closeAll:
						return
						//case <-t.Closed():
						//	return
						//case <-t.closeAll:
						//	return
					}
				}
			}(t.InfoHash(), n)
		case <-timer_1.C:
			if tm.fc != nil {
				log.Debug("Cache status", "total", common.StorageSize(tm.fc.FileSize()), "itms", tm.fc.Size())
				if tm.mode == params.LAZY {
					for _, itm := range tm.fc.MostAccessed(4) {
						log.Debug("Cache status", "key", itm.Key(), "acc", itm.AccessCount, "dur", common.PrettyDuration(itm.Dur()))
					}
				}
			}

			//if tm.dur() > 0 {
			//stopped := int32(tm.torrents.Len()) - tm.seeds.Load() - tm.actives.Load() - tm.pends.Load()
			log.Info("Fs status", "pending", tm.pends.Load(), "downloading", tm.actives.Load(), "seeding", tm.seeds.Load(), "stopping", tm.stops.Load(), "all", tm.torrents.Len(), "recovery", tm.recovery.Load(), "metrics", common.PrettyDuration(tm.Updates), "job", job.SEQ()) //, "total", common.StorageSize(tm.total()), "cost", common.PrettyDuration(time.Duration(tm.dur())), "speed", common.StorageSize(float64(tm.total()*1000*1000*1000)/float64(tm.dur())).String()+"/s")
			//}
		//case <-timer_2.C:
		//	go tm.updateGlobalTrackers()
		case <-timer.C:
			/*for ih, t := range tm.activeTorrents {
				if t.Torrent.BytesMissing() == 0 {
					tm.finish(ih, t)
					tm.cost(uint64(time.Duration(mclock.Now()) - time.Duration(t.start)))
					continue
				}

				if t.Torrent.BytesCompleted() < t.BytesRequested() {
					t.Leech()
				}
			}*/

			tm.torrents.Range(func(ih string, t *caffe.Torrent) bool {
				if t.Running() {
					if t.Torrent.BytesMissing() == 0 {
						//clean = append(clean, t)
						tm.finish(t)
					} else {
						if t.Torrent.BytesCompleted() < t.BytesRequested() {
							t.Leech()
						}
					}
				}

				return true
			})

			/*for _, i := range clean {
				tm.finish(i)
			}

			clean = []*Torrent{}*/
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
			//tm.seeding_lock.Lock()
			//tm.seedingTorrents[t.InfoHash()] = t
			//tm.seeding_lock.Unlock()

			//tm.seedingTorrents.Set(t.InfoHash(), t)

			if t.Seed() {
				// count
				tm.actives.Add(-1)
				tm.seeds.Add(1)

				// TODO
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
				tm.dropTorrent(t)

				elapsed := time.Duration(mclock.Now()) - time.Duration(t.Birth())
				log.Debug("Seed has been dropped", "ih", ih, "cited", t.Cited(), "status", t.Status(), "elapsed", common.PrettyDuration(elapsed))
			} else {
				log.Warn("Drop seed not found", "ih", ih)
			}
		case <-tm.closeAll:
			log.Info("Dropping loop closed")
			return
		}
	}
}

func (tm *TorrentManager) Dropping(ih string) error {
	select {
	case tm.droppingChan <- ih:
	case <-tm.closeAll:
	}
	return nil
}

/*
func (tm *TorrentManager) Exists(ih string, rawSize uint64) (bool, uint64, mclock.AbsTime, error) {
	availableMeter.Mark(1)

	if !common.IsHexAddress(ih) {
		return false, 0, 0, errors.New("invalid infohash format")
	}

	ih = strings.TrimPrefix(strings.ToLower(ih), common.Prefix)

	if t := tm.getTorrent(ih); t == nil {
		dir := filepath.Join(tm.DataDir, ih)
		if _, err := os.Stat(dir); err == nil {
			return true, 0, 0, ErrInactiveTorrent
		}
		return false, 0, 0, ErrInactiveTorrent
	} else {
		if !t.Ready() {
			if t.Torrent.Info() == nil {
				return false, 0, 0, ErrTorrentNotFound
			}
			return false, uint64(t.Torrent.BytesCompleted()), mclock.Now() - t.Birth(), ErrUnfinished
		}

		// TODO
		ok := t.Torrent.BytesCompleted() <= int64(rawSize)

		return ok, uint64(t.Torrent.BytesCompleted()), mclock.Now() - t.Birth(), nil
	}
}

func (tm *TorrentManager) GetFile(ctx context.Context, infohash, subpath string) (data []byte, err error) {
	getfileMeter.Mark(1)
	if tm.metrics {
		defer func(start time.Time) { tm.Updates += time.Since(start) }(time.Now())
	}

	if !common.IsHexAddress(infohash) {
		return nil, errors.New("invalid infohash format")
	}

	infohash = strings.TrimPrefix(strings.ToLower(infohash), common.Prefix)
	subpath = strings.TrimPrefix(subpath, "/")
	subpath = strings.TrimSuffix(subpath, "/")

	var key = filepath.Join(infohash, subpath)

	log.Debug("Get File", "dir", tm.DataDir, "key", key)

	if t := tm.getTorrent(infohash); t != nil {
		if !t.Ready() {
			return nil, ErrUnfinished
		}

		// Data protection when torrent is active
		t.RLock()
		defer t.RUnlock()

	}

	diskReadMeter.Mark(1)
	dir := filepath.Join(tm.DataDir, key)
	if tm.fc != nil && tm.fc.Active() {
		start := mclock.Now()
		if data, err = tm.fc.ReadFile(dir); err == nil {
			log.Debug("Load data from file cache", "ih", infohash, "dir", dir, "elapsed", common.PrettyDuration(time.Duration(mclock.Now()-start)))
		}
	} else {
		// local read
		data, err = os.ReadFile(dir)
	}

	return
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
}*/
