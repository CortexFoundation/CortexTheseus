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
	//"bytes"
	"context"
	"reflect"
	//"crypto/sha1"
	"errors"
	//"fmt"
	//"io"
	//"math"
	"math/rand"
	//"net"
	"os"
	"path/filepath"
	//"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/event"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/wormhole"
	//"github.com/anacrolix/dht/v2"
	//"github.com/anacrolix/dht/v2/int160"
	//peer_store "github.com/anacrolix/dht/v2/peer-store"
	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/analysis"
	"github.com/anacrolix/torrent/bencode"
	//"github.com/anacrolix/torrent/iplist"
	"github.com/anacrolix/torrent/metainfo"
	//"github.com/anacrolix/torrent/mmap_span"
	//pp "github.com/anacrolix/torrent/peer_protocol"
	"github.com/anacrolix/torrent/storage"
	//"github.com/bradfitz/iter"
	"github.com/edsrzf/mmap-go"
	"github.com/ucwong/filecache"
	"github.com/ucwong/golang-kv"
	"github.com/ucwong/shard"

	"github.com/CortexFoundation/torrentfs/backend/caffe"
	//"github.com/CortexFoundation/torrentfs/backend/job"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/types"

	xlog "github.com/anacrolix/log"

	"golang.org/x/sync/errgroup"
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
	dataDir    string
	tmpDataDir string
	closeAll   chan struct{}
	//taskChan   chan any
	lock sync.RWMutex
	//pending_lock sync.RWMutex
	//active_lock  sync.RWMutex
	//seeding_lock sync.RWMutex
	wg sync.WaitGroup
	//seedingChan chan *caffe.Torrent
	//activeChan chan *caffe.Torrent
	//pendingChan chan *caffe.Torrent

	taskEvent *event.TypeMux
	//pendingRemoveChan chan string
	//droppingChan chan string
	mode string
	//boost               bool
	id   uint64
	slot int

	//fileLock sync.RWMutex
	//fileCache *bigcache.BigCache
	cache    bool
	compress bool

	metrics bool
	updates time.Duration

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

	//filter map[string]int64
}

func (tm *TorrentManager) getLimitation(value int64) int64 {
	return ((value + block - 1) / block) * block
}

func (tm *TorrentManager) blockCaculate(value int64) int64 {
	return ((value + block - 1) / block)
}

func (tm *TorrentManager) register(t *torrent.Torrent, requested int64, ih string, spec *torrent.TorrentSpec) *caffe.Torrent {
	tt := caffe.NewTorrent(t, requested, ih, filepath.Join(tm.tmpDataDir, ih), tm.slot, spec)
	tm.setTorrent(tt)

	if err := tm.Pending(tt); err != nil {
		return nil
	}
	return tt
}

func (tm *TorrentManager) getTorrent(ih string) *caffe.Torrent {
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

	//delete(tm.filter, t.InfoHash())

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

		if tm.client != nil {
			tm.client.Close()
			tm.client.WaitAll()
		}

		if tm.closeAll != nil {
			close(tm.closeAll)
		}
		tm.wg.Wait()

		if tm.kvdb != nil {
			log.Info("Nas engine close", "engine", tm.kvdb.Name())
			tm.kvdb.Close()
		}

		if tm.fc != nil {
			tm.fc.Stop()
		}

		if tm.taskEvent != nil {
			tm.taskEvent.Stop()
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

/*func (tm *TorrentManager) verifyTorrent(info *metainfo.Info, root string) error {
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
		good := bytes.Equal(hash.Sum(nil), p.V1Hash().Unwrap().Bytes())
		if !good {
			return fmt.Errorf("hash mismatch at piece %d", i)
		}
	}
	return nil
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
		TmpDir = filepath.Join(tm.tmpDataDir, ih)
		//ExistDir = filepath.Join(tm.dataDir, ih)
	)

	//if _, err := os.Stat(ExistDir); err == nil {
	//	spec.Storage = storage.NewMMap(ExistDir) //storage.NewFile(ExistDir)
	//} else {
	spec.Storage = storage.NewMMap(TmpDir)
	//}

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

	// Start wormhole tunneling in a goroutine
	if !server && enableWorm {
		tm.wg.Add(1)
		go func() {
			defer tm.wg.Done()
			if err := tm.worm.Tunnel(ih); err != nil {
				log.Error("Wormhole error", "err", err)
			}
		}()
	}

	// Try to load torrent spec from different locations
	var spec *torrent.TorrentSpec
	var infoBytes []byte

	if tm.kvdb != nil {
		infoBytes = tm.kvdb.Get([]byte(SEED_PRE + ih))
	}

	// If not found in KVDB, check local files
	if infoBytes == nil {
		seedTorrentPath := filepath.Join(tm.dataDir, ih, TORRENT)
		if _, err := os.Stat(seedTorrentPath); err == nil {
			spec = tm.loadSpec(ih, seedTorrentPath)
		}

		if spec == nil {
			tmpTorrentPath := filepath.Join(tm.tmpDataDir, ih, TORRENT)
			if _, err := os.Stat(tmpTorrentPath); err == nil {
				spec = tm.loadSpec(ih, tmpTorrentPath)
			}
		}
	}

	// If spec is still not found, create a new one
	if spec == nil {
		tmpDataPath := filepath.Join(tm.tmpDataDir, ih)
		if err := os.MkdirAll(tmpDataPath, 0777); err != nil {
			log.Warn("nas path create failed", "err", err)
			return nil
		}

		opts := torrent.AddTorrentOpts{
			InfoHash:  metainfo.NewHashFromHex(ih),
			Storage:   storage.NewMMap(tmpDataPath),
			InfoBytes: infoBytes,
		}
		spec = &torrent.TorrentSpec{
			AddTorrentOpts: opts,
		}
	}

	// Inject the spec and register the torrent
	t, err := tm.injectSpec(ih, spec)
	if err != nil {
		return nil
	}
	return tm.register(t, bytesRequested, ih, spec)
}

// Start the torrent leeching
func (tm *TorrentManager) injectSpec(ih string, spec *torrent.TorrentSpec) (*torrent.Torrent, error) {
	if spec != nil {
		spec.Trackers = tm.trackers
	} else {
		return nil, errors.New("Nil spec")
	}

	if t, _, err := tm.client.AddTorrentSpec(spec); err == nil {
		/*if !n {
			log.Warn("Try to add a dupliated torrent", "ih", ih)
			return t, errors.New("Try to add a dupliated torrent")
		}*/

		/*if len(spec.Trackers) == 0 {
		          t.AddTrackers(tm.trackers)
		  } else {
		          t.ModifyTrackers(tm.trackers)
		  }*/

		if t.Info() == nil && tm.kvdb != nil {
			if v := tm.kvdb.Get([]byte(SEED_PRE + ih)); v != nil {
				t.SetInfoBytes(v)
			}
		}

		log.Debug("Meta", "ih", ih, "mi", t.Metainfo().AnnounceList)

		return t, nil
	} else {
		return nil, err
	}
}

func (tm *TorrentManager) updateGlobalTrackers() (uint64, float32, error) {
	tm.lock.Lock()
	defer tm.lock.Unlock()

	// TODO light mode without global trackers
	if tm.mode == params.LAZY {
		//return nil
	}

	var (
		total  uint64
		health float32
	)

	if global := tm.worm.BestTrackers(); len(global) > 0 {
		/*if len(tm.trackers) > 1 {
			tm.trackers = append(tm.trackers[:1], global)
		} else {
			tm.trackers = append(tm.trackers, global)
		}*/

		tm.globalTrackers = [][]string{global}

		for _, url := range global {
			score, _ := tm.wormScore(url)
			log.Debug("Tracker status", "url", url, "score", score)
			total += score
		}
		health = float32(len(global)) / float32(wormhole.CAP)
		log.Info("Global trackers update", "size", len(global), "cap", wormhole.CAP, "health", health, "total", total)
	} else {
		// TODO
		return total, health, errors.New("best trackers failed")
	}

	// TODO

	return total, health, nil
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
		//if t.BytesRequested() < bytesRequested {
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
		//}
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

	/*cfg.IPBlocklist = iplist.New([]iplist.Range{
		{First: net.ParseIP("10.0.0.1"), Last: net.ParseIP("10.0.0.255")}})

	if blocklist, err := iplist.MMapPackedFile("packed-blocklist"); err == nil {
		log.Info("Block list loaded")
		cfg.IPBlocklist = blocklist
	}*/

	//cfg.MinPeerExtensions.SetBit(pp.ExtensionBitFast, true)
	//cfg.DisableWebtorrent = false
	//cfg.DisablePEX = false
	//cfg.DisableWebseeds = false
	//cfg.DisableIPv4 = false
	//cfg.DisableAcceptRateLimiting = true
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

	//cfg.EstablishedConnsPerTorrent = 128 //int(math.Min(float64(runtime.NumCPU()*2), float64(50))) //4 //len(config.DefaultTrackers)
	//cfg.HalfOpenConnsPerTorrent = cfg.EstablishedConnsPerTorrent / 2

	cfg.ListenPort = config.Port
	if !config.Quiet {
		var pieceOrdering analysis.PeerUploadOrder
		pieceOrdering.Init()
		pieceOrdering.Install(&cfg.Callbacks)

		//cfg.Debug=true
	}
	cfg.Logger.SetHandlers(xlog.DiscardHandler)
	//cfg.DropDuplicatePeerIds = true
	cfg.Bep20 = params.ClientVersion //"-COLA01-"
	//id := strconv.FormatUint(fsid, 16)[0:14]
	//cfg.PeerID = "cortex" + id
	//cfg.ListenHost = torrent.LoopbackListenHost
	//cfg.DhtStartingNodes = dht.GlobalBootstrapAddrs //func() ([]dht.Addr, error) { return nil, nil }

	/*cfg.ConfigureAnacrolixDhtServer = func(cfg *dht.ServerConfig) {
		cfg.InitNodeId()
		if cfg.PeerStore == nil {
			cfg.PeerStore = &peer_store.InMemory{
				RootId: int160.FromByteArray(cfg.NodeId),
			}
		}
	}*/
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
		dataDir:    config.DataDir,
		tmpDataDir: tmpFilePath,
		//boostFetcher:        NewBoostDataFetcher(config.BoostNodes),
		closeAll: make(chan struct{}),
		//initCh:              make(chan struct{}),
		//simulate:          false,
		//taskChan: make(chan any), //, taskChanBuffer),
		//seedingChan: make(chan *caffe.Torrent, torrentChanSize),
		//activeChan:  make(chan *caffe.Torrent, torrentChanSize),
		//pendingChan: make(chan *caffe.Torrent, torrentChanSize),

		taskEvent: new(event.TypeMux),
		//pendingRemoveChan: make(chan string, torrentChanSize),
		//droppingChan: make(chan string, 1),
		mode: config.Mode,
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
	score, health, _ := torrentManager.updateGlobalTrackers()
	//torrentManager.updateColaList()

	log.Info("Fs client initialized", "config", config, "trackers", torrentManager.trackers, "score", score, "health", health)

	return torrentManager, nil
}

/*func (tm *TorrentManager) Start() (err error) {
	tm.startOnce.Do(func() {
		if tm.fc != nil {
			if err := tm.fc.Start(); err != nil {
				log.Error("File cache start", "err", err)
				return
			}
		}

		//tm.wg.Add(1)
		//go tm.droppingLoop()
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
}*/

func (tm *TorrentManager) Start() error {
	var startErr error

	tm.startOnce.Do(func() {
		if tm.fc != nil {
			if err := tm.fc.Start(); err != nil {
				log.Error("File cache start failed", "err", err)
				startErr = err
				return
			}
		}

		if err := tm.init(); err != nil {
			log.Error("TorrentManager init failed", "err", err)
			startErr = err
			return
		}

		tm.runLoop("seedingLoop", tm.seedingLoop)
		tm.runLoop("activeLoop", tm.activeLoop)
		tm.runLoop("pendingLoop", tm.pendingLoop)
		tm.runLoop("mainLoop", tm.mainLoop)
	})

	return startErr
}

func (tm *TorrentManager) runLoop(name string, fn func()) {
	tm.wg.Add(1)
	go func() {
		defer tm.wg.Done()
		log.Debug("Loop started", "loop", name)
		fn()
		log.Debug("Loop exited", "loop", name)
	}()
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

func (tm *TorrentManager) Pending(t *caffe.Torrent) error {
	/*select {
	case tm.pendingChan <- t:
		log.Trace("Stable pending", "ih", t.InfoHash())
	case <-tm.closeAll:
	default:
	}*/
	return tm.taskEvent.Post(pendingEvent{t})
}

func (tm *TorrentManager) Running(t *caffe.Torrent) error {
	/*select {
	case tm.activeChan <- t:
		log.Trace("Stable running", "ih", t.InfoHash())
	case <-tm.closeAll:
	default:
		log.Trace("Unstable running", "ih", t.InfoHash())
	}*/
	return tm.taskEvent.Post(runningEvent{t})
}

type pendingEvent struct {
	T *caffe.Torrent
}

type runningEvent struct {
	T *caffe.Torrent
}

type seedingEvent struct {
	T *caffe.Torrent
}

type droppingEvent struct {
	S string
}

func (tm *TorrentManager) Seeding(t *caffe.Torrent) error {
	return tm.taskEvent.Post(seedingEvent{t})
	/*select {
	case tm.seedingChan <- t:
		log.Debug("Stable seeding", "ih", t.InfoHash())
	case <-tm.closeAll:
	default:
		log.Debug("Unstable seeding", "ih", t.InfoHash())
	}*/
}

func (tm *TorrentManager) Dropping(ih string) error {
	return tm.taskEvent.Post(droppingEvent{ih})
	/*select {
	case tm.droppingChan <- ih:
	case <-tm.closeAll:
	default:
	}
	return nil*/
}

type mainEvent struct {
	B *types.BitsFlow
}

func (tm *TorrentManager) mainLoop() {
	//defer tm.wg.Done()

	// Use a more readable timer duration.
	const twelveHours = time.Hour * 12
	timer := time.NewTimer(twelveHours)
	defer timer.Stop()

	// Unsubscribe from event channel on function exit.
	sub := tm.taskEvent.Subscribe(mainEvent{})
	defer sub.Unsubscribe()

	for {
		select {
		case ev := <-sub.Chan():
			// Skip nil events or events with non-mainEvent data.
			m, ok := ev.Data.(mainEvent)
			if !ok {
				continue
			}
			meta := m.B
			ih := meta.InfoHash()

			if params.IsBad(ih) || (tm.mode == params.LAZY && meta.Request() == 0) {
				continue
			}

			t := tm.addInfoHash(ih, int64(meta.Request()))
			if t == nil {
				log.Error("Seed [create] failed", "ih", ih, "request", meta.Request())
				continue
			}

			// Handle NAS recovery for a torrent that's stopping.
			if t.Stopping() {
				tm.handleNasRecovery(t)
			}

		case <-timer.C:
			tm.handleTrackerUpdate(timer)

		case <-tm.closeAll:
			return
		}
	}
}

// handleNasRecovery encapsulates the logic for recovering a torrent.
func (tm *TorrentManager) handleNasRecovery(t *caffe.Torrent) {
	log.Debug("Nas recovery", "ih", t.InfoHash(), "status", t.Status(), "complete", common.StorageSize(t.Torrent.BytesCompleted()))

	tt, err := tm.injectSpec(t.InfoHash(), t.Spec())
	if err != nil {
		log.Warn("Nas recovery failed", "ih", t.InfoHash(), "status", t.Status(), "complete", common.StorageSize(t.Torrent.BytesCompleted()), "err", err)
		return
	}

	t.SetStatus(caffe.TorrentPending)
	t.Lock()
	t.Torrent = tt
	t.SetStart(mclock.Now())
	t.Unlock()

	if err := tm.Pending(t); err == nil {
		tm.recovery.Add(1)
		tm.stops.Add(-1)
	}
}

// handleTrackerUpdate encapsulates the logic for updating global trackers.
func (tm *TorrentManager) handleTrackerUpdate(timer *time.Timer) {
	tm.wg.Add(1)
	go func() {
		defer tm.wg.Done()

		const oneHour = time.Hour
		const oneDay = time.Hour * 24

		score, health, err := tm.updateGlobalTrackers()
		if err == nil && score > wormhole.CAP && health > 0.66 {
			timer.Reset(oneDay)
		} else {
			log.Warn("Network weak, rescan one hour later", "score", score, "health", health, "err", err)
			timer.Reset(oneHour)
		}
	}()
}

func (tm *TorrentManager) pendingLoop() {
	//defer tm.wg.Done()

	sub := tm.taskEvent.Subscribe(pendingEvent{})
	defer sub.Unsubscribe()

	for {
		select {
		case ev := <-sub.Chan():
			if ev == nil {
				continue
			}

			m, ok := ev.Data.(pendingEvent)
			if !ok {
				continue
			}
			t := m.T

			if t.Torrent.Info() != nil {
				// Already got info (recovery)
				t.AddTrackersOnce(tm.globalTrackers)
				tm.activeTor(t)
				continue
			}

			tm.pends.Add(1)
			tm.wg.Add(1)
			go func() {
				defer tm.pends.Add(-1)
				defer tm.wg.Done()

				tm.handlePendingTorrent(t)
			}()

		case <-tm.closeAll:
			log.Info("Pending seed loop closed")
			return
		}
	}
}

func (tm *TorrentManager) handlePendingTorrent(t *caffe.Torrent) {
	//defer func() {
	//tm.wg.Done()
	//tm.pends.Add(-1)
	//}()

	timeout := (10 + time.Duration(tm.slot&9)) * time.Minute
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	timer := time.NewTimer(5 * time.Second)
	defer timer.Stop()

	addedTrackers := false

	for {
		select {
		case <-t.Torrent.GotInfo():
			log.Debug("Searching", "ih", t.InfoHash(), "elapsed", common.PrettyDuration(time.Duration(mclock.Now())-time.Duration(t.Birth())), "wait", tm.pends.Load())
			tm.activeTor(t)
			return

		case <-t.Closed():
			return

		case <-tm.closeAll:
			return

		case <-ctx.Done():
			tm.Dropping(t.InfoHash())
			return

		case <-timer.C:
			if !addedTrackers {
				log.Debug("Adding global trackers to pending torrent",
					"infoHash", t.InfoHash(),
					"pending", tm.pends.Load(),
				)
				t.AddTrackersOnce(tm.globalTrackers)
				addedTrackers = true
			}
		}
	}
}

func (tm *TorrentManager) activeTor(t *caffe.Torrent) error {
	if b, err := bencode.Marshal(t.Torrent.Info()); err == nil {
		if tm.kvdb != nil && tm.kvdb.Get([]byte(SEED_PRE+t.InfoHash())) == nil {
			if tm.mode != params.LAZY {
				tm.wg.Add(1)
				go func() {
					defer tm.wg.Done()
					t.WriteTorrent()
				}()
			}
			tm.kvdb.Set([]byte(SEED_PRE+t.InfoHash()), b)
		}
	} else {
		log.Error("Meta info marshal failed", "ih", t.InfoHash(), "err", err)
		tm.Dropping(t.InfoHash())
		return err
	}

	if err := t.Start(); err != nil {
		tm.Dropping(t.InfoHash())
		log.Error("Nas start failed", "ih", t.InfoHash(), "err", err)
		return err
	}

	log.Debug("Meta found", "ih", t.InfoHash(), "len", t.Length())
	if (params.IsGood(t.InfoHash()) && tm.mode != params.LAZY) || tm.mode == params.FULL {
		t.SetBytesRequested(t.Length()) // request bytes fix after meta information got
	}

	return tm.Running(t)
}

func (tm *TorrentManager) finish(t *caffe.Torrent) error {
	t.Lock()
	defer t.Unlock()

	dataPath := filepath.Join(tm.dataDir, t.InfoHash())
	tmpPath := filepath.Join(params.DefaultTmpPath, t.InfoHash())

	// If data already exists, seed directly
	if _, err := os.Stat(dataPath); err == nil {
		return tm.Seeding(t)
	}

	// Try creating a symlink
	if err := os.Symlink(tmpPath, dataPath); err != nil {
		log.Warn("Symbolic link creation failed", "ih", t.InfoHash(), "tmpPath", tmpPath, "err", err)
		return err
	}

	return tm.Seeding(t)
}

/*
	func (tm *TorrentManager) dur() uint64 {
		return tm.seconds.Load()
	}

	func (tm *TorrentManager) cost(s uint64) {
		tm.seconds.Add(s)
	}
*/
func (tm *TorrentManager) activeLoop() {
	const (
		shortInterval = 100 * time.Millisecond
		longInterval  = time.Duration(params.QueryTimeInterval) * time.Minute
	)

	ticker := time.NewTicker(shortInterval)
	statusTicker := time.NewTicker(longInterval)
	defer func() {
		ticker.Stop()
		statusTicker.Stop()
		//tm.wg.Done()
	}()

	sub := tm.taskEvent.Subscribe(runningEvent{})
	defer sub.Unsubscribe()

	var counter int64

	for {
		select {
		case ev := <-sub.Chan():
			if ev == nil {
				continue
			}
			m, ok := ev.Data.(runningEvent)
			if !ok {
				continue
			}
			t := m.T
			if t.Dirty() {
				log.Debug("Leech", "ih", t.InfoHash(), "request", t.BytesRequested(), "total", t.Length())
				t.Leech()
			}

			n := tm.blockCaculate(t.Torrent.Length())
			if n < 300 {
				n = 300
			}
			n += rand.Int63n(300)
			if tm.mode == params.FULL {
				n *= 2
			}

			tm.actives.Add(1)
			tm.wg.Add(1)
			go func() {
				defer tm.wg.Done()
				tm.monitorActiveTorrent(t.InfoHash(), n)
			}()

		case <-statusTicker.C:
			log.Info("Fs status",
				"pending", tm.pends.Load(),
				"downloading", tm.actives.Load(),
				"seeding", tm.seeds.Load(),
				"stopping", tm.stops.Load(),
				"all", tm.torrents.Len(),
				"recovery", tm.recovery.Load(),
				"metrics", common.PrettyDuration(tm.updates),
			)

		case <-ticker.C:
			var workers errgroup.Group

			tm.torrents.Range(func(ih string, t *caffe.Torrent) bool {
				if !t.Running() {
					return true
				}
				if t.Torrent.BytesMissing() == 0 {
					log.Trace("Finish", "ih", ih)
					workers.Go(func() error { return tm.finish(t) })
				} else if t.Dirty() {
					log.Debug("Request bytes leech", "ih", ih, "request", t.BytesRequested())
					workers.Go(func() error { return t.Leech() })
				}

				if counter%60 == 0 {
					log.Debug("All torrents print", "ih", ih, "request", t.BytesRequested(), "complete", t.Torrent.BytesCompleted(), "total", t.Length(), "status", t.Status())
				}
				return true
			})

			if err := workers.Wait(); err != nil {
				log.Warn("Leech error", "err", err)
			}
			counter++

		case <-tm.closeAll:
			log.Info("Active seed loop closed")
			return
		}
	}
}

func (tm *TorrentManager) monitorActiveTorrent(ih string, intervalSec int64) {
	//defer tm.wg.Done()

	t := tm.getTorrent(ih)
	if t == nil {
		log.Error("Nil torrent on start", "ih", ih)
		return
	}

	timer := time.NewTimer(time.Duration(intervalSec) * time.Second)
	defer timer.Stop()

	for {
		select {
		case <-timer.C:
			if t == nil {
				log.Error("Nil torrent during loop", "ih", ih)
				return
			}

			switch {
			case t.Cited() == 0:
				if t.Paused() || t.IsSeeding() || tm.mode == params.LAZY {
					tm.Dropping(ih)
					return
				}
				t.CitedDec()
				log.Info("File can't be dropped for leeching",
					"ih", ih,
					"request", t.BytesRequested(),
					"complete", t.Torrent.BytesCompleted(),
					"total", t.Length(),
					"status", t.Status(),
				)

			case t.Cited() < 0:
				tm.Dropping(ih)
				log.Info("File timeout dropped",
					"ih", ih,
					"request", t.BytesRequested(),
					"complete", t.Torrent.BytesCompleted(),
					"total", t.Length(),
					"status", t.Status(),
				)
				return

			default:
				t.CitedDec()
				log.Debug("Seed cited decreased",
					"ih", ih,
					"cited", t.Cited(),
					"interval", intervalSec,
					"status", t.Status(),
					"elapsed", common.PrettyDuration(time.Duration(mclock.Now())-time.Duration(t.Birth())),
				)
			}

			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
			timer.Reset(time.Duration(intervalSec) * time.Second)

		case <-tm.closeAll:
			log.Info("Torrent manager closed", "ih", ih)
			return

		case <-t.Closed():
			log.Info("Torrent closed", "ih", ih)
			return
		}
	}
}

func (tm *TorrentManager) seedingLoop() {
	//defer tm.wg.Done()

	sub := tm.taskEvent.Subscribe(seedingEvent{}, droppingEvent{})
	defer sub.Unsubscribe()

	for {
		select {
		case ev := <-sub.Chan():
			if ev == nil || ev.Data == nil {
				continue
			}

			switch m := ev.Data.(type) {
			case seedingEvent:
				t := m.T
				if t.Seed() {
					tm.actives.Add(-1)
					tm.seeds.Add(1)

					evn := caffe.TorrentEvent{S: t.Status()}
					t.Mux().Post(evn)
				}

			case droppingEvent:
				ih := m.S
				t := tm.getTorrent(ih)
				if t != nil {
					tm.dropTorrent(t)
					elapsed := time.Duration(mclock.Now()) - time.Duration(t.Birth())
					log.Debug("Seed has been dropped",
						"ih", ih,
						"cited", t.Cited(),
						"status", t.Status(),
						"elapsed", common.PrettyDuration(elapsed),
					)
				} else {
					log.Warn("Drop seed not found", "ih", ih)
				}

			default:
				log.Warn("Unhandled event type in seedingLoop", "type", reflect.TypeOf(ev.Data))
			}

		case <-tm.closeAll:
			log.Info("Seeding loop closed")
			return
		}
	}
}

/*func (tm *TorrentManager) droppingLoop() {
	defer tm.wg.Done()
	sub := tm.taskEvent.Subscribe(droppingEvent{})
	defer sub.Unsubscribe()
	for {
		select {
		case ev := <-sub.Chan():
			if m, ok := ev.Data.(droppingEvent); ok {
				ih := m.S
				if t := tm.getTorrent(ih); t != nil { //&& t.Ready() {
					tm.dropTorrent(t)

					elapsed := time.Duration(mclock.Now()) - time.Duration(t.Birth())
					log.Debug("Seed has been dropped", "ih", ih, "cited", t.Cited(), "status", t.Status(), "elapsed", common.PrettyDuration(elapsed))
				} else {
					log.Warn("Drop seed not found", "ih", ih)
				}
			}
		case <-tm.closeAll:
			log.Info("Dropping loop closed")
			return
		}
	}
}

func (tm *TorrentManager) Exists(ih string, rawSize uint64) (bool, uint64, mclock.AbsTime, error) {
	availableMeter.Mark(1)

	if !common.IsHexAddress(ih) {
		return false, 0, 0, errors.New("invalid infohash format")
	}

	ih = strings.TrimPrefix(strings.ToLower(ih), common.Prefix)

	if t := tm.getTorrent(ih); t == nil {
		dir := filepath.Join(tm.dataDir, ih)
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

	log.Debug("Get File", "dir", tm.dataDir, "key", key)

	if t := tm.getTorrent(infohash); t != nil {
		if !t.Ready() {
			return nil, ErrUnfinished
		}

		// Data protection when torrent is active
		t.RLock()
		defer t.RUnlock()

	}

	diskReadMeter.Mark(1)
	dir := filepath.Join(tm.dataDir, key)
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

func (tm *TorrentManager) TmpDataDir() string {
	return tm.tmpDataDir
}
