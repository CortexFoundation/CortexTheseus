package torrentfs

import (
	"bytes"
	"crypto/sha1"
	"fmt"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/anacrolix/missinggo/slices"
	"github.com/bradfitz/iter"
	"github.com/edsrzf/mmap-go"
	"io"
	"math"
	"math/rand"
	"os"
	"path"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/anacrolix/torrent"
	//	"net"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/mmap_span"
	"github.com/anacrolix/torrent/storage"
)

const (
	bucket                  = params.Bucket //it is best size is 1/3 full nodes
	updateTorrentChanBuffer = batch
	torrentChanSize         = 1024

	torrentPending = iota //2
	torrentPaused
	torrentRunning
	torrentSeeding
	torrentSeedingInQueue
)

type Torrent struct {
	*torrent.Torrent
	maxEstablishedConns int
	currentConns        int
	bytesRequested      int64
	bytesLimitation     int64
	bytesCompleted      int64
	bytesMissing        int64
	status              int
	infohash            string
	filepath            string
	cited               int64
	weight              int
	loop                int
	maxPieces           int
	isBoosting          bool
	start               mclock.AbsTime
}

const block = int64(params.PER_UPLOAD_BYTES)

func (tm *TorrentManager) GetLimitation(value int64) int64 {
	return ((value + block - 1) / block) * block
}

func (t *Torrent) BytesLeft() int64 {
	if t.bytesRequested < t.bytesCompleted {
		return 0
	}
	return t.bytesRequested - t.bytesCompleted
}

func (t *Torrent) InfoHash() string {
	return t.infohash
}

func (t *Torrent) ReloadFile(files []string, datas [][]byte, tm *TorrentManager) {
	if len(files) > 1 {
		err := os.Mkdir(path.Join(t.filepath, "data"), os.ModePerm)
		if err != nil {
			return
		}
	}
	log.Info("Try to boost files", "files", files)
	for i, filename := range files {
		filePath := path.Join(t.filepath, filename)
		f, err := os.Create(filePath)
		if err != nil {
			return
		}
		defer f.Close()
		log.Debug("Write file (Boost mode)", "path", filePath)
		if _, err := f.Write(datas[i]); err != nil {
			log.Error("Error while write data file", "error", err)
		}
	}
	mi, err := metainfo.LoadFromFile(path.Join(t.filepath, "torrent"))
	if err != nil {
		log.Error("Error while loading torrent", "Err", err)
		return
	}
	spec := torrent.TorrentSpecFromMetaInfo(mi)
	spec.Storage = storage.NewFile(t.filepath)
	spec.Trackers = append(spec.Trackers, tm.trackers...)
	if torrent, _, err := tm.client.AddTorrentSpec(spec); err == nil {
		var ss []string
		slices.MakeInto(&ss, mi.Nodes)
		tm.client.AddDHTNodes(ss)
		//<-torrent.GotInfo()
		//torrent.VerifyData()
		t.Torrent = torrent
		//	t.Pause()
	}
}

func (t *Torrent) ReloadTorrent(data []byte, tm *TorrentManager) {
	torrentPath := path.Join(t.filepath, "torrent")
	os.Remove(path.Join(t.filepath, ".torrent.bolt.db"))
	f, err := os.Create(torrentPath)
	if err != nil {
		log.Warn("Create torrent path failed", "path", torrentPath)
		return
	}
	defer f.Close()
	log.Debug("Write seed file (Boost mode)", "path", torrentPath)
	if _, err := f.Write(data); err != nil {
		log.Error("Error while write torrent file", "error", err)
		return
	}
	mi, err := metainfo.LoadFromFile(torrentPath)
	if err != nil {
		log.Error("Error while adding torrent", "Err", err)
		return
	}
	spec := torrent.TorrentSpecFromMetaInfo(mi)
	spec.Storage = storage.NewFile(t.filepath)
	spec.Trackers = append(spec.Trackers, tm.trackers...)
	if torrent, _, err := tm.client.AddTorrentSpec(spec); err == nil {
		var ss []string
		slices.MakeInto(&ss, mi.Nodes)
		tm.client.AddDHTNodes(ss)
		//<-torrent.GotInfo()
		//torrent.VerifyData()
		t.Torrent = torrent
		//t.Pause()
	}
}

/*func (t *Torrent) GetFile(subpath string) ([]byte, error) {
	if !t.IsAvailable() {
		return nil, errors.New(fmt.Sprintf("InfoHash %s not Available", t.infohash))
	}
	filepath := path.Join(t.filepath, subpath)
	if _, cfgErr := os.Stat(filepath); os.IsNotExist(cfgErr) {
		return nil, errors.New(fmt.Sprintf("File %s not Available", filepath))
	}
	data, data_err := ioutil.ReadFile(filepath)
	return data, data_err
}*/

var maxCited int64 = 1

func (t *Torrent) IsAvailable() bool {
	t.cited += 1
	if t.cited > maxCited {
		maxCited = t.cited
	}
	if t.Seeding() {
		return true
	}
	return false
}

//func (t *Torrent) HasTorrent() bool {
//	return t.status != torrentPending
//}

func (t *Torrent) WriteTorrent() {
	//log.Info("Write seed", "hash", t.infohash)
	if _, err := os.Stat(path.Join(t.filepath, "torrent")); err == nil {
		t.Pause()
		return
	}

	if f, err := os.Create(path.Join(t.filepath, "torrent")); err == nil {
		defer f.Close()
		log.Debug("Write seed file", "path", t.filepath)
		if err := t.Metainfo().Write(f); err == nil {
			t.Pause()
		} else {
			log.Warn("Write seed error", "err", err)
		}
	} else {
		log.Warn("Create Path error", "err", err)
	}
}

func (t *Torrent) SeedInQueue() {
	t.status = torrentSeedingInQueue
	if t.currentConns != 0 {
		t.currentConns = 0
		t.Torrent.SetMaxEstablishedConns(0)
	}
	t.Torrent.CancelPieces(0, t.Torrent.NumPieces())
	//t.Torrent.Drop()
}

func (t *Torrent) BoostOff() {
	t.isBoosting = false
}

func (t *Torrent) Seed() {
	if t.status == torrentSeeding {
		return
	}
	//t.status = torrentSeeding
	//if t.currentConns == 0 {
	//	t.currentConns = t.maxEstablishedConns
	//	t.Torrent.SetMaxEstablishedConns(t.currentConns)
	//}

	//t.Torrent.DownloadAll()
	if t.Torrent.Seeding() {
		t.status = torrentSeeding
		if t.currentConns == 0 {
			t.currentConns = t.maxEstablishedConns
			t.Torrent.SetMaxEstablishedConns(t.currentConns)
		}
		elapsed := time.Duration(mclock.Now()) - time.Duration(t.start)
		log.Info("Download success, seeding(s)", "hash", t.InfoHash(), "size", common.StorageSize(t.BytesCompleted()), "files", len(t.Files()), "pieces", t.Torrent.NumPieces(), "seg", len(t.Torrent.PieceStateRuns()), "cited", t.cited, "elapsed", elapsed)
	} else {
		t.Torrent.DownloadAll()
	}
}

func (t *Torrent) Seeding() bool {
	return (t.status == torrentSeeding ||
		t.status == torrentSeedingInQueue) && t.BytesMissing() == 0
}

func (t *Torrent) Pause() {
	if t.currentConns != 0 {
		t.currentConns = 0
		t.Torrent.SetMaxEstablishedConns(0)
	}
	if t.status != torrentPaused {
		t.status = torrentPaused
		t.maxPieces = 0
		t.Torrent.CancelPieces(0, t.Torrent.NumPieces())
		//t.Torrent.Drop()
	}
}

func (t *Torrent) Paused() bool {
	return t.status == torrentPaused
}

//func (t *Torrent) Length() int64 {
//	return t.bytesCompleted + t.bytesMissing
//}

//func (t *Torrent) NumPieces() int {
//	return t.Torrent.NumPieces()
//}

func (t *Torrent) Run(slot int) {
	limitPieces := int((t.bytesRequested*int64(t.Torrent.NumPieces()) + t.Length() - 1) / t.Length())
	if limitPieces > t.Torrent.NumPieces() {
		limitPieces = t.Torrent.NumPieces()
	}
	if t.currentConns == 0 {
		t.currentConns = t.maxEstablishedConns
		t.Torrent.SetMaxEstablishedConns(t.currentConns)
	}
	t.status = torrentRunning
	if limitPieces > t.maxPieces {
		t.maxPieces = limitPieces
		//t.Torrent.DownloadPieces(0, limitPieces)
		t.download(limitPieces, slot)
	}
}

func (t *Torrent) download(p, slot int) {
	//if p >= t.Torrent.NumPieces() {
	//	t.Torrent.DownloadAll()
	//	return
	//}

	var s, e int
	/*if mod == 0 {
		e = p
	} else if mod == 1 {
		s = (t.Torrent.NumPieces() - p) / 2
		e = (t.Torrent.NumPieces() + p) / 2
	} else if mod == 2 {
		if  t.Torrent.NumPieces() < mod {
			s = mod - t.Torrent.NumPieces()
		}
		if t.Torrent.NumPieces() < mod + p {
			s = t.Torrent.NumPieces() - p
		}
		s = mod
		e = s + p
	} else {
		s = t.Torrent.NumPieces() - p
		e = t.Torrent.NumPieces()
	}*/
	s = (t.Torrent.NumPieces() * slot) / bucket
	if t.Torrent.NumPieces() < s {
		s = s - t.Torrent.NumPieces()
	}
	if t.Torrent.NumPieces() < s+p {
		s = t.Torrent.NumPieces() - p
	}
	e = s + p
	log.Info("Download slot", "hash", t.infohash, "b", s, "e", e, "p", p, "t", t.Torrent.NumPieces(), "s", slot, "b", bucket)
	t.Torrent.DownloadPieces(s, e)
}

func (t *Torrent) Running() bool {
	return t.status == torrentRunning
}

func (t *Torrent) Finished() bool {
	//for _, file := range t.Files() {
	//	if file.BytesCompleted() <= 0 {
	//		return false
	//	}
	//log.Info("File", "hash", t.InfoHash(), "name", file.Path(), "complete", file.BytesCompleted(), "file", file)
	//}
	return t.bytesMissing == 0 && t.bytesRequested > 0 && t.bytesCompleted > 0
}

func (t *Torrent) Pending() bool {
	return t.status == torrentPending
}

type TorrentManager struct {
	client              *torrent.Client
	bytes               map[metainfo.Hash]int64
	torrents            map[metainfo.Hash]*Torrent
	seedingTorrents     map[metainfo.Hash]*Torrent
	activeTorrents      map[metainfo.Hash]*Torrent
	pendingTorrents     map[metainfo.Hash]*Torrent
	maxSeedTask         int
	maxEstablishedConns int
	maxActiveTask       int
	trackers            [][]string
	boostFetcher        *BoostDataFetcher
	DataDir             string
	TmpDataDir          string
	closeAll            chan struct{}
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
	id       uint64
	slot     int
	//bucket int
}

func (tm *TorrentManager) CreateTorrent(t *torrent.Torrent, requested int64, status int, ih metainfo.Hash) *Torrent {
	tt := &Torrent{
		t,
		tm.maxEstablishedConns, tm.maxEstablishedConns,
		requested,
		//int64(float64(requested) * expansionFactor),
		tm.GetLimitation(requested),
		0, 0, status,
		ih.String(),
		path.Join(tm.TmpDataDir, ih.String()),
		0, 1, 0, 0, false, mclock.Now(),
	}
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
	tm.torrents[ih] = torrent
	tm.lock.Unlock()
	tm.pendingChan <- torrent
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

func (tm *TorrentManager) SetTrackers(trackers []string) {
	tm.lock.Lock()
	defer tm.lock.Unlock()
	tm.trackers = append(tm.trackers, trackers)
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
		log.Trace("Seed was already existed. Skip", "InfoHash", ih.HexString())
		return t
	}
	TmpDir := path.Join(tm.TmpDataDir, ih.HexString())
	ExistDir := path.Join(tm.DataDir, ih.HexString())

	useExistDir := false
	if _, err := os.Stat(ExistDir); err == nil {
		log.Debug("Seeding from existing file.", "InfoHash", ih.HexString())
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
		spec.Trackers = append(spec.Trackers, tm.trackers...)
		if t, _, err := tm.client.AddTorrentSpec(spec); err == nil {
			var ss []string
			slices.MakeInto(&ss, mi.Nodes)
			tm.client.AddDHTNodes(ss)
			//<-t.GotInfo()
			//t.VerifyData()
			torrent := tm.CreateTorrent(t, BytesRequested, torrentPending, ih)
			//torrent.Pause() //SeedInQueue()
			return torrent
		} else {
			log.Warn("Create error")
		}
	} else {
		spec.Storage = storage.NewFile(TmpDir)
		/*for _, tracker := range tm.trackers {
			spec.Trackers = append(spec.Trackers, tracker)
		}*/
		spec.Trackers = append(spec.Trackers, tm.trackers...)
		if t, _, err := tm.client.AddTorrentSpec(spec); err == nil {
			var ss []string
			slices.MakeInto(&ss, mi.Nodes)
			tm.client.AddDHTNodes(ss)
			//<-t.GotInfo()
			//t.VerifyData()
			torrent := tm.CreateTorrent(t, BytesRequested, torrentPending, ih)
			//torrent.Pause()
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

	log.Trace("Get file from infohash", "InfoHash", ih.HexString())

	spec := &torrent.TorrentSpec{
		Trackers:    [][]string{},
		DisplayName: ih.String(),
		InfoHash:    ih,
		Storage:     storage.NewFile(dataPath),
	}

	//for _, tracker := range tm.trackers {
	//	spec.Trackers = append(spec.Trackers, tracker)
	//}
	spec.Trackers = append(spec.Trackers, tm.trackers...)
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
	log.Debug("Update seed", "InfoHash", ih, "bytes", BytesRequested)
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
func NewTorrentManager(config *Config, fsid uint64) *TorrentManager {
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

	//cfg.HeaderObfuscationPolicy.Preferred = true
	//cfg.HeaderObfuscationPolicy.RequirePreferred = true

	cfg.DataDir = config.DataDir
	//cfg.DisableEncryption = true
	//cfg.ExtendedHandshakeClientVersion = params.VersionWithMeta
	//listenAddr := &net.TCPAddr{}
	//log.Info("Torrent client listening on", "addr", listenAddr)
	//cfg.SetListenAddr(listenAddr.String())
	cfg.HTTPUserAgent = "Cortex"
	cfg.Seed = true
	//cfg.EstablishedConnsPerTorrent = 10
	//cfg.HalfOpenConnsPerTorrent = 5
	cfg.ListenPort = config.Port
	//cfg.DropDuplicatePeerIds = true
	//cfg.ListenHost = torrent.LoopbackListenHost
	//cfg.DhtStartingNodes = dht.GlobalBootstrapAddrs //func() ([]dht.Addr, error) { return nil, nil }
	//log.Info("Torrent client configuration", "config", cfg)
	cl, err := torrent.NewClient(cfg)
	if err != nil {
		log.Error("Error while create torrent client", "err", err)
		return nil
	}

	tmpFilePath := path.Join(config.DataDir, defaultTmpFilePath)
	/*if _, err := os.Stat(tmpFilePath); err == nil {
		err := os.Remove(tmpFilePath)
		if err != nil {
			log.Warn("Purge the current file path failed", "path", tmpFilePath, "err", err)
		}
	}*/

	if _, err := os.Stat(tmpFilePath); err != nil {
		err = os.Mkdir(tmpFilePath, os.FileMode(os.ModePerm))
		if err != nil {
			log.Error("Mkdir failed", "path", tmpFilePath)
			return nil
		}
	}

	TorrentManager := &TorrentManager{
		client:              cl,
		torrents:            make(map[metainfo.Hash]*Torrent),
		pendingTorrents:     make(map[metainfo.Hash]*Torrent),
		seedingTorrents:     make(map[metainfo.Hash]*Torrent),
		activeTorrents:      make(map[metainfo.Hash]*Torrent),
		bytes:               make(map[metainfo.Hash]int64),
		maxSeedTask:         config.MaxSeedingNum,
		maxActiveTask:       config.MaxActiveNum,
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
		id:       fsid,
		//bucket:1024
		slot: int(fsid % bucket),
	}

	if len(config.DefaultTrackers) > 0 {
		log.Debug("Tracker list", "trackers", config.DefaultTrackers)
		TorrentManager.SetTrackers(config.DefaultTrackers)
	}
	log.Info("Fs client initialized")

	//CurrentTorrentManager = TorrentManager
	//cl.WaitAll()
	return TorrentManager
}

func (tm *TorrentManager) Start() error {
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
			t.Seed()
			if len(tm.seedingTorrents) > tm.maxSeedTask {
				tm.seedingTask()
			}
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

func (tm *TorrentManager) mainLoop() {
	defer tm.wg.Done()
	for {
		select {
		case msg := <-tm.updateTorrent:
			meta := msg.(FlowControlMeta)
			if meta.IsCreate {
				counter := 0
				for {
					if t := tm.AddInfoHash(meta.InfoHash, int64(meta.BytesRequested)); t != nil {
						log.Debug("Seed [create] success", "hash", meta.InfoHash, "request", meta.BytesRequested)
						if int64(meta.BytesRequested) > 0 {
							tm.UpdateInfoHash(meta.InfoHash, int64(meta.BytesRequested))
						}
						break
					} else {
						if counter > 10 {
							break
						}
						log.Error("Seed [create] failed", "hash", meta.InfoHash, "request", meta.BytesRequested, "counter", counter)
						counter++
					}
				}
			} else {
				log.Debug("Seed [update] success", "hash", meta.InfoHash, "request", meta.BytesRequested)
				tm.UpdateInfoHash(meta.InfoHash, int64(meta.BytesRequested))
			}
		case <-tm.closeAll:
			return
		}
	}
}

const (
	loops = 30
)

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
	timer := time.NewTimer(time.Second * defaultTimerInterval)
	defer timer.Stop()
	for {
		select {
		case t := <-tm.pendingChan:
			tm.pendingTorrents[t.Torrent.InfoHash()] = t
		case <-timer.C:
			for _, t := range tm.pendingTorrents {
				ih := t.Torrent.InfoHash()
				if _, ok := BadFiles[t.infohash]; ok {
					continue
				}
				t.loop += 1
				if !t.Pending() {
					if len(tm.activeChan) < cap(tm.activeChan) {
						delete(tm.pendingTorrents, ih)
						t.loop = 0
						tm.activeChan <- t
					}
				} else if t.Torrent.Info() != nil {
					t.WriteTorrent()
				} else if t.loop > torrentWaitingTime/queryTimeInterval {
					if !t.isBoosting {
						t.loop = 0
						t.isBoosting = true
						go func(t *Torrent) {
							defer t.BoostOff()
							log.Info("Try to boost seed", "hash", t.infohash)
							if data, err := tm.boostFetcher.GetTorrent(t.infohash); err == nil {
								if t.Torrent.Info() != nil {
									log.Warn("Seed already exist", "hash", t.infohash)
									return
								}
								t.Torrent.Drop()
								t.ReloadTorrent(data, tm)

								bytesRequested := t.bytesRequested
								tm.UpdateTorrent(FlowControlMeta{
									InfoHash:       ih,
									BytesRequested: uint64(bytesRequested),
									IsCreate:       true,
								})
							} else {
								log.Warn("Boost failed", "hash", t.infohash, "err", err)
							}
						}(t)
					}
				} else {
					if t.loop%20 == 0 {
						log.Debug("Searching file", "hash", t.infohash, "status", t.status, "file", t, "info", t.Torrent.Info(), "loops", t.loop)
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
	timer := time.NewTimer(time.Second * defaultTimerInterval)
	defer timer.Stop()
	var total_size, current_size, counter, log_counter uint64
	for {
		counter++
		select {
		case t := <-tm.activeChan:
			tm.activeTorrents[t.Torrent.InfoHash()] = t
		case <-timer.C:
			for _, t := range tm.torrents {
				t.weight = 1 + int(t.cited*10/maxCited)
			}
			log_counter++
			var all, active_paused, active_wait, active_boost, active_running int
			var activeTorrents []*Torrent

			for _, t := range tm.activeTorrents {
				ih := t.Torrent.InfoHash()
				tm.lock.RLock()
				BytesRequested := int64(0)
				if tm.fullSeed {
					BytesRequested = t.BytesCompleted() + t.BytesMissing()
				} else {
					BytesRequested = tm.bytes[ih]
				}
				tm.lock.RUnlock()
				if t.bytesRequested < BytesRequested {
					t.bytesRequested = BytesRequested
					t.bytesLimitation = tm.GetLimitation(BytesRequested)
				}

				t.bytesCompleted = t.BytesCompleted()
				t.bytesMissing = t.BytesMissing()

				if t.Finished() {
					tm.lock.Lock()
					if _, err := os.Stat(path.Join(tm.DataDir, t.InfoHash())); err == nil {
						if len(tm.seedingChan) < cap(tm.seedingChan) {
							log.Debug("Path exist", "hash", t.Torrent.InfoHash(), "path", path.Join(tm.DataDir, t.InfoHash()))
							delete(tm.activeTorrents, ih)
							tm.seedingChan <- t
							t.loop = defaultSeedInterval / queryTimeInterval
							total_size += uint64(t.bytesCompleted)
							current_size += uint64(t.bytesCompleted)
						}
					} else {
						err := os.Symlink(
							path.Join(defaultTmpFilePath, t.InfoHash()),
							path.Join(tm.DataDir, t.InfoHash()),
						)
						if err != nil {
							err = os.Remove(
								path.Join(tm.DataDir, t.InfoHash()),
							)
							if err == nil {
								log.Debug("Fix path success", "hash", t.Torrent.InfoHash(), "size", t.bytesCompleted, "miss", t.bytesMissing, "loop", log_counter)
							}
						} else {
							if len(tm.seedingChan) < cap(tm.seedingChan) {
								delete(tm.activeTorrents, ih)
								tm.seedingChan <- t
								t.loop = defaultSeedInterval / queryTimeInterval
								total_size += uint64(t.bytesCompleted)
								current_size += uint64(t.bytesCompleted)
							}
						}
					}

					tm.lock.Unlock()
					continue
				}

				if t.bytesRequested == 0 {
					active_wait += 1
					continue
				}

				all += len(t.Torrent.PieceStateRuns())

				if t.bytesCompleted >= t.bytesLimitation {
					t.Pause()
					active_paused += 1
					if log_counter%20 == 0 {
						log.Info("[Pausing]", "hash", ih.String(), "complete", common.StorageSize(t.bytesCompleted), "quota", common.StorageSize(t.bytesRequested), "total", common.StorageSize(t.bytesMissing+t.bytesCompleted), "prog", math.Min(float64(t.bytesCompleted), float64(t.bytesRequested))/float64(t.bytesCompleted+t.bytesMissing), "seg", len(t.Torrent.PieceStateRuns()), "max", t.Torrent.NumPieces(), "status", t.status, "boost", t.isBoosting, "s", t.Seeding())
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
						go func(t *Torrent) {
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
								}
							}
							t.Torrent.Drop()
							t.ReloadFile(filepaths, filedatas, tm)
						}(t)
						active_boost += 1
						if log_counter%20 == 0 {
							log.Info("[Boosting]", "hash", ih.String(), "complete", common.StorageSize(t.bytesCompleted), "quota", common.StorageSize(t.bytesRequested), "total", common.StorageSize(t.bytesMissing+t.bytesCompleted), "prog", math.Min(float64(t.bytesCompleted), float64(t.bytesRequested))/float64(t.bytesCompleted+t.bytesMissing), "seg", len(t.Torrent.PieceStateRuns()), "max", t.Torrent.NumPieces(), "status", t.status, "boost", t.isBoosting)
						}
						continue
					}
				}

				if log_counter%20 == 0 {
					log.Info("[Downloading]", "hash", ih.String(), "complete", common.StorageSize(t.bytesCompleted), "quota", common.StorageSize(t.bytesRequested), "total", common.StorageSize(t.bytesMissing+t.bytesCompleted), "prog", math.Min(float64(t.bytesCompleted), float64(t.bytesRequested))/float64(t.bytesCompleted+t.bytesMissing), "seg", len(t.Torrent.PieceStateRuns()), "max", t.Torrent.NumPieces(), "status", t.status, "boost", t.isBoosting, "s", t.Seeding())
				}

				if t.bytesCompleted < t.bytesLimitation && !t.isBoosting {
					activeTorrents = append(activeTorrents, t)
				}
			}

			if len(activeTorrents) <= tm.maxActiveTask {
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
			}

			if counter >= loops {
				log.Info("Fs status", "pending", len(tm.pendingTorrents), "active", len(tm.activeTorrents), "wait", active_wait, "downloading", active_running, "paused", active_paused, "boost", active_boost, "seeding", len(tm.seedingTorrents), "pieces", all, "size", common.StorageSize(total_size), "speed_a", common.StorageSize(total_size/log_counter*queryTimeInterval).String()+"/s", "speed_b", common.StorageSize(current_size/counter*queryTimeInterval).String()+"/s", "channel", len(tm.updateTorrent), "slot", tm.slot)
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
			timer.Reset(time.Second * queryTimeInterval)
		case <-tm.closeAll:
			log.Info("Active seed loop closed")
			return
		}
	}
}

func (tm *TorrentManager) seedingTask() error {
	//nSeed := 0
	//if len(tm.seedingTorrents) <= tm.maxSeedTask {
	//	for _, t := range tm.seedingTorrents {
	//		t.Seed()
	//		t.loop = 0
	//		nSeed++
	//	}
	//} else {
	var totalWeight int = 0
	var nSeedTask int = tm.maxSeedTask
	for _, t := range tm.seedingTorrents {
		if t.loop == 0 {
			totalWeight += t.weight
		} else if t.status == torrentSeeding {
			nSeedTask -= 1
		}
	}

	for _, t := range tm.seedingTorrents {
		if t.loop > 0 {
			t.loop -= 1
		} else {
			t.loop = defaultSeedInterval / queryTimeInterval
			prob := float32(t.weight) * float32(nSeedTask) / float32(totalWeight)
			if rand.Float32() < prob {
				t.Seed()
				//			nSeed++
			} else {
				t.SeedInQueue()
			}
		}
	}
	//}

	//return nSeed
	return nil
}
