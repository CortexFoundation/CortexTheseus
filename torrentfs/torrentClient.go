package torrentfs

import (
	"bytes"
	"crypto/sha1"
	//"errors"
	"fmt"
	//"github.com/anacrolix/missinggo/slices"
	"github.com/bradfitz/iter"
	"github.com/edsrzf/mmap-go"
	"io"
	//"io/ioutil"
	"math"
	"math/rand"
	//"net"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/mmap_span"
	"github.com/anacrolix/torrent/storage"
	//"github.com/anacrolix/dht/v2"
)

const (
	//removeTorrentChanBuffer = 1
	updateTorrentChanBuffer = 2048

	torrentPending = iota //2
	torrentPaused
	torrentRunning
	torrentSeeding
	torrentSeedingInQueue
)

// Torrent ...
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
}

const block = int64(params.PER_UPLOAD_BYTES)

func GetLimitation(value int64) int64 {
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
	//  os.Remove(path.Join(t.filepath, ".torrent.bolt.db"))
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
		//f.Close()
	}
	mi, err := metainfo.LoadFromFile(path.Join(t.filepath, "torrent"))
	if err != nil {
		log.Error("Error while loading torrent", "Err", err)
		return
	}
	spec := torrent.TorrentSpecFromMetaInfo(mi)
	spec.Storage = storage.NewFile(t.filepath)
	for _, tracker := range tm.trackers {
		spec.Trackers = append(spec.Trackers, tracker)
	}
	torrent, _, err := tm.client.AddTorrentSpec(spec)
	if err != nil {
		return
	}
	<-torrent.GotInfo()
	torrent.VerifyData()
	t.Torrent = torrent
	t.Pause()
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
	log.Debug("Write torrent file (Boost mode)", "path", torrentPath)
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
	for _, tracker := range tm.trackers {
		spec.Trackers = append(spec.Trackers, tracker)
	}
	torrent, _, err := tm.client.AddTorrentSpec(spec)
	if err != nil {
		return
	}
	<-torrent.GotInfo()
	torrent.VerifyData()
	t.Torrent = torrent
	t.Pause()
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

func (t *Torrent) HasTorrent() bool {
	return t.status != torrentPending
}

func (t *Torrent) WriteTorrent() {
	f, err := os.Create(path.Join(t.filepath, "torrent"))
	if err != nil {
		return
	}
	defer f.Close()
	log.Debug("Write torrent file", "path", t.filepath)
	if err := t.Metainfo().Write(f); err != nil {
		log.Error("Error while write torrent file", "error", err)
		return
	}

	t.Pause()
}

func (t *Torrent) SeedInQueue() {
	if t.currentConns != 0 {
		t.currentConns = 0
		t.Torrent.SetMaxEstablishedConns(0)
	}
	t.Torrent.CancelPieces(0, t.Torrent.NumPieces())
	t.status = torrentSeedingInQueue
}

func (t *Torrent) BoostOff() {
	t.isBoosting = false
}

func (t *Torrent) Seed() {
	if t.status == torrentSeeding {
		return
	}

	if t.currentConns == 0 {
		t.currentConns = t.maxEstablishedConns
		t.Torrent.SetMaxEstablishedConns(t.currentConns)
	}

	t.Torrent.DownloadAll()
	t.status = torrentSeeding
	log.Info("Download success, seeding(s)", "hash", t.InfoHash(), "size", common.StorageSize(t.BytesCompleted()), "files", len(t.Files()), "pieces", t.Torrent.NumPieces(), "seg", len(t.Torrent.PieceStateRuns()), "cited", t.cited)
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
		t.maxPieces = 0
		t.Torrent.CancelPieces(0, t.Torrent.NumPieces())
		t.status = torrentPaused
	}
}

func (t *Torrent) Paused() bool {
	return t.status == torrentPaused
}

func (t *Torrent) Length() int64 {
	return t.bytesCompleted + t.bytesMissing
}

func (t *Torrent) NumPieces() int {
	return t.Torrent.NumPieces()
}

func (t *Torrent) Run() {
	maxPieces := int((t.bytesRequested*int64(t.NumPieces()) + t.Length() - 1) / t.Length())
	if maxPieces > t.NumPieces() {
		maxPieces = t.NumPieces()
	}
	if t.currentConns == 0 {
		t.currentConns = t.maxEstablishedConns
		t.Torrent.SetMaxEstablishedConns(t.currentConns)
	}
	t.status = torrentRunning
	if maxPieces > t.maxPieces {
		t.maxPieces = maxPieces
		t.Torrent.DownloadPieces(0, maxPieces)
	}
}

func (t *Torrent) Running() bool {
	return t.status == torrentRunning
}

func (t *Torrent) Finished() bool {
	return t.bytesMissing == 0
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
	lock      sync.RWMutex
	wg        sync.WaitGroup
	closeOnce sync.Once
}

func (tm *TorrentManager) CreateTorrent(t *torrent.Torrent, requested int64, status int, ih metainfo.Hash) *Torrent {
	tt := &Torrent{
		t,
		tm.maxEstablishedConns, tm.maxEstablishedConns,
		requested,
		//int64(float64(requested) * expansionFactor),
		GetLimitation(requested),
		0, 0, status,
		ih.String(),
		path.Join(tm.TmpDataDir, ih.String()),
		0, 1, 0, 0, false,
	}
	tm.SetTorrent(ih, tt)
	return tt
}

func (tm *TorrentManager) GetTorrent(ih metainfo.Hash) *Torrent {
	tm.lock.RLock()
	defer tm.lock.RUnlock()
	torrent, ok := tm.torrents[ih]
	if !ok {
		return nil
	}
	return torrent
}

func (tm *TorrentManager) SetTorrent(ih metainfo.Hash, torrent *Torrent) {
	tm.lock.Lock()
	defer tm.lock.Unlock()
	tm.torrents[ih] = torrent
	tm.pendingTorrents[ih] = torrent
}

func (tm *TorrentManager) Close() error {
	log.Info("Torrent Download Manager Closing")
	close(tm.closeAll)
	tm.wg.Wait()
	tm.dropAll()
	/*tm.wg.Add(1)
	tm.closeOnce.Do(func() {
		defer tm.wg.Done()
		tm.dropAll()
	})
	tm.wg.Wait()*/
	log.Info("Torrent Download Manager Closed")
	return nil
}

func (tm *TorrentManager) dropAll() {
	tm.lock.Lock()
	tm.lock.Unlock()
	defer tm.client.Close()
	for _, t := range tm.torrents {
		stats := t.Stats()
		log.Info("torrent statics", "hash", t.infohash, "total", stats.TotalPeers, "pending", stats.PendingPeers, "active", stats.ActivePeers, "seeder", stats.ConnectedSeeders, "half", stats.HalfOpenPeers)
		t.Drop()
	}
}

//func (tm *TorrentManager) RemoveTorrent(input metainfo.Hash) error {
//	tm.removeTorrent <- input
//	return nil
//}

func (tm *TorrentManager) UpdateTorrent(input interface{}) error {
	go func() { tm.updateTorrent <- input }()
	//tm.updateTorrent <- input
	return nil
}

func isMagnetURI(uri string) bool {
	return strings.HasPrefix(uri, "magnet:?xt=urn:btih:")
}

func GetMagnetURI(infohash metainfo.Hash) string {
	return "magnet:?xt=urn:btih:" + infohash.String()
}

func (tm *TorrentManager) UpdateDynamicTrackers(trackers []string) {
	tm.lock.Lock()
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

	tm.lock.Unlock()
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

func verifyTorrent(info *metainfo.Info, root string) error {
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
	log.Trace("Get torrent from local file", "InfoHash", ih.HexString())

	if t := tm.GetTorrent(ih); t != nil {
		log.Trace("Torrent was already existed. Skip", "InfoHash", ih.HexString())
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
		if err := verifyTorrent(&info, ExistDir); err != nil {
			log.Warn("torrent failed verification:", "err", err)
		} else {
			useExistDir = true
		}
	}

	if useExistDir {
		log.Trace("existing dir", "dir", ExistDir)
		spec.Storage = storage.NewFile(ExistDir)
		for _, tracker := range tm.trackers {
			spec.Trackers = append(spec.Trackers, tracker)
		}
		t, _, err := tm.client.AddTorrentSpec(spec)
		if err != nil {
			return nil
		}
		//var ss []string
		//slices.MakeInto(&ss, mi.Nodes)
		//tm.client.AddDHTNodes(ss)
		<-t.GotInfo()
		t.VerifyData()
		torrent := tm.CreateTorrent(t, BytesRequested, torrentPending, ih)
		torrent.Pause() //SeedInQueue()
		return torrent
	} else {
		spec.Storage = storage.NewFile(TmpDir)
		for _, tracker := range tm.trackers {
			spec.Trackers = append(spec.Trackers, tracker)
		}
		t, _, err := tm.client.AddTorrentSpec(spec)
		if err != nil {
			return nil
		}
		//var ss []string
		//slices.MakeInto(&ss, mi.Nodes)
		//tm.client.AddDHTNodes(ss)
		<-t.GotInfo()
		t.VerifyData()
		torrent := tm.CreateTorrent(t, BytesRequested, torrentPending, ih)
		torrent.Pause()
		return torrent
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
	//log.Info("Get torrent from infohash", "InfoHash", ih.HexString())

	spec := &torrent.TorrentSpec{
		Trackers:    [][]string{},
		DisplayName: ih.String(),
		InfoHash:    ih,
		Storage:     storage.NewFile(dataPath),
	}

	for _, tracker := range tm.trackers {
		spec.Trackers = append(spec.Trackers, tracker)
	}
	//log.Info("Torrent specific info", "spec", spec)

	t, _, err := tm.client.AddTorrentSpec(spec)
	if err != nil {
		return nil
	}
	tt := tm.CreateTorrent(t, BytesRequested, torrentPending, ih)
	//tm.mu.Unlock()
	//log.Info("Torrent is waiting for gotInfo", "InfoHash", ih.HexString())
	return tt
}

// UpdateInfoHash ...
func (tm *TorrentManager) UpdateInfoHash(ih metainfo.Hash, BytesRequested int64) {
	log.Debug("Update torrent", "InfoHash", ih, "bytes", BytesRequested)
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
func NewTorrentManager(config *Config) *TorrentManager {
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
	cfg.DisableUTP = true //config.DisableUTP
	//cfg.NoDHT = true//config.DisableDHT
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
	cfg.ListenPort = 0
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
		//updateTorrent:       make(chan interface{}),
	}

	if len(config.DefaultTrackers) > 0 {
		log.Info("Tracker list", "trackers", config.DefaultTrackers)
		TorrentManager.SetTrackers(config.DefaultTrackers)
	}
	log.Info("Torrent client initialized")

	//CurrentTorrentManager = TorrentManager
	//cl.WaitAll()
	return TorrentManager
}

func (tm *TorrentManager) Start() error {
	tm.wg.Add(1)
	go tm.mainLoop()
	tm.wg.Add(1)
	go tm.listenTorrentProgress()

	return nil
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
		//case torrent := <-tm.removeTorrent:
		//	tm.DropInfoHash(torrent)
		case msg := <-tm.updateTorrent:
			meta := msg.(FlowControlMeta)
			if meta.IsCreate {
				counter := 0
				for {
					if t := tm.AddInfoHash(meta.InfoHash, int64(meta.BytesRequested)); t != nil {
						log.Debug("Torrent [create] success", "hash", meta.InfoHash, "request", meta.BytesRequested)
						break
					} else {
						if counter > 10 {
							break
						}
						log.Error("Torrent [create] failed", "hash", meta.InfoHash, "request", meta.BytesRequested, "counter", counter)
						counter++
					}
				}
			} else {
				log.Debug("Torrent [update] success", "hash", meta.InfoHash, "request", meta.BytesRequested)
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

type ActiveTorrentList []*Torrent

func (s ActiveTorrentList) Len() int      { return len(s) }
func (s ActiveTorrentList) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s ActiveTorrentList) Less(i, j int) bool {
	return s[i].BytesLeft() > s[j].BytesLeft() || (s[i].BytesLeft() == s[j].BytesLeft() && s[i].weight > s[j].weight)
}

type seedingTorrentList []*Torrent

func (s seedingTorrentList) Len() int           { return len(s) }
func (s seedingTorrentList) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s seedingTorrentList) Less(i, j int) bool { return s[i].weight > s[j].weight }

func (tm *TorrentManager) listenTorrentProgress() {
	defer tm.wg.Done()
	var total_size, current_size, counter, log_counter uint64
	timer := time.NewTimer(time.Second * defaultTimerInterval)
	for counter = 0; ; counter++ {
		select {
		case <-timer.C:
			tm.lock.RLock()
			log_counter++

			for _, t := range tm.torrents {
				t.weight = 1 + int(t.cited*10/maxCited)
			}

			var pendingTorrents []*Torrent
			for _, t := range tm.pendingTorrents {
				pendingTorrents = append(pendingTorrents, t)
			}
			tm.lock.RUnlock()

			for _, t := range pendingTorrents {
				ih := t.Torrent.InfoHash()
				t.loop += 1
				if t.Seeding() {
					delete(tm.pendingTorrents, ih)
					tm.seedingTorrents[ih] = t
					t.loop = 0
				} else if !t.Pending() {
					delete(tm.pendingTorrents, ih)
					tm.activeTorrents[ih] = t
					t.loop = 0
				} else if t.Torrent.Info() != nil {
					t.WriteTorrent()
				} else if t.loop > torrentWaitingTime/queryTimeInterval {
					if !t.isBoosting {
						t.loop = 0
						t.isBoosting = true
						go func(t *Torrent) {
							defer t.BoostOff()
							log.Info("Try to boost torrent", "hash", t.infohash)
							if data, err := tm.boostFetcher.GetTorrent(t.infohash); err == nil {
								if t.Torrent.Info() != nil {
									log.Warn("Torrent already exist", "hash", t.infohash)
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
				}
			}

			var activeTorrents, activeTorrentsCandidate []*Torrent
			for _, t := range tm.activeTorrents {
				activeTorrentsCandidate = append(activeTorrentsCandidate, t)
			}

			var all, active_paused, active_wait, active_boost, active_running int
			for _, t := range activeTorrentsCandidate {
				ih := t.Torrent.InfoHash()
				tm.lock.RLock()
				BytesRequested := tm.bytes[ih]
				tm.lock.RUnlock()
				if t.bytesRequested < BytesRequested {
					t.bytesRequested = BytesRequested
					t.bytesLimitation = GetLimitation(BytesRequested)
				}

				t.bytesCompleted = t.BytesCompleted()
				t.bytesMissing = t.BytesMissing()

				if t.Finished() {
					tm.lock.Lock()
					err := os.Symlink(
						path.Join(defaultTmpFilePath, t.InfoHash()),
						path.Join(tm.DataDir, t.InfoHash()),
					)

					if err != nil {
						//log.Warn("Seeding path error", "hash", t.Torrent.InfoHash(), "size", t.bytesCompleted, "miss", t.bytesMissing, "loop", log_counter)
						err = os.Remove(
							path.Join(tm.DataDir, t.InfoHash()),
						)
						if err != nil {
							//	log.Warn("Fix path error", "hash", t.Torrent.InfoHash(), "size", t.bytesCompleted, "miss", t.bytesMissing, "loop", log_counter)
						} else {
							log.Debug("Fix path success", "hash", t.Torrent.InfoHash(), "size", t.bytesCompleted, "miss", t.bytesMissing, "loop", log_counter)
						}
					} else {
						delete(tm.activeTorrents, ih)
						tm.seedingTorrents[ih] = t
						t.Seed()
						t.loop = defaultSeedInterval / queryTimeInterval
						total_size += uint64(t.bytesCompleted)
						current_size += uint64(t.bytesCompleted)
					}
					tm.lock.Unlock()
					continue
				}

				if t.bytesRequested == 0 {
					if log_counter%20 == 0 {
						log.Debug("[Waiting]", "hash", ih.String(), "complete", common.StorageSize(t.bytesCompleted), "quota", common.StorageSize(t.bytesRequested), "total", common.StorageSize(t.bytesMissing+t.bytesCompleted), "prog", math.Min(float64(t.bytesCompleted), float64(t.bytesRequested))/float64(t.bytesCompleted+t.bytesMissing), "seg", len(t.Torrent.PieceStateRuns()), "max", t.NumPieces(), "status", t.status, "boost", t.isBoosting)
					}
					active_wait += 1
					continue
				}

				all += len(t.Torrent.PieceStateRuns())

				if t.bytesCompleted >= t.bytesLimitation {
					t.Pause()
					active_paused += 1
					if log_counter%20 == 0 {
						log.Info("[Pausing]", "hash", ih.String(), "complete", common.StorageSize(t.bytesCompleted), "quota", common.StorageSize(t.bytesRequested), "total", common.StorageSize(t.bytesMissing+t.bytesCompleted), "prog", math.Min(float64(t.bytesCompleted), float64(t.bytesRequested))/float64(t.bytesCompleted+t.bytesMissing), "seg", len(t.Torrent.PieceStateRuns()), "max", t.NumPieces(), "status", t.status, "boost", t.isBoosting)
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
							log.Trace("Try to boost files", "infohash", ih.String())
							if t.Files() != nil {
								filepaths := []string{}
								filedatas := [][]byte{}
								for _, file := range t.Files() {
									subpath := file.Path()
									if data, err := tm.boostFetcher.GetFile(ih.String(), subpath); err == nil {
										filedatas = append(filedatas, data)
										filepaths = append(filepaths, subpath)
									} else {
										continue
									}
								}
								t.Torrent.Drop()
								t.ReloadFile(filepaths, filedatas, tm)
							}
						}(t)
						active_boost += 1
						if log_counter%20 == 0 {
							log.Info("[Boosting]", "hash", ih.String(), "complete", common.StorageSize(t.bytesCompleted), "quota", common.StorageSize(t.bytesRequested), "total", common.StorageSize(t.bytesMissing+t.bytesCompleted), "prog", math.Min(float64(t.bytesCompleted), float64(t.bytesRequested))/float64(t.bytesCompleted+t.bytesMissing), "seg", len(t.Torrent.PieceStateRuns()), "max", t.NumPieces(), "status", t.status, "boost", t.isBoosting)
						}
						continue
					}
				}

				if log_counter%20 == 0 && t.bytesCompleted > 0 {
					log.Info("[Downloading]", "hash", ih.String(), "complete", common.StorageSize(t.bytesCompleted), "quota", common.StorageSize(t.bytesRequested), "total", common.StorageSize(t.bytesMissing+t.bytesCompleted), "prog", math.Min(float64(t.bytesCompleted), float64(t.bytesRequested))/float64(t.bytesCompleted+t.bytesMissing), "seg", len(t.Torrent.PieceStateRuns()), "max", t.NumPieces(), "status", t.status, "boost", t.isBoosting)
				}

				if t.bytesCompleted < t.bytesLimitation && !t.isBoosting {
					activeTorrents = append(activeTorrents, t)
				}
			}

			if len(activeTorrents) <= tm.maxActiveTask {
				for _, t := range activeTorrents {
					t.Run()
					active_running += 1
				}
			} else {
				sort.Stable(ActiveTorrentList(activeTorrents))
				for i := 0; i < tm.maxActiveTask; i++ {
					activeTorrents[i].Run()
					active_running += 1
				}
				for i := tm.maxActiveTask; i < len(activeTorrents); i++ {
					activeTorrents[i].Pause()
					active_paused += 1
				}
			}

			nSeed := tm.seedingTask()

			if counter >= loops {
				log.Info("Torrent status", "pending", len(tm.pendingTorrents), "active", len(tm.activeTorrents), "wait", active_wait, "downloading", active_running, "paused", active_paused, "boost", active_boost, "seeding", nSeed, "queue", len(tm.seedingTorrents)-nSeed, "pieces", all, "size", common.StorageSize(total_size), "speed_a", common.StorageSize(total_size/log_counter*queryTimeInterval).String()+"/s", "speed_b", common.StorageSize(current_size/counter*queryTimeInterval).String()+"/s")
				counter = 0
				current_size = 0
			}
			timer.Reset(time.Second * queryTimeInterval)
		case <-tm.closeAll:
			log.Info("Listen torrent progress closed")
			return
		}
	}
}

func (tm *TorrentManager) seedingTask() int {
	nSeed := 0
	if len(tm.seedingTorrents) <= tm.maxSeedTask {
		for _, t := range tm.seedingTorrents {
			t.Seed()
			t.loop = 0
			nSeed++
		}
	} else {
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
					nSeed++
				} else {
					t.SeedInQueue()
				}
			}
		}
	}

	return nSeed
}
