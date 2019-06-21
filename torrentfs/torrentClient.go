package torrentfs

import (
	"bytes"
	"errors"
	"crypto/sha1"
	"io/ioutil"
	"fmt"
	"github.com/anacrolix/missinggo/slices"
	"github.com/bradfitz/iter"
	"github.com/edsrzf/mmap-go"
	"io"
	"net"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/mmap_span"
	"github.com/anacrolix/torrent/storage"

	"github.com/anacrolix/dht"
)

const (
	defaultBytesLimitation          = 512 * 1024
	queryTimeInterval               = 3
	removeTorrentChanBuffer         = 16
	newTorrentChanBuffer            = 32
	updateTorrentChanBuffer         = 32
	expansionFactor         float64 = 1.2
	torrentPending     = 0
	torrentPaused      = 1
	torrentRunning     = 2
	torrentSeeding     = 3
	torrentSeedingInQueue = 4
	defaultTmpFilePath = ".tmp"
)

// Torrent ...
type Torrent struct {
	*torrent.Torrent
	bytesRequested  int64
	bytesLimitation int64
	bytesCompleted  int64
	bytesMissing    int64
	status          int
	infohash        string
	filepath        string
	priority        int64
}

func (t *Torrent) InfoHash() string {
	return t.infohash
}

func (t *Torrent) GetFile(subpath string) ([]byte, error) {
	if !t.IsAvailable() {
		return nil,  errors.New(fmt.Sprintf("InfoHash %s not Available", t.infohash))
	}
	filepath := path.Join(t.filepath, subpath)
	// fmt.Println("modelCfg = ", modelCfg)
	if _, cfgErr := os.Stat(filepath); os.IsNotExist(cfgErr) {
		return nil, errors.New(fmt.Sprintf("File %s not Available", filepath))
	}
	data, data_err := ioutil.ReadFile(filepath)
	return data, data_err
}

func NewTorrent(t *torrent.Torrent, requested int64, status int, infohash string, filepath string) *Torrent {
	return &Torrent{t, requested, int64(float64(requested) * expansionFactor), 0, 0, status, infohash, filepath, 0}
}

func (t *Torrent) IsAvailable() bool {
	if (t.status == torrentPending) {
		return false
	}
	return t.bytesMissing == 0
}

func (t *Torrent) HasTorrent() bool {
	return t.status != torrentPending
}

func (t *Torrent) GetTorrent() {
	<-t.GotInfo()
	if t.status != torrentPending {
		return
	}

	// log.Debug("Torrent gotInfo finished")
	f, _ := os.Create(path.Join(t.filepath, "torrent"))
	log.Trace("Write torrent file", "path", t.filepath)
	if err := t.Metainfo().Write(f); err != nil {
		log.Error("Error while write torrent file", "error", err)
	}

	defer f.Close()
	t.status = torrentPaused
}

func (t *Torrent) Seed() {
	t.Torrent.DownloadAll()
	t.status = torrentSeeding
}

func (t *Torrent) Seeding() bool {
	return t.status == torrentSeeding
}

// Pause ...
func (t *Torrent) Pause() {
	if t.status != torrentPaused {
		t.status = torrentPaused
		t.Torrent.Drop()
	}
}

// Paused ...
func (t *Torrent) Paused() bool {
	return t.status == torrentPaused
}

// Run ...
func (t *Torrent) Run() {
	if t.status == torrentRunning {
		return
	}
	t.Torrent.DownloadAll()
	t.status = torrentRunning
}

// Running ...
func (t *Torrent) Running() bool {
	return t.status == torrentRunning
}

// Pending ...
func (t *Torrent) Pending() bool {
	return t.status == torrentPending
}

// TorrentManager ...
type TorrentManager struct {
	client        *torrent.Client
	torrents      map[metainfo.Hash]*Torrent
	trackers      [][]string
	DataDir       string
	TmpDataDir    string
	closeAll      chan struct{}
	newTorrent    chan interface{}
	removeTorrent chan metainfo.Hash
	updateTorrent chan interface{}
	halt          bool
	mu            sync.Mutex
	lock          sync.RWMutex
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
}

func (tm *TorrentManager) Close() error {
	close(tm.closeAll)
	log.Info("Torrent Download Manager Closed")
	return nil
}

func (tm *TorrentManager) NewTorrent(input interface{}) error {
	fmt.Println("NewTorrent", input.(FlowControlMeta))
	tm.newTorrent <- input
	return nil
}

func (tm *TorrentManager) RemoveTorrent(input metainfo.Hash) error {
	tm.removeTorrent <- input
	return nil
}

func (tm *TorrentManager) UpdateTorrent(input interface{}) error {
	tm.updateTorrent <- input
	return nil
}

func isMagnetURI(uri string) bool {
	return strings.HasPrefix(uri, "magnet:?xt=urn:btih:")
}

func GetMagnetURI(infohash metainfo.Hash) string {
	return "magnet:?xt=urn:btih:" + infohash.String()
}

func (tm *TorrentManager) SetTrackers(trackers []string) {
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

func (tm *TorrentManager) AddTorrent(filePath string, BytesRequested int64) {
	if _, err := os.Stat(filePath); err != nil {
		return
	}
	mi, err := metainfo.LoadFromFile(filePath)
	if err != nil {
		log.Error("Error while adding torrent", "Err", err)
		return
	}
	spec := torrent.TorrentSpecFromMetaInfo(mi)
	ih := spec.InfoHash
	log.Debug("Get torrent from local file", "InfoHash", ih.HexString())

	if tm.GetTorrent(ih) != nil {
		log.Debug("Torrent was already existed. Skip", "InfoHash", ih.HexString())
		return
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
		spec.Storage = storage.NewFile(ExistDir)

		for _, tracker := range tm.trackers {
			spec.Trackers = append(spec.Trackers, tracker)
		}
		t, _, _ := tm.client.AddTorrentSpec(spec)
		var ss []string
		slices.MakeInto(&ss, mi.Nodes)
		tm.client.AddDHTNodes(ss)
		torrent := NewTorrent(t, BytesRequested, torrentPaused, ih.String(), path.Join(tm.TmpDataDir, ih.HexString()))
		tm.SetTorrent(ih, torrent)
		<-t.GotInfo()
	  t.VerifyData()
		torrent.Seed()
	} else {
		spec.Storage = storage.NewFile(TmpDir)

		for _, tracker := range tm.trackers {
			spec.Trackers = append(spec.Trackers, tracker)
		}
		t, _, _ := tm.client.AddTorrentSpec(spec)
		var ss []string
		slices.MakeInto(&ss, mi.Nodes)
		tm.client.AddDHTNodes(ss)
		torrent := NewTorrent(t, BytesRequested, torrentPaused, ih.String(), path.Join(tm.TmpDataDir, ih.HexString()))
		tm.SetTorrent(ih, torrent)
		<-t.GotInfo()
		torrent.Run()
	}
}

func (tm *TorrentManager) AddInfoHash(ih metainfo.Hash, BytesRequested int64) {
	if tm.GetTorrent(ih) != nil {
		tm.UpdateInfoHash(ih, BytesRequested)
		return
	}

	dataPath := path.Join(tm.TmpDataDir, ih.HexString())
	torrentPath := path.Join(tm.TmpDataDir, ih.HexString(), "torrent")
	seedTorrentPath := path.Join(tm.DataDir, ih.HexString(), "torrent")
	
	if _, err := os.Stat(seedTorrentPath); err == nil {
		tm.AddTorrent(seedTorrentPath, BytesRequested)
		return
	} else if _, err := os.Stat(torrentPath); err == nil {
		tm.AddTorrent(torrentPath, BytesRequested)
		return
	}
	log.Debug("Get torrent from infohash", "InfoHash", ih.HexString())

	if tm.GetTorrent(ih) != nil {
		log.Warn("Torrent was already existed. Skip", "InfoHash", ih.HexString())
		//tm.mu.Unlock()
		return
	}
  
	spec := &torrent.TorrentSpec{
		Trackers: [][]string{},
		DisplayName: ih.String(),
		InfoHash: ih,
		Storage: storage.NewFile(dataPath),
	}

	for _, tracker := range tm.trackers {
		spec.Trackers = append(spec.Trackers, tracker)
	}
	log.Debug("Torrent specific info", "spec", spec)

	t, _, _ := tm.client.AddTorrentSpec(spec)
	torrent := NewTorrent(t, BytesRequested, torrentPending, ih.String(), path.Join(tm.TmpDataDir, ih.HexString()))
	tm.SetTorrent(ih, torrent)
	//tm.mu.Unlock()
	log.Debug("Torrent is waiting for gotInfo", "InfoHash", ih.HexString())
	go torrent.GetTorrent()
}

// UpdateInfoHash ...
func (tm *TorrentManager) UpdateInfoHash(ih metainfo.Hash, BytesRequested int64) {
	log.Debug("Update torrent", "InfoHash", ih, "bytes", BytesRequested)
	if t := tm.GetTorrent(ih); t != nil {
		if BytesRequested < t.bytesRequested {
			return
		}
		t.bytesRequested = BytesRequested
		if t.bytesRequested > t.bytesLimitation {
			t.bytesLimitation = int64(float64(BytesRequested) * expansionFactor)
		}
	}
	//tm.mu.Unlock()
}

// DropMagnet ...
func (tm *TorrentManager) DropMagnet(ih metainfo.Hash) bool {
	if t := tm.GetTorrent(ih); t != nil {
		t.Torrent.Drop()
		tm.lock.Lock()
		delete(tm.torrents, ih)
		tm.lock.Unlock()
		return true
	}
	return false
}

var CurrentTorrentManager *TorrentManager = nil
// NewTorrentManager ...
func NewTorrentManager(config *Config) *TorrentManager {
	cfg := torrent.NewDefaultClientConfig()
	cfg.DisableUTP = config.DisableUTP
	cfg.NoDHT = false
	cfg.DhtStartingNodes = dht.GlobalBootstrapAddrs
	cfg.DataDir = config.DataDir
	cfg.DisableEncryption = true
	cfg.ExtendedHandshakeClientVersion = params.VersionWithMeta
	listenAddr := &net.TCPAddr{}
	log.Info("Torrent client listening on", "addr", listenAddr)
	//cfg.SetListenAddr(listenAddr.String())
	cfg.HTTPUserAgent = "Cortex"
	cfg.Seed = true
	cfg.EstablishedConnsPerTorrent = 15
	cfg.HalfOpenConnsPerTorrent = 10
	log.Info("Torrent client configuration", "config", cfg)
	cl, err := torrent.NewClient(cfg)
	if err != nil {
		log.Error("Error while create torrent client", "err", err)
	}

	tmpFilePath := path.Join(config.DataDir, defaultTmpFilePath)
	if _, err := os.Stat(tmpFilePath); err == nil {
		os.Remove(tmpFilePath)
	}
	os.Mkdir(tmpFilePath, os.FileMode(os.ModePerm))

	TorrentManager := &TorrentManager{
		client:        cl,
		torrents:      make(map[metainfo.Hash]*Torrent),
		DataDir:       config.DataDir,
		TmpDataDir:    tmpFilePath,
		closeAll:      make(chan struct{}),
		newTorrent:    make(chan interface{}, newTorrentChanBuffer),
		removeTorrent: make(chan metainfo.Hash, removeTorrentChanBuffer),
		updateTorrent: make(chan interface{}, updateTorrentChanBuffer),
	}

	if len(config.DefaultTrackers) > 0 {
		log.Info("Tracker list", "trackers", config.DefaultTrackers)
		TorrentManager.SetTrackers(config.DefaultTrackers)
		TorrentManager.SetTrackers(params.MainnetTrackers)
	}
	log.Info("Torrent client initialized")

	CurrentTorrentManager = TorrentManager
	return TorrentManager
}

func (tm *TorrentManager) Start() error {

	go tm.mainLoop()
	go tm.listenTorrentProgress()

	return nil
}

func (tm *TorrentManager) mainLoop() {
	for {
		select {
		case msg := <-tm.newTorrent:
		  meta := msg.(FlowControlMeta)
			log.Debug("TorrentManager", "newTorrent", meta.InfoHash.String())
			go tm.AddInfoHash(meta.InfoHash, int64(meta.BytesRequested))
		case torrent := <-tm.removeTorrent:
			go tm.DropMagnet(torrent)
		case msg := <-tm.updateTorrent:
		  meta := msg.(FlowControlMeta)
			go tm.UpdateInfoHash(meta.InfoHash, int64(meta.BytesRequested))
		case <-tm.closeAll:
			tm.halt = true
			tm.client.Close()
			return
		}
	}
}

const (
	loops = 5
)

func (tm *TorrentManager) listenTorrentProgress() {
	var counter uint64
	for counter = 0; ; counter++ {
		if tm.halt {
			return
		}
		var seeding_n int = 0
		var pending_n int = 0
		var progress_n int = 0
		var pause_n int = 0
		var pendingTorrents []*Torrent
		var activeTorrents []*Torrent

		for _, t := range tm.torrents {
			if t.Pending() {
				pendingTorrents = append(pendingTorrents, t)
			} else {
				activeTorrents = append(activeTorrents, t)
			}
		}

		for _, t := range activeTorrents {
			t.bytesCompleted = t.BytesCompleted()
			t.bytesMissing = t.BytesMissing()
			if t.Seeding() {
				if counter >= loops {
					log.Trace("Torrent seeding",
						"InfoHash", t.InfoHash(),
						"completed", t.bytesCompleted,
						"total", t.bytesCompleted+t.bytesMissing,
						"seeding", t.Torrent.Seeding(),
					)
					if t.Torrent.Seeding() {
						seeding_n += 1
					}
				}
			} else if !t.Pending() {
				if t.bytesMissing == 0 {
					os.Symlink(
						path.Join(defaultTmpFilePath, t.InfoHash()),
						path.Join(tm.DataDir, t.InfoHash()),
					)
					t.Seed()
				} else if t.bytesCompleted >= t.bytesLimitation {
					t.Pause()
				} else if t.bytesCompleted < t.bytesLimitation {
					t.Run()
				}
				if counter >= loops {
					log.Trace("Torrent progress",
						"InfoHash", t.InfoHash(),
						"completed", t.bytesCompleted,
						"requested", t.bytesLimitation,
						"total", t.bytesCompleted+t.bytesMissing,
						"status", t.status)
					if t.bytesCompleted != 0 {
						if t.bytesCompleted >= t.bytesLimitation {
							pause_n += 1
						} else {
							progress_n += 1
						}
					}
				}
			} 
		}
		if counter >= loops {
			log.Info("Torrent tasks working status", "pause", pause_n, "progress", progress_n, "pending", pending_n, "seeding", seeding_n)
			counter = 0
		}
		time.Sleep(time.Second * queryTimeInterval)
	}
}
