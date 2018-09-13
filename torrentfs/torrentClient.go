package torrentfs

import (
	"net"
	"os"
	"path"
	"strings"
	"sync"
	"time"

	"github.com/anacrolix/missinggo/slices"
	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/storage"
	"github.com/ethereum/go-ethereum/log"
)

const (
	defaultBytesLimitation          = 512 * 1024
	queryTimeInterval               = 5
	removeTorrentChanBuffer         = 16
	newTorrentChanBuffer            = 32
	updateTorrentChanBuffer         = 32
	expansionFactor         float64 = 1.5
	// Pending for gotInfo
	torrentPending = 0
	torrentPaused  = 1
	torrentRunning = 2
)

// Torrent ...
type Torrent struct {
	*torrent.Torrent
	bytesRequested  int64
	bytesLimitation int64
	bytesCompleted  int64
	bytesMissing    int64
	status          int64
}

// Pause ...
func (t *Torrent) Pause() {
	if t.status != torrentPaused {
		t.status = torrentPaused
		t.Drop()
	}
}

// Paused ...
func (t *Torrent) Paused() bool {
	return t.status == torrentPaused
}

// Run ...
func (t *Torrent) Run() {
	if t.status != torrentRunning {
		t.status = torrentRunning
		t.DownloadAll()
	}
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
	trackers      []string
	DataDir       string
	closeAll      chan struct{}
	newTorrent    chan string
	removeTorrent chan string
	updateTorrent chan interface{}
	mu            sync.Mutex
}

func (tm *TorrentManager) CloseAll(input struct{}) error {
	tm.closeAll <- input
	return nil
}

func (tm *TorrentManager) NewTorrent(input string) error {
	tm.newTorrent <- input
	return nil
}

func (tm *TorrentManager) RemoveTorrent(input string) error {
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

// SetTrackers ...
func (tm *TorrentManager) SetTrackers(trackers []string) {
	for _, tracker := range trackers {
		tm.trackers = append(tm.trackers, tracker)
	}
}

// AddTorrent ...
func (tm *TorrentManager) AddTorrent(filePath string) {
	mi, err := metainfo.LoadFromFile(filePath)
	if err != nil {
		log.Error("Error while adding torrent", "Err", err)
		return
	}
	spec := torrent.TorrentSpecFromMetaInfo(mi)
	ih := spec.InfoHash
	log.Info("Get torrent from local file", "InfoHash", ih.HexString())

	tm.mu.Lock()
	if _, ok := tm.torrents[ih]; ok {
		log.Info("Torrent was already existed. Skip", "InfoHash", ih.HexString())
		tm.mu.Unlock()
		return
	}

	spec.Storage = storage.NewFile(path.Join(tm.DataDir, ih.HexString()))
	if len(spec.Trackers) == 0 {
		spec.Trackers = append(spec.Trackers, []string{})
	}
	for _, tracker := range tm.trackers {
		spec.Trackers[0] = append(spec.Trackers[0], tracker)
	}
	var ss []string
	slices.MakeInto(&ss, mi.Nodes)
	tm.client.AddDHTNodes(ss)
	t, _, err := tm.client.AddTorrentSpec(spec)
	tm.torrents[ih] = &Torrent{
		t,
		defaultBytesLimitation,
		int64(defaultBytesLimitation * expansionFactor),
		0,
		0,
		torrentPending,
	}
	tm.mu.Unlock()
	log.Info("Torrent is waiting for gotInfo", "InfoHash", ih.HexString())
	<-t.GotInfo()
	tm.torrents[ih].Run()
}

// AddMagnet ...
func (tm *TorrentManager) AddMagnet(uri string) {
	spec, err := torrent.TorrentSpecFromMagnetURI(uri)
	if err != nil {
		log.Error("Error while adding magnet uri", "Err", err)
	}
	ih := spec.InfoHash
	dataPath := path.Join(tm.DataDir, ih.HexString())
	torrentPath := path.Join(dataPath, "torrent")
	if _, err := os.Stat(torrentPath); err == nil {
		log.Info("Torrent was already existed. Skip", "InfoHash", ih.HexString())
		tm.AddTorrent(torrentPath)
		return
	}
	log.Info("Get torrent from magnet uri", "InfoHash", ih.HexString())

	tm.mu.Lock()
	if _, ok := tm.torrents[ih]; ok {
		log.Info("Torrent was already existed. Skip", "InfoHash", ih.HexString())
		tm.mu.Unlock()
		return
	}

	spec.Storage = storage.NewFile(dataPath)
	if len(spec.Trackers) == 0 {
		spec.Trackers = append(spec.Trackers, []string{})
	}
	for _, tracker := range tm.trackers {
		spec.Trackers[0] = append(spec.Trackers[0], tracker)
	}
	t, _, err := tm.client.AddTorrentSpec(spec)
	tm.torrents[ih] = &Torrent{
		t,
		defaultBytesLimitation,
		int64(defaultBytesLimitation * expansionFactor),
		0,
		0,
		torrentPending,
	}
	tm.mu.Unlock()
	log.Info("Torrent is waiting for gotInfo", "InfoHash", ih.HexString())

	<-t.GotInfo()
	log.Info("Torrent gotInfo finished", "InfoHash", ih.HexString())
	tm.torrents[ih].Run()

	f, _ := os.Create(torrentPath)
	log.Info("Write torrent file", "InfoHash", ih.HexString(), "path", torrentPath)
	if err := t.Metainfo().Write(f); err != nil {
		log.Error("Error while write torrent file", "error", err)
	}
	defer f.Close()
}

// UpdateMagnet ...
func (tm *TorrentManager) UpdateMagnet(ih metainfo.Hash, BytesRequested int64) {
	log.Info("Update torrent", "InfoHash", ih, "bytes", BytesRequested)

	if t, ok := tm.torrents[ih]; ok {
		t.bytesRequested = BytesRequested
		if t.bytesRequested > t.bytesLimitation {
			t.bytesLimitation = int64(float64(BytesRequested) * expansionFactor)
		}
	}
}

// DropMagnet ...
func (tm *TorrentManager) DropMagnet(uri string) bool {
	spec, err := torrent.TorrentSpecFromMagnetURI(uri)
	if err != nil {
		log.Info("error while removing magnet", "error", err)
	}
	ih := spec.InfoHash
	if t, ok := tm.torrents[ih]; ok {
		t.Drop()
		delete(tm.torrents, ih)
		return true
	}
	return false
}

// NewTorrentManager ...
func NewTorrentManager(config *Config) *TorrentManager {
	cfg := torrent.NewDefaultClientConfig()
	cfg.DisableTCP = true
	cfg.DataDir = config.DataDir
	cfg.DisableEncryption = true
	listenAddr := &net.TCPAddr{}
	log.Info("Torrent client listening on", "addr", listenAddr)
	cfg.SetListenAddr(listenAddr.String())
	cl, err := torrent.NewClient(cfg)
	if err != nil {
		log.Error("Error while create torrent client", "err", err)
	}

	TorrentManager := &TorrentManager{
		client:        cl,
		torrents:      make(map[metainfo.Hash]*Torrent),
		DataDir:       config.DataDir,
		closeAll:      make(chan struct{}),
		newTorrent:    make(chan string, newTorrentChanBuffer),
		removeTorrent: make(chan string, removeTorrentChanBuffer),
		updateTorrent: make(chan interface{}, updateTorrentChanBuffer),
	}

	if len(config.DefaultTrackers) > 0 {
		TorrentManager.SetTrackers(strings.Split(config.DefaultTrackers, ","))
	}
	log.Info("Torrent client created")

	go func() {
		for {
			select {
			case torrent := <-TorrentManager.newTorrent:
				if isMagnetURI(torrent) {
					go TorrentManager.AddMagnet(torrent)
				} else {
					go TorrentManager.AddTorrent(torrent)
				}
			case torrent := <-TorrentManager.removeTorrent:
				if isMagnetURI(torrent) {
					go TorrentManager.DropMagnet(torrent)
				} else {
				}
			case msg := <-TorrentManager.updateTorrent:
				meta := msg.(FlowControlMeta)
				go TorrentManager.UpdateMagnet(meta.InfoHash, int64(meta.BytesRequested))
			}
		}
	}()

	go func() {
		var counter uint64
		for counter = 0;; counter++ {
			for ih, t := range TorrentManager.torrents {
				if !t.Pending() {
					t.bytesCompleted = t.BytesCompleted()
					t.bytesMissing = t.BytesMissing()
					if t.bytesCompleted >= t.bytesLimitation {
						t.Pause()
					} else if t.bytesCompleted < t.bytesLimitation {
						t.Run()
					}
					if counter >= 20 {
						counter = 0
						log.Info("Torrent progress",
							"InfoHash", ih.HexString(),
							"completed", t.bytesCompleted,
							"requested", t.bytesLimitation,
							"total", t.bytesCompleted+t.bytesMissing,
						)
					}
				}
			}
			time.Sleep(time.Second * queryTimeInterval)
		}
	}()

	return TorrentManager
}
