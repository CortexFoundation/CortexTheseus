package main

import (
	"log"
	"net"
	"os"
	"path"
	"strings"
	"sync"
	"time"

	"github.com/CortexFoundation/CortexTheseus/torrentfs/types"

	"github.com/anacrolix/missinggo/slices"
	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/storage"
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
		log.Printf("error adding torrent: %s", err)
		return
	}
	spec := torrent.TorrentSpecFromMetaInfo(mi)
	ih := spec.InfoHash
	log.Println(ih.HexString(), "get torrent from local file.")

	tm.mu.Lock()
	if _, ok := tm.torrents[ih]; ok {
		log.Println(ih.HexString(), "torrent was already existed. Skip.")
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
	log.Println(ih, "waiting for gotInfo")
	<-t.GotInfo()
	tm.torrents[ih].Run()
}

// AddMagnet ...
func (tm *TorrentManager) AddMagnet(uri string) {
	spec, err := torrent.TorrentSpecFromMagnetURI(uri)
	if err != nil {
		log.Printf("error adding magnet: %s", err)
	}
	ih := spec.InfoHash
	dataPath := path.Join(tm.DataDir, ih.HexString())
	torrentPath := path.Join(dataPath, "torrent")
	if _, err := os.Stat(torrentPath); err == nil {
		log.Println(ih.HexString(), "torrent file exists: ", torrentPath)
		tm.AddTorrent(torrentPath)
		return
	}
	log.Println(ih, "get torrent from magnet uri.")

	tm.mu.Lock()
	if _, ok := tm.torrents[ih]; ok {
		log.Println(ih, "torrent was already existed. Skip.")
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
	log.Println(ih, "waiting for gotInfo")

	<-t.GotInfo()
	log.Println(ih, "gotInfo finish")
	tm.torrents[ih].Run()

	f, _ := os.Create(torrentPath)
	log.Println(ih.HexString(), "write torrent file to", torrentPath)
	if err := t.Metainfo().Write(f); err != nil {
		log.Println(err)
	}
	defer f.Close()
}

// UpdateMagnet ...
func (tm *TorrentManager) UpdateMagnet(ih metainfo.Hash, BytesRequested int64) {
	log.Println(ih, "update torrent from infohash.")

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
		log.Printf("error removing magnet: %s", err)
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
func NewTorrentManager(flag *types.Flag) *TorrentManager {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	cfg := torrent.NewDefaultClientConfig()
	cfg.DisableTCP = true
	cfg.DataDir = *flag.DataDir
	cfg.DisableEncryption = true
	listenAddr := &net.TCPAddr{}
	log.Println(listenAddr)
	cfg.SetListenAddr(listenAddr.String())
	cl, err := torrent.NewClient(cfg)
	if err != nil {
		log.Println(err)
	}

	TorrentManager := &TorrentManager{
		client:        cl,
		torrents:      make(map[metainfo.Hash]*Torrent),
		DataDir:       *flag.DataDir,
		closeAll:      make(chan struct{}),
		newTorrent:    make(chan string, newTorrentChanBuffer),
		removeTorrent: make(chan string, removeTorrentChanBuffer),
		updateTorrent: make(chan interface{}, updateTorrentChanBuffer),
	}

	if flag.DefaultTrackers != nil {
		TorrentManager.SetTrackers(*flag.DefaultTrackers)
	}

	go func() {
		for {
			select {
			case torrent := <-TorrentManager.newTorrent:
				log.Println("Add", torrent)
				if isMagnetURI(torrent) {
					go TorrentManager.AddMagnet(torrent)
				} else {
					go TorrentManager.AddTorrent(torrent)
				}
			case torrent := <-TorrentManager.removeTorrent:
				log.Println("Drop", torrent)
				if isMagnetURI(torrent) {
					go TorrentManager.DropMagnet(torrent)
				} else {
				}
			case msg := <-TorrentManager.updateTorrent:
				meta := msg.(types.FlowControlMeta)
				go TorrentManager.UpdateMagnet(meta.InfoHash, int64(meta.BytesRequested))
			}
		}
	}()

	go func() {
		for {
			for ih, t := range TorrentManager.torrents {
				if !t.Pending() {
					t.bytesCompleted = t.BytesCompleted()
					t.bytesMissing = t.BytesMissing()
					if t.bytesCompleted >= t.bytesLimitation {
						t.Pause()
					} else if t.bytesCompleted < t.bytesLimitation {
						t.Run()
					}
					log.Printf("Torrent %s: %d/%d, limit=%d", ih.HexString(), t.bytesCompleted, t.bytesCompleted+t.bytesMissing, t.bytesLimitation)
				}
			}
			time.Sleep(time.Second * queryTimeInterval)
		}
	}()

	return TorrentManager
}
