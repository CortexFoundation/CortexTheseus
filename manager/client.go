package downloadmanager

import (
	"io"
	"log"
	"net"
	"os"
	"path"
	"strings"
	"sync"
	"time"

	"../common"

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
	torrentPending                  = 0
	torrentPaused                   = 1
	torrentRunning                  = 2
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
	torrents      map[string]*Torrent
	trackers      []string
	DataDir       string
	CloseAll      chan struct{}
	NewTorrent    chan string
	RemoveTorrent chan string
	UpdateTorrent chan interface{}
	mu            sync.Mutex
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
	ih := spec.InfoHash.HexString()
	log.Println(ih, "get torrent from local file.")

	tm.mu.Lock()
	if _, ok := tm.torrents[ih]; ok {
		log.Println(ih, "torrent was already existed. Skip.")
		tm.mu.Unlock()
		return
	}

	spec.Storage = storage.NewFile(path.Join(tm.DataDir, ih))
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
	log.Println(uri, "spec:", spec)
	if err != nil {
		log.Printf("error adding magnet: %s", err)
	}
	ih := spec.InfoHash.HexString()
	dataPath := path.Join(tm.DataDir, ih)
	torrentPath := path.Join(dataPath, "torrent")
	if _, err := os.Stat(torrentPath); err == nil {
		log.Println(ih, "torrent file exists: ", torrentPath)
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
	tm.torrents[ih].Run()

	f, _ := os.Create(torrentPath)
	torrent := t.Metainfo().Encoding
	io.WriteString(f, torrent)
	defer f.Close()
}

// UpdateMagnet ...
func (tm *TorrentManager) UpdateMagnet(uri string, BytesRequested int64) {
	spec, err := torrent.TorrentSpecFromMagnetURI(uri)
	if err != nil {
		log.Printf("error while parsing magnet uri: %s", err)
	}
	ih := spec.InfoHash.HexString()
	log.Println(ih, "update torrent from magnet uri.")

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
	ih := spec.InfoHash.HexString()
	if t, ok := tm.torrents[ih]; ok {
		t.Drop()
		delete(tm.torrents, ih)
		return true
	}
	return false
}

// NewTorrentManager ...
func NewTorrentManager(DataDir string) *TorrentManager {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	cfg := torrent.NewDefaultClientConfig()
	cfg.DisableTCP = true
	cfg.DataDir = DataDir
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
		torrents:      make(map[string]*Torrent),
		DataDir:       DataDir,
		CloseAll:      make(chan struct{}),
		NewTorrent:    make(chan string, newTorrentChanBuffer),
		RemoveTorrent: make(chan string, removeTorrentChanBuffer),
		UpdateTorrent: make(chan interface{}, updateTorrentChanBuffer),
	}

	go func() {
		for {
			select {
			case torrent := <-TorrentManager.NewTorrent:
				log.Println("Add", torrent)
				if isMagnetURI(torrent) {
					go TorrentManager.AddMagnet(torrent)
				} else {
					go TorrentManager.AddTorrent(torrent)
				}
			case torrent := <-TorrentManager.RemoveTorrent:
				log.Println("Drop", torrent)
				if isMagnetURI(torrent) {
					go TorrentManager.DropMagnet(torrent)
				} else {
				}
			case msg := <-TorrentManager.UpdateTorrent:
				meta := msg.(common.FlowControlMeta)
				if isMagnetURI(meta.URI) {
					go TorrentManager.UpdateMagnet(meta.URI, int64(meta.BytesRequested))
				}
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
					log.Printf("Torrent %s: %d/%d, limit=%d", ih, t.bytesCompleted, t.bytesCompleted+t.bytesMissing, t.bytesLimitation)
				}
			}
			time.Sleep(time.Second * queryTimeInterval)
		}
	}()

	return TorrentManager
}
