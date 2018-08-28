package downloadmanager

import (
	"log"
	"net"
	"path"
	"strings"
	"sync"
	"time"

	"github.com/anacrolix/missinggo/slices"
	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/storage"
)

const (
	defaultBytesLimitation  = 512 * 1024
	queryTimeInterval       = 5
	newTorrentChanBuffer    = 32
	updateTorrentChanBuffer = 32
	removeTorrentChanBuffer = 16
)

// Torrent ...
type Torrent struct {
	*torrent.Torrent
	bytesLimitation int64
	bytesCompleted  int64
	bytesMissing    int64
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
	lock          sync.Mutex
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
func (tm *TorrentManager) AddTorrent(filename string) {
	mi, err := metainfo.LoadFromFile(filename)
	if err != nil {
		log.Printf("error adding torrent: %s", err)
		return
	}
	spec := torrent.TorrentSpecFromMetaInfo(mi)
	ih := spec.InfoHash.HexString()
	log.Println(ih, "get torrent from local file.")

	tm.lock.Lock()
	if _, ok := tm.torrents[ih]; ok {
		log.Println(ih, "torrent was already existed. Skip.")
		tm.lock.Unlock()
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
	tm.torrents[ih] = &Torrent{t, defaultBytesLimitation, 0, 0}
	tm.lock.Unlock()
	log.Println(ih, "wait for gotInfo")

	<-t.GotInfo()
	t.DownloadAll()
	tm.torrents[ih].bytesCompleted = t.BytesCompleted()
	tm.torrents[ih].bytesMissing = t.BytesMissing()
	log.Println(ih, "start to download.")
}

// AddMagnet ...
func (tm *TorrentManager) AddMagnet(mURI string) {
	spec, err := torrent.TorrentSpecFromMagnetURI(mURI)
	if err != nil {
		log.Printf("error adding magnet: %s", err)
	}
	ih := spec.InfoHash.HexString()
	log.Println(ih, "get torrent from magnet uri.")

	tm.lock.Lock()
	if _, ok := tm.torrents[ih]; ok {
		log.Println(ih, "torrent was already existed. Skip.")
		tm.lock.Unlock()
		return
	}

	spec.Storage = storage.NewFile(path.Join(tm.DataDir, ih))
	if len(spec.Trackers) == 0 {
		spec.Trackers = append(spec.Trackers, []string{})
	}
	for _, tracker := range tm.trackers {
		spec.Trackers[0] = append(spec.Trackers[0], tracker)
	}
	t, _, err := tm.client.AddTorrentSpec(spec)
	tm.torrents[ih] = &Torrent{t, defaultBytesLimitation, 0, 0}
	tm.lock.Unlock()
	log.Println(ih, "wait for gotInfo")

	<-t.GotInfo()
	t.DownloadAll()
	tm.torrents[ih].bytesCompleted = t.BytesCompleted()
	tm.torrents[ih].bytesMissing = t.BytesMissing()
	log.Println(ih, "start to download.")
}

// DropMagnet ...
func (tm *TorrentManager) DropMagnet(mURI string) bool {
	spec, err := torrent.TorrentSpecFromMagnetURI(mURI)
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
			case <-TorrentManager.UpdateTorrent:
				continue
			}
		}
	}()

	go func() {
		for {
			for ih, t := range TorrentManager.torrents {
				if !(t.bytesCompleted == 0 && t.bytesMissing == 0) {
					t.bytesCompleted = t.BytesCompleted()
					t.bytesMissing = t.BytesMissing()
					log.Println(ih, t.bytesCompleted, t.bytesCompleted+t.bytesMissing)
				}
			}
			time.Sleep(time.Second * queryTimeInterval)
		}
	}()

	return TorrentManager
}
