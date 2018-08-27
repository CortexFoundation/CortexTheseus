package downloadmanager

import (
	"log"
	"net"
	"path"
	"strings"
	"sync"

	"github.com/anacrolix/missinggo/slices"
	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/storage"
)

// Torrent ...
type Torrent struct {
	*torrent.Torrent
	RawSize     uint64
	CurrentSize uint64
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
		return
	}
	spec := torrent.TorrentSpecFromMetaInfo(mi)
	ih := spec.InfoHash.HexString()

	tm.lock.Lock()
	if _, ok := tm.torrents[ih]; ok {
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
	tm.torrents[ih] = &Torrent{t, 0, 0}
	tm.lock.Unlock()

	<-t.GotInfo()
	t.DownloadAll()
}

// AddMagnet ...
func (tm *TorrentManager) AddMagnet(mURI string) {
	spec, err := torrent.TorrentSpecFromMagnetURI(mURI)
	if err != nil {
		log.Printf("error adding magnet: %s", err)
	}
	ih := spec.InfoHash.HexString()

	tm.lock.Lock()
	if _, ok := tm.torrents[ih]; ok {
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
	tm.torrents[ih] = &Torrent{t, 0, 0}
	tm.lock.Unlock()

	<-t.GotInfo()
	t.DownloadAll()
}

// DropMagnet ...
func (tm *TorrentManager) DropMagnet(mURI string) {
	spec, err := torrent.TorrentSpecFromMagnetURI(mURI)
	if err != nil {
		log.Printf("error adding magnet: %s", err)
	}
	ih := spec.InfoHash.HexString()
	if ts, ok := tm.torrents[ih]; ok {
		ts.Drop()
		delete(tm.torrents, ih)
	} else {
		return
	}
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
	t, err := torrent.NewClient(cfg)
	if err != nil {
		log.Println(err)
	}

	TorrentManager := &TorrentManager{
		client:        t,
		torrents:      make(map[string]*Torrent),
		DataDir:       DataDir,
		CloseAll:      make(chan struct{}),
		NewTorrent:    make(chan string),
		RemoveTorrent: make(chan string),
		UpdateTorrent: make(chan interface{}),
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

	return TorrentManager
}
