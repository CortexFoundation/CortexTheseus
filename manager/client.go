package downloadmanager

import (
	"log"
	"net"
	"strings"

	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/metainfo"
)

// Manager ...
type Manager struct {
	client          *torrent.Client
	torrentSessions map[metainfo.Hash]*torrent.Torrent
	torrentProgress map[metainfo.Hash]int
	CloseAll        chan struct{}
	NewTorrent      chan string
	RemoveTorrent   chan string
	UpdateTorrent   chan interface{}
}

func isMagnetURI(uri string) bool {
	return strings.HasPrefix(uri, "magnet:?xt=urn:btih:")
}

// AddTorrent ...
func (m *Manager) AddTorrent(filename string) {
	tm, err := m.client.AddTorrentFromFile(filename)
	if err != nil {
		log.Printf("error adding torrent: %s", err)
	}
	<-tm.GotInfo()
	infohash := tm.InfoHash()
	_, ok := m.torrentSessions[infohash]
	if ok {
		tm.Drop()
	} else {
		tm.DownloadAll()
	}
}

// Add ...
func (m *Manager) Add(mURI string) {
	tm, err := m.client.AddMagnet(mURI)
	if err != nil {
		log.Printf("error adding magnet: %s", err)
	}
	<-tm.GotInfo()
	infohash := tm.InfoHash()
	_, ok := m.torrentSessions[infohash]
	if ok {
		tm.Drop()
	} else {
		tm.DownloadAll()
	}
}

// Drop ...
func (m *Manager) Drop(mURI string) {
	tm, err := m.client.AddMagnet(mURI)
	if err != nil {
		log.Printf("error adding magnet: %s", err)
	}
	<-tm.GotInfo()
	infohash := tm.InfoHash()
	_, ok := m.torrentSessions[infohash]
	if ok {
		tm.Drop()
		m.torrentSessions[infohash].Drop()
	}
}

// NewManager ...
func NewManager(storageDir string, torrents chan string) *Manager {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	cfg := torrent.NewDefaultClientConfig()
	cfg.DisableTCP = true
	cfg.DataDir = storageDir
	cfg.DisableEncryption = true
	listenAddr := &net.TCPAddr{}
	log.Println(listenAddr)
	cfg.SetListenAddr(listenAddr.String())
	t, err := torrent.NewClient(cfg)
	if err != nil {
		log.Println(err)
	}

	manager := &Manager{
		client:          t,
		torrentSessions: make(map[metainfo.Hash]*torrent.Torrent),
		torrentProgress: make(map[metainfo.Hash]int),
		CloseAll:        make(chan struct{}),
		NewTorrent:      make(chan string),
		RemoveTorrent:   make(chan string),
		UpdateTorrent:   make(chan interface{}),
	}

	go func() {
		for {
			select {
			case torrent := <-manager.NewTorrent:
				log.Println("torrent", torrent, "added")
				if isMagnetURI(torrent) {
					go manager.Add(torrent)
				} else {
					go manager.AddTorrent(torrent)
				}
			case torrent := <-manager.RemoveTorrent:
				go manager.Drop(torrent)
			case <-manager.UpdateTorrent:
				continue
			}
		}
	}()

	return manager
}
