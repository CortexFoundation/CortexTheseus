package downloadmanager

import (
	"log"
	"net"
	"path"
	"strings"

	"github.com/anacrolix/missinggo/slices"
	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/storage"
)

// Manager ...
type Manager struct {
	client          *torrent.Client
	torrentSessions map[string]*torrent.Torrent
	torrentProgress map[string]int
	trackers        []string
	DataDir         string
	CloseAll        chan struct{}
	NewTorrent      chan string
	RemoveTorrent   chan string
	UpdateTorrent   chan interface{}
}

func isMagnetURI(uri string) bool {
	return strings.HasPrefix(uri, "magnet:?xt=urn:btih:")
}

// SetBuiltinTrackers ...
func (m *Manager) SetBuiltinTrackers(trackers []string) {
	for _, tracker := range trackers {
		m.trackers = append(m.trackers, tracker)
	}
}

// AddTorrent ...
func (m *Manager) AddTorrent(filename string) {
	mi, err := metainfo.LoadFromFile(filename)
	if err != nil {
		return
	}
	spec := torrent.TorrentSpecFromMetaInfo(mi)
	ih := spec.InfoHash.HexString()
	if _, ok := m.torrentSessions[ih]; ok {
		return
	}
	spec.Storage = storage.NewFile(path.Join(m.DataDir, ih))

	if len(spec.Trackers) == 0 {
		spec.Trackers = append(spec.Trackers, []string{})
	}

	for _, tracker := range m.trackers {
		spec.Trackers[0] = append(spec.Trackers[0], tracker)
	}

	var ss []string
	slices.MakeInto(&ss, mi.Nodes)
	m.client.AddDHTNodes(ss)
	t, _, err := m.client.AddTorrentSpec(spec)
	m.torrentSessions[ih] = t
	<-t.GotInfo()
	t.DownloadAll()
}

// AddMagnet ...
func (m *Manager) AddMagnet(mURI string) {
	spec, err := torrent.TorrentSpecFromMagnetURI(mURI)
	if err != nil {
		log.Printf("error adding magnet: %s", err)
	}
	ih := spec.InfoHash.HexString()
	if _, ok := m.torrentSessions[ih]; ok {
		return
	}
	spec.Storage = storage.NewFile(path.Join(m.DataDir, ih))

	if len(spec.Trackers) == 0 {
		spec.Trackers = append(spec.Trackers, []string{})
	}

	for _, tracker := range m.trackers {
		spec.Trackers[0] = append(spec.Trackers[0], tracker)
	}

	t, _, err := m.client.AddTorrentSpec(spec)
	<-t.GotInfo()
	m.torrentSessions[ih] = t
	t.DownloadAll()
}

// DropMagnet ...
func (m *Manager) DropMagnet(mURI string) {
	spec, err := torrent.TorrentSpecFromMagnetURI(mURI)
	if err != nil {
		log.Printf("error adding magnet: %s", err)
	}
	ih := spec.InfoHash.HexString()
	if ts, ok := m.torrentSessions[ih]; ok {
		ts.Drop()
		delete(m.torrentSessions, ih)
	} else {
		return
	}
}

// NewManager ...
func NewManager(DataDir string) *Manager {
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

	manager := &Manager{
		client:          t,
		torrentSessions: make(map[string]*torrent.Torrent),
		torrentProgress: make(map[string]int),
		DataDir:         DataDir,
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
					go manager.AddMagnet(torrent)
				} else {
					go manager.AddTorrent(torrent)
				}
			case torrent := <-manager.RemoveTorrent:
				go manager.DropMagnet(torrent)
			case <-manager.UpdateTorrent:
				continue
			}
		}
	}()

	return manager
}
