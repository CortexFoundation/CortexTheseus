package torrentfs

import (
	"fmt"
	"sync"

	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/p2p"
	"github.com/ethereum/go-ethereum/params"
	"github.com/ethereum/go-ethereum/rpc"
)

type GeneralMessage struct {
	Version string `json:"version,omitempty"`
	Commit  string `json:"commit,omitempty"`
}

// TorrentFS contains the torrent file system internals.
type TorrentFS struct {
	config   *Config
	lock     sync.RWMutex // Lock protecting the torrentfs' internals
	history  *GeneralMessage

	quit chan chan error // Channel used for graceful exit
	monitor  *Monitor
	tm       *TorrentManager
}

// New creates a new dashboard instance with the given configuration.
func New(config *Config, commit string) *TorrentFS {
	versionMeta := ""
	if len(params.VersionMeta) > 0 {
		versionMeta = fmt.Sprintf(" (%s)", params.VersionMeta)
	}
	return &TorrentFS{
		config: config,
		history: &GeneralMessage{
			Commit:  commit,
			Version: fmt.Sprintf("v%d.%d.%d%s", params.VersionMajor, params.VersionMinor, params.VersionPatch, versionMeta),
		},
		quit:   make(chan chan error),
	}
}

// Protocols implements the node.Service interface.
func (db *TorrentFS) Protocols() []p2p.Protocol { return nil }

// APIs implements the node.Service interface.
func (db *TorrentFS) APIs() []rpc.API { return nil }

// Start starts the data collection thread and the listening server of the dashboard.
// Implements the node.Service interface.
func (db *TorrentFS) Start(server *p2p.Server) error {
	go func(){
		db.tm = NewTorrentManager(db.config)
		db.monitor = NewMonitor(db.config)
		db.monitor.SetDownloader(db.tm)
		db.monitor.Start()
	}()
	return nil
}

// Stop stops the data collection thread and the connection listener of the dashboard.
// Implements the node.Service interface.
func (db *TorrentFS) Stop() error {
	// Wait until every goroutine terminates.
	db.monitor.Terminate() <- struct{}{}
	log.Info("TorrentFs stopped")

	return nil
}
