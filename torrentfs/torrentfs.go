package torrentfs

import (
	"fmt"
	"io/ioutil"

	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/CortexTheseus/rpc"
)

type GeneralMessage struct {
	Version string `json:"version,omitempty"`
	Commit  string `json:"commit,omitempty"`
}

// TorrentFS contains the torrent file system internals.
type TorrentFS struct {
	config  *Config
	history *GeneralMessage

	monitor *Monitor
}

var Torrentfs_handle CVMStorage

// New creates a new torrentfs instance with the given configuration.
func New(config *Config, commit string) (*TorrentFS, error) {
	versionMeta := ""
	TorrentAPIAvailable.Lock()
	if len(params.VersionMeta) > 0 {
		versionMeta = fmt.Sprintf(" (%s)", params.VersionMeta)
	}

	msg := &GeneralMessage{
		Commit:  commit,
		Version: fmt.Sprintf("v%d.%d.%d%s", params.VersionMajor, params.VersionMinor, params.VersionPatch, versionMeta),
	}

	monitor, moErr := NewMonitor(config)
	if moErr != nil {
		log.Error("Failed create monitor")
		return nil, moErr
	}
	tmp := &TorrentFS{
		config:  config,
		history: msg,
		monitor: monitor,
	}

	Torrentfs_handle = *tmp

	return tmp, nil
}

// Protocols implements the node.Service interface.
func (tfs *TorrentFS) Protocols() []p2p.Protocol { return nil }

// APIs implements the node.Service interface.
func (tfs *TorrentFS) APIs() []rpc.API { return nil }

// Start starts the data collection thread and the listening server of the dashboard.
// Implements the node.Service interface.
func (tfs *TorrentFS) Start(server *p2p.Server) error {
	log.Info("Torrent monitor starting", "torrentfs", tfs)
	if tfs.monitor == nil {
		log.Error("Monitor is error")
		return nil
	}
	return tfs.monitor.Start()
}

// Stop stops the data collection thread and the connection listener of the dashboard.
// Implements the node.Service interface.
func (tfs *TorrentFS) Stop() error {
	// Wait until every goroutine terminates.
	tfs.monitor.Stop()
	return nil
}


func (fs TorrentFS) Available(infohash string, rawSize int64) bool {
	// modelDir := fs.DataDir + "/" + infoHash
	// if (os.Stat)
	return Available(infohash, rawSize)
}

func (fs TorrentFS) Exist(infohash string) bool {
	return Exist(infohash)
}

func (fs TorrentFS) GetFile(infohash string, path string) ([]byte, error) {
	fn := fs.config.DataDir + "/" + infohash  + "/" + path
	data, err := ioutil.ReadFile(fn)
	fmt.Println("InfoHashFileSystem", "GetFile", fn)
	return data, err

}
func (fs TorrentFS) ExistTorrent(infohash string) bool {
	return ExistTorrent(infohash)
}
