package torrentfs

import (
	"context"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"sync"
	"time"
)

// TorrentFS contains the torrent file system internals.
type TorrentFS struct {
	//protocol p2p.Protocol // Protocol description and parameters
	config  *Config
	monitor *Monitor

	peerMu sync.RWMutex       // Mutex to sync the active peer set
	peers  map[*Peer]struct{} // Set of currently active peers
}

func (t *TorrentFS) storage() *TorrentManager {
	return t.monitor.dl
}

var torrentInstance *TorrentFS = nil

func GetStorage() CortexStorage {
	return torrentInstance //GetTorrentInstance()
}

// New creates a new torrentfs instance with the given configuration.
func New(config *Config, commit string, cache, compress bool) (*TorrentFS, error) {
	if torrentInstance != nil {
		return torrentInstance, nil
	}

	monitor, moErr := NewMonitor(config, cache, compress)
	if moErr != nil {
		log.Error("Failed create monitor")
		return nil, moErr
	}

	torrentInstance = &TorrentFS{
		config:  config,
		monitor: monitor,
		peers:   make(map[*Peer]struct{}),
	}

	/*torrentInstance.protocol = p2p.Protocol{
		Name:    ProtocolName,
		Version: uint(ProtocolVersion),
		Length:  NumberOfMessageCodes,
		Run:     torrentInstance.HandlePeer,
		NodeInfo: func() interface{} {
			return map[string]interface{}{
				"version": ProtocolVersionStr,
				"utp":    !config.DisableUTP,
				"tcp":    !config.DisableTCP,
				"dht":    !config.DisableDHT,
				"listen": config.Port,
			}
		},
	}*/

	return torrentInstance, nil
}

func (tfs *TorrentFS) MaxMessageSize() uint64 {
	return NumberOfMessageCodes
}

func (tfs *TorrentFS) HandlePeer(peer *p2p.Peer, rw p2p.MsgReadWriter) error {
	tfsPeer := newPeer(tfs, peer, rw)

	tfs.peerMu.Lock()
	tfs.peers[tfsPeer] = struct{}{}
	tfs.peerMu.Unlock()

	defer func() {
		tfs.peerMu.Lock()
		delete(tfs.peers, tfsPeer)
		tfs.peerMu.Unlock()
	}()

	if err := tfsPeer.handshake(); err != nil {
		return err
	}

	tfsPeer.Start()
	defer func() {
		tfsPeer.Stop()
	}()

	return tfs.runMessageLoop(tfsPeer, rw)
}
func (tfs *TorrentFS) runMessageLoop(p *Peer, rw p2p.MsgReadWriter) error {
	return nil
}

// Protocols implements the node.Service interface.
func (tfs *TorrentFS) Protocols() []p2p.Protocol { return nil } //return []p2p.Protocol{tfs.protocol} }

// APIs implements the node.Service interface.
func (tfs *TorrentFS) APIs() []rpc.API {
	//return []rpc.API{
	//	{
	//		Namespace: ProtocolName,
	//		Version:   ProtocolVersionStr,
	//		Service:   NewPublicTorrentAPI(tfs),
	//		Public: false,
	//	},
	//}
	return nil
}

func (tfs *TorrentFS) Version() uint {
	//return tfs.protocol.Version
	return 0
}

type PublicTorrentAPI struct {
	w *TorrentFS

	lastUsed map[string]time.Time // keeps track when a filter was polled for the last time.
}

// NewPublicWhisperAPI create a new RPC whisper service.
func NewPublicTorrentAPI(w *TorrentFS) *PublicTorrentAPI {
	api := &PublicTorrentAPI{
		w:        w,
		lastUsed: make(map[string]time.Time),
	}
	return api
}

// Start starts the data collection thread and the listening server of the dashboard.
// Implements the node.Service interface.
func (tfs *TorrentFS) Start(server *p2p.Server) error {
	log.Info("Started nas v.1.0", "config", tfs)
	if tfs == nil || tfs.monitor == nil {
		return nil
	}
	return tfs.monitor.Start()
}

// Stop stops the data collection thread and the connection listener of the dashboard.
// Implements the node.Service interface.
func (tfs *TorrentFS) Stop() error {
	if tfs == nil || tfs.monitor == nil {
		return nil
	}
	// Wait until every goroutine terminates.
	tfs.monitor.Stop()
	return nil
}

func (fs *TorrentFS) Available(ctx context.Context, infohash string, rawSize int64) (bool, error) {
	return fs.storage().Available(infohash, rawSize)
}

func (fs *TorrentFS) GetFile(ctx context.Context, infohash, subpath string) ([]byte, error) {
	return fs.storage().GetFile(infohash, subpath)
}
