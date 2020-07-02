package torrentfs

import (
	"context"
	"errors"
	"fmt"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/CortexFoundation/CortexTheseus/p2p/enode"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"sync"
)

// TorrentFS contains the torrent file system internals.
type TorrentFS struct {
	protocol p2p.Protocol // Protocol description and parameters
	config   *Config
	monitor  *Monitor

	peerMu sync.RWMutex     // Mutex to sync the active peer set
	peers  map[string]*Peer // Set of currently active peers
}

func (t *TorrentFS) storage() *TorrentManager {
	return t.monitor.dl
}

func (t *TorrentFS) chain() *ChainDB {
	return t.monitor.fs
}

var inst *TorrentFS = nil

func GetStorage() CortexStorage {
	return inst //GetTorrentInstance()
}

// New creates a new torrentfs instance with the given configuration.
func New(config *Config, commit string, cache, compress bool) (*TorrentFS, error) {
	if inst != nil {
		return inst, nil
	}

	monitor, moErr := NewMonitor(config, cache, compress)
	if moErr != nil {
		log.Error("Failed create monitor")
		return nil, moErr
	}

	inst = &TorrentFS{
		config:  config,
		monitor: monitor,
		peers:   make(map[string]*Peer),
	}

	inst.protocol = p2p.Protocol{
		Name:    ProtocolName,
		Version: uint(ProtocolVersion),
		Length:  NumberOfMessageCodes,
		Run:     inst.HandlePeer,
		NodeInfo: func() interface{} {
			return map[string]interface{}{
				"version": ProtocolVersion,
				"status": map[string]interface{}{
					"dht":            !config.DisableDHT,
					"listen":         inst.LocalPort(),
					"root":           inst.chain().Root().Hex(),
					"files":          inst.Congress(),
					"active":         inst.Candidate(),
					"leafs":          len(inst.chain().Blocks()),
					"number":         monitor.currentNumber,
					"maxMessageSize": inst.MaxMessageSize(),
				},
			}
		},
		PeerInfo: func(id enode.ID) interface{} {
			inst.peerMu.Lock()
			defer inst.peerMu.Unlock()
			if p := inst.peers[fmt.Sprintf("%x", id[:8])]; p != nil {
				return map[string]interface{}{
					"version": p.version,
					"listen":  p.Info().Listen,
					"root":    p.Info().Root.Hex(),
					"files":   p.Info().Files,
					"leafs":   p.Info().Leafs,
				}
			}
			return nil
		},
	}

	return inst, nil
}

func (tfs *TorrentFS) MaxMessageSize() uint32 {
	return DefaultMaxMessageSize
}

func (tfs *TorrentFS) HandlePeer(peer *p2p.Peer, rw p2p.MsgReadWriter) error {
	tfsPeer := newPeer(fmt.Sprintf("%x", peer.ID().Bytes()[:8]), tfs, peer, rw)

	tfs.peerMu.Lock()
	tfs.peers[tfsPeer.id] = tfsPeer
	tfs.peerMu.Unlock()

	defer func() {
		tfs.peerMu.Lock()
		delete(tfs.peers, tfsPeer.id)
		tfs.peerMu.Unlock()
	}()

	if err := tfsPeer.handshake(); err != nil {
		return err
	}

	tfsPeer.start()
	defer func() {
		tfsPeer.stop()
	}()

	return tfs.runMessageLoop(tfsPeer, rw)
}
func (tfs *TorrentFS) runMessageLoop(p *Peer, rw p2p.MsgReadWriter) error {
	for {
		// fetch the next packet
		packet, err := rw.ReadMsg()
		if err != nil {
			log.Debug("message loop", "peer", p.peer.ID(), "err", err)
			return err
		}

		if packet.Size > tfs.MaxMessageSize() {
			log.Warn("oversized message received", "peer", p.peer.ID())
			packet.Discard()
			return errors.New("oversized message received")
		}

		log.Debug("Nas package", "size", packet.Size)

		switch packet.Code {
		case statusCode:
			var info *PeerInfo
			if err := packet.Decode(&info); err != nil {
				log.Warn("failed to decode peer state, peer will be disconnected", "peer", p.peer.ID(), "err", err)
				packet.Discard()
				return errors.New("invalid peer state")
			}
			p.peerInfo = info
		case messagesCode:
			//
		default:
		}
		packet.Discard()
	}
}

// Protocols implements the node.Service interface.
func (tfs *TorrentFS) Protocols() []p2p.Protocol { return []p2p.Protocol{tfs.protocol} }

// APIs implements the node.Service interface.
func (tfs *TorrentFS) APIs() []rpc.API {
	return []rpc.API{
		{
			Namespace: ProtocolName,
			Version:   ProtocolVersionStr,
			Service:   NewPublicTorrentAPI(tfs),
			Public:    false,
		},
	}
}

func (tfs *TorrentFS) Version() uint {
	return tfs.protocol.Version
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

func (fs *TorrentFS) LocalPort() int {
	return fs.storage().LocalPort()
}

func (fs *TorrentFS) Congress() int {
	return fs.storage().Congress()
}

func (fs *TorrentFS) Candidate() int {
	return fs.storage().Candidate()
}
