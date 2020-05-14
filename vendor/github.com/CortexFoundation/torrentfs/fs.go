package torrentfs

import (
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	//"github.com/anacrolix/torrent/metainfo"
	//"io/ioutil"
	//"path"
	//"github.com/anacrolix/torrent/metainfo"
	"sync"
	"time"
	//"errors"
	//"github.com/CortexFoundation/CortexTheseus/common/compress"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	//lru "github.com/hashicorp/golang-lru"
)

type CVMStorage interface {
	Available(infohash string, rawSize int64) (bool, error)
	GetFile(infohash, path string) ([]byte, error)
	Stop() error
}

type StorageAPI interface {
	//Start() error
	//Close() error
	//RemoveTorrent(metainfo.Hash) error
	//UpdateTorrent(interface{}) error
	//UpdateDynamicTrackers(trackers []string)
	//GetTorrent(ih metainfo.Hash) *Torrent
	Available(ih string, raw int64) (bool, error)
	GetFile(infohash, subpath string) ([]byte, error)
	//Metrics() time.Duration
}

// TorrentFS contains the torrent file system internals.
type TorrentFS struct {
	//protocol p2p.Protocol // Protocol description and parameters
	config *Config
	//history  *GeneralMessage
	monitor *Monitor

	//fileLock  sync.Mutex
	//fileCache *lru.Cache
	//fileCh    chan bool
	//cache     bool
	//compress  bool

	peerMu sync.RWMutex       // Mutex to sync the active peer set
	peers  map[*Peer]struct{} // Set of currently active peers

	//metrics   bool
	//fsUpdates time.Duration
}

//func (t *TorrentFS) Config() *Config {
//	return t.config
//}

//func (t *TorrentFS) Monitor() *Monitor {
//	return t.monitor
//}

func (t *TorrentFS) storage() StorageAPI {
	return t.monitor.dl
}

var torrentInstance *TorrentFS = nil

//func GetTorrentInstance() *TorrentFS {
//if torrentInstance == nil {
//	torrentInstance, _ = New(&DefaultConfig, "")
//}
//	return torrentInstance
//}

func GetStorage() CVMStorage {
	return torrentInstance //GetTorrentInstance()
}

/*func GetConfig() *Config {
	if torrentInstance != nil {
		return torrentInstance.Config()
	} else {
		return &DefaultConfig
	}
	return nil
}*/

// New creates a new dashboard instance with the given configuration.
//var Torrentfs_handle CVMStorage

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
		config: config,
		//history: msg,
		monitor: monitor,
		peers:   make(map[*Peer]struct{}),
	}
	//torrentInstance.fileCache, _ = lru.New(8)
	//torrentInstance.fileCh = make(chan bool, 4)
	//torrentInstance.compress = compress
	//torrentInstance.cache = cache

	//torrentInstance.metrics = config.Metrics

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
	//if tfs.cache {
	//	tfs.fileCache.Purge()
	//}
	return nil
}

//func (fs *TorrentFS) Metrics() time.Duration {
//	return fs.fsUpdates
//}

func (fs *TorrentFS) Available(infohash string, rawSize int64) (bool, error) {
	/*if fs.metrics {
		defer func(start time.Time) { fs.fsUpdates += time.Since(start) }(time.Now())
	}
	ih := metainfo.NewHashFromHex(infohash)
	tm := fs.monitor.dl
	if torrent := tm.GetTorrent(ih); torrent == nil {
		log.Debug("Seed not found", "hash", infohash)
		return false, errors.New("download not completed")
	} else {
		if !torrent.IsAvailable() {
			log.Debug("[Not available] Download not completed", "hash", infohash, "raw", rawSize, "complete", torrent.bytesCompleted)
			return false, errors.New("download not completed")
		}
		return torrent.BytesCompleted() <= rawSize, nil
	}*/
	return fs.storage().Available(infohash, rawSize)
}

/*func (fs *TorrentFS) release() {
	<-torrentInstance.fileCh
}

func (fs *TorrentFS) unzip(data []byte, c bool) ([]byte, error) {
	if c {
		return compress.UnzipData(data)
	} else {
		return data, nil
	}
}

func (fs *TorrentFS) zip(data []byte, c bool) ([]byte, error) {
	if c {
		return compress.ZipData(data)
	} else {
		return data, nil
	}
}*/

func (fs *TorrentFS) GetFile(infohash, subpath string) ([]byte, error) {
	return fs.storage().GetFile(infohash, subpath)
	/*if fs.metrics {
		defer func(start time.Time) { fs.fsUpdates += time.Since(start) }(time.Now())
	}
	ih := metainfo.NewHashFromHex(infohash)
	tm := fs.monitor.dl
	if torrent := tm.GetTorrent(ih); torrent == nil {
		log.Debug("Torrent not found", "hash", infohash)
		return nil, errors.New("download not completed")
	} else {

		if !torrent.IsAvailable() {
			log.Error("Read unavailable file", "hash", infohash, "subpath", subpath)
			return nil, errors.New("download not completed")
		}
		torrentInstance.fileCh <- true
		defer fs.release()
		var key = infohash + subpath
		if fs.cache {
			if cache, ok := fs.fileCache.Get(key); ok {
				if c, err := fs.unzip(cache.([]byte), fs.compress); err != nil {
					return nil, err
				} else {
					if fs.compress {
						log.Info("File cache", "hash", infohash, "path", subpath, "size", fs.fileCache.Len(), "compress", len(cache.([]byte)), "origin", len(c), "compress", fs.compress)
					}
					return c, nil
				}
			}
		}

		fs.fileLock.Lock()
		defer fs.fileLock.Unlock()
		fn := path.Join(fs.config.DataDir, infohash, subpath)
		data, err := ioutil.ReadFile(fn)
		for _, file := range torrent.Files() {
			log.Debug("File path info", "path", file.Path(), "subpath", subpath)
			if file.Path() == subpath[1:] {
				if int64(len(data)) != file.Length() {
					log.Error("Read file not completed", "hash", infohash, "len", len(data), "total", file.Path())
					return nil, errors.New("not a complete file")
				} else {
					log.Debug("Read data success", "hash", infohash, "size", len(data), "path", file.Path())
					if c, err := fs.zip(data, fs.compress); err != nil {
						log.Warn("Compress data failed", "hash", infohash, "err", err)
					} else {
						if fs.cache {
							fs.fileCache.Add(key, c)
						}
					}
					break
				}
			}
		}
		return data, err
	}*/
}
