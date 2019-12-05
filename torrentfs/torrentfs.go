package torrentfs

import (
	"fmt"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"github.com/anacrolix/torrent/metainfo"
	"io/ioutil"
	"path"
	"sync"
	//"strings"
	"errors"
	"github.com/CortexFoundation/CortexTheseus/common/compress"
	lru "github.com/hashicorp/golang-lru"
)

type CVMStorage interface {
	Available(infohash string, rawSize int64) (bool, error)
	GetFile(infohash string, path string) ([]byte, error)
	Stop() error
}
type GeneralMessage struct {
	Version string `json:"version,omitempty"`
	Commit  string `json:"commit,omitempty"`
}

// TorrentFS contains the torrent file system internals.
type TorrentFS struct {
	config  *Config
	history *GeneralMessage
	monitor *Monitor

	fileLock  sync.Mutex
	fileCache *lru.Cache
	fileCh    chan bool
	compress  bool
}

func (t *TorrentFS) Config() *Config {
	return t.config
}

func (t *TorrentFS) Monitor() *Monitor {
	return t.monitor
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

func GetConfig() *Config {
	if torrentInstance != nil {
		return torrentInstance.Config()
	} else {
		return &DefaultConfig
	}
	return nil
}

// New creates a new dashboard instance with the given configuration.
//var Torrentfs_handle CVMStorage

// New creates a new torrentfs instance with the given configuration.
func New(config *Config, commit string) (*TorrentFS, error) {
	if torrentInstance != nil {
		return torrentInstance, nil
	}

	//versionMeta := ""
	//TorrentAPIAvailable.Lock()
	//if len(params.VersionMeta) > 0 {
	//	versionMeta = fmt.Sprintf(" (%s)", params.VersionMeta)
	//}

	msg := &GeneralMessage{
		Commit:  commit,
		Version: fmt.Sprintf("v%d.%d.%d-%s", params.VersionMajor, params.VersionMinor, params.VersionPatch, params.VersionMeta),
	}

	log.Info("Fs version info", "version", msg.Version)

	monitor, moErr := NewMonitor(config)
	if moErr != nil {
		log.Error("Failed create monitor")
		return nil, moErr
	}

	torrentInstance = &TorrentFS{
		config:  config,
		history: msg,
		monitor: monitor,
	}
	torrentInstance.fileCache, _ = lru.New(8)
	torrentInstance.fileCh = make(chan bool, 4)
	//Torrentfs_handle = torrentInstance
	//torrentInstance.compress = true

	return torrentInstance, nil
}

// Protocols implements the node.Service interface.
func (tfs *TorrentFS) Protocols() []p2p.Protocol { return nil }

// APIs implements the node.Service interface.
func (tfs *TorrentFS) APIs() []rpc.API { return nil }

// Start starts the data collection thread and the listening server of the dashboard.
// Implements the node.Service interface.
func (tfs *TorrentFS) Start(server *p2p.Server) error {
	log.Info("Fs monitor starting", "torrentfs", tfs)
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

func (fs *TorrentFS) Available(infohash string, rawSize int64) (bool, error) {
	// modelDir := fs.DataDir + "/" + infoHash
	// if (os.Stat)
	//return Available(infohash, rawSize)
	//log.Info("Available", "infohash", infohash, "RawSize", rawSize)
	//TorrentAPIAvailable.Lock()
	//defer TorrentAPIAvailable.Unlock()
	//if !strings.HasPrefix(infohash, "0x") {
	//      return false, errors.New("invalid info hash format")
	//}
	ih := metainfo.NewHashFromHex(infohash)
	tm := fs.monitor.dl //CurrentTorrentManager
	//log.Debug("storage", "ih", ih)
	if torrent := tm.GetTorrent(ih); torrent == nil {
		//log.Debug("storage", "ih", ih, "torrent", torrent)
		log.Debug("Seed not found", "hash", infohash)
		return false, errors.New("download not completed")
	} else {
		if !torrent.IsAvailable() {
			log.Debug("[Not available] Download not completed", "hash", infohash, "raw", rawSize, "complete", torrent.BytesCompleted())
			return false, fmt.Errorf("download not completed: %d %d", torrent.BytesCompleted(), rawSize)
		}
		//log.Debug("storage", "Available", torrent.IsAvailable(), "torrent.BytesCompleted()", torrent.BytesCompleted(), "rawSize", rawSize)
		//log.Info("download not completed", "complete", torrent.BytesCompleted(), "miss", torrent.BytesMissing(), "raw", rawSize)
		return torrent.BytesCompleted() <= rawSize, nil
	}
}

func (fs *TorrentFS) release() {
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
}

func (fs *TorrentFS) GetFile(infohash string, subpath string) ([]byte, error) {
	ih := metainfo.NewHashFromHex(infohash)
	tm := fs.monitor.dl //CurrentTorrentManager
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

		if cache, ok := fs.fileCache.Get(key); ok {
			//log.Trace("File cache", "hash", infohash, "path", subpath, "size", fs.fileCache.Len())
			if c, err := fs.unzip(cache.([]byte), fs.compress); err != nil {
				return nil, err
			} else {
				if fs.compress {
					log.Info("File cache", "hash", infohash, "path", subpath, "size", fs.fileCache.Len(), "compress", len(cache.([]byte)), "origin", len(c), "compress", fs.compress)
				}
				return c, nil
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
					log.Debug("Read data sucess", "hash", infohash, "size", len(data), "path", file.Path())
					if c, err := fs.zip(data, fs.compress); err != nil {
						log.Warn("Compress data failed", "hash", infohash, "err", err)
					} else {
						fs.fileCache.Add(key, c)
					}
					break
				}
			}
		}
		/*
			if subpath == "/data" {
				if int64(len(data)) != torrent.BytesCompleted() {
					log.Error("Read file not completed", "hash", infohash, "len", len(data), "total", torrent.BytesCompleted())
					return nil, errors.New("not a complete file")
				} else {
					log.Warn("Read data sucess", "hash", infohash, "size", len(data), "path", subpath)
				}
			} else if subpath == "/data/symbol" {
				for _, file := range torrent.Files() {
					if file.Path() == "/data/symbol" {
						if int64(len(data)) != file.Length() {
							log.Error("Read file not completed", "hash", infohash, "len", len(data), "total", file.Path())
							return nil, errors.New("not a complete file")
						} else {
							log.Warn("Read data sucess", "hash", infohash, "size", len(data), "path", file.Path())
						}
					}
				}
			} else if subpath == "/data/params" {
				for _, file := range torrent.Files() {
					if file.Path() == "/data/params" {
						if int64(len(data)) != file.Length() {
							log.Error("Read file not completed", "hash", infohash, "len", len(data), "total", file.Path())
							return nil, errors.New("not a complete file")
						} else {
							log.Warn("Read data sucess", "hash", infohash, "size", len(data), "path", file.Path())
						}
					}
				}
			}*/
		return data, err
	}
}
