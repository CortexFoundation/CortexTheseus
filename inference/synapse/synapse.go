package synapse

import (
	"fmt"
	"github.com/CortexFoundation/CortexTheseus/common/lru"
	"github.com/CortexFoundation/CortexTheseus/cvm-runtime/kernel"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/torrentfs"
	resty "github.com/go-resty/resty/v2"
	"math/big"
	"strconv"
	"sync"
)

var (
	synapseInstance *Synapse = nil

	DefaultConfig Config = Config{
		// StorageDir:    "",
		IsNotCache:     false,
		DeviceType:     "cpu",
		DeviceId:       0,
		IsRemoteInfer:  false,
		InferURI:       "",
		Debug:          false,
		MaxMemoryUsage: 4 * 1024 * 1024 * 1024,
	}
)

const (
	PLUGIN_PATH         string = "plugins/"
	PLUGIN_POST_FIX     string = "lib_cvm.so"
	MinMemoryUsage      int64  = 2 * 1024 * 1024 * 1024
	ReservedMemoryUsage int64  = 512 * 1024 * 1024
)

type Config struct {
	// StorageDir    string `toml:",omitempty"`
	IsNotCache     bool   `toml:",omitempty"`
	DeviceType     string `toml:",omitempty"`
	DeviceId       int    `toml:",omitempty"`
	IsRemoteInfer  bool   `toml:",omitempty"`
	InferURI       string `toml:",omitempty"`
	Debug          bool   `toml:",omitempty"`
	MaxMemoryUsage int64
	Storagefs      torrentfs.CortexStorage
}

type Synapse struct {
	config      *Config
	simpleCache sync.Map
	gasCache    sync.Map
	//modelLock   sync.Map
	mutex  sync.Mutex
	lib    *kernel.LibCVM
	caches map[int]*lru.Cache
	//exitCh chan struct{}
	client *resty.Client
	//ctx    context.Context
}

func Engine() *Synapse {
	/*if synapseInstance == nil {
		log.Error("Synapse Engine has not been initalized")
		return New(&DefaultConfig)
	}*/

	return synapseInstance
}

func New(config *Config) *Synapse {
	// path := PLUGIN_PATH + config.DeviceType + PLUGIN_POST_FIX
	path := PLUGIN_PATH + PLUGIN_POST_FIX
	if synapseInstance != nil {
		log.Warn("Synapse Engine has been initalized")
		if config.Debug {
			fmt.Println("Synapse Engine has been initalized")
		}
		return synapseInstance
	}
	var lib *kernel.LibCVM
	var status int
	if !config.IsRemoteInfer {
		lib, status = kernel.LibOpen(path)
		if status != kernel.SUCCEED {
			log.Error("infer helper", "init cvm plugin error", "")
			if config.Debug {
				fmt.Println("infer helper", "init cvm plugin error", "")
			}
			return nil
		}
		if lib == nil {
			panic("lib_path = " + PLUGIN_PATH + config.DeviceType + PLUGIN_POST_FIX + " config.IsRemoteInfer = " + strconv.FormatBool(config.IsRemoteInfer))
		}
	}

	synapseInstance = &Synapse{
		config: config,
		lib:    lib,
		//exitCh: make(chan struct{}),
		caches: make(map[int]*lru.Cache),
	}

	if synapseInstance.config.IsRemoteInfer {
		synapseInstance.client = resty.New()
	}

	//synapseInstance.ctx = context.Background()

	log.Info("Initialising Synapse Engine", "Cache Disabled", config.IsNotCache)
	return synapseInstance
}

func (s *Synapse) Close() {
	//close(s.exitCh)
	if s.config.Storagefs != nil {
		s.config.Storagefs.Stop()
	}
	for _, c := range s.caches {
		if c != nil {
			c.Clear()
		}
	}
	log.Info("Synapse Engine Closed")
}

func CVMVersion(config *params.ChainConfig, num *big.Int) int {
	// TODO(ryt): For Istanbul and versions after Istanbul, return CVM_VERSION_TWO
	version := kernel.CVM_VERSION_ONE
	if config.IsIstanbul(num) {
		version = kernel.CVM_VERSION_TWO
	}
	return version
}

func (s *Synapse) InferByInfoHashWithSize(modelInfoHash, inputInfoHash string, modelSize uint64, inputSize uint64, cvmVersion int, cvmNetworkID int64) ([]byte, error) {
	if s.config.IsRemoteInfer {
		return s.remoteInferByInfoHashWithSize(modelInfoHash, inputInfoHash, modelSize, inputSize, cvmVersion, cvmNetworkID)
	}
	return s.inferByInfoHashWithSize(modelInfoHash, inputInfoHash, modelSize, inputSize, cvmVersion, cvmNetworkID)
}

func (s *Synapse) InferByInputContentWithSize(modelInfoHash string, inputContent []byte, modelSize uint64, cvmVersion int, cvmNetworkID int64) ([]byte, error) {
	if s.config.IsRemoteInfer {
		return s.remoteInferByInputContentWithSize(modelInfoHash, inputContent, modelSize, cvmVersion, cvmNetworkID)
	}
	return s.inferByInputContentWithSize(modelInfoHash, inputContent, modelSize, cvmVersion, cvmNetworkID)
}

func (s *Synapse) GetGasByInfoHashWithSize(modelInfoHash string, modelSize uint64, cvmNetworkID int64) (gas uint64, err error) {
	if s.config.IsRemoteInfer {
		return s.remoteGasByModelHashWithSize(modelInfoHash, modelSize, cvmNetworkID)
	}
	return s.getGasByInfoHashWithSize(modelInfoHash, modelSize, cvmNetworkID)
}

func (s *Synapse) Available(infoHash string, rawSize uint64, cvmNetworkID int64) error {
	if s.config.IsRemoteInfer {
		return s.remoteAvailable(infoHash, rawSize, cvmNetworkID)
	}
	return s.available(infoHash, rawSize, cvmNetworkID)
}

// Download is used to control the torrentfs, not for remote invoked now
func (s *Synapse) Download(infohash string, request uint64) error {
	if s.config.IsRemoteInfer {
		return nil
	}
	return s.download(infohash, request)
}
