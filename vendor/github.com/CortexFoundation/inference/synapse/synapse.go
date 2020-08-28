package synapse

import (
	"fmt"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/lru"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/cvm-runtime/kernel"
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
	PLUGIN_POST_FIX     string = "libcvm_runtime.so"
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
	if synapseInstance == nil {
		log.Warn("Synapse Engine has not been initalized, should new it first")
	}

	return synapseInstance
}

var mut sync.Mutex

func New(config *Config) *Synapse {
	mut.Lock()
	defer mut.Unlock()
	if synapseInstance != nil {
		log.Warn("Synapse Engine has been initalized", "synapse", synapseInstance, "config", config)
		return synapseInstance
	}
	var lib *kernel.LibCVM
	var status int
	if !config.IsRemoteInfer {
		path := PLUGIN_PATH + PLUGIN_POST_FIX
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

func (s *Synapse) InferByInfoHashWithSize(model, input common.StorageEntry, cvmVersion int, cvmNetworkID int64) ([]byte, error) {
	if s.config.IsRemoteInfer {
		return s.remoteInferByInfoHashWithSize(model.Hash, input.Hash, model.Size, input.Size, cvmVersion, cvmNetworkID)
	}
	return s.inferByInfoHashWithSize(model.Hash, input.Hash, model.Size, input.Size, cvmVersion, cvmNetworkID)
}

func (s *Synapse) InferByInputContentWithSize(model common.StorageEntry, inputContent []byte, cvmVersion int, cvmNetworkID int64) ([]byte, error) {
	if s.config.IsRemoteInfer {
		return s.remoteInferByInputContentWithSize(model.Hash, inputContent, model.Size, cvmVersion, cvmNetworkID)
	}
	return s.inferByInputContentWithSize(model.Hash, inputContent, model.Size, cvmVersion, cvmNetworkID)
}

func (s *Synapse) GetGasByInfoHashWithSize(model common.StorageEntry, cvmNetworkID int64) (gas uint64, err error) {
	if s.config.IsRemoteInfer {
		return s.remoteGasByModelHashWithSize(model.Hash, model.Size, cvmNetworkID)
	}
	return s.getGasByInfoHashWithSize(model.Hash, model.Size, cvmNetworkID)
}

func (s *Synapse) Available(entry common.StorageEntry, cvmNetworkID int64) error {
	if s.config.IsRemoteInfer {
		return s.remoteAvailable(entry.Hash, entry.Size, cvmNetworkID)
	}
	return s.available(entry.Hash, entry.Size, cvmNetworkID)
}

// Download is used to control the torrentfs, not for remote invoked now
func (s *Synapse) Download(info common.StorageEntry) error {
	if s.config.IsRemoteInfer {
		return nil
	}
	return s.download(info.Hash, info.Size)
}
