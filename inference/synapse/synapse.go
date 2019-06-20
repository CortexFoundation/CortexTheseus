package synapse

import (
	"fmt"
	"os"
	"plugin"
	"strconv"
	"strings"
	"sync"
	//	"github.com/CortexFoundation/CortexTheseus/inference/synapse/parser"
	"github.com/CortexFoundation/CortexTheseus/common/lru"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/torrentfs"
)

var synapseInstance *Synapse = nil

const PLUGIN_PATH string = "plugins/"
const PLUGIN_POST_FIX string = "_cvm.so"


type Config struct {
	StorageDir    string `toml:",omitempty"`
	IsNotCache    bool   `toml:",omitempty"`
	DeviceType    string `toml:",omitempty"`
	DeviceId      int    `toml:",omitempty"`
	IsRemoteInfer bool   `toml:",omitempty"`
	InferURI      string `toml:",omitempty"`
	Debug         bool   `toml:",omitempty"`
	Storagefs torrentfs.CVMStorage
}

var DefaultConfig Config = Config{
	StorageDir:    "",
	IsNotCache:    false,
	DeviceType:    "cpu",
	DeviceId:      0,
	IsRemoteInfer: false,
	InferURI:      "",
	Debug:         false,
}

type Synapse struct {
	config      *Config
	simpleCache sync.Map
	lib         *plugin.Plugin
	caches      map[int]*lru.Cache
	exitCh      chan struct{}
}

func Engine() *Synapse {
	if synapseInstance == nil {
		log.Error("Synapse Engine has not been initalized")
		return New(&DefaultConfig)
	}

	return synapseInstance
}

func New(config *Config) *Synapse {
	path := PLUGIN_PATH + config.DeviceType + PLUGIN_POST_FIX
	var lib *plugin.Plugin = nil
	// fmt.Println("config ", config, "synapseInstance ", synapseInstance)
	if synapseInstance != nil {
		log.Warn("Synapse Engine has been initalized")
		if config.Debug {
			fmt.Println("Synapse Engine has been initalized")
		}
		return synapseInstance
	}

	if !config.IsRemoteInfer {
		var err error = nil
		// fmt.Println("path ", path)
		lib, err = plugin.Open(path)
		if err != nil {
			log.Error("infer helper", "init cvm plugin error", err)
			if config.Debug {
				fmt.Println("infer helper", "init cvm plugin error", err)
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
		exitCh: make(chan struct{}),
		caches: make(map[int]*lru.Cache),
	}

	log.Info("Initialising Synapse Engine", "Storage Dir", config.StorageDir, "Cache Disabled", config.IsNotCache)
	return synapseInstance
}

func (s *Synapse) Close() {
	close(s.exitCh)
	log.Info("Synapse Engine Closed")
}

func (s *Synapse) VerifyInput(inputInfoHash string) error {
	inputHash := strings.ToLower(string(inputInfoHash[2:]))
	inputDir := s.config.StorageDir + "/" + inputHash

	image := inputDir + "/data"
	_, imageErr := os.Stat(image)

	return imageErr
}
