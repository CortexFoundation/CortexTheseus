package synapse

import (
	"os"
	"strings"
	"sync"
	"plugin"
//	"github.com/CortexFoundation/CortexTheseus/inference/synapse/parser"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/common/lru"
)

var synapseInstance *Synapse = nil

const PLUGIN_PATH string = "plugins/"
const PLUGIN_POST_FIX string = "_cvm.so"

var defaultConfig Config = Config{
	StorageDir: "",
	IsNotCache: false,
	DeviceType: "cpu",
	DeviceId: 0,
}

type Config struct {
	StorageDir string
	IsNotCache bool
	DeviceType string
	DeviceId int
}

type Synapse struct {
	config Config
	simpleCache sync.Map
	lib *plugin.Plugin
	caches map[int]*lru.Cache
	exitCh chan struct{}
}

func Engine() *Synapse {
	if synapseInstance == nil {
		log.Error("Synapse Engine has not been initalized")
		return New(defaultConfig)
	}

	return synapseInstance
}

func New(config Config) *Synapse {
	if synapseInstance != nil {
		log.Warn("Synapse Engine has been initalized")
		return synapseInstance
	}

	lib, err := plugin.Open(PLUGIN_PATH + config.DeviceType + PLUGIN_POST_FIX)
	if err != nil {
		log.Error("infer helper", "init cvm plugin error", err)
		return nil         
	}

	synapseInstance = &Synapse{
		config: config,
		lib: lib,
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

func (s *Synapse) VerifyModel(modelInfoHash string) error {
	return nil
	// modelHash := strings.ToLower(string(modelInfoHash[2:]))
	// modelDir := s.config.StorageDir + "/" + modelHash

	// modelCfg := modelDir + "/data/symbol"
	// modelBin := modelDir + "/data/params"

	// return parser.CheckModel(modelCfg, modelBin)
}

func (s *Synapse) VerifyInput(inputInfoHash string) error {
	inputHash := strings.ToLower(string(inputInfoHash[2:]))
	inputDir := s.config.StorageDir + "/" + inputHash

	image := inputDir + "/data"
	_, imageErr := os.Stat(image)

	return imageErr
}
