package synapse

import (
	"os"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/ethereum/go-ethereum/infernet/parser"
	"github.com/ethereum/go-ethereum/log"
)

var synapseInstance *Synapse = nil
var defaultConfig Config = Config{
	StorageDir: "",
	IsNotCache: false,
}

type Config struct {
	StorageDir string
	IsNotCache bool
}

type Synapse struct {
	config Config

	simpleCache sync.Map

	exitCh  chan struct{}
	stopSig int32
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

	synapseInstance = &Synapse{
		config:  config,
		exitCh:  make(chan struct{}),
		stopSig: 0,
	}

	log.Info("Initialising Synapse Engine", "Storage Dir", config.StorageDir, "Cache Disabled", config.IsNotCache)
	return synapseInstance
}

func (s *Synapse) Close() {
	atomic.StoreInt32(&s.stopSig, 1)
	close(s.exitCh)
	log.Info("Synapse Engine Closed")
}

func (s *Synapse) VerifyModel(modelInfoHash string) error {
	modelHash := strings.ToLower(string(modelInfoHash[2:]))
	modelDir := s.config.StorageDir + "/" + modelHash

	modelCfg := modelDir + "/data/symbol"
	modelBin := modelDir + "/data/params"

	return parser.CheckModel(modelCfg, modelBin)
}

func (s *Synapse) VerifyInput(inputInfoHash string) error {
	inputHash := strings.ToLower(string(inputInfoHash[2:]))
	inputDir := s.config.StorageDir + "/" + inputHash

	image := inputDir + "/data"
	_, imageErr := os.Stat(image)

	return imageErr
}
