package cuckoo

import (
	"errors"
	"math/big"
	"math/rand"
	"sync"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/consensus"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/metrics"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"plugin"
)

var sharedCuckoo = New(Config{PowMode: ModeNormal})

var ErrInvalidDumpMagic = errors.New("invalid dump magic")

var (
	// maxUint256 is a big integer representing 2^256-1
	maxUint256 = new(big.Int).Exp(big.NewInt(2), big.NewInt(256), big.NewInt(0))
)

type Mode uint

const (
	ModeNormal Mode = iota
	ModeShared
	ModeTest
	ModeFake
	ModeFullFake
)

// mineResult wraps the pow solution parameters for the specified block.
type mineResult struct {
	nonce types.BlockNonce
	//mixDigest common.Hash
	hash     common.Hash
	solution types.BlockSolution

	errc chan error
}

// hashrate wraps the hash rate submitted by the remote sealer.
type hashrate struct {
	id   common.Hash
	ping time.Time
	rate uint64

	done chan struct{}
}

// sealWork wraps a seal work package for remote sealer.
type sealWork struct {
	errc chan error
	res  chan [4]string
}

// compatiable with cuckoo interface
type Config struct {
	CacheDir     string
	CachesInMem  int
	CachesOnDisk int

	DatasetDir     string
	DatasetsInMem  int
	DatasetsOnDisk int

	PowMode Mode

	UseCuda      bool
	UseOpenCL    bool
	StrDeviceIds string
	Threads      int
	Algorithm    string
}

type Cuckoo struct {
	config Config

	rand *rand.Rand
	// Current version allows single thread only
	threads  int
	update   chan struct{}
	hashrate metrics.Meter

	// Remote sealer related fields
	workCh       chan *types.Block // Notification channel to push new work to remote sealer
	resultCh     chan *types.Block // Channel used by mining threads to return result
	fetchWorkCh  chan *sealWork    // Channel used for remote sealer to fetch mining work
	submitWorkCh chan *mineResult  // Channel used for remote sealer to submit their mining result
	fetchRateCh  chan chan uint64  // Channel used to gather submitted hash rate for local or remote sealer.
	submitRateCh chan *hashrate    // Channel used for remote sealer to submit their mining hashrate

	shared *Cuckoo

	fakeFail  uint64        // Block number which fails PoW check even in fake mode
	fakeDelay time.Duration // Time delay to sleep for before returning from verify

	lock          sync.Mutex      // Ensures thread safety for the in-memory caches and mining fields
	once          sync.Once       // Ensures cuckoo-cycle algorithm initialize once
	closeOnce     sync.Once       // Ensures exit channel will not be closed twice.
	exitCh        chan chan error // Notification channel to exiting backend threads
	cMutex        sync.Mutex
	minerPlugin   *plugin.Plugin
	xcortexPlugin *plugin.Plugin
}

func New(config Config) *Cuckoo {
	// C.CuckooInit()
	// CuckooInit(2)

	cuckoo := &Cuckoo{
		config:       config,
		update:       make(chan struct{}),
		hashrate:     metrics.NewMeter(),
		threads:      1,
		workCh:       make(chan *types.Block),
		resultCh:     make(chan *types.Block),
		fetchWorkCh:  make(chan *sealWork),
		submitWorkCh: make(chan *mineResult),
		fetchRateCh:  make(chan chan uint64),
		submitRateCh: make(chan *hashrate),
		exitCh:       make(chan chan error),
	}
	cuckoo.InitPlugin()
	go cuckoo.remote()
	return cuckoo
}

func NewTester() *Cuckoo {
	cuckoo := New(Config{PowMode: ModeTest})
	// go cuckoo.remote()
	return cuckoo
}

func DeleteTester() {
	// C.CuckooRelease()

	//	CuckooFinalize()
}

// NewShared() func in tests/block_tests_util.go
func NewShared() *Cuckoo {
	return &Cuckoo{shared: sharedCuckoo}
}

const PLUGIN_PATH string = "plugins/"
const PLUGIN_POST_FIX string = "_helper_for_node.so"
const XCORTEX_PLUGIN string = "plugins/xcortex_helper.so"
const CUCKAROO = "cuckaroo"
const XCORTEX = "xcortex"

func (cuckoo *Cuckoo) InitPlugin() error {
	var errc error
	if cuckoo.config.Algorithm == CUCKAROO {
		var minerName string = "cpu"
		if cuckoo.config.UseCuda == true {
			minerName = "cuda"
			cuckoo.threads = 1
		} else if cuckoo.config.UseOpenCL == true {
			minerName = "opencl"
			cuckoo.threads = 1
		}
		if cuckoo.config.StrDeviceIds == "" {
			cuckoo.config.StrDeviceIds = "0" //default gpu device 0
		}
		so_path := PLUGIN_PATH + minerName + PLUGIN_POST_FIX
		log.Info("Cuckoo Init Plugin", "name", minerName, "library path", so_path,
			"threads", cuckoo.threads, "device ids", cuckoo.config.StrDeviceIds)
		if cuckoo.minerPlugin == nil {
			cuckoo.minerPlugin, errc = plugin.Open(so_path)
		}
	} else {
		if cuckoo.xcortexPlugin == nil {
			log.Info("Cuckoo Init Plugin", "name", "xcortex", "library path", XCORTEX_PLUGIN)
			cuckoo.xcortexPlugin, errc = plugin.Open(XCORTEX_PLUGIN)
			if errc != nil {
				log.Info("cuckoo init plugin", "error", errc)
			}
		}
	}
	return errc
}

/*func (cuckoo *Cuckoo) InitOnce() error {
	var err error
	cuckoo.once.Do(func() {
		errc := cuckoo.InitPlugin()
		if errc != nil {
			log.Error("Cuckoo Init Plugin", "error", errc)
			err = errc
			return
		} else {
			m, errc := cuckoo.minerPlugin.Lookup("CuckooInitialize")
			if err != nil {
				log.Error("Cuckoo Init Plugin", "error", errc)
				err = errc
				return
			}
			// miner algorithm use cuckaroo by default.
			errc = m.(func(int, string, string) error)(cuckoo.config.Threads, cuckoo.config.StrDeviceIds, "cuckaroo")
			err = errc
		}
	})
	return err
}*/

// Close closes the exit channel to notify all backend threads exiting.
func (cuckoo *Cuckoo) Close() error {
	close(cuckoo.exitCh)

	if cuckoo.minerPlugin == nil {
		return nil
	}
	m, e := cuckoo.minerPlugin.Lookup("CuckooFinalize")
	if e != nil {
		return e
	}
	m.(func())()
	return nil
	/*
		var err error
		cuckoo.closeOnce.Do(func() {
			// Short circuit if the exit channel is not allocated.
			if cuckoo.exitCh == nil {
				return
			}
			errc := make(chan error)
			cuckoo.exitCh <- errc
			err = <-errc
			close(cuckoo.exitCh)

			if cuckoo.minerPlugin == nil {
				return
			}
			m, e := cuckoo.minerPlugin.Lookup("CuckooFinalize")
			if e != nil {
				err = e
				return
			}
			m.(func())()
		})
		return err
	*/
}

func (cuckoo *Cuckoo) Threads() int {
	cuckoo.lock.Lock()
	defer cuckoo.lock.Unlock()

	return cuckoo.threads
}

func (cuckoo *Cuckoo) Hashrate() float64 {
	return cuckoo.hashrate.Rate1()
}

func (cuckoo *Cuckoo) APIs(chain consensus.ChainReader) []rpc.API {
	// In order to ensure backward compatibility, we exposes cuckoo RPC APIs
	// to both ctxc and cuckoo namespaces.
	return []rpc.API{
		{
			Namespace: "ctxc",
			Version:   "1.0",
			Service:   &API{cuckoo},
			Public:    true,
		},
	}
}

func SeedHash(block uint64) []byte {
	seed := make([]byte, 32)
	return seed
}
