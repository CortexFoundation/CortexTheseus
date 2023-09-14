package cuckoo

import (
	"errors"
	"math/big"
	"math/rand"
	"path/filepath"
	"plugin"
	"sync"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/consensus"
	// "github.com/CortexFoundation/CortexTheseus/consensus/cuckoo/plugins"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/metrics"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	gopsutil "github.com/shirou/gopsutil/mem"
)

var sharedCuckoo = New(Config{PowMode: ModeNormal})

var two256 = new(big.Int).Exp(big.NewInt(2), big.NewInt(256), big.NewInt(0))

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

	UseCuda bool
	//UseOpenCL    bool
	StrDeviceIds string
	Threads      int
	Algorithm    string

	Mine bool
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

	lock        sync.Mutex      // Ensures thread safety for the in-memory caches and mining fields
	once        sync.Once       // Ensures cuckoo-cycle algorithm initialize once
	closeOnce   sync.Once       // Ensures exit channel will not be closed twice.
	exitCh      chan chan error // Notification channel to exiting backend threads
	cMutex      sync.Mutex
	minerPlugin *plugin.Plugin

	wg sync.WaitGroup
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
	//if config.Mine {
	// miner algorithm use cuckaroo by default.
	cuckoo.wg.Add(1)
	go func() {
		defer cuckoo.wg.Done()
		cuckoo.remote()
	}()
	//}
	cuckoo.InitOnce()
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

func (cuckoo *Cuckoo) initPlugin() error {
	if !cuckoo.config.UseCuda {
		return nil
	}
	start := mclock.Now()
	var minerName string = "cpu"
	if cuckoo.config.UseCuda {
		minerName = "cuda"
		cuckoo.threads = 1
		//} else if cuckoo.config.UseOpenCL {
		//	minerName = "opencl"
		//	cuckoo.threads = 1
	}
	if cuckoo.config.StrDeviceIds == "" {
		cuckoo.config.StrDeviceIds = "0" //default gpu device 0
	}
	var errc error
	so_path := filepath.Join(PLUGIN_PATH, minerName+PLUGIN_POST_FIX)
	cuckoo.minerPlugin, errc = plugin.Open(so_path)
	if errc != nil {
		panic(errc)
	}

	elapsed := time.Duration(mclock.Now() - start)
	log.Info("Cuckoo Init Plugin", "name", minerName, "library path", so_path,
		"threads", cuckoo.threads, "device ids", cuckoo.config.StrDeviceIds, "elapsed", common.PrettyDuration(elapsed))
	return errc
}

func (cuckoo *Cuckoo) InitOnce() error {
	var err error
	cuckoo.once.Do(func() {
		if cuckoo.minerPlugin != nil {
			return
		}
		errc := cuckoo.initPlugin()
		if errc != nil {
			log.Error("Cuckoo Init Plugin", "error", errc)
			err = errc //errors.New("Cuckoo plugins init failed")
			return
		} else {
			// miner algorithm use cuckaroo by default.
			if cuckoo.config.Threads > 0 && cuckoo.config.UseCuda {
				m, errc := cuckoo.minerPlugin.Lookup("CuckooInitialize")
				if errc != nil {
					panic(errc)
				}
				errc = m.(func(int, string, string) error)(cuckoo.config.Threads, cuckoo.config.StrDeviceIds, cuckoo.config.Algorithm)
			} else {
				cuckoo.threads = 0
			}

			if mem, err := gopsutil.VirtualMemory(); err == nil {
				allowance := int(mem.Total / 1024 / 1024 / 3)
				log.Warn("Memory status", "total", mem.Total/1024/1024, "allowance", allowance, "cuda", cuckoo.config.UseCuda, "device", cuckoo.config.StrDeviceIds, "threads", cuckoo.config.Threads, "algo", cuckoo.config.Algorithm)
			}

			err = errc
		}
	})
	return err
}

// Close closes the exit channel to notify all backend threads exiting.
func (cuckoo *Cuckoo) Close() error {
	if cuckoo.exitCh != nil {
		close(cuckoo.exitCh)
	}

	cuckoo.wg.Wait()
	cuckoo.closeOnce.Do(func() {
		if cuckoo.minerPlugin != nil {
			m, e := cuckoo.minerPlugin.Lookup("CuckooFinalize")
			if e != nil {
				panic(e)
			}
			m.(func())()
			//		} else {
			//			plugins.CuckooFinalize()
		}
	})
	return nil
}

func (cuckoo *Cuckoo) Threads() int {
	cuckoo.lock.Lock()
	defer cuckoo.lock.Unlock()

	return cuckoo.threads
}

func (cuckoo *Cuckoo) Hashrate() float64 {
	return cuckoo.hashrate.Snapshot().Rate1()
}

func (cuckoo *Cuckoo) APIs(chain consensus.ChainHeaderReader) []rpc.API {
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
