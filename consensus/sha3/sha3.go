package sha3

import (
	"errors"
	"math/big"
	"math/rand"
	"sync"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/consensus"

	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/metrics"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	gopsutil "github.com/shirou/gopsutil/mem"
)

var sharedSHAThree = New(Config{PowMode: ModeNormal})

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
	ModeSha3
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

// compatiable with sha3 interface
type Config struct {
	CacheDir     string
	CachesInMem  int
	CachesOnDisk int

	DatasetDir     string
	DatasetsInMem  int
	DatasetsOnDisk int

	PowMode Mode

	BlockInterval time.Duration

	StrDeviceIds string
	Threads      int
	Algorithm    string

	Mine bool
}

type SHAThree struct {
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

	shared *SHAThree

	fakeFail  uint64        // Block number which fails PoW check even in fake mode
	fakeDelay time.Duration // Time delay to sleep for before returning from verify

	lock      sync.Mutex      // Ensures thread safety for the in-memory caches and mining fields
	once      sync.Once       // Ensures sha3-cycle algorithm initialize once
	closeOnce sync.Once       // Ensures exit channel will not be closed twice.
	exitCh    chan chan error // Notification channel to exiting backend threads
	cMutex    sync.Mutex

	wg sync.WaitGroup
}

func New(config Config) *SHAThree {
	sha3 := &SHAThree{
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
	sha3.wg.Add(1)
	go func() {
		defer sha3.wg.Done()
		sha3.remote()
	}()
	//}
	sha3.InitOnce()
	return sha3
}

func NewTester() *SHAThree {
	sha3 := New(Config{PowMode: ModeTest})
	// go sha3.remote()
	return sha3
}

func DeleteTester() {
	// C.SHAThreeRelease()

	//	SHAThreeFinalize()
}

// NewShared() func in tests/block_tests_util.go
func NewShared() *SHAThree {
	return &SHAThree{shared: sharedSHAThree}
}

func (sha3 *SHAThree) initPlugin() error {
	start := mclock.Now()
	var minerName string = "cpu"
	if sha3.config.StrDeviceIds == "" {
		sha3.config.StrDeviceIds = "0" //default gpu device 0
	}
	var errc error
	if errc != nil {
		panic(errc)
	}

	elapsed := time.Duration(mclock.Now() - start)
	log.Info("SHAThree Init Plugin", "name", minerName,
		"threads", sha3.threads, "device ids", sha3.config.StrDeviceIds, "elapsed", common.PrettyDuration(elapsed))
	return errc
}

func (sha3 *SHAThree) InitOnce() error {
	var err error
	sha3.once.Do(func() {
		errc := sha3.initPlugin()
		if errc != nil {
			log.Error("SHAThree Init Plugin", "error", errc)
			err = errc //errors.New("SHAThree plugins init failed")
			return
		} else {
			sha3.threads = 0
			if mem, err := gopsutil.VirtualMemory(); err == nil {
				allowance := int(mem.Total / 1024 / 1024 / 3)
				log.Warn("Memory status", "total", mem.Total/1024/1024, "allowance", allowance, "device", sha3.config.StrDeviceIds, "threads", sha3.config.Threads, "algo", sha3.config.Algorithm)
			}
			err = errc
		}
	})
	return err
}

// Close closes the exit channel to notify all backend threads exiting.
func (sha3 *SHAThree) Close() error {
	if sha3.exitCh != nil {
		close(sha3.exitCh)
	}

	sha3.wg.Wait()
	sha3.closeOnce.Do(func() {
	})
	return nil
}

func (sha3 *SHAThree) Threads() int {
	sha3.lock.Lock()
	defer sha3.lock.Unlock()

	return sha3.threads
}

func (sha3 *SHAThree) Hashrate() float64 {
	return sha3.hashrate.Rate1()
}

func (sha3 *SHAThree) APIs(chain consensus.ChainHeaderReader) []rpc.API {
	// In order to ensure backward compatibility, we exposes sha3 RPC APIs
	// to both ctxc and sha3 namespaces.
	return []rpc.API{
		{
			Namespace: "ctxc",
			Version:   "1.13",
			Service:   &API{sha3},
			Public:    true,
		},
	}
}

func SeedHash(block uint64) []byte {
	seed := make([]byte, 32)
	return seed
}
