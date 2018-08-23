package cuckoo

/*
#cgo LDFLAGS:  -lstdc++ -lgominer
#include "gominer.h"
*/
import "C"
import (
	"errors"
	"math/big"
	"math/rand"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/metrics"
	"github.com/ethereum/go-ethereum/rpc"
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
	nonce     types.BlockNonce
	mixDigest common.Hash
	hash      common.Hash

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
	res  chan [3]string
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

	lock      sync.Mutex      // Ensures thread safety for the in-memory caches and mining fields
	closeOnce sync.Once       // Ensures exit channel will not be closed twice.
	exitCh    chan chan error // Notification channel to exiting backend threads
	cMutex    sync.Mutex
}

func New(config Config) *Cuckoo {
	C.CuckooInit()

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
	go cuckoo.remote()
	return cuckoo
}

func NewTester() *Cuckoo {
	cuckoo := New(Config{PowMode: ModeTest})
	// go cuckoo.remote()
	return cuckoo
}
func DeleteTester() {
	C.CuckooRelease()
}
func NewFaker() *Cuckoo {
	return &Cuckoo{
		config: Config{
			PowMode: ModeFake,
		},
	}
}

func NewFullFaker() *Cuckoo {
	return &Cuckoo{
		config: Config{
			PowMode: ModeFullFake,
		},
	}
}

// NewShared() func in tests/block_tests_util.go
func NewShared() *Cuckoo {
	return &Cuckoo{shared: sharedCuckoo}
}

// Close closes the exit channel to notify all backend threads exiting.
func (cuckoo *Cuckoo) Close() error {
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
	})
	return err
}

func (cuckoo *Cuckoo) Threads() int {
	cuckoo.lock.Lock()
	defer cuckoo.lock.Unlock()

	return cuckoo.threads
}

func (cuckoo *Cuckoo) SetThreads(threads int) {
	cuckoo.lock.Lock()
	defer cuckoo.lock.Unlock()

	// If we're running a shared PoW, set the thread count on that instead
	if cuckoo.shared != nil {
		cuckoo.shared.SetThreads(threads)
		return
	}
	// Update the threads and ping any running seal to pull in any changes
	cuckoo.threads = threads
	select {
	case cuckoo.update <- struct{}{}:
	default:
	}
}

func (cuckoo *Cuckoo) Hashrate() float64 {
	return cuckoo.hashrate.Rate1()
}

func Release(cuckoo *Cuckoo) {
	C.CuckooRelease()
}

func (cuckoo *Cuckoo) APIs(chain consensus.ChainReader) []rpc.API {
	// In order to ensure backward compatibility, we exposes cuckoo RPC APIs
	// to both eth and cuckoo namespaces.
	return []rpc.API{
		{
			Namespace: "cortex",
			Version:   "1.0",
			Service:   &API{cuckoo},
			Public:    true,
		},
		{
			Namespace: "cuckoo",
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
