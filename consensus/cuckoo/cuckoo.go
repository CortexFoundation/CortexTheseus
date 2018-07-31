package cuckoo

/*
#cgo LDFLAGS:  -lstdc++ -lgominer
#include "gominer.h"
*/
import "C"
import (
	"math/rand"
	"sync"

	"github.com/ethereum/go-ethereum/metrics"
)

type Mode uint

const (
	ModeNormal Mode = iota
	ModeShared
	ModeTest
	ModeFake
	ModeFullFake
)

type Config struct {
	PowMode Mode
}

type Cuckoo struct {
	config Config

	rand *rand.Rand
	// Current version allows single thread only
	threads  int
	update   chan struct{}
	hashrate metrics.Meter

	lock sync.Mutex // Ensures thread safety for the in-memory caches and mining fields
}

func New(config Config) *Cuckoo {
	C.CuckooInit()

	return &Cuckoo{
		config:   config,
		update:   make(chan struct{}),
		hashrate: metrics.NewMeter(),
		threads:  1,
	}
}

func NewTester() *Cuckoo {
	return New(Config{PowMode: ModeTest})
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

func (cuckoo *Cuckoo) Threads() int {
	cuckoo.lock.Lock()
	defer cuckoo.lock.Unlock()

	return cuckoo.threads
}

func (cuckoo *Cuckoo) SetThreads(threads int) {
	cuckoo.lock.Lock()
	defer cuckoo.lock.Unlock()

	// cuckoo.threads = threads
	cuckoo.threads = 1
}

func (cuckoo *Cuckoo) Hashrate() float64 {
	return cuckoo.hashrate.Rate1()
}

func Release(cuckoo *Cuckoo) {
	C.CuckooRelease()
}
