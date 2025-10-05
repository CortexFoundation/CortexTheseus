// SPDX-FileCopyrightText: 2023 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

package vnet

import (
	"errors"
	"math/rand"
	"sync"
	"time"
)

// Static errors for better error handling.
var (
	ErrInvalidChance           = errors.New("chance must be between 0 and 100 inclusive")
	ErrInvalidShuffleBlockSize = errors.New("shuffleBlockSize must be greater than 0")
)

type LossFilterHandler interface {
	shouldDrop() bool
	setLossRate(chance int, resetImmediately bool)
}

// LossFilter is a wrapper around NICs, that drops some of the packets passed to
// onInboundChunk.
type LossFilter struct {
	NIC
	LossFilterHandler
}

// RandomLossHandler drops packets randomly with a probability determined by the chance parameter.
type RandomLossHandler struct {
	chance int
	mutex  sync.RWMutex
}

// NewRandomLossHandler creates a new RandomLossHandler with the given drop chance.
func NewRandomLossHandler(chance int) (*RandomLossHandler, error) {
	if !validateChance(chance) {
		return nil, ErrInvalidChance
	}

	return &RandomLossHandler{
		chance: chance,
	}, nil
}

func (r *RandomLossHandler) shouldDrop() bool {
	r.mutex.RLock()
	chance := r.chance
	r.mutex.RUnlock()

	return rand.Intn(100) < chance //nolint:gosec
}

func (r *RandomLossHandler) setLossRate(chance int, _ bool) {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	r.chance = chance
}

// RandomShuffleLossHandler drops packets with a deterministic probability for every 100 packets
// That is, for every 100 packets, it guarantees that the number of packets dropped is equal to the chance parameter.
type RandomShuffleLossHandler struct {
	blockIdx      int
	shuffledBlock []bool
	currentChance int
	pendingChance int
	mutex         sync.Mutex
}

// NewRandomShuffleLossHandler creates a new RandomShuffleLossHandler with the given drop chance and shuffle block size.
// The default shuffle block size should be 100.
func NewRandomShuffleLossHandler(chance int, shuffleBlockSize int) (*RandomShuffleLossHandler, error) {
	if !validateChance(chance) {
		return nil, ErrInvalidChance
	}

	if shuffleBlockSize < 1 {
		return nil, ErrInvalidShuffleBlockSize
	}

	filter := RandomShuffleLossHandler{
		shuffledBlock: make([]bool, shuffleBlockSize),
		blockIdx:      0,
		currentChance: chance,
		pendingChance: chance,
	}

	for i := 0; i < filter.currentChance; i++ {
		filter.shuffledBlock[i] = true
	}

	filter.shuffleBlock()

	return &filter, nil
}

func (r *RandomShuffleLossHandler) setLossRate(chance int, resetImmediately bool) {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	r.pendingChance = chance

	if resetImmediately {
		r.shuffleBlock()
	}
}

func (r *RandomShuffleLossHandler) shuffleBlock() {
	for idx := 0; idx < len(r.shuffledBlock); idx++ {
		switch {
		case r.pendingChance == r.currentChance:
			goto shuffleComplete
		case r.pendingChance > r.currentChance && !r.shuffledBlock[idx]:
			r.shuffledBlock[idx] = true
			r.currentChance++
		case r.pendingChance < r.currentChance && r.shuffledBlock[idx]:
			r.shuffledBlock[idx] = false
			r.currentChance--
		}
	}

shuffleComplete:

	rand.Shuffle(len(r.shuffledBlock), func(i, j int) {
		r.shuffledBlock[i], r.shuffledBlock[j] = r.shuffledBlock[j], r.shuffledBlock[i]
	})
	r.blockIdx = 0
}

func (r *RandomShuffleLossHandler) shouldDrop() bool {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if r.blockIdx == len(r.shuffledBlock) {
		r.shuffleBlock()
	}

	res := r.shuffledBlock[r.blockIdx]
	r.blockIdx++

	return res
}

// LossFilterOption represents a configuration option for LossFilter creation.
type LossFilterOption func(nic NIC, chance int) (LossFilterHandler, error)

// WithLossHandler sets a custom loss handler for the LossFilter.
func WithLossHandler(handler LossFilterHandler) LossFilterOption {
	return func(_ NIC, chance int) (LossFilterHandler, error) {
		// Set the chance on the provided handler
		handler.setLossRate(chance, false)

		return handler, nil
	}
}

// WithShuffleLossHandler creates a LossFilter with a RandomShuffleLossHandler
// with the specified block size for deterministic packet loss distribution.
func WithShuffleLossHandler(blockSize int) LossFilterOption {
	return func(_ NIC, chance int) (LossFilterHandler, error) {
		return NewRandomShuffleLossHandler(chance, blockSize)
	}
}

// NewLossFilter creates a new LossFilter that drops every packet with a
// probability of chance/100 using the default RandomLossHandler.
// This maintains backward compatibility with the original API.
func NewLossFilter(nic NIC, chance int) (*LossFilter, error) {
	return NewLossFilterWithOptions(nic, chance)
}

// NewLossFilterWithOptions creates a new LossFilter that drops every packet with a
// probability of chance/100. You can provide custom options to override the
// default behavior. This follows the Pion options pattern for extensibility.
func NewLossFilterWithOptions(nic NIC, chance int, options ...LossFilterOption) (*LossFilter, error) {
	if !validateChance(chance) {
		return nil, ErrInvalidChance
	}

	var lossHandler LossFilterHandler
	var err error

	// If options are provided, use the first one to create the handler
	if len(options) > 0 {
		lossHandler, err = options[0](nic, chance)
		if err != nil {
			return nil, err
		}
	} else {
		// Create default handler
		lossHandler, err = NewRandomLossHandler(chance)
		if err != nil {
			return nil, err
		}
	}

	lossFilter := &LossFilter{
		NIC:               nic,
		LossFilterHandler: lossHandler,
	}

	//nolint:staticcheck
	rand.Seed(time.Now().UTC().UnixNano())

	return lossFilter, nil
}

func (f *LossFilter) onInboundChunk(c Chunk) {
	if f.LossFilterHandler.shouldDrop() {
		return
	}

	f.NIC.onInboundChunk(c)
}

// SetLossRate sets the loss rate for the loss filter.
// The chance parameter is an integer out of 100.
// The resetImmediately parameter is a boolean that indicates whether to reset the loss rate immediately.
// If resetImmediately is true, the loss rate will be reset immediately.
// If resetImmediately is false, the loss rate will be reset after the next shuffle for RandomShuffleLossHandler
// Note that for random loss handler, the loss rate will be reset immediately
// regardless of the resetImmediately parameter.
func (f *LossFilter) SetLossRate(chance int, resetImmediately bool) error {
	if !validateChance(chance) {
		return ErrInvalidChance
	}

	f.LossFilterHandler.setLossRate(chance, resetImmediately)

	return nil
}

func validateChance(chance int) bool {
	return chance >= 0 && chance <= 100
}
