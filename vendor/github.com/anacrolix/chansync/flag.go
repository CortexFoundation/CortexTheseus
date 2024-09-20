package chansync

import (
	"sync"
	"sync/atomic"

	"github.com/anacrolix/chansync/events"
)

type ReadOnlyFlag interface {
	Bool() bool
	On() events.Active
	Off() events.Active
}

// Flag wraps a boolean value that starts as false (off). You can wait for it to be on or off, and
// set the value as needed.
type Flag struct {
	mu sync.Mutex
	// It could be possible to optimize this to only allocate channels when one doesn't exist.
	on    chan struct{}
	off   chan struct{}
	state atomic.Bool
	// It could be possible to optimize this away to just checking if the desired on or off channel
	// is present.
	inited bool
}

// To match the SetOnce API.
func (me *Flag) IsSet() bool {
	return me.Bool()
}

func (me *Flag) Bool() bool {
	return me.state.Load()
}

func (me *Flag) On() events.Active {
	me.mu.Lock()
	defer me.mu.Unlock()
	me.init()
	return me.on
}

func (me *Flag) Off() events.Active {
	me.mu.Lock()
	defer me.mu.Unlock()
	me.init()
	return me.off
}

// Everywhere this is called then checks state and potentially modifies channels so it's probably
// not worth using sync.Once.
func (me *Flag) init() {
	if me.inited {
		return
	}
	me.on = make(chan struct{})
	me.off = make(chan struct{})
	close(me.off)
	me.inited = true
}

func (me *Flag) SetBool(b bool) {
	if b {
		me.Set()
	} else {
		me.Clear()
	}
}

func (me *Flag) Set() {
	if me.state.Load() {
		return
	}
	me.mu.Lock()
	defer me.mu.Unlock()
	me.init()
	if !me.state.CompareAndSwap(false, true) {
		return
	}
	close(me.on)
	me.off = make(chan struct{})
}

func (me *Flag) Clear() {
	if !me.state.Load() {
		return
	}
	me.mu.Lock()
	defer me.mu.Unlock()
	me.init()
	if !me.state.CompareAndSwap(true, false) {
		return
	}
	close(me.off)
	me.on = make(chan struct{})
}
