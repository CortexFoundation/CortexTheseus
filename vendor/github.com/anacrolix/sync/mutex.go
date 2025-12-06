//go:build !disable_pprof_sync

package sync

import (
	"runtime"
	"sync"
	"time"
	"unique"
)

type Mutex struct {
	hold profileKey // Unique value for passing to pprof.
	// Values if lockTimes tracking is enabled.
	*lockTimes
	// Last for struct size reasons.
	mu sync.Mutex
}

func (me *Mutex) Lock() {
	if !contentionOn {
		me.mu.Lock()
		return
	}
	withBlocked(me.mu.Lock, me.mu.TryLock)
	me.hold = addHolderProfile(0)
	me.startLockTime()
}

func (me *Mutex) TryLock() bool {
	if !me.mu.TryLock() {
		return false
	}
	if contentionOn {
		me.hold = addHolderProfile(0)
		me.startLockTime()
	}
	return true
}

func (me *Mutex) startLockTime() {
	if !lockTimesOn {
		return
	}
	// We're holding the lock here so it's safe to check.
	if me.lockTimes == nil {
		me.lockTimes = new(lockTimes)
	}
	var stack callerArray
	me.entries = runtime.Callers(2, stack[:])
	me.stack = unique.Make(stack)
	me.start = time.Now()
}

func (me *Mutex) Unlock() {
	if lockTimesOn {
		d := time.Since(me.start)
		key := me.stack.Value()
		lockStatsMu.Lock()
		v, ok := lockStatsByStack[key]
		if !ok {
			v.Init()
		}
		v.Add(d)
		lockStatsByStack[key] = v
		lockStatsMu.Unlock()
	}
	removeHolder(me.hold)
	me.mu.Unlock()
}
