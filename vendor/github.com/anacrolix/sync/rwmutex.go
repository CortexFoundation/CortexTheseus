//go:build !disable_pprof_sync

package sync

import (
	"sync"
)

// TODO: No lock times currently, was in the wrapped sync.Mutex before.
type RWMutex struct {
	inner   sync.RWMutex // Real McCoy
	holders mutexHolderSet
}

func (me *RWMutex) Lock() {
	if !contentionOn {
		me.inner.Lock()
		return
	}
	withBlocked(me.inner.Lock, me.inner.TryLock)
	me.addHolder()
}

func (me *RWMutex) TryLock() bool {
	if !me.inner.TryLock() {
		return false
	}
	if contentionOn {
		me.addHolder()
	}
	return true
}

func (me *RWMutex) Unlock() {
	me.removeHolder()
	me.inner.Unlock()
}

func (me *RWMutex) RLock() {
	if !contentionOn {
		me.inner.RLock()
		return
	}
	withBlocked(me.inner.RLock, me.inner.TryRLock)
	me.addHolder()
}
func (me *RWMutex) RUnlock() {
	me.removeHolder()
	me.inner.RUnlock()
}

func (me *RWMutex) TryRLock() bool {
	if !me.inner.TryRLock() {
		return false
	}
	if contentionOn {
		me.addHolder()
	}
	return true
}

func (me *RWMutex) addHolder() {
	key := addHolderProfile(1)
	me.holders.Add(key)
}

// Currently we just evict the last profile added. If it's a write lock there should only be one. If
// it's a read lock... Well that needs more context.
func (me *RWMutex) removeHolder() {
	if !contentionOn {
		return
	}
	// TODO: This could push to a special routine to reduce overhead.
	lockHolders.Remove(me.holders.Pop())
}
