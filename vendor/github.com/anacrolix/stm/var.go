package stm

import (
	"sync"
	"sync/atomic"
)

// Holds an STM variable.
type Var struct {
	state    atomic.Value
	watchers sync.Map
	mu       sync.Mutex
}

func (v *Var) loadState() varSnapshot {
	return v.state.Load().(varSnapshot)
}

func (v *Var) changeValue(new interface{}) {
	version := v.loadState().version
	v.state.Store(varSnapshot{version: version + 1, val: new})
	v.wakeWatchers()
}

func (v *Var) wakeWatchers() {
	v.watchers.Range(func(k, _ interface{}) bool {
		tx := k.(*Tx)
		tx.mu.Lock()
		tx.cond.Broadcast()
		tx.mu.Unlock()
		return true
	})
}

type varSnapshot struct {
	val     interface{}
	version uint64
}

// Returns a new STM variable.
func NewVar(val interface{}) *Var {
	v := &Var{}
	v.state.Store(varSnapshot{version: 0, val: val})
	return v
}
