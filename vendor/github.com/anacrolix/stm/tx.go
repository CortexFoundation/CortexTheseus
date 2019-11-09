package stm

import (
	"sync"
)

// A Tx represents an atomic transaction.
type Tx struct {
	reads  map[*Var]uint64
	writes map[*Var]interface{}
	cond   sync.Cond
}

// Check that none of the logged values have changed since the transaction began.
func (tx *Tx) verify() bool {
	for v, version := range tx.reads {
		v.mu.Lock()
		changed := v.version != version
		v.mu.Unlock()
		if changed {
			return false
		}
	}
	return true
}

// Writes the values in the transaction log to their respective Vars.
func (tx *Tx) commit() {
	for v, val := range tx.writes {
		v.mu.Lock()
		v.val = val
		v.version++
		v.mu.Unlock()
		for tx := range v.watchers {
			tx.cond.Broadcast()
			delete(v.watchers, tx)
		}
	}
}

// wait blocks until another transaction modifies any of the Vars read by tx.
func (tx *Tx) wait() {
	globalLock.Lock()
	for v := range tx.reads {
		v.watchers[tx] = struct{}{}
	}
	for tx.verify() {
		tx.cond.Wait()
	}
	for v := range tx.reads {
		delete(v.watchers, tx)
	}
	globalLock.Unlock()
}

// Get returns the value of v as of the start of the transaction.
func (tx *Tx) Get(v *Var) interface{} {
	// If we previously wrote to v, it will be in the write log.
	if val, ok := tx.writes[v]; ok {
		return val
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	// If we haven't previously read v, record its version
	if _, ok := tx.reads[v]; !ok {
		tx.reads[v] = v.version
	}
	return v.val
}

// Set sets the value of a Var for the lifetime of the transaction.
func (tx *Tx) Set(v *Var, val interface{}) {
	if v == nil {
		panic("nil Var")
	}
	tx.writes[v] = val
}

// Retry aborts the transaction and retries it when a Var changes.
func (tx *Tx) Retry() {
	panic(Retry)
}

// Assert is a helper function that retries a transaction if the condition is
// not satisfied.
func (tx *Tx) Assert(p bool) {
	if !p {
		tx.Retry()
	}
}

func (tx *Tx) Return(v interface{}) {
	panic(_return{v})
}

type _return struct {
	value interface{}
}

func (tx *Tx) reset() {
	for k := range tx.reads {
		delete(tx.reads, k)
	}
	for k := range tx.writes {
		delete(tx.writes, k)
	}
}

func (tx *Tx) recycle() {
	tx.reset()
	txPool.Put(tx)
}
