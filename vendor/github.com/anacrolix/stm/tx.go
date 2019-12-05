package stm

import (
	"fmt"
	"sort"
	"sync"
	"unsafe"
)

// A Tx represents an atomic transaction.
type Tx struct {
	reads  map[*Var]uint64
	writes map[*Var]interface{}
	locks  []*sync.Mutex
	mu     sync.Mutex
	cond   sync.Cond
}

// Check that none of the logged values have changed since the transaction began.
func (tx *Tx) verify() bool {
	for v, version := range tx.reads {
		changed := v.loadState().version != version
		if changed {
			return false
		}
	}
	return true
}

// Writes the values in the transaction log to their respective Vars.
func (tx *Tx) commit() {
	for v, val := range tx.writes {
		v.changeValue(val)
	}
}

// wait blocks until another transaction modifies any of the Vars read by tx.
func (tx *Tx) wait() {
	for v := range tx.reads {
		v.watchers.Store(tx, nil)
	}
	tx.mu.Lock()
	for tx.verify() {
		expvars.Add("waits", 1)
		tx.cond.Wait()
	}
	tx.mu.Unlock()
	for v := range tx.reads {
		v.watchers.Delete(tx)
	}
}

// Get returns the value of v as of the start of the transaction.
func (tx *Tx) Get(v *Var) interface{} {
	// If we previously wrote to v, it will be in the write log.
	if val, ok := tx.writes[v]; ok {
		return val
	}
	state := v.loadState()
	// If we haven't previously read v, record its version
	if _, ok := tx.reads[v]; !ok {
		tx.reads[v] = state.version
	}
	return state.val
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

func (tx *Tx) reset() {
	for k := range tx.reads {
		delete(tx.reads, k)
	}
	for k := range tx.writes {
		delete(tx.writes, k)
	}
	tx.resetLocks()
}

func (tx *Tx) recycle() {
	txPool.Put(tx)
}

func (tx *Tx) lockAllVars() {
	tx.resetLocks()
	tx.collectAllLocks()
	tx.sortLocks()
	tx.lock()
}

func (tx *Tx) resetLocks() {
	tx.locks = tx.locks[:0]
}

func (tx *Tx) collectReadLocks() {
	for v := range tx.reads {
		tx.locks = append(tx.locks, &v.mu)
	}
}

func (tx *Tx) collectAllLocks() {
	tx.collectReadLocks()
	for v := range tx.writes {
		if _, ok := tx.reads[v]; !ok {
			tx.locks = append(tx.locks, &v.mu)
		}
	}
}

func (tx *Tx) sortLocks() {
	sort.Slice(tx.locks, func(i, j int) bool {
		return uintptr(unsafe.Pointer(tx.locks[i])) < uintptr(unsafe.Pointer(tx.locks[j]))
	})
}

func (tx *Tx) lock() {
	for _, l := range tx.locks {
		l.Lock()
	}
}

func (tx *Tx) unlock() {
	for _, l := range tx.locks {
		l.Unlock()
	}
}

func (tx *Tx) String() string {
	return fmt.Sprintf("%[1]T %[1]p", tx)
}
