package stm

import (
	"sync"
)

var (
	txPool = sync.Pool{New: func() interface{} {
		tx := &Tx{
			reads:  make(map[*Var]uint64),
			writes: make(map[*Var]interface{}),
		}
		tx.cond.L = &globalLock
		return tx
	}}
)

// Atomically executes the atomic function fn.
func Atomically(fn func(*Tx)) interface{} {
	// run the transaction
	tx := txPool.Get().(*Tx)
retry:
	tx.reset()
	var ret interface{}
	if func() (retry bool) {
		defer func() {
			r := recover()
			if r == nil {
				return
			}
			if _ret, ok := r.(_return); ok {
				ret = _ret.value
			} else if r == Retry {
				// wait for one of the variables we read to change before retrying
				tx.wait()
				retry = true
			} else {
				panic(r)
			}
		}()
		fn(tx)
		return false
	}() {
		goto retry
	}
	// verify the read log
	globalLock.Lock()
	if !tx.verify() {
		globalLock.Unlock()
		goto retry
	}
	// commit the write log and broadcast that variables have changed
	tx.commit()
	globalLock.Unlock()
	tx.recycle()
	return ret
}

// AtomicGet is a helper function that atomically reads a value.
func AtomicGet(v *Var) interface{} {
	// since we're only doing one operation, we don't need a full transaction
	globalLock.Lock()
	v.mu.Lock()
	val := v.val
	v.mu.Unlock()
	globalLock.Unlock()
	return val
}

// AtomicSet is a helper function that atomically writes a value.
func AtomicSet(v *Var, val interface{}) {
	Atomically(func(tx *Tx) {
		tx.Set(v, val)
	})
}

// Compose is a helper function that composes multiple transactions into a
// single transaction.
func Compose(fns ...func(*Tx)) func(*Tx) {
	return func(tx *Tx) {
		for _, f := range fns {
			f(tx)
		}
	}
}

// Select runs the supplied functions in order. Execution stops when a
// function succeeds without calling Retry. If no functions succeed, the
// entire selection will be retried.
func Select(fns ...func(*Tx)) func(*Tx) {
	return func(tx *Tx) {
		switch len(fns) {
		case 0:
			// empty Select blocks forever
			tx.Retry()
		case 1:
			fns[0](tx)
		default:
			if catchRetry(fns[0], tx) {
				Select(fns[1:]...)(tx)
			}
		}
	}
}
