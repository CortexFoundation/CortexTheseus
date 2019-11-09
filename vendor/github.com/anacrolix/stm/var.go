package stm

import "sync"

// Holds an STM variable.
type Var struct {
	mu      sync.Mutex
	val     interface{}
	version uint64

	watchers map[*Tx]struct{}
}

// Returns a new STM variable.
func NewVar(val interface{}) *Var {
	return &Var{
		val:      val,
		watchers: make(map[*Tx]struct{}),
	}
}
