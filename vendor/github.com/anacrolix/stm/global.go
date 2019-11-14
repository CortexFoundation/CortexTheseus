package stm

import "sync"

// The globalLock serializes transaction verification/committal. globalCond is
// used to signal that at least one Var has changed.
var globalLock sync.Mutex
