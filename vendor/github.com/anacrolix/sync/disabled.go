//go:build disable_pprof_sync

package sync

import (
	"sync"
)

type (
	Mutex   = sync.Mutex
	RWMutex = sync.RWMutex
)
