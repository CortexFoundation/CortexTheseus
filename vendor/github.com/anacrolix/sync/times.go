package sync

import (
	"time"
	"unique"
)

// Data for tracking lock time on a Mutex.
type lockTimes struct {
	stack   unique.Handle[callerArray] // The stack for the current holder.
	start   time.Time                  // When the lock was obtained.
	entries int                        // Number of entries returned from runtime.Callers.
}
