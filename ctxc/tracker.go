package ctxc

import (
	"time"

	"github.com/CortexFoundation/CortexTheseus/p2p/tracker"
)

// requestTracker is a singleton tracker for request times.
var requestTracker = tracker.New("ctxc", time.Minute)
