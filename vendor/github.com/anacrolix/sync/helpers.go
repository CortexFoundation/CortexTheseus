package sync

import (
	"sync/atomic"
)

type profileKey = int64

var nextKey atomic.Int64

// Avoid using this as an actual key (by adding 1 first before using always).
const noKey = 0

func withBlocked(slow func(), fast func() bool) {
	if fast() {
		return
	}
	v := nextKey.Add(1)
	lockBlockers.Add(v, 2)
	slow()
	lockBlockers.Remove(v)
}

func addHolderProfile(skip int) profileKey {
	v := nextKey.Add(1)
	lockHolders.Add(v, 2+skip)
	return v
}

func removeHolder(key profileKey) {
	if !contentionOn {
		return
	}
	lockHolders.Remove(key)
}
