package sync

import (
	"sync"
)

type mutexHolderSet struct {
	mu    sync.Mutex
	slots []profileKey
}

func (me *mutexHolderSet) Add(key profileKey) {
	me.mu.Lock()
	me.slots = append(me.slots, key)
	me.mu.Unlock()
}

func (me *mutexHolderSet) Pop() profileKey {
	me.mu.Lock()
	key := me.slots[len(me.slots)-1]
	me.slots = me.slots[:len(me.slots)-1]
	me.mu.Unlock()
	return key
}

//type atomicHolderSet struct {
//	// First for optimistic alignment.
//	len   atomic.Int32
//	mu    sync.RWMutex
//	slots []profileKey
//}
//
//func (me *atomicHolderSet) Add(key profileKey) {
//	slices.Grow()
//	newLen := me.len.Add(1)
//	me.slots[newLen-1] = key
//}
//
//func (me *atomicHolderSet) Pop() profileKey {
//	for {
//		x := me.len.Load()
//		key := me.slots[x-1]
//		if me.len.CompareAndSwap(x, x-1) {
//			return key
//		}
//	}
//}
