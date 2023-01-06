package filecache

import (
	"time"

	"github.com/anacrolix/missinggo/orderedmap"
)

type lru struct {
	o     orderedmap.OrderedMap
	oKeys map[policyItemKey]lruKey
}

type lruKey struct {
	item policyItemKey
	used time.Time
}

func (me lruKey) Before(other lruKey) bool {
	if me.used.Equal(other.used) {
		return me.item.Before(other.item)
	}
	return me.used.Before(other.used)
}

var _ Policy = (*lru)(nil)

func (me *lru) Choose() (ret policyItemKey) {
	any := false
	me.o.Iter(func(i interface{}) bool {
		ret = i.(lruKey).item
		any = true
		return false
	})
	if !any {
		panic("cache empty")
	}
	return
}

func (me *lru) Used(k policyItemKey, at time.Time) {
	if me.o == nil {
		me.o = orderedmap.NewGoogleBTree(func(l, r interface{}) bool {
			return l.(lruKey).Before(r.(lruKey))
		})
	} else {
		me.o.Unset(me.oKeys[k])
	}
	lk := lruKey{k, at}
	me.o.Set(lk, lk)
	if me.oKeys == nil {
		me.oKeys = make(map[policyItemKey]lruKey)
	}
	me.oKeys[k] = lk
}

func (me *lru) Forget(k policyItemKey) {
	if me.o != nil {
		me.o.Unset(me.oKeys[k])
	}
	delete(me.oKeys, k)
}

func (me *lru) NumItems() int {
	return len(me.oKeys)
}
