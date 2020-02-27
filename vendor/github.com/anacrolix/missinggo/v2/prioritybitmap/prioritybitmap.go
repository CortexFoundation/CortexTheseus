// Package prioritybitmap implements a set of integers ordered by attached priorities.
package prioritybitmap

import (
	"sync"

	"github.com/anacrolix/missinggo/bitmap"
	"github.com/anacrolix/missinggo/iter"
	"github.com/anacrolix/missinggo/orderedmap"
)

// The interface used for non-singleton bit-sets for each priority level.
type Set interface {
	Has(bit int) bool
	Delete(bit int)
	Len() int
	Set(bit int)
	Range(f func(int) bool)
}

// Maintains set of ints ordered by priority.
type PriorityBitmap struct {
	// From priority to singleton or set of bit indices.
	om orderedmap.OrderedMap
	// From bit index to priority
	priorities map[int]int
	// If not set, is initialized to the default map[int]struct{} implementation on first use.
	NewSet  func() Set
	bitSets sync.Pool
}

var _ bitmap.Interface = (*PriorityBitmap)(nil)

func (me *PriorityBitmap) Contains(bit int) bool {
	_, ok := me.priorities[bit]
	return ok
}

func (me *PriorityBitmap) Len() int {
	return len(me.priorities)
}

func (me *PriorityBitmap) Clear() {
	me.om = nil
	me.priorities = nil
}

func (me *PriorityBitmap) deleteBit(bit int) (priority int, ok bool) {
	priority, ok = me.priorities[bit]
	if !ok {
		return
	}
	switch v := me.om.Get(priority).(type) {
	case int:
		if v != bit {
			panic("invariant broken")
		}
	case Set:
		if !v.Has(bit) {
			panic("invariant broken")
		}
		v.Delete(bit)
		if v.Len() != 0 {
			return
		}
		me.bitSets.Put(v)
	default:
		panic(v)
	}
	me.om.Unset(priority)
	if me.om.Len() == 0 {
		me.om = nil
	}
	return
}

func bitLess(l, r interface{}) bool {
	return l.(int) < r.(int)
}

// Returns true if the priority is changed, or the bit wasn't present.
func (me *PriorityBitmap) Set(bit int, priority int) bool {
	if p, ok := me.priorities[bit]; ok && p == priority {
		return false
	}
	if oldPriority, deleted := me.deleteBit(bit); deleted && oldPriority == priority {
		panic("should have already returned")
	}
	if me.priorities == nil {
		me.priorities = make(map[int]int)
	}
	me.priorities[bit] = priority
	if me.om == nil {
		me.om = orderedmap.New(bitLess)
	}
	_v, ok := me.om.GetOk(priority)
	if !ok {
		// No other bits with this priority, set it to a lone int.
		me.om.Set(priority, bit)
		return true
	}
	switch v := _v.(type) {
	case int:
		newV := func() Set {
			i := me.bitSets.Get()
			if i == nil {
				if me.NewSet == nil {
					me.NewSet = newMapSet
				}
				return me.NewSet()
			} else {
				return i.(Set)
			}
		}()
		newV.Set(v)
		newV.Set(bit)
		me.om.Set(priority, newV)
	case Set:
		v.Set(bit)
	default:
		panic(v)
	}
	return true
}

func (me *PriorityBitmap) Remove(bit int) bool {
	if _, ok := me.deleteBit(bit); !ok {
		return false
	}
	delete(me.priorities, bit)
	if len(me.priorities) == 0 {
		me.priorities = nil
	}
	if me.om != nil && me.om.Len() == 0 {
		me.om = nil
	}
	return true
}

func (me *PriorityBitmap) Iter(f iter.Callback) {
	me.IterTyped(func(i int) bool {
		return f(i)
	})
}

func (me *PriorityBitmap) IterTyped(_f func(i bitmap.BitIndex) bool) bool {
	if me == nil || me.om == nil {
		return true
	}
	f := func(i int) bool {
		return _f(i)
	}
	return iter.All(func(key interface{}) bool {
		value := me.om.Get(key)
		switch v := value.(type) {
		case int:
			return f(v)
		case Set:
			v.Range(func(i int) bool {
				return f(i)
			})
		default:
			panic(v)
		}
		return true
	}, me.om.Iter)
}

func (me *PriorityBitmap) IsEmpty() bool {
	if me.om == nil {
		return true
	}
	return me.om.Len() == 0
}

// ok is false if the bit is not set.
func (me *PriorityBitmap) GetPriority(bit int) (prio int, ok bool) {
	prio, ok = me.priorities[bit]
	return
}
