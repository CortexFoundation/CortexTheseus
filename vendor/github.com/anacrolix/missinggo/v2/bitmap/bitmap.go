// Package bitmap provides a []bool/bitmap implementation with standardized
// iteration. Bitmaps are the equivalent of []bool, with improved compression
// for runs of similar values, and faster operations on ranges and the like.
package bitmap

import (
	"math"

	"github.com/RoaringBitmap/roaring"

	"github.com/anacrolix/missinggo/iter"
)

const MaxInt = -1

type BitIndex = int

type Interface interface {
	Len() int
}

// Bitmaps store the existence of values in [0,math.MaxUint32] more
// efficiently than []bool. The empty value starts with no bits set.
type Bitmap struct {
	RB *roaring.Bitmap
}

var ToEnd int = -1

// The number of set bits in the bitmap. Also known as cardinality.
func (me Bitmap) Len() int {
	if me.RB == nil {
		return 0
	}
	return int(me.RB.GetCardinality())
}

func (me Bitmap) ToSortedSlice() (ret []int) {
	if me.RB == nil {
		return
	}
	for _, ui32 := range me.RB.ToArray() {
		ret = append(ret, int(int32(ui32)))
	}
	return
}

func (me *Bitmap) lazyRB() *roaring.Bitmap {
	if me.RB == nil {
		me.RB = roaring.NewBitmap()
	}
	return me.RB
}

func (me Bitmap) Iter(cb iter.Callback) {
	me.IterTyped(func(i int) bool {
		return cb(i)
	})
}

// Returns true if all values were traversed without early termination.
func (me Bitmap) IterTyped(f func(int) bool) bool {
	if me.RB == nil {
		return true
	}
	it := me.RB.Iterator()
	for it.HasNext() {
		if !f(int(it.Next())) {
			return false
		}
	}
	return true
}

func checkInt(i BitIndex) {
	if i < math.MinInt32 || i > math.MaxInt32 {
		panic("out of bounds")
	}
}

func (me *Bitmap) Add(is ...BitIndex) {
	rb := me.lazyRB()
	for _, i := range is {
		checkInt(i)
		rb.AddInt(i)
	}
}

func (me *Bitmap) AddRange(begin, end BitIndex) {
	if begin >= end {
		return
	}
	me.lazyRB().AddRange(uint64(begin), uint64(end))
}

func (me *Bitmap) Remove(i BitIndex) bool {
	if me.RB == nil {
		return false
	}
	return me.RB.CheckedRemove(uint32(i))
}

func (me *Bitmap) Union(other Bitmap) {
	me.lazyRB().Or(other.lazyRB())
}

func (me Bitmap) Contains(i int) bool {
	if me.RB == nil {
		return false
	}
	return me.RB.Contains(uint32(i))
}

func (me *Bitmap) Sub(other Bitmap) {
	if other.RB == nil {
		return
	}
	if me.RB == nil {
		return
	}
	me.RB.AndNot(other.RB)
}

func (me *Bitmap) Clear() {
	if me.RB == nil {
		return
	}
	me.RB.Clear()
}

func (me Bitmap) Copy() (ret Bitmap) {
	ret = me
	if ret.RB != nil {
		ret.RB = ret.RB.Clone()
	}
	return
}

func (me *Bitmap) FlipRange(begin, end BitIndex) {
	me.lazyRB().FlipInt(begin, end)
}

func (me Bitmap) Get(bit BitIndex) bool {
	return me.RB != nil && me.RB.ContainsInt(bit)
}

func (me *Bitmap) Set(bit BitIndex, value bool) {
	if value {
		me.lazyRB().AddInt(bit)
	} else {
		if me.RB != nil {
			me.RB.Remove(uint32(bit))
		}
	}
}

func (me *Bitmap) RemoveRange(begin, end BitIndex) *Bitmap {
	if me.RB == nil {
		return me
	}
	rangeEnd := uint64(end)
	if end == ToEnd {
		rangeEnd = 0x100000000
	}
	me.RB.RemoveRange(uint64(begin), rangeEnd)
	return me
}

func (me Bitmap) IsEmpty() bool {
	return me.RB == nil || me.RB.IsEmpty()
}
