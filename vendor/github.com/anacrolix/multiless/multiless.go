package multiless

type (
	// A helper for long chains of "less-than" comparisons, where later comparisons are only
	// required if earlier ones haven't resolved the comparison.
	Computation struct {
		ok   bool
		less bool
	}
)

func New() Computation {
	return Computation{}
}

func (me Computation) eagerSameLess(same, less bool) Computation {
	if me.ok || same {
		return me
	}
	return Computation{
		ok:   true,
		less: less,
	}
}

func (me Computation) Bool(l, r bool) Computation {
	return me.eagerSameLess(l == r, r)
}

func (me Computation) Uint32(l, r uint32) Computation {
	return me.eagerSameLess(l == r, l < r)
}

func (me Computation) Int64(l, r int64) Computation {
	return me.eagerSameLess(l == r, l < r)
}

func (me Computation) CmpInt64(i int64) Computation {
	return me.eagerSameLess(i == 0, i < 0)
}

func (me Computation) Uintptr(l, r uintptr) Computation {
	return me.eagerSameLess(l == r, l < r)
}

func (me Computation) Less() bool {
	return me.less
}

func (me Computation) LessOk() (less, ok bool) {
	return me.less, me.ok
}
