package generics

type Option[V any] struct {
	ok    bool
	value V
}

func (me *Option[V]) Ok() bool {
	return me.ok
}

func (me *Option[V]) Value() V {
	if !me.ok {
		panic("not set")
	}
	return me.value
}

func Some[V any](value V) Option[V] {
	return Option[V]{ok: true, value: value}
}

func None[V any]() Option[V] {
	return Option[V]{}
}
