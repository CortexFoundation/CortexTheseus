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

func (me Option[V]) AndThen(f func(V) Option[V]) Option[V] {
	if me.Ok() {
		return f(me.value)
	}
	return me
}

func Some[V any](value V) Option[V] {
	return Option[V]{ok: true, value: value}
}

func None[V any]() Option[V] {
	return Option[V]{}
}
