package generics

type Option[V any] struct {
	Ok    bool
	Value V
}

func (me Option[V]) Unwrap() V {
	if !me.Ok {
		panic("not set")
	}
	return me.Value
}

func (me Option[V]) AndThen(f func(V) Option[V]) Option[V] {
	if me.Ok {
		return f(me.Value)
	}
	return me
}

func (me Option[V]) UnwrapOr(or V) V {
	if me.Ok {
		return me.Value
	} else {
		return or
	}
}

func (me *Option[V]) Set(v V) {
	me.Ok = true
	me.Value = v
}

func Some[V any](value V) Option[V] {
	return Option[V]{Ok: true, Value: value}
}

func None[V any]() Option[V] {
	return Option[V]{}
}
