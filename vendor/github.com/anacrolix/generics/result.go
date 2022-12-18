package generics

type Result[T any] struct {
	Ok  T
	Err error
}

func ResultFromTuple[T any](t T, err error) Result[T] {
	return Result[T]{
		Ok:  t,
		Err: err,
	}
}
