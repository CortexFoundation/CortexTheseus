package generics

func InitNew[T any](p **T) {
	*p = new(T)
}
