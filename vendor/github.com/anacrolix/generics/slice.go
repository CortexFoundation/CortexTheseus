package generics

// Pops the last element from the slice and returns it. Panics if the slice is empty, or if the
// slice is nil.
func SlicePop[T any](slice *[]T) T {
	lastIndex := len(*slice) - 1
	last := (*slice)[lastIndex]
	*slice = (*slice)[:lastIndex]
	return last
}
