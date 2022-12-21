package generics

import (
	"golang.org/x/exp/constraints"
)

// Pops the last element from the slice and returns it. Panics if the slice is empty, or if the
// slice is nil.
func SlicePop[T any](slice *[]T) T {
	lastIndex := len(*slice) - 1
	last := (*slice)[lastIndex]
	*slice = (*slice)[:lastIndex]
	return last
}

func MakeSliceWithLength[T any, L constraints.Integer](slice *[]T, length L) {
	*slice = make([]T, length)
}

func Reversed[T any](slice []T) []T {
	reversed := make([]T, len(slice))
	for i := range reversed {
		reversed[i] = slice[len(slice)-1-i]
	}
	return reversed
}
