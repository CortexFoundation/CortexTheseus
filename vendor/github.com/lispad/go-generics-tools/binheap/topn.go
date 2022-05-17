package binheap

import (
	"golang.org/x/exp/constraints"
	"golang.org/x/exp/slices"
)

// TopN mutates data slice, moving TopN elements of slice to the beginning and returns subslice of first n elements.
// O(M *ln(N)), where M is slice size
// Mutates source data slice. No additional allocations are done.
func TopN[T any](data []T, n int, comparator func(x, y T) bool) []T {
	if n > len(data) {
		n = len(data)
	}
	heap := FromSlice(data[0:n], reverse(comparator))
	for i := n; i < len(data); i++ {
		if !comparator(data[0], data[i]) {
			data[0], data[i] = data[i], data[0]
			heap.fix(0)
		}
	}
	slices.SortFunc(data[0:n], comparator)
	return data[0:n]
}

// MinN moves N Min elements of slice to the beginning and returns subslice of first n elements.
// Mutates source data slice.
// No additional allocations are done.
func MinN[T constraints.Ordered](data []T, n int) []T {
	return TopN(data, n, func(x, y T) bool {
		return x < y
	})
}

// MaxN moves N Max elements of slice to the beginning and returns subslice of first n elements.
// Mutates source data slice.
// No additional allocations are done.
func MaxN[T constraints.Ordered](data []T, n int) []T {
	return TopN(data, n, func(x, y T) bool {
		return x > y
	})
}

func reverse[T any](comparator func(x, y T) bool) func(x, y T) bool {
	return func(x, y T) bool {
		return !comparator(x, y)
	}
}
