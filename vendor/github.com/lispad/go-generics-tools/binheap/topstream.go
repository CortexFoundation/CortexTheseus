package binheap

import (
	"golang.org/x/exp/constraints"
	"golang.org/x/exp/slices"
)

// TopNHeap keeps N top elements in reverse order.
type TopNHeap[T any] struct {
	comparator func(x, y T) bool
	heap       Heap[T]
	maxSize    int
}

// EmptyTopNHeap creates new heap for storing n top elements.
func EmptyTopNHeap[T any](n int, comparator func(x, y T) bool) TopNHeap[T] {
	return TopNHeap[T]{
		comparator: comparator,
		heap:       EmptyHeap(reverse(comparator)),
		maxSize:    n,
	}
}

// Push stores x, if x is better than top n, or ignores otherwise.
func (h *TopNHeap[T]) Push(x T) {
	if h.heap.Len() < h.maxSize {
		h.heap.Push(x)
		return
	}

	if !h.comparator(h.heap.data[0], x) {
		h.heap.Replace(x)
	}
}

// PopTopN returns current top N elements, and empties slice.
func (h *TopNHeap[T]) PopTopN() []T {
	result := h.PeekTopN()
	h.heap.data = h.heap.data[:0]

	return result
}

// PeekTopN returns current top N elements.
func (h *TopNHeap[T]) PeekTopN() []T {
	result := make([]T, h.heap.Len())
	copy(result, h.heap.data)
	slices.SortFunc(result, h.comparator)

	return result
}

// TopNImmutable returns top n elements from slice.
// O(M *ln(N)), where M is slice size
// Allocates new slice, source data isn't mutated. O(N) additional allocations.
func TopNImmutable[T any](data []T, n int, comparator func(x, y T) bool) []T {
	h := EmptyTopNHeap(n, comparator)
	for _, x := range data {
		h.Push(x)
	}
	return h.PopTopN()
}

// MinNImmutable return n minimal elements from slice.
// O(M *ln(N)), where M is slice size
// Allocates new slice, source data isn't mutated. O(N) additional allocations.
func MinNImmutable[T constraints.Ordered](data []T, n int) []T {
	return TopNImmutable(data, n, func(x, y T) bool {
		return x < y
	})
}

// MaxNImmutable return n maximal elements from slice.
// O(M *ln(N)), where M is slice size
// Allocates new slice, source data isn't mutated. O(N) additional allocations.
func MaxNImmutable[T constraints.Ordered](data []T, n int) []T {
	return TopNImmutable(data, n, func(x, y T) bool {
		return x > y
	})
}
