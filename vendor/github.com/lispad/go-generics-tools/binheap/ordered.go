package binheap

import (
	"golang.org/x/exp/constraints"
)

// ComparableHeap provides additional Search and Delete methods.
// Could be used for comparable types.
type ComparableHeap[T comparable] struct {
	Heap[T]
}

// EmptyComparableHeap creates heap for comparable types.
func EmptyComparableHeap[T comparable](comparator func(x, y T) bool) ComparableHeap[T] {
	return ComparableHeap[T]{
		Heap: EmptyHeap(comparator),
	}
}

// ComparableFromSlice creates heap, based on provided slice.
// Slice could be reordered.
func ComparableFromSlice[T comparable](data []T, comparator func(x, y T) bool) ComparableHeap[T] {
	return ComparableHeap[T]{
		Heap: FromSlice(data, comparator),
	}
}

// Search returns if element presents in heap.
func (h *ComparableHeap[T]) Search(x T) bool {
	return h.search(x) != -1
}

// Delete removes element from heap, and returns true if x presents in heap.
func (h *ComparableHeap[T]) Delete(x T) bool {
	pos := h.search(x)
	if pos == -1 {
		return false
	}
	newLen := h.Len() - 1
	h.swap(pos, newLen)
	h.data = h.data[:newLen]

	return true
}

func (h *ComparableHeap[T]) search(x T) int {
	for i := range h.data {
		if h.data[i] == x {
			return i
		}
	}
	return -1
}

// EmptyMinHeap creates empty Min-Heap for ordered types.
func EmptyMinHeap[T constraints.Ordered]() ComparableHeap[T] {
	return EmptyComparableHeap(func(x, y T) bool {
		return x < y
	})
}

// EmptyMaxHeap creates empty Max-Heap for ordered types.
func EmptyMaxHeap[T constraints.Ordered]() ComparableHeap[T] {
	return EmptyComparableHeap(func(x, y T) bool {
		return x > y
	})
}

// EmptyMinHeap creates Min-Heap, based on provided slice.
// Slice could be reordered.
func MinHeapFromSlice[T constraints.Ordered](data []T) ComparableHeap[T] {
	return ComparableFromSlice(data, func(x, y T) bool {
		return x < y
	})
}

// EmptyMaxHeap creates Max-Heap, based on provided slice.
// Slice could be reordered.
func MaxHeapFromSlice[T constraints.Ordered](data []T) ComparableHeap[T] {
	return ComparableFromSlice(data, func(x, y T) bool {
		return x > y
	})
}
