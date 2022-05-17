// Package binheap provides implementation of binary heap for any type
// with use of golang generics, and top-N heap usecases.
package binheap

// Heap provides basic methods: Push, Peak, Pop, Replace top element and PushPop.
type Heap[T any] struct {
	comparator func(x, y T) bool
	data       []T
}

// EmptyHeap creates empty heap with provided comparator.
func EmptyHeap[T any](comparator func(x, y T) bool) Heap[T] {
	return Heap[T]{
		comparator: comparator,
	}
}

// FromSlice creates heap, based on provided slice.
// Slice could be reordered.
func FromSlice[T any](data []T, comparator func(x, y T) bool) Heap[T] {
	h := Heap[T]{
		data:       data,
		comparator: comparator,
	}
	n := h.Len()
	for i := n/2 - 1; i >= 0; i-- {
		h.down(i, n)
	}

	return h
}

// Push inserts element to heap.
func (h *Heap[T]) Push(x T) {
	h.data = append(h.data, x)
	h.up(h.Len() - 1)
}

// Len returns count of elements in heap.
func (h *Heap[T]) Len() int {
	return len(h.data)
}

// Peak returns top element without deleting.
func (h *Heap[T]) Peak() T {
	return h.data[0]
}

// Pop returns top element with removing it.
func (h *Heap[T]) Pop() T {
	n := h.Len() - 1
	h.swap(0, n)
	h.down(0, n)
	result := h.data[n]
	h.data = h.data[0:n]

	return result
}

// PushPop pushes x to the heap and then pops top element.
func (h *Heap[T]) PushPop(x T) T {
	if h.Len() > 0 && h.comparator(h.data[0], x) {
		x, h.data[0] = h.data[0], x
		h.down(0, h.Len())
	}

	return x
}

// Replace extracts the root of the heap, and push a new item.
func (h *Heap[T]) Replace(x T) (result T) {
	result, h.data[0] = h.data[0], x
	h.fix(0)
	return
}

func (h *Heap[T]) fix(i int) (result T) {
	if !h.down(i, h.Len()) {
		h.up(i)
	}

	return result
}

func (h *Heap[T]) swap(i, j int) {
	h.data[i], h.data[j] = h.data[j], h.data[i]
}

func (h *Heap[T]) up(j int) {
	for {
		i := (j - 1) / 2 // parent
		if i == j || !h.comparator(h.data[j], h.data[i]) {
			break
		}
		h.swap(i, j)
		j = i
	}
}

func (h *Heap[T]) down(i0, n int) bool {
	i := i0
	for {
		j1 := 2*i + 1
		if j1 >= n || j1 < 0 { // j1 < 0 after int overflow
			break
		}
		j := j1 // left child
		if j2 := j1 + 1; j2 < n && h.comparator(h.data[j2], h.data[j1]) {
			j = j2 // = 2*i + 2  // right child
		}
		if !h.comparator(h.data[j], h.data[i]) {
			break
		}
		h.swap(i, j)
		i = j
	}

	return i > i0
}
