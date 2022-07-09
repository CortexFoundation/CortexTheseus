# Heap structure, using go generics
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Introduction
------------

The Heap package contains simple [binary heap](https://en.wikipedia.org/wiki/Binary_heap) implementation, using Golang
generics. There are several heap implementations

- generic Heap implementation, that could be used for `any` type,
- `ComparableHeap` for [comparable](https://go.dev/ref/spec#Comparison_operators) types. Additional `Search`
  and `Delete` are implemented,
- for [`constraints.Ordered`](https://pkg.go.dev/golang.org/x/exp/constraints#Ordered) there are
  constructors for min, max heaps;

Also use-cases provided:

- `TopN` that allows getting N top elements from slice.
  `TopN` swaps top N elements to first N elements of slice, no additional allocations are done. All slice elements are
  kept, only order is changed.
- `TopNHeap` allows to get N top, pushing elements from stream without allocation slice for all elements. Only O(N)
  memory is used.
- `TopNImmutable` allocated new slice for heap, input slice is not mutated.

Both TopN and TopNImmutable has methods for creating min and max tops for `constraints.Ordered`.

Usage Example
-----------------

    package main
    
    import (
        "fmt"

        "github.com/lispad/go-generics-tools/binheap"
    )
    
    func main() {
        someData := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}
        mins := binheap.MinN[float64](someData, 3)
        fmt.Printf("--- top 3 min elements: %v\n", mins)
        maxs := binheap.MaxN[float64](someData, 3)
        fmt.Printf("--- top 3 max elements: %v\n\n", maxs)
  
        heap := binheap.EmptyMaxHeap[string]()
        heap.Push("foo")
        heap.Push("zzz")
        heap.Push("bar")
        heap.Push("baz")
        heap.Push("foobar")
        heap.Push("foobaz")
        fmt.Printf("--- heap has %d elements, max element:\n%s\n\n", heap.Len(), heap.Peak())
    }

A bit more examples could be found in `examples` directory

Benchmark
-----------------
Theoretical complexity for getting TopN from slice with size M, N <= M: O(N*ln(M)). When N << M, the heap-based TopN
could be much faster than sorting slice and getting top. E.g. For top-3 from 10k elements approach is ln(10^5)/ln(3) ~=
8.38 times faster.

#### Benchmark

    BenchmarkSortedMaxN-8      	10303648	   136.0 ns/op	   0 B/op	   0 allocs/op
    BenchmarkMaxNImmutable-8   	398996316	   3.029 ns/op	   0 B/op	   0 allocs/op
    BenchmarkMaxN-8            	804041455	   1.819 ns/op	   0 B/op	   0 allocs/op

Compatibility
-------------
Minimal Golang version is 1.18. Generics and fuzz testing are used.

Installation
----------------------

To install package, run:

    go get github.com/lispad/go-generics-tools/binheap

License
-------

The binheap package is licensed under the MIT license. Please see the LICENSE file for details.
