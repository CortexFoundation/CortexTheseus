// Copyright 2019 The go-ethereum Authors
// This file is part of The go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with The go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package enode

import (
	"context"
	"sync"
	"time"
)

// Iterator represents a sequence of nodes. The Next method moves to the next node in the
// sequence. It returns false when the sequence has ended or the iterator is closed. Close
// may be called concurrently with Next and Node, and interrupts Next if it is blocked.
type Iterator interface {
	Next() bool  // moves to next node
	Node() *Node // returns current node
	Close()      // ends the iterator
}

// SourceIterator represents a sequence of nodes like [Iterator]
// Each node also has a named 'source'.
type SourceIterator interface {
	Iterator
	NodeSource() string // source of current node
}

// WithSource attaches a 'source name' to an iterator.
func WithSourceName(name string, it Iterator) SourceIterator {
	return sourceIter{it, name}
}

func ensureSourceIter(it Iterator) SourceIterator {
	if si, ok := it.(SourceIterator); ok {
		return si
	}
	return WithSourceName("", it)
}

type sourceIter struct {
	Iterator
	name string
}

// NodeSource implements IteratorSource.
func (it sourceIter) NodeSource() string {
	return it.name
}

type iteratorItem struct {
	n      *Node
	source string
}

// ReadNodes reads at most n nodes from the given iterator. The return value contains no
// duplicates and no nil values. To prevent looping indefinitely for small repeating node
// sequences, this function calls Next at most n times.
func ReadNodes(it Iterator, n int) []*Node {
	seen := make(map[ID]*Node, n)
	for i := 0; i < n && it.Next(); i++ {
		// Remove duplicates, keeping the node with higher seq.
		node := it.Node()
		prevNode, ok := seen[node.ID()]
		if ok && prevNode.Seq() > node.Seq() {
			continue
		}
		seen[node.ID()] = node
	}
	result := make([]*Node, 0, len(seen))
	for _, node := range seen {
		result = append(result, node)
	}
	return result
}

// IterNodes makes an iterator which runs through the given nodes once.
func IterNodes(nodes []*Node) Iterator {
	return &sliceIter{nodes: nodes, index: -1}
}

// CycleNodes makes an iterator which cycles through the given nodes indefinitely.
func CycleNodes(nodes []*Node) Iterator {
	return &sliceIter{nodes: nodes, index: -1, cycle: true}
}

type sliceIter struct {
	mu    sync.Mutex
	nodes []*Node
	index int
	cycle bool
}

func (it *sliceIter) Next() bool {
	it.mu.Lock()
	defer it.mu.Unlock()

	if len(it.nodes) == 0 {
		return false
	}
	it.index++
	if it.index == len(it.nodes) {
		if it.cycle {
			it.index = 0
		} else {
			it.nodes = nil
			return false
		}
	}
	return true
}

func (it *sliceIter) Node() *Node {
	it.mu.Lock()
	defer it.mu.Unlock()
	if len(it.nodes) == 0 {
		return nil
	}
	return it.nodes[it.index]
}

func (it *sliceIter) Close() {
	it.mu.Lock()
	defer it.mu.Unlock()

	it.nodes = nil
}

// Filter wraps an iterator such that Next only returns nodes for which
// the 'check' function returns true.
func Filter(it Iterator, check func(*Node) bool) Iterator {
	return &filterIter{ensureSourceIter(it), check}
}

type filterIter struct {
	SourceIterator
	check func(*Node) bool
}

func (f *filterIter) Next() bool {
	for f.SourceIterator.Next() {
		if f.check(f.Node()) {
			return true
		}
	}
	return false
}

// asyncFilterIter wraps an iterator such that Next only returns nodes for which
// the 'check' function returns a (possibly modified) node.
type asyncFilterIter struct {
	it        SourceIterator    // the iterator to filter
	slots     chan struct{}     // the slots for parallel checking
	passed    chan iteratorItem // channel to collect passed nodes
	cur       iteratorItem      // buffer to serve the Node call
	cancel    context.CancelFunc
	closeOnce sync.Once
}

type AsyncFilterFunc func(context.Context, *Node) *Node

// AsyncFilter creates an iterator which checks nodes in parallel.
// The 'check' function is called on multiple goroutines to filter each node
// from the upstream iterator. When check returns nil, the node will be skipped.
// It can also return a new node to be returned by the iterator instead of the .
func AsyncFilter(it Iterator, check AsyncFilterFunc, workers int) Iterator {
	f := &asyncFilterIter{
		it:     ensureSourceIter(it),
		slots:  make(chan struct{}, workers+1),
		passed: make(chan iteratorItem),
	}
	for range cap(f.slots) {
		f.slots <- struct{}{}
	}
	ctx, cancel := context.WithCancel(context.Background())
	f.cancel = cancel

	go func() {
		select {
		case <-ctx.Done():
			return
		case <-f.slots:
		}
		// read from the iterator and start checking nodes in parallel
		// when a node is checked, it will be sent to the passed channel
		// and the slot will be released
		for f.it.Next() {
			node := f.it.Node()
			nodeSource := f.it.NodeSource()

			// check the node async, in a separate goroutine
			<-f.slots
			go func() {
				if nn := check(ctx, node); nn != nil {
					item := iteratorItem{nn, nodeSource}
					select {
					case f.passed <- item:
					case <-ctx.Done(): // bale out if downstream is already closed and not calling Next
					}
				}
				f.slots <- struct{}{}
			}()
		}
		// the iterator has ended
		f.slots <- struct{}{}
	}()

	return f
}

// Next blocks until a node is available or the iterator is closed.
func (f *asyncFilterIter) Next() bool {
	var ok bool
	f.cur, ok = <-f.passed
	return ok
}

// Node returns the current node.
func (f *asyncFilterIter) Node() *Node {
	return f.cur.n
}

// NodeSource implements IteratorSource.
func (f *asyncFilterIter) NodeSource() string {
	return f.cur.source
}

// Close ends the iterator, also closing the wrapped iterator.
func (f *asyncFilterIter) Close() {
	f.closeOnce.Do(func() {
		f.it.Close()
		f.cancel()
		for range cap(f.slots) {
			<-f.slots
		}
		close(f.slots)
		close(f.passed)
	})
}

// bufferIter wraps an iterator and buffers the nodes it returns.
// The buffer is pre-filled with the given size from the wrapped iterator.
type bufferIter struct {
	it        SourceIterator
	buffer    chan iteratorItem
	head      iteratorItem
	closeOnce sync.Once
}

// NewBufferIter creates a new pre-fetch buffer of a given size.
func NewBufferIter(it Iterator, size int) Iterator {
	b := bufferIter{
		it:     ensureSourceIter(it),
		buffer: make(chan iteratorItem, size),
	}

	go func() {
		// if the wrapped iterator ends, the buffer content will still be served.
		defer close(b.buffer)
		// If instead the bufferIterator is closed, we bail out of the loop.
		for b.it.Next() {
			item := iteratorItem{b.it.Node(), b.it.NodeSource()}
			b.buffer <- item
		}
	}()
	return &b
}

func (b *bufferIter) Next() bool {
	var ok bool
	b.head, ok = <-b.buffer
	return ok
}

func (b *bufferIter) Node() *Node {
	return b.head.n
}

func (b *bufferIter) NodeSource() string {
	return b.head.source
}

func (b *bufferIter) Close() {
	b.closeOnce.Do(func() {
		b.it.Close()
		// Drain buffer and wait for the goroutine to end.
		for range b.buffer {
		}
	})
}

// FairMix aggregates multiple node iterators. The mixer itself is an iterator which ends
// only when Close is called. Source iterators added via AddSource are removed from the
// mix when they end.
//
// The distribution of nodes returned by Next is approximately fair, i.e. FairMix
// attempts to draw from all sources equally often. However, if a certain source is slow
// and doesn't return a node within the configured timeout, a node from any other source
// will be returned.
//
// It's safe to call AddSource and Close concurrently with Next.
type FairMix struct {
	wg      sync.WaitGroup
	fromAny chan iteratorItem
	timeout time.Duration
	cur     iteratorItem

	mu      sync.Mutex
	closed  chan struct{}
	sources []*mixSource
	last    int
}

type mixSource struct {
	it      SourceIterator
	next    chan iteratorItem
	timeout time.Duration
}

// NewFairMix creates a mixer.
//
// The timeout specifies how long the mixer will wait for the next fairly-chosen source
// before giving up and taking a node from any other source. A good way to set the timeout
// is deciding how long you'd want to wait for a node on average. Passing a negative
// timeout makes the mixer completely fair.
func NewFairMix(timeout time.Duration) *FairMix {
	m := &FairMix{
		fromAny: make(chan iteratorItem),
		closed:  make(chan struct{}),
		timeout: timeout,
	}
	return m
}

// AddSource adds a source of nodes.
func (m *FairMix) AddSource(it Iterator) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed == nil {
		return
	}
	m.wg.Add(1)
	source := &mixSource{
		it:      ensureSourceIter(it),
		next:    make(chan iteratorItem),
		timeout: m.timeout,
	}
	m.sources = append(m.sources, source)
	go m.runSource(m.closed, source)
}

// Close shuts down the mixer and all current sources.
// Calling this is required to release resources associated with the mixer.
func (m *FairMix) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed == nil {
		return
	}
	for _, s := range m.sources {
		s.it.Close()
	}
	close(m.closed)
	m.wg.Wait()
	close(m.fromAny)
	m.sources = nil
	m.closed = nil
}

// Next returns a node from a random source.
func (m *FairMix) Next() bool {
	m.cur = iteratorItem{}

	for {
		source := m.pickSource()
		if source == nil {
			return m.nextFromAny()
		}

		var timeout <-chan time.Time
		if source.timeout >= 0 {
			timer := time.NewTimer(source.timeout)
			timeout = timer.C
			defer timer.Stop()
		}

		select {
		case item, ok := <-source.next:
			if ok {
				// Here, the timeout is reset to the configured value
				// because the source delivered a node.
				source.timeout = m.timeout
				m.cur = item
				return true
			}
			// This source has ended.
			m.deleteSource(source)
		case <-timeout:
			// The selected source did not deliver a node within the timeout, so the
			// timeout duration is halved for next time. This is supposed to improve
			// latency with stuck sources.
			source.timeout /= 2
			return m.nextFromAny()
		}
	}
}

// Node returns the current node.
func (m *FairMix) Node() *Node {
	return m.cur.n
}

// NodeSource returns the current node's source name.
func (m *FairMix) NodeSource() string {
	return m.cur.source
}

// nextFromAny is used when there are no sources or when the 'fair' choice
// doesn't turn up a node quickly enough.
func (m *FairMix) nextFromAny() bool {
	item, ok := <-m.fromAny
	if ok {
		m.cur = item
	}
	return ok
}

// pickSource chooses the next source to read from, cycling through them in order.
func (m *FairMix) pickSource() *mixSource {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.sources) == 0 {
		return nil
	}
	m.last = (m.last + 1) % len(m.sources)
	return m.sources[m.last]
}

// deleteSource deletes a source.
func (m *FairMix) deleteSource(s *mixSource) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for i := range m.sources {
		if m.sources[i] == s {
			copy(m.sources[i:], m.sources[i+1:])
			m.sources[len(m.sources)-1] = nil
			m.sources = m.sources[:len(m.sources)-1]
			break
		}
	}
}

// runSource reads a single source in a loop.
func (m *FairMix) runSource(closed chan struct{}, s *mixSource) {
	defer m.wg.Done()
	defer close(s.next)
	for s.it.Next() {
		item := iteratorItem{s.it.Node(), s.it.NodeSource()}
		select {
		case s.next <- item:
		case m.fromAny <- item:
		case <-closed:
			return
		}
	}
}
