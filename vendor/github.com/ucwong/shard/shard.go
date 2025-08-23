package shard

import (
	"runtime"
	"sync"

	"github.com/tidwall/hashmap"
	"github.com/zeebo/xxh3"
	//"sync/atomic"
	//"github.com/cespare/xxhash"
)

// Map is a hashmap. Like map[string]any, but sharded and thread-safe.
type Map[V any] struct {
	once    sync.Once
	capcity int
	shards  int
	//seed    uint32
	mus  []sync.RWMutex
	maps []*hashmap.Map[string, V]
}

// New returns a new hashmap with the specified capacity.
func New[V any]() (m *Map[V]) {
	m = NewWithCapcity[V](0)
	return
}

// New returns a new hashmap with the specified capacity.
func NewWithCapcity[V any](capcity int) (m *Map[V]) {
	m = &Map[V]{capcity: capcity}
	m.initDo()
	return
}

// Clear out all values from map
func (m *Map[V]) Clear() {
	for i := 0; i < m.shards; i++ {
		m.mus[i].Lock()
		m.maps[i] = hashmap.New[string, V](m.capcity / m.shards)
		m.mus[i].Unlock()
	}
}

// Set assigns a value to a key.
// Returns the previous value, or false when no value was assigned.
func (m *Map[V]) Set(key string, value V) (prev any, replaced bool) {
	shard := m.choose(key)
	m.mus[shard].Lock()
	prev, replaced = m.maps[shard].Set(key, value)
	m.mus[shard].Unlock()
	return
}

// SetAccept assigns a value to a key. The "accept" function can be used to
// inspect the previous value, if any, and accept or reject the change.
// It's also provides a safe way to block other others from writing to the
// same shard while inspecting.
// Returns the previous value, or false when no value was assigned.
func (m *Map[V]) SetAccept(
	key string, value V,
	accept func(prev V, replaced bool) bool,
) (prev V, replaced bool) {
	shard := m.choose(key)
	m.mus[shard].Lock()
	defer m.mus[shard].Unlock()
	prev, replaced = m.maps[shard].Set(key, value)
	if accept != nil {
		if !accept(prev, replaced) {
			// revert unaccepted change
			if !replaced {
				// delete the newly set data
				m.maps[shard].Delete(key)
			} else {
				// reset updated data
				m.maps[shard].Set(key, prev)
			}
			//prev = nil
			replaced = false
		}
	}
	return
}

// Get returns a value for a key.
// Returns false when no value has been assign for key.
func (m *Map[V]) Get(key string) (value V, ok bool) {
	shard := m.choose(key)
	m.mus[shard].RLock()
	value, ok = m.maps[shard].Get(key)
	m.mus[shard].RUnlock()
	return
}

// Looks up an item under specified key
func (m *Map[V]) Has(key string) (ok bool) {
	shard := m.choose(key)
	m.mus[shard].RLock()
	_, ok = m.maps[shard].Get(key)
	m.mus[shard].RUnlock()
	return
}

// Delete deletes a value for a key.
// Returns the deleted value, or false when no value was assigned.
func (m *Map[V]) Delete(key string) (prev V, deleted bool) {
	shard := m.choose(key)
	m.mus[shard].Lock()
	prev, deleted = m.maps[shard].Delete(key)
	m.mus[shard].Unlock()
	return
}

// DeleteAccept deletes a value for a key. The "accept" function can be used to
// inspect the previous value, if any, and accept or reject the change.
// It's also provides a safe way to block other others from writing to the
// same shard while inspecting.
// Returns the deleted value, or false when no value was assigned.
func (m *Map[V]) DeleteAccept(
	key string,
	accept func(prev V, replaced bool) bool,
) (prev V, deleted bool) {
	shard := m.choose(key)
	m.mus[shard].Lock()
	defer m.mus[shard].Unlock()
	prev, deleted = m.maps[shard].Delete(key)
	if accept != nil {
		if !accept(prev, deleted) {
			// revert unaccepted change
			if deleted {
				// reset updated data
				m.maps[shard].Set(key, prev)
			}
			deleted = false
		}
	}

	return
}

// Len returns the number of values in map.
func (m *Map[V]) Len() (length int) {
	for i := 0; i < m.shards; i++ {
		m.mus[i].RLock()
		length += m.maps[i].Len()
		m.mus[i].RUnlock()
	}
	return
}

/*func (m *Map[V]) Len() int {
	var length atomic.Int32
	var wg sync.WaitGroup

	wg.Add(m.shards)
	for i := 0; i < m.shards; i++ {
		go func(i int) {
			defer wg.Done()
			shardLen := int32(m.maps[i].Len())
			length.Add(shardLen)
		}(i)
	}
	wg.Wait()
	return int(length.Load())
}*/

// Range calls the provided callback function for each key-value pair in the map until the
// callback returns false or all pairs have been processed.
func (m *Map[V]) Range(callback func(key string, value V) bool) {
	// Loop through all shards sequentially
	for i := 0; i < m.shards; i++ {
		// Lock and defer unlock for the current shard within its own scope
		m.mus[i].RLock()

		// The callback function returns false to stop the iteration.
		// If the inner Scan loop is terminated, we also need to stop the outer loop.
		// Therefore, we use a simple check and an early exit.
		// The `done` flag is no longer necessary.
		keepIterating := true
		m.maps[i].Scan(func(key string, value V) bool {
			if !callback(key, value) {
				keepIterating = false
				return false
			}
			return true
		})

		m.mus[i].RUnlock()

		// If the inner scan was terminated, we stop the outer loop.
		if !keepIterating {
			break
		}
	}
}

// choose selects a shard index based on the given key.
// It uses the xxh3 hash function to hash the key and then masks the result with
// (m.shards-1) to ensure the index falls within the range of shard indices.
func (m *Map[V]) choose(key string) int {
	return int(xxh3.HashString(key) & uint64(m.shards-1))
}

/*func (m *Map[V]) initDo() {
	m.once.Do(func() {
		m.shards = 1
		for m.shards < runtime.NumCPU()*16 {
			m.shards *= 2
		}
		scap := m.capcity / m.shards
		m.mus = make([]sync.RWMutex, m.shards)
		m.maps = make([]*hashmap.Map[string, V], m.shards)
		for i := 0; i < len(m.maps); i++ {
			m.maps[i] = hashmap.New[string, V](scap)
		}
	})
}*/

// initDo initializes the Map with the appropriate number of shards and mutexes.
// It ensures that this initialization is performed only once using sync.Once.
func (m *Map[V]) initDo() {
	// once.Do ensures the initialization block runs only once, even with concurrent calls.
	var once sync.Once
	once.Do(func() {
		// Step 1: Calculate the optimal number of shards.
		// The number of shards is chosen to be a power of 2 for efficient bitwise operations.
		numShards := runtime.NumCPU() * 16
		m.shards = 1
		for m.shards < numShards {
			m.shards *= 2
		}

		// Step 2: Initialize the mutexes and hash maps sequentially.
		m.mus = make([]sync.RWMutex, m.shards)
		m.maps = make([]*hashmap.Map[string, V], m.shards)

		// Calculate the capacity for each shard based on the total capacity.
		scap := m.capcity / m.shards

		// Loop through each shard and initialize it directly, which is simpler and avoids Goroutine overhead.
		for i := 0; i < m.shards; i++ {
			// A sync.RWMutex's zero value is the unlocked state, so explicit initialization is not needed.
			m.maps[i] = hashmap.New[string, V](scap)
		}
	})
}
