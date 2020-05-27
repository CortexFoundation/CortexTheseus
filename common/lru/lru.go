package lru

import "container/list"

// Cache is an LRU cache. It is not safe for concurrent access.
type Cache struct {
	// MaxWeight is the maximum number of cache entries before
	// an item is evicted. Zero means no limit.
	MaxWeight     int64
	CurrentWeight int64

	// OnEvicted optionally specifies a callback function to be
	// executed when an entry is purged from the cache.
	OnEvicted func(key Key, value interface{})

	ll          *list.List
	cache       map[interface{}]*list.Element
	cacheWeight map[interface{}]int64
}

// A Key may be any value that is comparable. See http://golang.org/ref/spec#Comparison_operators
type Key interface{}

type entry struct {
	key   Key
	value interface{}
}

// New creates a new Cache.
// If maxWeight is zero, the cache has no limit and it's assumed
// that eviction is done by the caller.
func New(maxWeight int64) *Cache {
	return &Cache{
		MaxWeight:     maxWeight,
		CurrentWeight: 0,
		ll:            list.New(),
		cache:         make(map[interface{}]*list.Element),
		cacheWeight:   make(map[interface{}]int64),
	}
}

// Add adds a value to the cache.
func (c *Cache) Add(key Key, value interface{}, weight int64) {
	if c.cache == nil {
		c.cache = make(map[interface{}]*list.Element)
		c.cacheWeight = make(map[interface{}]int64)
		c.ll = list.New()
	}
	if ee, ok := c.cache[key]; ok {
		c.ll.MoveToFront(ee)
		ee.Value.(*entry).value = value
		c.cacheWeight[key] = weight
		return
	}
	ele := c.ll.PushFront(&entry{key, value})
	c.CurrentWeight += weight
	c.cache[key] = ele
	c.cacheWeight[key] = weight
	if c.MaxWeight != 0 && c.CurrentWeight > c.MaxWeight {
		c.RemoveOldest()
	}
}

// Get looks up a key's value from the cache.
func (c *Cache) Get(key Key) (value interface{}, ok bool) {
	if c.cache == nil {
		return
	}
	if ele, hit := c.cache[key]; hit {
		c.ll.MoveToFront(ele)
		return ele.Value.(*entry).value, true
	}
	return
}

// Remove removes the provided key from the cache.
func (c *Cache) Remove(key Key) {
	if c.cache == nil {
		return
	}
	if ele, hit := c.cache[key]; hit {
		c.removeElement(ele)
	}
}

// RemoveOldest removes the oldest item from the cache.
func (c *Cache) RemoveOldest() {
	if c.cache == nil {
		return
	}
	ele := c.ll.Back()
	if ele != nil {
		c.removeElement(ele)
	}
}

func (c *Cache) removeElement(e *list.Element) {
	c.ll.Remove(e)
	kv := e.Value.(*entry)
	c.CurrentWeight -= c.cacheWeight[kv.key]
	delete(c.cache, kv.key)
	delete(c.cacheWeight, kv.key)
	if c.OnEvicted != nil {
		c.OnEvicted(kv.key, kv.value)
	}
}

// Len returns the number of items in the cache.
func (c *Cache) Len() int {
	if c.cache == nil {
		return 0
	}
	return c.ll.Len()
}

// Clear purges all stored items from the cache.
func (c *Cache) Clear() {
	if c.OnEvicted != nil {
		for _, e := range c.cache {
			kv := e.Value.(*entry)
			c.OnEvicted(kv.key, kv.value)
		}
	}
	c.ll = nil
	c.cache = nil
}
