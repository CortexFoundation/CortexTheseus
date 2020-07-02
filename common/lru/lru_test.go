package lru

import (
	"container/list"
	"reflect"
	"testing"
)

func (c *Cache) Keys() []interface{} {
	keys := make([]interface{}, len(c.cache))
	i := 0
	for ent := c.ll.Back(); ent != nil; ent = ent.Prev() {
		keys[i] = ent.Value.(*entry).key
		i++
	}
	return keys
}

func TestLRU(t *testing.T) {
	evictCounter := 0
	onEvicted := func(k Key, v interface{}) {
		evictCounter++
	}
	l := New(128)
	l.OnEvicted = onEvicted
	for i := 0; i < 256; i++ {
		l.Add(i, i, 1)
	}
	if l.Len() != 128 {
		t.Fatalf("bad len: %v", l.Len())
	}

	if evictCounter != 128 {
		t.Fatalf("bad evict count: %v", evictCounter)
	}

	for i, k := range l.Keys() {
		if v, ok := l.Get(k); !ok || v != k || v != i+128 {
			t.Fatalf("bad key: %v", k)
		}
	}
	for i := 0; i < 128; i++ {
		_, ok := l.Get(i)
		if ok {
			t.Fatalf("should be evicted")
		}
	}
	for i := 128; i < 256; i++ {
		_, ok := l.Get(i)
		if !ok {
			t.Fatalf("should not be evicted")
		}
	}
	for i := 128; i < 192; i++ {
		l.Remove(i)
		l.Remove(i)
		_, ok := l.Get(i)
		if ok {
			t.Fatalf("should be deleted")
		}
	}

	l.Get(192) // expect 192 to be last key in l.Keys()

	for i, k := range l.Keys() {
		if (i < 63 && k != i+193) || (i == 63 && k != 192) {
			t.Fatalf("out of order key: %v", k)
		}
	}

	l.Clear()
	if l.Len() != 0 {
		t.Fatalf("bad len: %v", l.Len())
	}
	if _, ok := l.Get(200); ok {
		t.Fatalf("should contain nothing")
	}
}

func TestLRU_RemoveOldest(t *testing.T) {
	l := New(128)
	for i := 0; i < 256; i++ {
		l.Add(i, i, 1)
	}
}

// Test that Add returns true/false if an eviction occurred
func TestLRU_Add(t *testing.T) {

	l := New(2)
	l.Add(1, 1, 1)
	l.Add(1, 1, 1)
	if l.Len() > 1 {
		t.Fatalf("the same cache should not add again")
	}

	l.Add(1, 1, 1)
	l.Add(2, 2, 2)
}
func TestCache_Add(t *testing.T) {
	type fields struct {
		MaxWeight     int64
		CurrentWeight int64
		OnEvicted     func(key Key, value interface{})
		ll            *list.List
		cache         map[interface{}]*list.Element
		cacheWeight   map[interface{}]int64
	}
	type args struct {
		key    Key
		value  interface{}
		weight int64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		{"case1",
			fields{MaxWeight: 4},
			args{1, 1, 1},
		},
		{"case2",
			fields{},
			args{1, 1, 1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &Cache{
				MaxWeight:     tt.fields.MaxWeight,
				CurrentWeight: tt.fields.CurrentWeight,
				OnEvicted:     tt.fields.OnEvicted,
				ll:            tt.fields.ll,
				cache:         tt.fields.cache,
				cacheWeight:   tt.fields.cacheWeight,
			}
			c.Add(tt.args.key, tt.args.value, tt.args.weight)
		})
	}
}

func TestCache_Get(t *testing.T) {

	l := New(128)
	l.Add(1, 2, 1)
	l.Add(2, 4, 2)

	type fields struct {
		MaxWeight     int64
		CurrentWeight int64
		OnEvicted     func(key Key, value interface{})
		ll            *list.List
		cache         map[interface{}]*list.Element
		cacheWeight   map[interface{}]int64
	}
	type args struct {
		key Key
	}
	tests := []struct {
		name      string
		fields    fields
		args      args
		wantValue interface{}
		wantOk    bool
	}{
		{
			"case1",
			fields{l.MaxWeight, l.CurrentWeight, l.OnEvicted, l.ll, l.cache, l.cacheWeight},
			args{1},
			2,
			true,
		},
		{
			"case2",
			fields{l.MaxWeight, l.CurrentWeight, l.OnEvicted, l.ll, l.cache, l.cacheWeight},
			args{4},
			nil,
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &Cache{
				MaxWeight:     tt.fields.MaxWeight,
				CurrentWeight: tt.fields.CurrentWeight,
				OnEvicted:     tt.fields.OnEvicted,
				ll:            tt.fields.ll,
				cache:         tt.fields.cache,
				cacheWeight:   tt.fields.cacheWeight,
			}
			gotValue, gotOk := c.Get(tt.args.key)
			if !reflect.DeepEqual(gotValue, tt.wantValue) {
				t.Errorf("Cache.Get() gotValue = %v, want %v", gotValue, tt.wantValue)
			}
			if gotOk != tt.wantOk {
				t.Errorf("Cache.Get() gotOk = %v, want %v", gotOk, tt.wantOk)
			}
		})
	}
}

func TestCache_Remove(t *testing.T) {
	//the start of the cache before remove

	l1 := New(128)
	l1.Add(1, 11, 1)
	l1.Add(2, 22, 2)
	l1.Add(3, 33, 3)

	l1r := New(128)
	l1r.Add(1, 11, 1)
	l1r.Add(3, 33, 3)
	l2 := New(128)
	l2.cache = nil
	tests := []struct {
		ca  *Cache
		key Key
		cll *list.List
	}{
		{l1, 2, l1r.ll},
		{l2, 2, l2.ll},
	}
	for _, tt := range tests {
		tt.ca.Remove(tt.key)
		if tt.ca.ll != tt.cll {
			t.Errorf("there is sonething wrong happend when remove")
		}
	}
}
