package legacy

import (
	"testing"
)

func TestLRUAdd(t *testing.T) {
	evictCounter := 0
	onEvicted := func(k Key, v interface{}) {
		evictCounter++
	}
	l := New(7)
	l.OnEvicted = onEvicted

	//test :when the weight is overflow after add
	l.Add(1, 1, 1)
	if evictCounter != 0 {
		t.Errorf("should not have an eviction")
	}
	l.Add(2, 2, 1)
	l.Add(3, 3, 7)
	if evictCounter != 1 {
		t.Errorf("should have an eviction")
	}
	//test : add the same key but different value and weight
	l.Add(1, 2, 3)
	if ee, ok := l.cache[1]; ok {
		if ee.Value.(*entry).value != 2 || l.cacheWeight[1] != 3 {
			t.Errorf("update the value of the key %v failed", 1)
		}
	}
	//test : add the exactly same key and value
	l.Add(1, 2, 3)
	if l.CurrentWeight != 10 {
		t.Errorf("deal with the same entry wrong")
	}
}

func TestLRUGet(t *testing.T) {
	l := New(7)
	//test : get from empty
	if _, ok := l.Get(1); ok {
		t.Errorf("empty cache should not get anything")
	}
	tests := []struct {
		key       Key
		wantValue interface{}
		wantOK    bool
	}{
		{1, 2, true},
		{2, nil, false},
	}
	l.Add(1, 2, 1)
	for _, tt := range tests {
		value, ok := l.Get(tt.key)
		if value != tt.wantValue {
			t.Errorf("get the value wrong")
		}
		if ok != tt.wantOK {
			t.Errorf("dont't get the value in right way")
		}
	}
}

func TestLRURemove(t *testing.T) {
	evictCounter := 0
	onEvicted := func(k Key, v interface{}) {
		evictCounter++
	}
	l := New(128)
	l.OnEvicted = onEvicted
	l.Remove(1)
	l.RemoveOldest()
	if evictCounter != 0 {
		t.Errorf("cant remove from empty cache")
	}
	for i := 0; i < 256; i++ {
		l.Add(i, i, 1)
	}
	evictCounter = 0
	l.Remove(1)
	if l.Len() != 128 {
		t.Errorf("remove the element that dont't exited ")
	}
	l.Remove(255)
	_, ok := l.Get(255)
	if l.Len() != 127 && ok {
		t.Errorf("don't remove")
	}
	l.RemoveOldest()
	_, ok = l.Get(128)
	if l.Len() != 126 {
		t.Errorf("don't remove")
	} else if ok {
		t.Errorf("remove the wrong entry")
	}

}

func TestLRULen(t *testing.T) {
	l := New(128)
	if l.Len() != 0 {
		t.Errorf("bad len")
	}
	for i := 0; i < 128; i++ {
		l.Add(i, i, 1)
	}
	if l.Len() != 128 {
		t.Errorf("bad len")
	}
}

func TestLRUClear(t *testing.T) {
	evictCounter := 0
	onEvicted := func(k Key, v interface{}) {
		evictCounter++
	}
	l := New(128)
	l.OnEvicted = onEvicted
	for i := 0; i < 128; i++ {
		l.Add(i, i, 1)
	}
	l.Clear()
	if evictCounter != 128 {
		t.Errorf("don't clear completely")
	}
	if l.ll != nil || l.cache != nil {
		t.Errorf("bad clear")
	}
}
