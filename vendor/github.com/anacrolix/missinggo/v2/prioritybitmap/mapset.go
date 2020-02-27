package prioritybitmap

func newMapSet() Set {
	return mapSet{
		m: make(map[int]struct{}),
	}
}

type mapSet struct {
	m map[int]struct{}
}

func (m mapSet) Has(bit int) bool {
	_, ok := m.m[bit]
	return ok
}

func (m mapSet) Delete(bit int) {
	delete(m.m, bit)
}

func (m mapSet) Len() int {
	return len(m.m)
}

func (m mapSet) Set(bit int) {
	m.m[bit] = struct{}{}
}

func (m mapSet) Range(f func(int) bool) {
	for bit := range m.m {
		if !f(bit) {
			break
		}
	}
}
