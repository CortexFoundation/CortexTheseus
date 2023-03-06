package generics

import "golang.org/x/exp/constraints"

func MakeMapIfNilAndSet[K comparable, V any](pm *map[K]V, k K, v V) {
	m := *pm
	if m == nil {
		m = make(map[K]V)
		*pm = m
	}
	m[k] = v
}

// Does this exist in the maps package?
func MakeMap[K comparable, V any, M ~map[K]V](pm *M) {
	*pm = make(M)
}

func MakeMapWithCap[K comparable, V any, M ~map[K]V, C constraints.Integer](pm *M, cap C) {
	*pm = make(M, cap)
}

func MakeMapIfNil[K comparable, V any, M ~map[K]V](pm *M) {
	if *pm == nil {
		MakeMap(pm)
	}
}
