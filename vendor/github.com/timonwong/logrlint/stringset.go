package logrlint

import "sort"

type stringSet map[string]struct{}

func newStringSet(items ...string) stringSet {
	s := make(stringSet)
	s.Insert(items...)
	return s
}

func (s stringSet) Insert(items ...string) {
	for _, item := range items {
		s[item] = struct{}{}
	}
}

func (s stringSet) Has(item string) bool {
	_, contained := s[item]
	return contained
}

func (s stringSet) List() []string {
	if len(s) == 0 {
		return nil
	}

	res := make([]string, 0, len(s))
	for key := range s {
		res = append(res, key)
	}
	sort.Strings(res)
	return res
}
