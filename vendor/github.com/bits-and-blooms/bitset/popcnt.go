package bitset

import "math/bits"

func popcntSlice(s []uint64) uint64 {
	var cnt int
	for _, x := range s {
		cnt += bits.OnesCount64(x)
	}
	return uint64(cnt)
}

func popcntMaskSlice(s, m []uint64) uint64 {
	var cnt int
	// this explicit check eliminates a bounds check in the loop
	if len(m) < len(s) {
		panic("mask slice is too short")
	}
	for i := range s {
		cnt += bits.OnesCount64(s[i] &^ m[i])
	}
	return uint64(cnt)
}

func popcntAndSlice(s, m []uint64) uint64 {
	var cnt int
	// this explicit check eliminates a bounds check in the loop
	if len(m) < len(s) {
		panic("mask slice is too short")
	}
	for i := range s {
		cnt += bits.OnesCount64(s[i] & m[i])
	}
	return uint64(cnt)
}

func popcntOrSlice(s, m []uint64) uint64 {
	var cnt int
	// this explicit check eliminates a bounds check in the loop
	if len(m) < len(s) {
		panic("mask slice is too short")
	}
	for i := range s {
		cnt += bits.OnesCount64(s[i] | m[i])
	}
	return uint64(cnt)
}

func popcntXorSlice(s, m []uint64) uint64 {
	var cnt int
	// this explicit check eliminates a bounds check in the loop
	if len(m) < len(s) {
		panic("mask slice is too short")
	}
	for i := range s {
		cnt += bits.OnesCount64(s[i] ^ m[i])
	}
	return uint64(cnt)
}
