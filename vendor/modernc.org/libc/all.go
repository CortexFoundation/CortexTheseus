// Copyright 202 The Libc Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package libc // import "modernc.org/libc"

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"
	"unsafe"
)

const (
	heapAlign = 16
	heapGuard = 16
)

const FALSE = 0
const TRUE = 1

var (
	pid = fmt.Sprintf("[%v %v] ", os.Getpid(), filepath.Base(os.Args[0]))
)

func origin(skip int) string {
	pc, fn, fl, _ := runtime.Caller(skip)
	f := runtime.FuncForPC(pc)
	var fns string
	if f != nil {
		fns = f.Name()
		if x := strings.LastIndex(fns, "."); x > 0 {
			fns = fns[x+1:]
		}
	}
	return fmt.Sprintf("%s:%d:%s", filepath.Base(fn), fl, fns)
}

func trc(s string, args ...interface{}) string { //TODO-
	switch {
	case s == "":
		s = fmt.Sprintf(strings.Repeat("%v ", len(args)), args...)
	default:
		s = fmt.Sprintf(s, args...)
	}
	r := fmt.Sprintf("%s: TRC %s (%v:)", origin(2), s, origin(3))
	fmt.Fprintf(os.Stdout, "%s %s\n", pid, r)
	os.Stdout.Sync()
	return r
}

func todo(s string, args ...interface{}) string { //TODO-
	switch {
	case s == "":
		s = fmt.Sprintf(strings.Repeat("%v ", len(args)), args...)
	default:
		s = fmt.Sprintf(s, args...)
	}
	r := fmt.Sprintf("%s %s: TODOTODO %s (%v:)", pid, origin(2), s, origin(3)) //TODOOK
	if dmesgs {
		dmesg("%s", r)
	}
	fmt.Fprintf(os.Stdout, "%s\n", r)
	fmt.Fprintf(os.Stdout, "%s\n", debug.Stack()) //TODO-
	os.Stdout.Sync()
	os.Exit(1)
	panic("unrechable")
}

type sorter struct {
	len  int
	base uintptr
	sz   uintptr
	f    func(*TLS, uintptr, uintptr) int32
	t    *TLS
}

func (s *sorter) Len() int { return s.len }

func (s *sorter) Less(i, j int) bool {
	return s.f(s.t, s.base+uintptr(i)*s.sz, s.base+uintptr(j)*s.sz) < 0
}

func (s *sorter) Swap(i, j int) {
	p := s.base + uintptr(i)*s.sz
	q := s.base + uintptr(j)*s.sz
	for i := 0; i < int(s.sz); i++ {
		*(*byte)(unsafe.Pointer(p)), *(*byte)(unsafe.Pointer(q)) = *(*byte)(unsafe.Pointer(q)), *(*byte)(unsafe.Pointer(p))
		p++
		q++
	}
}
