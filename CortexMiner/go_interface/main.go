package main

/*
#cgo  CFLAGS:   -I ..
#cgo  LDFLAGS:  -lstdc++  -L. -lgominer
#include "gominer.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func main() {
	C.CuckooInit()
	var result [42]uint32
	for i, _ := range result {
		result[i] = 1
	}
	var result_len uint32
	var a [10]byte
	// a[0] = 97
	C.CuckooSolve((*C.char)(unsafe.Pointer(&a[0])), C.uint(80), C.uint(63), (*C.uint)(unsafe.Pointer(&result[0])), (*C.uint)(unsafe.Pointer(&result_len)))

	r := C.CuckooVerify((*C.char)(unsafe.Pointer(&a[0])), C.uint(80), C.uint(63), (*C.uint)(unsafe.Pointer(&result[0])))
	var pass bool
	if byte(r) == 0 {
		pass = false
	} else {
		pass = true
	}
	fmt.Println(pass)

}
