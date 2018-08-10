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
	var target [32]uint8
	var result_hash [32]uint8
	// a[0] = 97
	start := 0
	for {
		C.CuckooSolve((*C.char)(unsafe.Pointer(&a[0])), C.uint(80), C.uint(start), (*C.uint)(unsafe.Pointer(&result[0])), (*C.uint)(unsafe.Pointer(&result_len)), (*C.uchar)(unsafe.Pointer(&target[0])), (*C.uchar)(unsafe.Pointer(&result_hash[0])))
		start += 1
	}

	r := C.CuckooVerify((*C.char)(unsafe.Pointer(&a[0])), C.uint(80), C.uint(start), (*C.uint)(unsafe.Pointer(&result[0])), (*C.uchar)(unsafe.Pointer(&target[0])), (*C.uchar)(unsafe.Pointer(&result_hash[0])))
	var pass bool
	if byte(r) == 0 {
		pass = false
	} else {
		pass = true
	}
	fmt.Println(pass)

}
