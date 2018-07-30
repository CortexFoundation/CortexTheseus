package main

/*
#cgo  LDFLAGS:  -lstdc++  -lgominer
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
	fmt.Println("######### DEBUG ############")
	fmt.Println(a)
	fmt.Println("############################")
	C.CuckooSolve((*C.char)(unsafe.Pointer(&a[0])), C.uint(80), C.uint(63), (*C.uint)(unsafe.Pointer(&result[0])), (*C.uint)(unsafe.Pointer(&result_len)))

	fmt.Println("######### DEBUG ############")
	fmt.Println(result, result_len)
	fmt.Println("############################")

	r := C.CuckooVerify((*C.char)(unsafe.Pointer(&a[0])), C.uint(80), C.uint(63), (*C.uint)(unsafe.Pointer(&result[0])))

	fmt.Println("######### DEBUG ############")
	fmt.Println(r)
	fmt.Println("############################")

	var pass bool
	if byte(r) == 0 {
		pass = false
	} else {
		pass = true
	}
	fmt.Println(pass)

}
