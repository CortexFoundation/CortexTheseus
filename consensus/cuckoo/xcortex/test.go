package main

// #cgo LDFLAGS: -L . -lverify -lstdc++ -L./cvm-runtime/build/cpu/ -lcvm_runtime_cpu
// #cgo CFLAGS: -I ./
// #include "verify.h"
import "C"
import "unsafe"
import "fmt"

func Verify() {
	var nonce uint64 = 0
	header := []uint8{0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6}
	difficulty := []uint8{0, 1, 2, 3, 34, 4, 5, 5, 5, 5}
	ret := C.Verify(C.uint64_t(nonce),
		(*C.uint8_t)(unsafe.Pointer(&header[0])),
		(*C.uint8_t)(unsafe.Pointer(&difficulty[0])))
	fmt.Println(ret)
}
