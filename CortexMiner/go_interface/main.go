package main

/*
#cgo  LDFLAGS:  -lstdc++  -lgominer
#include "gominer.h"
*/
import "C"
import (
	"encoding/binary"
	"fmt"
	"os"
	"unsafe"
)

type Header struct {
	data   []byte
	nonce  [8]byte
	result [42]uint32
}

func main() {
	C.CuckooInit()
	var header Header
	result := header.result
	for i, _ := range result {
		result[i] = 1
	}

	fmt.Println("")
	for nonce := 0; ; nonce++ {
		fmt.Println("Trying solve with nonce ", nonce)

		old := os.Stdout
		// _, w, _ := os.Pipe()
		os.Stdout = nil

		fmt.Println("Check")
		var result_len uint32

		var a = &header
		var a_len = unsafe.Sizeof(*a) * 8
		C.CuckooSolve(
			(*C.char)(unsafe.Pointer(a)),
			C.uint(a_len),
			C.uint(nonce),
			(*C.uint)(unsafe.Pointer(&result[0])),
			(*C.uint)(unsafe.Pointer(&result_len)))

		r := C.CuckooVerify(
			(*C.char)(unsafe.Pointer(a)),
			C.uint(80),
			C.uint(nonce),
			(*C.uint)(unsafe.Pointer(&result[0])))

		// w.Close()
		os.Stdout = old

		if byte(r) != 0 {
			binary.BigEndian.PutUint64(header.nonce[:], 63)
			header.result = result
			fmt.Println(header)
			break
		}
	}

}
