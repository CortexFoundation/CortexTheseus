package main

/*
#cgo  CFLAGS:   -I ..
#cgo  LDFLAGS:  -lstdc++  -L. -lgominer
#include "gominer.h"
*/
import (
	"C"
)
import (
	"fmt"
	"sync"
	"unsafe"
)

func main() {
	C.CuckooInit(C.uint(16))
	var wg sync.WaitGroup
	for nthread := 0; nthread < 3; nthread++ {
		wg.Add(1)
		go func() {
			var result [42]uint32
			for i, _ := range result {
				result[i] = 1
			}
			var result_len uint32
			var a = [32]byte{
				byte(nthread), 117, 240, 250, 183, 93, 137, 77, 80, 16, 154, 194, 62, 215, 43, 42, 208, 205, 34, 99, 158, 215, 220, 48, 125, 58, 2, 5, 40, 0, 0, 0,
			}
			var target [32]uint8
			target[0] = 0
			for i := 1; i < 32; i++ {
				target[i] = 255
			}
			var result_hash [32]uint8
			start := 0
			for {
				r := C.CuckooSolve((*C.char)(unsafe.Pointer(&a[0])), C.uint(32), C.uint(start), (*C.uint)(unsafe.Pointer(&result[0])), (*C.uint)(unsafe.Pointer(&result_len)), (*C.uchar)(unsafe.Pointer(&target[0])), (*C.uchar)(unsafe.Pointer(&result_hash[0])))
				if byte(r) == 1 {
					break
				}
				start += 1
			}
			r := C.CuckooVerify((*C.char)(unsafe.Pointer(&a[0])), C.uint(32), C.uint(start), (*C.uint)(unsafe.Pointer(&result[0])), (*C.uchar)(unsafe.Pointer(&target[0])), (*C.uchar)(unsafe.Pointer(&result_hash[0])))
			var pass bool
			if byte(r) == 0 {
				pass = false
			} else {
				pass = true
			}
			fmt.Println(pass)
			wg.Done()
		}()
	}
	wg.Wait()
	C.CuckooFinalize()
}
