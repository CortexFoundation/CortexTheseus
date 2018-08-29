package main

/*
#cgo  CFLAGS:   -I ..
#cgo  LDFLAGS:  -lstdc++  -L. -lgominer
#include "gominer.h"
*/
import (
	cc "C"
)
import (
	"fmt"
	"unsafe"
)

func main() {

	// head := &types.Header{
	// 	Number:     big.NewInt(1),
	// 	Difficulty: big.NewInt(0),
	// 	Time:       big.NewInt(0),
	// }
	// target := big.NewInt(0)
	// target.Exp(big.NewInt(2), big.NewInt(256), nil)
	// target.Sub(target, head.Difficulty)
	// target.Sub(target, big.NewInt(1))
	// head.Difficulty = target
	// cuckoo := NewTester()
	// nb := types.NewBlockWithHeader(head)
	// block, err := cuckoo.Seal(nil, nb, nil)
	// if err != nil {
	// 	t.Fatalf("failed to seal block: %v", err)
	// }
	// t.Fatalf("1")
	// head.Nonce = types.EncodeNonce(block.Nonce())
	// head.Solution = block.Solution()
	// head.SolutionHash = block.Header().SolutionHash

	// if err := cuckoo.VerifySeal(nil, head); err != nil {
	// 	t.Fatalf("unexpected verification error: %v", err)
	// }

	C.minerBot * bot = C.CuckooCreate()
	C.CuckooInit(bot)
	var result [42]uint32
	for i, _ := range result {
		result[i] = 1
	}
	var result_len uint32
	var a = [32]byte{
		2, 117, 240, 250, 183, 93, 137, 77, 80, 16, 154, 194, 62, 215, 43, 42, 208, 205, 34, 99, 158, 215, 220, 48, 125, 58, 2, 5, 40, 0, 0, 0,
	}
	var target [32]uint8
	for i := 0; i < 32; i++ {
		target[i] = 255
	}
	var result_hash [32]uint8
	// a[0] = 97
	start := 0
	for {
		r := C.CuckooSolve(bot, (*C.char)(unsafe.Pointer(&a[0])), C.uint(32), C.uint(start), (*C.uint)(unsafe.Pointer(&result[0])), (*C.uint)(unsafe.Pointer(&result_len)), (*C.uchar)(unsafe.Pointer(&target[0])), (*C.uchar)(unsafe.Pointer(&result_hash[0])))
		if byte(r) == 1 {
			break
		}
		start += 1
	}

	r := C.CuckooVerify(bot, (*C.char)(unsafe.Pointer(&a[0])), C.uint(32), C.uint(start), (*C.uint)(unsafe.Pointer(&result[0])), (*C.uchar)(unsafe.Pointer(&target[0])), (*C.uchar)(unsafe.Pointer(&result_hash[0])))
	var pass bool
	if byte(r) == 0 {
		pass = false
	} else {
		pass = true
	}
	fmt.Println(pass)

}
