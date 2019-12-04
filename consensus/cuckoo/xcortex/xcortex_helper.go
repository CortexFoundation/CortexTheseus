package main

// #cgo LDFLAGS: -L . -lverify -lstdc++ -L../../../infernet/build/cpu/ -lcvm_runtime_cpu
// #cgo CFLAGS: -I ./
// #include "verify.h"
import "C"
import "unsafe"

//import "fmt"

func Verify(header *byte, nonce uint64, difficulty string) bool {
	//	fmt.Printf("nonce=%v, header=%v, difficulty=%s", nonce, header, difficulty)
	bdiff := []byte(difficulty)
	ret := C.Verify(
		C.uint64_t(nonce),
		(*C.uint8_t)(unsafe.Pointer(header)),
		(*C.uint8_t)(unsafe.Pointer(&bdiff[0])))
	return (ret <= 0)
}

func Verify_remote(header *byte, nonce uint64, strDifficulty string, strShareTarget string, strBlockTarget string) (bool, bool, bool) {
	//	fmt.Printf("difficulty=%s, shareTarget=%s, blockTarget=%s\n", strDifficulty, strShareTarget, strBlockTarget)
	difficulty := []byte(strDifficulty)
	shareTarget := []byte(strShareTarget)
	blockTarget := []byte(strBlockTarget)
	var ret [3]int32
	C.Verify_remote(
		C.uint64_t(nonce),
		(*C.uint8_t)(unsafe.Pointer(header)),
		(*C.uint8_t)(unsafe.Pointer(&difficulty[0])),
		(*C.uint8_t)(unsafe.Pointer(&shareTarget[0])),
		(*C.uint8_t)(unsafe.Pointer(&blockTarget[0])),
		(*C.int32_t)(unsafe.Pointer(&ret[0])))
	//	fmt.Printf("verify result : %v %v %v\n", ret[0], ret[1], ret[2])
	return (ret[0] <= 0), (ret[1] <= 0), (ret[2] <= 0)
}
