package cuckoo

/*
#cgo LDFLAGS: -L../../cminer -lgominer -lstdc++
#cgo CFLAGS: -I../../cminer

#include "gominer.h"
*/
import "C"
import (
	"unsafe"
)

func CuckooInit(threads C.uint) {
	C.CuckooInit(threads)
}

func CuckooFinalize() {
	C.CuckooFinalize()
}

func CuckooSolve(hash *byte, hash_len int, nonce uint32, result *uint32, result_len *uint32, diff *byte, result_hash *byte) byte {
	r := C.CuckooSolve(
		(*C.char)(unsafe.Pointer(hash)),
		C.uint(hash_len),
		C.uint(nonce),
		(*C.uint)(unsafe.Pointer(result)),
		(*C.uint)(unsafe.Pointer(result_len)),
		(*C.uchar)(unsafe.Pointer(diff)),
		(*C.uchar)(unsafe.Pointer(result_hash)))

	return byte(r)
}

func CuckooVerify(hash *byte, hash_len int, nonce uint32, result *uint32, diff *byte, result_hash *byte) byte {
	r := C.CuckooVerify(
		(*C.char)(unsafe.Pointer(hash)),
		C.uint(hash_len),
		C.uint(uint32(nonce)),
		(*C.uint)(unsafe.Pointer(result)),
		(*C.uchar)(unsafe.Pointer(diff)),
		(*C.uchar)(unsafe.Pointer(result_hash)))

	return byte(r)
}

func CuckooVerifyHeaderNonceAndSolutions(hash []byte, nonce uint32, result *uint32) int {
	tmpHash := hash
	r := C.CuckooVerifyHeaderNonceAndSolutions(
		(*C.char)(unsafe.Pointer(&tmpHash[0])),
		C.uint(len(hash)),
		C.uint(uint32(nonce)),
		(*C.uint)(unsafe.Pointer(result)))

	return int(r)
}

func CuckooVerifySolutions(hash *byte, hash_len int, result *uint32) int {
	r := C.CuckooVerifySolutions(
		(*C.char)(unsafe.Pointer(hash)),
		C.uint(hash_len),
		(*C.uint)(unsafe.Pointer(result)))

	return int(r)
}
