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

func CuckooInit(threads uint32) {
	CuckooInitialize(threads, 5)
}

func CuckooInitialize(threads uint32, nInstances uint32) {
	C.CuckooInit(C.uint32_t(threads), C.uint32_t(nInstances))
}

func CuckooFinalize() {
	C.CuckooFinalize()
}

func CuckooSolve(hash *byte, hash_len int, nonce uint32, result *uint32, result_len *uint32, diff *byte, result_hash *byte) byte {

	r := C.CuckooSolve(
		(*C.uint8_t)(unsafe.Pointer(hash)),
		C.uint32_t(hash_len),
		C.uint32_t(nonce),
		(*C.uint32_t)(unsafe.Pointer(result)),
		(*C.uint32_t)(unsafe.Pointer(result_len)),
		(*C.uint8_t)(unsafe.Pointer(diff)),
		(*C.uint8_t)(unsafe.Pointer(result_hash)))

	return byte(r)
}

func CuckooFindSolutions(hash []byte, nonce uint32, result *[]uint32) (status_code uint32, solLength uint32, numSols uint32) {
	var (
		_solLength uint32
		_numSols   uint32
	)
	var tmpHash = make([]byte, len(hash))
	copy(tmpHash[:], hash)
	r := C.CuckooFindSolutions(
		(*C.uint8_t)(unsafe.Pointer(&tmpHash[0])),
		C.uint32_t(len(tmpHash)),
		C.uint32_t(nonce),
		(*C.uint32_t)(unsafe.Pointer(&((*result)[0]))),
		C.uint32_t(len(*result)),
		(*C.uint32_t)(unsafe.Pointer(&_solLength)),
		(*C.uint32_t)(unsafe.Pointer(&_numSols)),
	)

	return uint32(r), _solLength, _numSols
}

func CuckooVerify(hash *byte, hash_len int, nonce uint32, result *uint32, diff *byte, result_hash *byte) byte {
	r := C.CuckooVerify(
		(*C.uint8_t)(unsafe.Pointer(hash)),
		C.uint32_t(hash_len),
		C.uint32_t(uint32(nonce)),
		(*C.result_t)(unsafe.Pointer(result)),
		(*C.uint8_t)(unsafe.Pointer(diff)),
		(*C.uint8_t)(unsafe.Pointer(result_hash)))

	return byte(r)
}
