package cuckoo

/*
#cgo LDFLAGS: -L../../cminer -lcuckoo -lstdc++
#cgo CFLAGS: -I../../cminer

#include "gominer.h"
*/
import "C"
import (
	"log"
	"unsafe"
)

func CuckooVerifyHeaderNonceAndSolutions(hash []byte, nonce uint32, result *uint32) int {
	tmpHash := hash
	log.Println("CuckooVerifyHeaderNonceAndSolutions: ", hash, nonce, result)
	r := C.CuckooVerifyHeaderNonceAndSolutions(
		(*C.uchar)(unsafe.Pointer(&tmpHash[0])),
		C.uint(len(hash)),
		C.uint(uint32(nonce)),
		(*C.uint)(unsafe.Pointer((result))))

	return int(r)
}
