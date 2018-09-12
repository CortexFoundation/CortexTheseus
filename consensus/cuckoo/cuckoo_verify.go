package cuckoo

/*
#cgo LDFLAGS: -L../../cminer -lcuckoo -lstdc++
#cgo CFLAGS: -I../../cminer

#include "gominer.h"
*/
import "C"
import (
//	"log"
	"unsafe"
)

func CuckooVerifyHeaderNonceAndSolutions(hash []byte, nonce uint32, result *uint32) int {
	tmpHash := hash
	//log.Println("CuckooVerifyHeaderNonceAndSolutions: hash = ", hash, "\nnonce = ", nonce, "\nresult = ", result)
	r := C.CuckooVerifyHeaderNonceAndSolutions(
		(*C.uint8_t)(unsafe.Pointer(&tmpHash[0])),
		C.uint32_t(len(hash)),
		C.uint32_t(uint32(nonce)),
		(*C.uint32_t)(unsafe.Pointer((result))))

	return int(r)
}
