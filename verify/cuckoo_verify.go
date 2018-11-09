package verify

/*
#include "gominer.h"
*/
import "C"
import (
	"unsafe"
)

func CuckooVerifyProof(hash []byte, nonce uint64, result *uint32, proofSize uint8, edgeBits uint8) int {
	tmpHash := hash
	r := C.CuckooVerifyProof(
		(*C.uint8_t)(unsafe.Pointer(&tmpHash[0])),
		C.uint64_t(uint(nonce)),
		(*C.uint32_t)(unsafe.Pointer((result))),
		(C.uint8_t)(proofSize),
		(C.uint8_t)(edgeBits))
	return int(r)
}
