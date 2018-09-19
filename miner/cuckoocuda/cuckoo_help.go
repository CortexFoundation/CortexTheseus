package cuckoo_mean_miner

/*
#cgo LDFLAGS: -L./ -lgpugominer -L/usr/local/cuda/lib64 -lcudart -lstdc++
#cgo CFLAGS: -I./

#include "miner.h"
*/
import "C"
import (
	"unsafe"
)

func CuckooFindSolutionsCuda(hash []byte, nonce uint64) (status_code uint32, ret [][]uint32) {
	var (
		_solLength uint32
		_numSols   uint32
		result     [128]uint32
	)
	var tmpHash = make([]byte, 32)
	copy(tmpHash[:], hash)
	r := C.CuckooFindSolutionsCuda(
		(*C.uint8_t)(unsafe.Pointer(&tmpHash[0])),
		C.uint64_t(nonce),
		(*C.uint32_t)(unsafe.Pointer(&result[0])),
		C.uint32_t(len(result)),
		(*C.uint32_t)(unsafe.Pointer(&_solLength)),
		(*C.uint32_t)(unsafe.Pointer(&_numSols)))
	for solIdx := uint32(0); solIdx < _numSols; solIdx++ {
		var sol = make([]uint32, _solLength)
		copy(sol[:], result[solIdx*_solLength:(solIdx+1)*_solLength])
		ret = append(ret, sol)
	}
	return uint32(r), ret
}

func CuckooInitialize() {
	C.CuckooInitialize()
}
