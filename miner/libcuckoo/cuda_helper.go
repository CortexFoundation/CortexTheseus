// +build !opencl

package libcuckoo

/*
#cgo LDFLAGS: -L./ -lcudaminer -L/usr/local/cuda/lib64 -lcudart -lstdc++
#cgo CFLAGS: -I./

#include "miner.h"
*/
import "C"
import (
	"fmt"
	"log"
//	"time"
	"unsafe"
)

func FindSolutionsByGPU(hash []byte, nonce uint64, threadId uint32) (status_code uint32, ret [][]uint32) {
	var (
		_solLength uint32
		_numSols   uint32
		result     [128]uint32
	)
	var tmpHash = make([]byte, 32)
	copy(tmpHash[:], hash)

//	start := time.Now()
	r := C.FindSolutionsByGPU(
		(*C.uint8_t)(unsafe.Pointer(&tmpHash[0])),
		C.uint64_t(nonce),
		C.uint32_t(threadId),
		(*C.uint32_t)(unsafe.Pointer(&result[0])),
		C.uint32_t(len(result)),
		(*C.uint32_t)(unsafe.Pointer(&_solLength)),
		(*C.uint32_t)(unsafe.Pointer(&_numSols)))

//	duration := time.Since(start)
//	log.Println(fmt.Sprintf("CuckooFindSolutionCuda | time=%v, status code=%v", duration, _numSols))

	// TODO add warning of discarding possible solutions
	if uint32(len(result)) < _solLength*_numSols {
		log.Println(fmt.Sprintf("WARNING: discard possible solutions, total sol num=%v, received number=%v", _numSols, uint32(len(result))/_solLength))
		_numSols = uint32(len(result)) / _solLength
	}

	for solIdx := uint32(0); solIdx < _numSols; solIdx++ {
		var sol = make([]uint32, _solLength)
		copy(sol[:], result[solIdx*_solLength:(solIdx+1)*_solLength])
		// log.Println(fmt.Sprintf("Index: %v, Solution: %v", solIdx, sol))
		ret = append(ret, sol)
	}

	return uint32(r), ret
}

func CuckooInitialize(devices []uint32, deviceNum uint32) {
	C.CuckooInitialize((*C.uint32_t)(unsafe.Pointer(&devices[0])), C.uint32_t(deviceNum))
}
