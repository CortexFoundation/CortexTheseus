//go:build cuda_miner
// +build cuda_miner

package main

/*
#cgo LDFLAGS: -L../../../../solution -lcudaminer -L/usr/local/cuda/lib64 -lcudart -lstdc++ -lnvidia-ml
#cgo CFLAGS: -I./

#include "../../../../solution/miner.h"
*/
import "C"
import (
	"fmt"
	"log"

	//	"time"
	"strconv"
	"strings"
	"unsafe"
)

func CuckooInitialize(threads int, strDeviceIds string, algorithm string) error {
	var arrayDeviceIds []string = strings.Split(strDeviceIds, ",")
	var deviceNum int = 1
	var devices []uint32
	var selected int = 1

	v, err := strconv.Atoi(arrayDeviceIds[0])
	if err != nil {
		return err
	}
	devices = append(devices, uint32(v))

	C.CuckooInitialize((*C.uint32_t)(unsafe.Pointer(&devices[0])), C.uint32_t(deviceNum), C.int(selected), 0)
	return nil
}

func CuckooFinalize() {
	C.CuckooFinalize()
}

func CuckooFindSolutions(hash []byte, nonce uint64) (status_code uint32, ret [][]uint32) {
	var (
		_solLength uint32
		_numSols   uint32
		result     [128]uint32
	)
	var tmpHash = make([]byte, 32)
	copy(tmpHash[:], hash)
	var threadId uint32 = 0
	nedges := C.FindSolutionsByGPU(
		(*C.uint8_t)(unsafe.Pointer(&tmpHash[0])),
		C.uint64_t(nonce),
		C.uint32_t(threadId))

	r := C.FindCycles(
		C.uint32_t(threadId),
		C.uint32_t(nedges),
		(*C.uint32_t)(unsafe.Pointer(&result[0])),
		C.uint32_t(len(result)),
		(*C.uint32_t)(unsafe.Pointer(&_solLength)),
		(*C.uint32_t)(unsafe.Pointer(&_numSols)))

	if uint32(len(result)) < _solLength*_numSols {
		log.Println(fmt.Sprintf("WARNING: discard possible solutions, total sol num=%v, received number=%v", _numSols, uint32(len(result))/_solLength))
		_numSols = uint32(len(result)) / _solLength
	}

	for solIdx := uint32(0); solIdx < _numSols; solIdx++ {
		var sol = make([]uint32, _solLength)
		copy(sol[:], result[solIdx*_solLength:(solIdx+1)*_solLength])
		ret = append(ret, sol)
	}

	return uint32(r), ret
}

/*func CuckooVerify(hash *byte, nonce uint64, result types.BlockSolution, result_sha3 []byte, diff *big.Int) bool {
	sha3hash := common.BytesToHash(result_sha3)

	if sha3hash.Big().Cmp(diff) <= 0 {
		r := C.CuckooVerifyProof(
			(*C.uint8_t)(unsafe.Pointer(hash)),
			C.uint64_t(nonce),
			(*C.result_t)(unsafe.Pointer(&result[0])))
		return (r == 1)
	}
	return false
}

func CuckooVerify_cuckaroo(hash *byte, nonce uint64, result types.BlockSolution, result_sha3 []byte, diff *big.Int) bool {
	sha3hash := common.BytesToHash(result_sha3)
	if sha3hash.Big().Cmp(diff) <= 0 {
		r := C.CuckooVerifyProof_cuckaroo(
			(*C.uint8_t)(unsafe.Pointer(hash)),
			C.uint64_t(nonce),
			(*C.result_t)(unsafe.Pointer(&result[0])))
		return (r == 1)
	}
	return false
}*/
