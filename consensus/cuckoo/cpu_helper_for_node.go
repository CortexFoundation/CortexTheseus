// +build cpu_miner

package main

/*
#cgo LDFLAGS: -L../../PoolMiner/miner/libcuckoo -lcpuminer -lstdc++
#cgo CFLAGS: -I./

#include "../../PoolMiner/miner/libcuckoo/miner.h"
*/
import "C"
import (
	"fmt"
	//	"time"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/log"
	"math/big"
	"unsafe"
)

func CuckooInit(threads uint32) {
	CuckooInitialize(0, "", "cuckaroo")
}

func CuckooInitialize(threads int, strDeviceIds string, algorithm string) error {
	//	var deviceNum int = 1
	var devices []uint32
	var selected int = 1
	devices = append(devices, 0)
	C.CuckooInitialize((*C.uint32_t)(unsafe.Pointer(&devices[0])), C.uint32_t(threads), C.int(selected), 0)
	return nil
}

func CuckooFinalize() {
	log.Debug("CuckooFinalize")
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
	r := C.RunSolverOnCPU(
		(*C.uint8_t)(unsafe.Pointer(&tmpHash[0])),
		C.uint64_t(nonce),
		(*C.uint32_t)(unsafe.Pointer(&result[0])),
		C.uint32_t(len(result)),
		(*C.uint32_t)(unsafe.Pointer(&_solLength)),
		(*C.uint32_t)(unsafe.Pointer(&_numSols)))

	if uint32(len(result)) < _solLength*_numSols {
		log.Warn(fmt.Sprintf("WARNING: discard possible solutions, total sol num=%v, received number=%v", _numSols, uint32(len(result))/_solLength))
		_numSols = uint32(len(result)) / _solLength
	}

	for solIdx := uint32(0); solIdx < _numSols; solIdx++ {
		var sol = make([]uint32, _solLength)
		copy(sol[:], result[solIdx*_solLength:(solIdx+1)*_solLength])
		//		 log.Println(fmt.Sprintf("Index: %v, Solution: %v", solIdx, sol))
		ret = append(ret, sol)
	}

	return uint32(r), ret
}
func CuckooVerify(hash *byte, nonce uint64, result types.BlockSolution, result_sha3 []byte, diff *big.Int) bool {
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
}
