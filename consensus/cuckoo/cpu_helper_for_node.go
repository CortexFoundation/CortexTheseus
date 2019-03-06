// +build cpu_miner

package cuckoo

/*
#cgo LDFLAGS: -L../../PoolMiner/miner/libcuckoo -lcpuminer -lstdc++
#cgo CFLAGS: -I./

#include "../../PoolMiner/miner/libcuckoo/miner.h"
*/
import "C"
import (
	"fmt"
	"log"
	//	"time"
	"unsafe"
	"github.com/ethereum/go-ethereum/PoolMiner/common"
	"github.com/ethereum/go-ethereum/PoolMiner/crypto"
)

func CuckooInit(threads uint32) {
	CuckooInitialize(0, 1)
}

func CuckooInitialize(threads uint32, nInstances uint32) {
	var devices []uint32
	var deviceNum uint32
	var selected int
	devices = append(devices, 0);
	deviceNum = 1
	selected = 0
	C.CuckooInitialize((*C.uint32_t)(unsafe.Pointer(&devices[0])), C.uint32_t(deviceNum), C.int(selected), 0)
}

func CuckooFinalize() {
	log.Println("finalize()")
	C.CuckooFinalize()
}


func CuckooSolve(hash []byte, hash_len int, nonce uint64, result []uint32, result_len *uint32, diff *byte, result_hash *byte) byte {
	_, ret := CuckooFindSolutions(hash, nonce);
	for i := 0; len(ret) > 0 && i < len(ret[0]); i++ {
		result[i] = ret[0][i]
	}
	*result_len = uint32(len(ret))
	return byte(len(ret))
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
		log.Println(fmt.Sprintf("WARNING: discard possible solutions, total sol num=%v, received number=%v", _numSols, uint32(len(result))/_solLength))
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
func CuckooVerify(hash *byte, hash_len int, nonce uint64, result []uint32, diff []byte, result_hash *byte) byte {
	var tmpret common.BlockSolution
	for i := 0; i < len(result); i++ {
		tmpret[i] = result[i]
	}
	var tmpdiff common.Hash
	for i := 0; diff != nil && i < len(diff); i++ {
		tmpdiff[i] = diff[i]
	}
	sha3hash := common.BytesToHash(crypto.Sha3Solution(&tmpret))
	if diff == nil || sha3hash.Big().Cmp(tmpdiff.Big()) <= 0{
		r := C.CuckooVerifyProof(
			(*C.uint8_t)(unsafe.Pointer(hash)),
			C.uint64_t(nonce),
			(*C.result_t)(unsafe.Pointer(&tmpret[0])),
			12,
			28)
		return byte(r)
	}
	return 0
}

func CuckooVerify_cuckaroo(hash *byte, hash_len int, nonce uint64, result []uint32, diff []byte, result_hash *byte) byte {
	var tmpret common.BlockSolution
	for i := 0; i < len(result); i++ {
		tmpret[i] = result[i]
	}
	var tmpdiff common.Hash
	for i := 0; diff != nil && i < len(diff); i++ {
		tmpdiff[i] = diff[i]
	}
	sha3hash := common.BytesToHash(crypto.Sha3Solution(&tmpret))
	if diff == nil || sha3hash.Big().Cmp(tmpdiff.Big()) <= 0{
		r := C.CuckooVerifyProof_cuckaroo(
			(*C.uint8_t)(unsafe.Pointer(hash)),
			C.uint64_t(nonce),
			(*C.result_t)(unsafe.Pointer(&tmpret[0])),
			12,
			28)
		return byte(r)
	}
	return 0
}
