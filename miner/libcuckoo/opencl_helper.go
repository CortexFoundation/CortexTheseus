// +build opencl

package libcuckoo

/*
#cgo LDFLAGS: -L./ -lopenclminer -L/usr/local/cuda-10.0/lib64 -lOpenCL -lstdc++
//#cgo LDFLAGS: -L./ -lopenclminer -L/usr/local/cuda-10.0/lib64 -lOpenCL -lstdc++ -lrocm_smi64
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

func FindSolutionsByGPU(hash []byte, nonce uint64, threadId uint32) (nedges uint32) {
	var tmpHash = make([]byte, 32)
	copy(tmpHash[:], hash)

	//	start := time.Now()
	r := C.FindSolutionsByGPU(
		(*C.uint8_t)(unsafe.Pointer(&tmpHash[0])),
		C.uint64_t(nonce),
		C.uint32_t(threadId))

	//	duration := time.Since(start)
	//	log.Println(fmt.Sprintf("CuckooFindSolutionCuda | time=%v, status code=%v", duration, _numSols))

	// TODO add warning of discarding possible solutions

	return uint32(r)
}

func FindCycles(threadId uint32, nedges uint32) (status_code uint32, ret [][]uint32) {
	var (
		_solLength uint32
		_numSols   uint32
		result     [128]uint32
	)
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
	//	 log.Println(fmt.Sprintf("Index: %v, Solution: %v", solIdx, sol))
		ret = append(ret, sol)
	}

	return uint32(r), ret
}
func CuckooInitialize(devices []uint32, deviceNum uint32, selected int) {
	C.CuckooInitialize((*C.uint32_t)(unsafe.Pointer(&devices[0])), C.uint32_t(deviceNum), C.int(selected))
}

func Monitor(device_count uint32) (fanSpeeds []uint32, temperatures []uint32) {
	var (
		_fanSpeeds    [128]uint32
		_temperatures [128]uint32
	)
	C.monitor(C.uint32_t(device_count), (*C.uint32_t)(unsafe.Pointer(&_fanSpeeds[0])), (*C.uint32_t)(unsafe.Pointer(&_temperatures[0])))
	for i := 0; i < int(device_count); i++ {
		fanSpeeds = append(fanSpeeds, _fanSpeeds[i])
		temperatures = append(temperatures, _temperatures[i])
	}
	return fanSpeeds, temperatures
}
func RunSolverOnCPU(hash []byte, nonce uint64) (status_code uint32, ret [][]uint32){
	return 0,ret
}
