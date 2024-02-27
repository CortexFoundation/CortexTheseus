package plugins

/*
#cgo LDFLAGS: -L../../../solution -lcpuminer -lstdc++
#cgo CFLAGS: -I./

#include "../../../solution/verify.h"
*/
import "C"
import (
	"math/big"
	//	"fmt"
	//	"time"
	"unsafe"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	//"github.com/CortexFoundation/CortexTheseus/log"
)

//func CuckooInit(threads uint32) {
//	CuckooInitialize(0, "", "cuckaroo")
//}

//func CuckooInitialize(threads int, strDeviceIds string, algorithm string) error {
//	var deviceNum int = 1
//	var devices []uint32
//	var selected int = 1
//	devices = append(devices, 0)
//	C.CuckooInitializeCPU((*C.uint32_t)(unsafe.Pointer(&devices[0])), C.uint32_t(threads), C.int(selected), 0)
//	return nil
//}

//func CuckooFinalize() {
//	log.Debug("CuckooFinalize")
//	C.CuckooFinalizeCPU()
//}

//func CuckooVerify(hash *byte, nonce uint64, result types.BlockSolution, result_sha3 []byte, diff *big.Int) bool {
//	sha3hash := common.BytesToHash(result_sha3)
//	if sha3hash.Big().Cmp(diff) <= 0 {
//		r := C.CuckooVerifyProof(
//			(*C.uint8_t)(unsafe.Pointer(hash)),
//			C.uint64_t(nonce),
//			(*C.result_t)(unsafe.Pointer(&result[0])))
//		return (r == 1)
//	}
//	return false
//}

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
