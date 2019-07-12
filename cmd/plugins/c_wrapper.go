package main

/*
#cgo LDFLAGS: -lm -pthread
#cgo gpu LDFLAGS:  -L../../infernet/build/gpu -lcvm_runtime_cuda -lcudart -lcuda
#cgo !gpu LDFLAGS: -L../../infernet/build/cpu -lcvm_runtime_cpu
#cgo LDFLAGS: -ldl -lstdc++

#cgo CFLAGS: -I../../infernet/include -O2
#cgo CFLAGS: -Wall -Wno-unused-result -Wno-unknown-pragmas -Wno-unused-variable

#include <cvm/c_api.h>
*/
import "C"
import (
	"github.com/CortexFoundation/CortexTheseus/infernet/kernel"
	"unsafe"
)

func LoadModel(modelCfg, modelBin []byte, device_type, deviceId int) (unsafe.Pointer, int) {
	jptr := (*C.char)(unsafe.Pointer(&(modelCfg[0])))
	pptr := (*C.char)(unsafe.Pointer(&(modelBin[0])))
	j_len := C.int(len(modelCfg))
	p_len := C.int(len(modelBin))
	var net unsafe.Pointer
	status := int(C.CVMAPILoadModel(jptr, j_len,
		pptr, p_len,
		&net,
		C.int(device_type), C.int(deviceId)))
	return net, status
}
func FreeModel(net unsafe.Pointer) int {
	return int(C.CVMAPIFreeModel(net))
}
func Inference(net unsafe.Pointer, input []byte) ([]byte, int) {
	out_size, status := GetOutputLength(net)
	if status != kernel.SUCCEED {
		return nil, status
	}

	in_size := C.int(len(input))
	data := (*C.char)(unsafe.Pointer(&input[0]))
	tmp := make([]byte, out_size)
	output := (*C.char)(unsafe.Pointer(&tmp[0]))

	status = int(C.CVMAPIInference(net, data, in_size, output))
	if status != kernel.SUCCEED {
		return nil, status
	}
	return tmp, status
}

// version and method string length not greater than 32
func GetVersion(net unsafe.Pointer) ([34]byte, int) {
	var version [34]byte
	sptr := (*C.char)(unsafe.Pointer(&version[0]))
	status := int(C.CVMAPIGetVersion(net, sptr))
	return version, status
}
func GetPreprocessMethod(net unsafe.Pointer) ([34]byte, int) {
	var method [34]byte
	sptr := (*C.char)(unsafe.Pointer(&method[0]))
	status := int(C.CVMAPIGetPreprocessMethod(net, sptr))
	return method, status
}

func GetInputLength(net unsafe.Pointer) (uint64, int) {
	var tmp C.ulonglong
	status := int(C.CVMAPIGetInputLength(net, &tmp))
	return uint64(tmp), status
}
func GetOutputLength(net unsafe.Pointer) (uint64, int) {
	var tmp C.ulonglong
	status := int(C.CVMAPIGetOutputLength(net, &tmp))
	return uint64(tmp), status
}
func GetInputTypeSize(net unsafe.Pointer) (uint64, int) {
	var tmp C.ulonglong
	status := int(C.CVMAPIGetInputTypeSize(net, &tmp))
	return uint64(tmp), status
}
func GetOutputTypeSize(net unsafe.Pointer) (uint64, int) {
	var tmp C.ulonglong
	status := int(C.CVMAPIGetOutputTypeSize(net, &tmp))
	return uint64(tmp), status
}

func GetStorageSize(net unsafe.Pointer) (uint64, int) {
	var tmp C.ulonglong
	status := int(C.CVMAPIGetStorageSize(net, &tmp))
	return uint64(tmp), status
}
func GetGasFromModel(net unsafe.Pointer) (uint64, int) {
	var tmp C.ulonglong
	status := int(C.CVMAPIGetGasFromModel(net, &tmp))
	return uint64(tmp), status
}
func GetGasFromGraphFile(json []byte) (uint64, int) {
	jptr := (*C.char)(unsafe.Pointer(&json[0]))
	var tmp C.ulonglong
	status := int(C.CVMAPIGetGasFromGraphFile(jptr, &tmp))
	return uint64(tmp), status
}
