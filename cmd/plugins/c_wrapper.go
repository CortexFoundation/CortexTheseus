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
	"github.com/CortexFoundation/CortexTheseus/log"
	"unsafe"
)

func LoadModel(modelCfg, modelBin []byte, deviceId int) (unsafe.Pointer, int) {
	jptr := (*C.char)(unsafe.Pointer(&(modelCfg[0])))
	pptr := (*C.char)(unsafe.Pointer(&(modelBin[0])))
	j_len := C.int(len(modelCfg))
	p_len := C.int(len(modelBin))
	var net C.ModelHandler = nil
	status := C.CVMAPILoadModel(jptr, j_len, pptr, p_len, &net, 0, C.int(deviceId))
	return unsafe.Pointer(net), int(status)
}

func GetModelOpsFromModel(net unsafe.Pointer) (uint64, int) {
	var gas C.ulonglong
	status := C.CVMAPIGetGasFromModel(C.ModelHandler(net), &gas)
	return uint64(gas), int(status)
}

func GetModelOps(file []byte) (uint64, int) {
	var gas C.ulonglong
	fptr := (*C.char)(unsafe.Pointer(&file[0]))
	status := C.CVMAPIGetGasFromGraphFile(fptr, &gas)
	return uint64(gas), int(status)
}

func FreeModel(net unsafe.Pointer) int {
	status := C.CVMAPIFreeModel(C.ModelHandler(net))
	return int(status)
}

func Predict(net unsafe.Pointer, data []byte) ([]byte, int) {
	var (
		resLen       C.ulonglong
		input_bytes  C.ulonglong
		output_bytes C.ulonglong
		status       C.enum_CVMStatus
	)

	status = C.CVMAPIGetOutputLength(C.ModelHandler(net), &resLen)
	if status != C.SUCCEED {
		return nil, int(status)
	}
	status = C.CVMAPIGetInputTypeSize(C.ModelHandler(net), &input_bytes)
	if status != C.SUCCEED {
		return nil, int(status)
	}
	status = C.CVMAPIGetOutputTypeSize(C.ModelHandler(net), &output_bytes)
	if status != C.SUCCEED {
		return nil, int(status)
	}

	data_aligned, data_aligned_err := kernel.ToAlignedData(data, int(input_bytes))
	if data_aligned_err != nil {
		return nil, int(C.ERROR_LOGIC)
	}

	res := make([]byte, uint64(resLen))
	input := (*C.char)(unsafe.Pointer(&data_aligned[0]))
	output := (*C.char)(unsafe.Pointer(&res[0]))

	status = C.CVMAPIInference(C.ModelHandler(net),
		input, C.int(len(data_aligned)), output)
	if status != C.SUCCEED {
		return nil, int(status)
	}

	if uint64(output_bytes) > 1 {
		var err error
		res, err = kernel.SwitchEndian(res, int(output_bytes))
		if err != nil {
			return nil, int(C.ERROR_LOGIC)
		}
	}
	log.Info("CPU Inference succeed", "res", res)
	return res, int(C.SUCCEED)
}

func GetStorageSize(net unsafe.Pointer) (uint64, int) {
	var size C.ulonglong
	status := C.CVMAPIGetStorageSize(C.ModelHandler(net), &size)
	return uint64(size), int(status)
}

func GetInputLength(net unsafe.Pointer) (uint64, int) {
	var size C.ulonglong
	status := C.CVMAPIGetInputLength(C.ModelHandler(net), &size)
	return uint64(size), int(status)
}
