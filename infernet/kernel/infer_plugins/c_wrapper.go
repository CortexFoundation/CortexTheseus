package main

/*
#cgo LDFLAGS: -lm -pthread
#cgo gpu LDFLAGS:  -L../../build/gpu -lcvm_runtime_cuda -lcudart -lcuda
#cgo !gpu LDFLAGS: -L../../build/cpu -lcvm_runtime_cpu
#cgo LDFLAGS: -ldl -lstdc++

#cgo CFLAGS: -I../include -O2
#cgo CFLAGS: -Wall -Wno-unused-result -Wno-unknown-pragmas -Wno-unused-variable

#include <cvm/c_api.h>
*/
import "C"
import (
	"github.com/CortexFoundation/CortexTheseus/inference/synapse"
	"github.com/CortexFoundation/CortexTheseus/log"
	"unsafe"
)

func StatusCheck(status C.enum_CVMStatus) error {
	switch int(status) {
	case int(C.SUCCEED):
		return nil
	case int(C.ERROR_LOGIC):
		return synapse.ErrLogic
	case int(C.ERROR_RUNTIME):
		return synapse.ErrRuntime
	}

	// status should not go here.
	return synapse.ErrRuntime
}

func LoadModel(modelCfg, modelBin []byte, deviceId int) (unsafe.Pointer, error) {
	jptr := (*C.char)(unsafe.Pointer(&(modelCfg[0])))
	pptr := (*C.char)(unsafe.Pointer(&(modelBin[0])))
	j_len := C.int(len(modelCfg))
	p_len := C.int(len(modelBin))
	var net C.ModelHandler
	status := C.CVMAPILoadModel(jptr, j_len, pptr, p_len, &net, 0, C.int(deviceId))
	if err := StatusCheck(status); err != nil {
		return nil, err
	}
	return unsafe.Pointer(net), nil
}

func GetModelOpsFromModel(net unsafe.Pointer) (uint64, error) {
	var gas C.ulonglong
	status := C.CVMAPIGetGasFromModel(C.ModelHandler(net), &gas)
	if err := StatusCheck(status); err != nil {
		return 0, err
	}
	return uint64(gas), nil
}

func GetModelOps(file []byte) (uint64, error) {
	var gas C.ulonglong
	fptr := (*C.char)(unsafe.Pointer(&file[0]))
	status := C.CVMAPIGetGasFromGraphFile(fptr, &gas)
	if err := StatusCheck(status); err != nil {
		return 0, err
	}
	return uint64(gas), nil
}

func FreeModel(net unsafe.Pointer) error {
	status := C.CVMAPIFreeModel(C.ModelHandler(net))
	if err := StatusCheck(status); err != nil {
		return err
	}
	return nil
}

func Predict(net unsafe.Pointer, data []byte) ([]byte, error) {
	var (
		resLen       C.ulonglong
		input_bytes  C.ulonglong
		output_bytes C.ulonglong
		status       C.enum_CVMStatus
	)

	status = C.CVMAPIGetOutputLength(C.ModelHandler(net), &resLen)
	if err := StatusCheck(status); err != nil {
		return nil, err
	}
	status = C.CVMAPIGetInputTypeSize(C.ModelHandler(net), &input_bytes)
	if err := StatusCheck(status); err != nil {
		return nil, err
	}
	status = C.CVMAPIGetOutputTypeSize(C.ModelHandler(net), &output_bytes)
	if err := StatusCheck(status); err != nil {
		return nil, err
	}

	data_aligned, data_aligned_err := synapse.ToAlignedData(data, int(input_bytes))
	if data_aligned_err != nil {
		return nil, synapse.ErrLogic
	}

	res := make([]byte, uint64(resLen))
	input := (*C.char)(unsafe.Pointer(&data_aligned[0]))
	output := (*C.char)(unsafe.Pointer(&res[0]))

	status = C.CVMAPIInference(C.ModelHandler(net), input, output)
	if err := StatusCheck(status); err != nil {
		return nil, err
	}

	if uint64(output_bytes) > 1 {
		var err error
		res, err = synapse.SwitchEndian(res, int(output_bytes))
		if err != nil {
			return nil, synapse.ErrLogic
		}
	}
	log.Info("CPU Inference succeed", "res", res)
	return res, nil
}

func GetStorageSize(net unsafe.Pointer) (uint64, error) {
	var size C.ulonglong
	status := C.CVMAPIGetStorageSize(C.ModelHandler(net), &size)
	if err := StatusCheck(status); err != nil {
		return 0, err
	}
	return uint64(size), nil
}

func GetInputLength(net unsafe.Pointer) (uint64, error) {
	var size C.ulonglong
	status := C.CVMAPIGetInputLength(C.ModelHandler(net), &size)
	if err := StatusCheck(status); err != nil {
		return 0, err
	}
	return uint64(size), nil
}
