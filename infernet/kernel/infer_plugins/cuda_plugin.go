package main

/*
#cgo LDFLAGS: -lm -pthread
#cgo LDFLAGS:  -L../../build/gpu -lcvm_runtime_cuda -lcudart -lcuda
#cgo LDFLAGS: -lstdc++

#cgo CFLAGS: -I../include -I/usr/local/cuda/include/ -O2

#cgo CFLAGS: -Wall -Wno-unused-result -Wno-unknown-pragmas -Wno-unused-variable

#include <cvm/c_api.h>
*/
import "C"
import (
	"errors"
	"unsafe"
	"fmt"
	"github.com/CortexFoundation/CortexTheseus/log"
	kernel "github.com/CortexFoundation/CortexTheseus/inference/synapse"
)


func LoadModel(modelCfg, modelBin []byte, deviceId int) (unsafe.Pointer, error) {
	fmt.Println("LoadModel\tisGPU:", 1, "DeviceId: ", deviceId)
	jptr := (*C.char)(unsafe.Pointer(&(modelCfg[0])))
	pptr := (*C.char)(unsafe.Pointer(&(modelBin[0])))
	j_len := C.int(len(modelCfg))
	p_len := C.int(len(modelBin))
	net := C.CVMAPILoadModel(jptr, j_len, pptr, p_len, 1, C.int(deviceId))

	if net == nil {
		return nil, errors.New("Model load error")
	}
	return net, nil
}

func GetModelOpsFromModel(net unsafe.Pointer) (int64, error) {
	ret := int64(C.CVMAPIGetGasFromModel(net))
	if ret < 0 {
		return 0, errors.New("Gas Error")
	} else {
		return ret, nil
	}
}

func GetModelOps(file []byte) (uint64, error) {
  ret := int64(C.CVMAPIGetGasFromGraphFile((*C.char)(unsafe.Pointer(&(file[0])))))
	if ret < 0 {
		return 0, errors.New("Gas Error")
	} else {
		return uint64(ret), nil
	}
}

func FreeModel(net unsafe.Pointer) {
	C.CVMAPIFreeModel(net)
}

func Predict(net unsafe.Pointer, data []byte) ([]byte, error) {
	if net == nil {
		return nil, errors.New("Internal error: network is null in InferProcess")
	}

	resLen := int(C.CVMAPIGetOutputLength(net))
	if resLen == 0 {
		return nil, errors.New("Model result len is 0")
	}
	input_bytes := C.CVMAPISizeOfInputType(net)
	data_aligned, data_aligned_err := kernel.ToAlignedData(data, int(input_bytes))
	if data_aligned_err != nil {
		return nil, data_aligned_err
	}
	input := (*C.char)(unsafe.Pointer(&data_aligned[0]))

	res := make([]byte, resLen)
	output := (*C.char)(unsafe.Pointer(&res[0]))
	output_bytes := C.CVMAPISizeOfOutputType(net)
	// TODO(tian) check input endian
  flag := C.CVMAPIInfer(net, input, output)
	if (output_bytes > 1) {
		fmt.Println("gpu_plugin", "output_bytes = ", output_bytes)
		var err error
		res, err = kernel.SwitchEndian(res, int(output_bytes))
		if err != nil {
			return nil, err
		}
	}
	log.Info("GPU Infernet", "flag", flag, "res", res)
	if flag != 0 {
		return nil, errors.New("Predict Error")
	}

	return res, nil
}

func GetStorageSize(net unsafe.Pointer) (int64, error) {
	if net == nil {
		return 0, errors.New("Internal error: network is null in InferProcess")
	}

	ret := int64(C.CVMAPIGetStorageSize(net))
	if ret == -1 {
		return 0, errors.New("Model size is 0")
	}

	return ret, nil
}

func GetInputLength(net unsafe.Pointer) (int, error) {
	if net == nil {
		return 0, errors.New("Internal error: network is null in InferProcess")
	}

	ret := int(C.CVMAPIGetInputLength(net))
	if ret == -1 {
		return 0, errors.New("Model result len is 0")
	}

	return ret, nil
}
