package main

/*
#cgo LDFLAGS: -lm -pthread
#cgo LDFLAGS: -L../../build/cpu -lcvm_runtime_cpu
#cgo LDFLAGS: -lstdc++ 

#cgo CFLAGS: -I../include -O2

#cgo CFLAGS: -Wall -Wno-unused-result -Wno-unknown-pragmas -Wno-unused-variable

#include <cvm/c_api.h>
*/
import "C"
import (
//  "os"
//  "time"
  "errors"
  "fmt"
  "unsafe"
//  "strings"
//  "strconv"
  "github.com/CortexFoundation/CortexTheseus/log"
  "github.com/CortexFoundation/CortexTheseus/inference/synapse/kernel"
)

func LoadModel(modelCfg, modelBin string,  deviceId int) (unsafe.Pointer, error) {
  net := C.CVMAPILoadModel(
    C.CString(modelCfg),
    C.CString(modelBin),
    0,
    0,
  )

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

func GetModelOps(filepath string) (uint64, error) {
  ret := int64(C.CVMAPIGetGasFromGraphFile(C.CString(filepath)))
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
  fmt.Println("CPU Infernet", "resLen = ", resLen)
  if resLen == 0 {
    return nil, errors.New("Model result len is 0")
  }

  res := make([]byte, resLen)
	input := (*C.char)(unsafe.Pointer(&data[0]))
	output := (*C.char)(unsafe.Pointer(&res[0]))
	input_bytes := C.CVMAPISizeOfInputType(net)
	output_bytes := C.CVMAPISizeOfOutputType(net)
	// TODO(tian) check input endian
  flag := C.CVMAPIInfer(net, input, output)
	if (input_bytes > 1) {
		fmt.Println("cpu_plugin", "input_bytes = ", input_bytes)
	}
	if (output_bytes > 1) {
		fmt.Println("cpu_plugin", "output_bytes = ", output_bytes)
		var err error
		res, err = kernel.SwitchEndian(res, int(output_bytes))
		if err != nil {
    return nil, err 
		}
	}
  log.Info("CPU Infernet", "flag", flag, "res", res)
  if flag != 0 {
    return nil, errors.New("Predict Error")
  }
  return res, nil
}

func GetStorageSize(net unsafe.Pointer)(int64, error) {
  if net == nil {
    return 0, errors.New("Internal error: network is null in InferProcess")
  }

  ret := int64(C.CVMAPIGetStorageSize(net))
  if ret == -1 {
    return 0, errors.New("Model size is 0")
  }

  return ret, nil
}

func GetInputLength(net unsafe.Pointer)(int, error) {
  if net == nil {
    return 0, errors.New("Internal error: network is null in InferProcess")
  }

  ret := int(C.CVMAPIGetInputLength(net))
  if ret == -1 {
    return 0, errors.New("Model result len is 0")
  }

  return ret, nil
}
