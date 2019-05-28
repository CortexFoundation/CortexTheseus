package kernel

// #cgo CFLAGS: -DDEBUG

/*
#cgo LDFLAGS: -lm -pthread
#cgo LDFLAGS: -L../../../infernet/build/ -lcvm_runtime -lcudart -lcuda
#cgo LDFLAGS: -lstdc++ 

#cgo CFLAGS: -I./include -I/usr/local/cuda/include/ -O2

#cgo CFLAGS: -Wall -Wno-unused-result -Wno-unknown-pragmas -Wno-unused-variable

#include <cvm/c_api.h>
*/
import "C"
import (
	"fmt"
//	"os"
//	"time"
	"errors"
	"unsafe"
//	"strings"
//	"strconv"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/common/lru"
)

var LRU map[int]*lru.Cache

func LoadModel(modelCfg, modelBin string) (unsafe.Pointer, error) {
	if LRU == nil {
		LRU = make(map[int]*lru.Cache)
	}
	if cache, ok := LRU[0]; !ok {
		cache = lru.New(4000000)
		LRU[0] = cache
		cache.OnEvicted = func(key lru.Key, value interface{}) {
			FreeModel(value.(unsafe.Pointer))
		}
	}
	cache, _ := LRU[0]

	if model, ok := cache.Get(modelCfg); ok {
		if model == nil {
			return nil, errors.New("Model error")
		} else {
			return model.(unsafe.Pointer), nil
		}
	}

	model := C.CVMAPILoadModel(
		C.CString(modelCfg),
		C.CString(modelBin),
		1, 1,
	)
	storage_size := C.CVMAPIGetStorageSize(model)
	storage_weight := int64(storage_size) / 1000
	log.Info("Model loaded", "memory", storage_size)

	if model == nil {
		return nil, errors.New("Model load error")
	}

	cache.Add(modelCfg, model, storage_weight)

	return model, nil
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

func Predict(net unsafe.Pointer, imageData []byte) ([]byte, error) {
	if net == nil {
		return nil, errors.New("Internal error: network is null in InferProcess")
	}

	resLen := int(C.CVMAPIGetOutputLength(net))
	if resLen == 0 {
		return nil, errors.New("Model result len is 0")
	}

	res := make([]byte, resLen)

	flag := C.CVMAPIInfer(
		net,
		(*C.char)(unsafe.Pointer(&imageData[0])),
		(*C.char)(unsafe.Pointer(&res[0])))
	log.Info("Infernet", "flag", flag, "res", res)
	if flag != 0 {
		return nil, errors.New("Predict Error")
	}

	return res, nil
}

func InferCore(modelCfg, modelBin string, imageData []byte) (ret []byte, err error) {
	net, loadErr := LoadModel(modelCfg, modelBin)
	// defer FreeModel(net)
	if loadErr != nil {
		return nil, errors.New("Model load error")
	}
	expectedInputSize := int(C.CVMAPIGetInputLength(net))
	if expectedInputSize != len(imageData) {
		return nil, errors.New(fmt.Sprintf("input size not match, Expected: %d, Have %d",
																		  expectedInputSize, len(imageData)))
	}
	ret, err = Predict(net, imageData)
	return ret, err
}
