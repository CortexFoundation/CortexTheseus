package kernel

import (
//	"fmt"
//	"os"
//	"time"
	"errors"
	"unsafe"
//	"strings"
//	"strconv"
	"github.com/CortexFoundation/CortexTheseus/log"
<<<<<<< HEAD
	"plugin"
)

const PLUGIN_PATH string = "plugins/"
const PLUGIN_POST_FIX string = "_cvm.so"
var cvm_plugin  *plugin.Plugin = nil

func Init(cortex string)(*plugin.Plugin, error){
	if cvm_plugin == nil{
		so_path := PLUGIN_PATH + cortex + PLUGIN_POST_FIX
		cvm_plugin, err := plugin.Open(so_path)
		if err != nil{
			log.Error("infer helper", "init cvm plugin error", err)
			return nil, err
		}
		return cvm_plugin, nil
	}
	return cvm_plugin, nil
}

func LoadModel(modelCfg, modelBin string, deviceType string) (unsafe.Pointer, error) {
	cvm_plugin, err := Init(deviceType)
	m, err := cvm_plugin.Lookup("LoadModel")
	if err != nil{
		log.Error("infer helper", "LoadModel", "error", err)
		return nil, err
	}
	net, err := m.(func(string, string, uint32)(unsafe.Pointer, error))(modelCfg, modelBin, 0)
	if net == nil || err != nil {
		log.Error("infer helper", "LoadModel", "error", err)
=======
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
>>>>>>> origin/dev-zhen
		return nil, errors.New("Model load error")
	}

	cache.Add(modelCfg, model, storage_weight)

	return model, nil
}

func GetModelOpsFromModel(net unsafe.Pointer, deviceType string) (int64, error) {
	cvm_plugin, err := Init(deviceType)
	m, err := cvm_plugin.Lookup("GetModelOpsFromModel")
	if err != nil{
		log.Error("infer helper", "GetModelOpsFromModel", "error", err)
		return -1, err
	}
	ret,err := m.(func(unsafe.Pointer)(int64, error))(net)
	if ret < 0 {
		return 0, errors.New("Gas Error")
	} else {
		return ret, nil
	}
}

func GetModelOps(filepath string, deviceType string) (uint64, error) {
	cvm_plugin, err := Init(deviceType)
	m, err := cvm_plugin.Lookup("GetModelOps")
	if err != nil{
		log.Error("infer helper", "GetModelOps", "error", err)
		return 0, err
	}
	ret, err := m.(func(string)(int64, error))(filepath)
	if ret < 0 {
		return 0, errors.New("Gas Error")
	} else {
		return uint64(ret), nil
	}
}

func FreeModel(net unsafe.Pointer, deviceType string) {
	cvm_plugin, err := Init(deviceType)
	m, err := cvm_plugin.Lookup("FreeModel")
	if err != nil{
		log.Error("infer helper", "FreeModel", "error", err)
		return
	}
	m.(func(unsafe.Pointer)())(net)
}

func Predict(net unsafe.Pointer, imageData []byte, deviceType string) ([]byte, error) {
	cvm_plugin, err := Init(deviceType)
	if net == nil {
		return nil, errors.New("Internal error: network is null in InferProcess")
	}
	m, err := cvm_plugin.Lookup("Predict")
	if err != nil{
		log.Error("infer helper", "Predict", "error", err)
		return nil, err
	}
	res, err := m.(func(unsafe.Pointer, []byte)([]byte, error))(net, imageData)
	return res, err
}

<<<<<<< HEAD
func InferCore(modelCfg, modelBin string, imageData []byte, deviceType string, deviceId int) (ret []byte, err error) {
	cvm_plugin, err := Init(deviceType)
	m, err := cvm_plugin.Lookup("InferCore")
	if err != nil{
		log.Error("infer helper", "InferCore", "error", err)
		return nil, err
=======
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
>>>>>>> origin/dev-zhen
	}
	ret, err = m.(func(string, string, []byte, int)([]byte, error))(modelCfg, modelBin, imageData, deviceId)
	return ret, err
}
