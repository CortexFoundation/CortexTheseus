package kernel

import (
	"fmt"
	"errors"
	"unsafe"
	"github.com/CortexFoundation/CortexTheseus/log"
	"plugin"
)


type Model struct {
	model unsafe.Pointer
	lib *plugin.Plugin
	ops int64
	size int64
}

func New(lib *plugin.Plugin, deviceId int, modelCfg, modelBin []byte) *Model {
	var model unsafe.Pointer
	var size int64
	var ops int64

	if m, err := lib.Lookup("LoadModel"); err != nil {
		log.Error("infer helper", "LoadModel", "error", err)
		return nil
	} else {
		model, err = m.(func([]byte, []byte, int)(unsafe.Pointer, error))(modelCfg, modelBin, deviceId)
		if model == nil || err != nil {
			log.Error("infer helper", "LoadModel", "error", err)
			return nil
		}
	}

    if m, err := lib.Lookup("GetStorageSize"); err != nil {
		log.Error("Error while get model size")
		return nil
	} else {
		ret, err := m.(func(unsafe.Pointer)(int64, error))(model)
		if err != nil {
			return nil
		}
		size = ret
	}
	if m, err := lib.Lookup("GetModelOpsFromModel"); err != nil {
		log.Error("infer helper", "GetModelOpsFromModel", "error", err)
		return nil
	} else {
		ret, err := m.(func(unsafe.Pointer)(int64, error))(model)
		if err != nil || ret < 0 {
			log.Error("infer helper", "GetModelOpsFromModel", "error", err)
			return nil
		}
		ops = ret
	}
	return &Model{
		model: model,
		lib: lib,
		ops: ops,
		size: size,
	}
}

func (m *Model) Ops()(int64) {
	return m.ops;
}

func (m *Model) Size()(int64) {
	return m.size;
}

func (m *Model) GetInputLength() int {
	f, err := m.lib.Lookup("GetInputLength")

	if err != nil {
		log.Error("infer helper", "GetInputLength", "error", err)
		return -1
	}
	ret, err := f.(func(unsafe.Pointer)(int, error))(m.model)
	if ret < 0 {
		return -1
	} else {
		return int(ret)
	}
}

func (m *Model) GetOutputLength() int {
	f, err := m.lib.Lookup("GetOutputLength")

	if err != nil {
		log.Error("infer helper", "GetOutputLength", "error", err)
		return -1
	}
	ret, err := f.(func(unsafe.Pointer)(int, error))(m.model)
	if ret < 0 {
		return -1
	} else {
		return int(ret)
	}
}

func GetModelOps(lib *plugin.Plugin, filepath string) (uint64, error) {
	m, err := lib.Lookup("GetModelOps")
	if err != nil{
		log.Error("infer helper", "GetModelOps", "error", err)
		return 0, err
	}
	ret, err := m.(func(string)(uint64, error))(filepath)
	if ret < 0 {
		return 0, errors.New("Gas Error")
	} else {
		return uint64(ret), nil
	}
}

func (m *Model) Free() {
	f, err := m.lib.Lookup("FreeModel")
	if err != nil {
		log.Error("infer helper", "FreeModel", "error", err)
		return
	}
	f.(func(unsafe.Pointer)())(m.model)
}

func (m *Model) Predict(data []byte) ([]byte, error) {
	expectedInputLength := m.GetInputLength()
	if expectedInputLength > len(data) {
		return nil, errors.New(fmt.Sprintf("input size not match, Expected at least %d, Got %d",
																			 expectedInputLength, len(data)))
	}

	f, err := m.lib.Lookup("Predict")
	if err != nil {
		log.Error("infer helper", "Predict", "error", err)
		return nil, err
	}
	res, err := f.(func(unsafe.Pointer, []byte)([]byte, error))(m.model, data)
	return res, err
}
