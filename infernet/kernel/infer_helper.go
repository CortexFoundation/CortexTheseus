package kernel

import (
	"github.com/CortexFoundation/CortexTheseus/log"
	"plugin"
	"unsafe"
)

type Model struct {
	model unsafe.Pointer
	lib   *plugin.Plugin
	ops   uint64
	size  uint64
}

// Exactly copy from c_api.h:
//	status code of cvm executor
var (
	SUCCEED       = 0
	ERROR_LOGIC   = 1
	ERROR_RUNTIME = 2
)

func New(lib *plugin.Plugin, deviceId int, modelCfg, modelBin []byte) (*Model, int) {
	var (
		model  unsafe.Pointer
		status int
		size   uint64
		ops    uint64
	)

	if m, err := lib.Lookup("LoadModel"); err != nil {
		log.Error("infer helper", "LoadModel", "error", err)
		return nil, ERROR_RUNTIME
	} else {
		model, status = m.(func([]byte, []byte, int) (unsafe.Pointer, int))(modelCfg, modelBin, deviceId)
		if status != SUCCEED {
			return nil, status
		}
	}

	if m, err := lib.Lookup("GetStorageSize"); err != nil {
		log.Error("Error while get model size")
		return nil, ERROR_RUNTIME
	} else {
		size, status = m.(func(unsafe.Pointer) (uint64, int))(model)
		if status != SUCCEED {
			return nil, status
		}
	}
	if m, err := lib.Lookup("GetModelOpsFromModel"); err != nil {
		log.Error("infer helper", "GetModelOpsFromModel", "error", err)
		return nil, ERROR_RUNTIME
	} else {
		ops, status = m.(func(unsafe.Pointer) (uint64, int))(model)
		if status != SUCCEED {
			return nil, status
		}
	}
	return &Model{
		model: model,
		lib:   lib,
		ops:   ops,
		size:  size,
	}, SUCCEED
}

func (m *Model) Ops() uint64 {
	return m.ops
}

func (m *Model) Size() uint64 {
	return m.size
}

func (m *Model) GetInputLength() (uint64, int) {
	f, err := m.lib.Lookup("GetInputLength")
	if err != nil {
		log.Error("infer helper", "GetInputLength", "error", err)
		return 0, ERROR_RUNTIME
	}
	return f.(func(unsafe.Pointer) (uint64, int))(m.model)
}

func GetModelOps(lib *plugin.Plugin, file []byte) (uint64, int) {
	m, err := lib.Lookup("GetModelOps")
	if err != nil {
		log.Error("infer helper", "GetModelOps", "error", err)
		return 0, ERROR_RUNTIME
	}
	return m.(func([]byte) (uint64, int))(file)
}

func (m *Model) Free() int {
	f, err := m.lib.Lookup("FreeModel")
	if err != nil {
		log.Error("infer helper", "FreeModel", "error", err)
		return ERROR_RUNTIME
	}
	return f.(func(unsafe.Pointer) int)(m.model)
}

func (m *Model) Predict(data []byte) ([]byte, int) {
	f, err := m.lib.Lookup("Predict")
	if err != nil {
		log.Error("infer helper", "Predict", "error", err)
		return nil, ERROR_RUNTIME
	}
	return f.(func(unsafe.Pointer, []byte) ([]byte, int))(m.model, data)
}
