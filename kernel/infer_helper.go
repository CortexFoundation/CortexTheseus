package kernel

import (
	"github.com/CortexFoundation/CortexTheseus/log"
	"plugin"
	"unsafe"
	"errors"
)

// Exactly copy from c_api.h:
//	status code of cvm executor
var (
	SUCCEED       = 0
	ERROR_LOGIC   = 1
	ERROR_RUNTIME = 2
	KERNEL_RUNTIME_ERROR error = errors.New("Kernel runtime error")
	KERNEL_LOGIC_ERROR error = errors.New("Kernel logic error")
)


type func_LoadModel func([]byte, []byte, int, int) (unsafe.Pointer, int)
type func_FreeModel func(unsafe.Pointer) int
type func_Inference func(unsafe.Pointer, []byte) ([]byte, int)
type func_GetVersion func(unsafe.Pointer) ([34]byte, int)
type func_GetPreprocessMethod func(unsafe.Pointer) ([34]byte, int)
type func_GetInputLength func(unsafe.Pointer) (uint64, int)
type func_GetOutputLength func(unsafe.Pointer) (uint64, int)
type func_GetInputTypeSize func(unsafe.Pointer) (uint64, int)
type func_GetOutputTypeSize func(unsafe.Pointer) (uint64, int)
type func_GetStorageSize func(unsafe.Pointer) (uint64, int)
type func_GetGasFromModel func(unsafe.Pointer) (uint64, int)
type func_GetGasFromGraphFile func([]byte) (uint64, int)

type Model struct {
	model unsafe.Pointer
	lib   *plugin.Plugin
	ops   uint64
	size  uint64

	input_size uint64
	input_byte uint64

	output_byte uint64
}

func lookUp(lib *plugin.Plugin, func_name string) interface{} {
	if f, err := lib.Lookup(func_name); err != nil {
		log.Error("lib cannot find function ", "name", func_name, "error", err)
		return nil
	} else {
		return f
	}
}

func New(lib *plugin.Plugin, device_type, deviceId int, modelCfg, modelBin []byte) (*Model, int) {
	var (
		model  *Model = &Model{lib: lib}
		status int
	)
	if func_ptr := lookUp(lib, "LoadModel"); func_ptr == nil {
		return nil, ERROR_RUNTIME
	} else if model.model, status = func_ptr.(func_LoadModel)(
		modelCfg, modelBin, device_type, deviceId); status != SUCCEED {
		return nil, status
	}

	if func_ptr := lookUp(lib, "GetStorageSize"); func_ptr == nil {
		return nil, ERROR_RUNTIME
	} else if model.size, status = func_ptr.(func_GetStorageSize)(model.model); status != SUCCEED {
		return nil, status
	}

	if func_ptr := lookUp(lib, "GetGasFromModel"); func_ptr == nil {
		return nil, ERROR_RUNTIME
	} else if model.ops, status = func_ptr.(func_GetGasFromModel)(model.model); status != SUCCEED {
		return nil, status
	}

	if func_ptr := lookUp(lib, "GetInputLength"); func_ptr == nil {
		return nil, ERROR_RUNTIME
	} else if model.input_size, status = func_ptr.(func_GetInputLength)(model.model); status != SUCCEED {
		return nil, status
	}

	if func_ptr := lookUp(lib, "GetInputTypeSize"); func_ptr == nil {
		return nil, ERROR_RUNTIME
	} else if model.input_byte, status = func_ptr.(func_GetInputTypeSize)(model.model); status != SUCCEED {
		return nil, status
	}

	if func_ptr := lookUp(lib, "GetOutputTypeSize"); func_ptr == nil {
		return nil, ERROR_RUNTIME
	} else if model.output_byte, status = func_ptr.(func_GetOutputTypeSize)(model.model); status != SUCCEED {
		return nil, status
	}

	return model, status
}

func (m *Model) Ops() uint64 {
	return m.ops
}

func (m *Model) Size() uint64 {
	return m.size
}

func (m *Model) GetInputLength() uint64 {
	return m.input_size
}

func GetModelGasFromGraphFile(lib *plugin.Plugin, file []byte) (uint64, int) {
	if func_ptr := lookUp(lib, "GetModelGasFromGraphFile"); func_ptr == nil {
		return 0, ERROR_RUNTIME
	} else {
		return func_ptr.(func_GetGasFromGraphFile)(file)
	}
}

func (m *Model) Free() int {
	if func_ptr := lookUp(m.lib, "FreeModel"); func_ptr == nil {
		return ERROR_RUNTIME
	} else {
		return func_ptr.(func_FreeModel)(m.model)
	}
}

func (m *Model) Predict(data []byte) ([]byte, int) {
	var (
		output   []byte
		status   int
		func_ptr interface{}
		err      error
	)
	if func_ptr = lookUp(m.lib, "Inference"); func_ptr == nil {
		return nil, ERROR_RUNTIME
	}
	if len(data) != int(m.input_size) {
		log.Debug("input length not matched")
		return nil, ERROR_LOGIC
	}
	if data, err = ToAlignedData(data, int(m.input_byte)); err != nil {
		return nil, ERROR_LOGIC
	}
	if output, status = func_ptr.(func_Inference)(m.model, data); status != SUCCEED {
		return nil, status
	}
	if m.output_byte > 1 {
		if output, err = SwitchEndian(output, int(m.output_byte)); err != nil {
			return nil, ERROR_LOGIC
		}
	}
	return output, status
}
