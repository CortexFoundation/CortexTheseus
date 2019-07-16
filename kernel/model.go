package kernel

import (
	"github.com/CortexFoundation/CortexTheseus/log"
	"unsafe"
)

type Model struct {
	model unsafe.Pointer
	lib   *LibCVM
	ops   uint64
	size  uint64

	input_size uint64
	input_byte uint64

	output_byte uint64
}

func New(lib *LibCVM, modelCfg, modelBin []byte, deviceType, deviceId int) (*Model, int) {
	var (
		model  *Model = &Model{lib: lib}
		status int
	)

	if model.model, status = lib.LoadModel(modelCfg, modelBin,
		deviceType, deviceId); status != SUCCEED {
		return nil, status
	}

	if model.size, status = lib.GetStorageSize(model.model); status != SUCCEED {
		return nil, status
	}
	if model.ops, status = lib.GetGasFromModel(model.model); status != SUCCEED {
		return nil, status
	}
	if model.input_size, status = lib.GetInputLength(model.model); status != SUCCEED {
		return nil, status
	}
	if model.input_byte, status = lib.GetInputTypeSize(model.model); status != SUCCEED {
		return nil, status
	}
	if model.output_byte, status = lib.GetOutputTypeSize(model.model); status != SUCCEED {
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

func (m *Model) Predict(data []byte) ([]byte, int) {
	var (
		output []byte
		status int
		err    error
	)
	if len(data) != int(m.input_size) {
		log.Warn("input length not matched",
			"input length", len(data), "expected", m.input_size)
		return nil, ERROR_LOGIC
	}
	if data, err = ToAlignedData(data, int(m.input_byte)); err != nil {
		log.Warn("input ToAlignedData invalid", "error", err)
		return nil, ERROR_LOGIC
	}
	if output, status = m.lib.Inference(m.model, data); status != SUCCEED {
		return nil, status
	}
	if m.output_byte > 1 {
		if output, err = SwitchEndian(output, int(m.output_byte)); err != nil {
			log.Warn("output SwitchEndian invalid", "error", err)
			return nil, ERROR_LOGIC
		}
	}
	return output, status
}

func (m *Model) Free() int {
	return m.lib.FreeModel(m.model)
}

func GetModelGasFromGraphFile(lib *LibCVM, json []byte) (gas uint64, status int) {
	return lib.GetGasFromGraphFile(json)
}
