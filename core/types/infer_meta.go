package types

import "errors"

var (
	ErrorCodeLengthNotEnough = errors.New("Code length should be larger than 2")
	ErrorCodeTypeModelMeta   = errors.New("Model meta should start with 0x0001")
	ErrorCodeTypeInputMeta   = errors.New("Input meta should start with 0x0002")
)

//InferMeta include ModelMeta struct and InputMeta type
type InferMeta interface {
	TypeCode() []byte
	RawSize() uint64
	Gas() uint64
}

type ModelMeta struct {
	typeCode []byte
	rawSize  uint64
	gas      uint64
}

type InputMeta struct {
	typeCode []byte
	rawSize  uint64
	gas      uint64
}

func ParseModelMeta(code []byte) (*ModelMeta, error) {
	if len(code) < 2 {
		return nil, ErrorCodeLengthNotEnough
	}
	if code[0] != 0x0 || code[1] != 0x1 {
		return nil, ErrorCodeTypeModelMeta
	}
	return &ModelMeta{
		typeCode: code[:2],
		rawSize:  uint64(len(code) - 2),
		gas:      123,
	}, nil
}
func ParseInputMeta(code []byte) (*InputMeta, error) {
	if len(code) < 2 {
		return nil, ErrorCodeLengthNotEnough
	}
	if code[0] != 0x0 || code[1] != 0x2 {
		return nil, ErrorCodeTypeInputMeta
	}

	return &InputMeta{
		typeCode: code[:2],
		rawSize:  uint64(len(code) - 2),
		gas:      123,
	}, nil
}

func (mm *ModelMeta) TypeCode() []byte {
	return mm.typeCode
}
func (mm *ModelMeta) RawSize() uint64 {
	return mm.rawSize
}
func (mm *ModelMeta) Gas() uint64 {
	return mm.gas
}
func (im *InputMeta) TypeCode() []byte {
	return im.typeCode
}
func (im *InputMeta) RawSize() uint64 {
	return im.rawSize
}
func (im *InputMeta) Gas() uint64 {
	return im.gas
}
