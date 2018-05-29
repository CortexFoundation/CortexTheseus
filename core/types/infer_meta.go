package types

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/ethereum/go-ethereum/common"
)

var (
	ErrorCodeLengthNotEnough = errors.New("Code length should be larger than 2")
	ErrorCodeTypeModelMeta   = errors.New("Model meta should start with 0x0001")
	ErrorCodeTypeInputMeta   = errors.New("Input meta should start with 0x0002")
)

//InferMeta include ModelMeta struct and InputMeta type
type InferMeta interface {
	TypeCode() []byte
	RawSize() uint64
	// Gas() uint64
	AuthorAddress() common.Address
}

type ModelMeta struct {
	typeCode []byte
	rawSize  uint64
	gas      uint64 `json:"gas"`
	author   common.Address
}

type InputMeta struct {
	typeCode []byte
	rawSize  uint64
	gas      uint64
	author   common.Address
}

func ParseModelMeta(code []byte) (*ModelMeta, error) {
	if len(code) < 2 {
		return nil, ErrorCodeLengthNotEnough
	}
	if !(code[0] == 0x0 && code[1] == 0x1) {
		return nil, ErrorCodeTypeModelMeta
	}
	var model_meta ModelMeta
	json.Unmarshal(code[2:], &model_meta)
	model_meta.typeCode = code[:2]
	model_meta.rawSize = uint64(len(code) - 2)
	model_meta.author = common.BytesToAddress([]byte{0x0})
	fmt.Println("ParseModelMeta", code, model_meta)
	return &model_meta, nil
}

func ParseInputMeta(code []byte) (*InputMeta, error) {
	if len(code) < 2 {
		return nil, ErrorCodeLengthNotEnough
	}
	if !(code[0] == 0x0 && code[1] == 0x2) {
		return nil, ErrorCodeTypeInputMeta
	}

	return &InputMeta{
		typeCode: code[:2],
		rawSize:  uint64(len(code) - 2),
		gas:      123,
		author:   common.BytesToAddress([]byte{0x1}),
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

func (mm *ModelMeta) AuthorAddress() common.Address {
	return mm.author
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

func (im *InputMeta) AuthorAddress() common.Address {
	return im.author
}
