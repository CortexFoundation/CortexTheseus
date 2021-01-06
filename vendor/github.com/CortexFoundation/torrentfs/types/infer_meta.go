package types

import (
	"encoding/json"
	"errors"
	"math/big"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/rlp"
)

var (
	ErrorCodeTypeMeta      = errors.New("meta should start with 0x0001 or 0x0002")
	ErrorCodeTypeModelMeta = errors.New("model meta should start with 0x0001")
	ErrorCodeTypeInputMeta = errors.New("input meta should start with 0x0002")
	ErrorDecodeModelMeta   = errors.New("model meta decode error")
	ErrorDecodeInputMeta   = errors.New("input meta decode error")
	ErrorNotMature         = errors.New("not mature")
	ErrorExpired           = errors.New("meta Expired")
	ErrorInvalidBlockNum   = errors.New("invalid block number")
)

type InferMeta interface {
	TypeCode() []byte
	RawSize() uint64
	// Gas() uint64
	AuthorAddress() common.Address
	InfoHash() string
	Comment() string
}

//go:generate gencodec -type Meta -out gen_meta_json.go
type Meta struct {
	Comment  string         `json:"comment"`
	Hash     common.Address `json:"hash"`
	RawSize  uint64         `json:"rawSize"`
	Shape    []uint64       `json:"shape"`
	BlockNum big.Int        `json:"blockNum"`
}

//go:generate gencodec -type ModelMeta -out gen_model_json.go

type ModelMeta struct {
	Comment       string         `json:"comment"`
	Hash          common.Address `json:"hash"`
	RawSize       uint64         `json:"rawSize"`
	InputShape    []uint64       `json:"inputShape"`
	OutputShape   []uint64       `json:"outputShape"`
	Gas           uint64         `json:"gas"`
	AuthorAddress common.Address `json:"authorAddress"`
	BlockNum      big.Int        `json:"blockNum"`

	//RawBytes []byte `json:"RawBytes"`
}

//go:generate gencodec -type InputMeta -out gen_input_json.go
type InputMeta struct {
	Comment string         `json:"comment"`
	Hash    common.Address `json:"hash"`
	RawSize uint64         `json:"rawSize"`
	Shape   []uint64       `json:"shape"`
	//AuthorAddress common.Address `json:"AuthorAddress"`
	BlockNum big.Int `json:"blockNum"`

	//RawBytes []byte `json:"RawBytes"`
}

func (mm *ModelMeta) InfoHash() string {
	ih := mm.Hash.String()[2:]
	return ih
}

func (mm *InputMeta) InfoHash() string {
	ih := mm.Hash.String()[2:]
	return ih
}

func (mm *Meta) InfoHash() string {
	ih := mm.Hash.String()[2:]
	return ih
}

func (mm *ModelMeta) SetBlockNum(num big.Int) error {
	mm.BlockNum = num
	return nil
}

func (mm *ModelMeta) SetGas(gas uint64) error {
	mm.Gas = gas
	return nil
}

/*func (im *InputMeta) SetRawBytes(rawBytes []byte) error {
	im.RawBytes = rawBytes
	return nil
}*/

func (im *InputMeta) SetBlockNum(num big.Int) error {
	im.BlockNum = num
	return nil
}

func (mm *ModelMeta) EncodeJSON() (string, error) {
	data, err := json.Marshal(mm)
	return string(data), err
}

func (mm *ModelMeta) DecodeJSON(s string) error {
	err := json.Unmarshal([]byte(s), mm)
	return err
}
func (im *InputMeta) EncodeJSON() (string, error) {
	data, err := json.Marshal(im)
	return string(data), err
}
func (im *InputMeta) DecodeJSON(s string) error {
	err := json.Unmarshal([]byte(s), im)
	return err
}

func (mm *ModelMeta) ToBytes() ([]byte, error) {
	if array, err := rlp.EncodeToBytes(mm); err != nil {
		return nil, err
	} else {
		return array, nil
	}
}

func (im *InputMeta) ToBytes() ([]byte, error) {
	if array, err := rlp.EncodeToBytes(im); err != nil {
		return nil, err
	} else {
		return array, nil
	}
}

func ParseModelMeta(code []byte) (*ModelMeta, error) {
	if len(code) < 2 {
		return nil, ErrorCodeTypeModelMeta
	}
	if !(code[0] == 0x0 && code[1] == 0x1) {
		return nil, ErrorCodeTypeModelMeta
	}
	var modelMeta ModelMeta
	err := rlp.DecodeBytes(code[2:], &modelMeta)
	if err != nil {
		return nil, err
	}
	return &modelMeta, nil
}

func (mm *ModelMeta) DecodeRLP(code []byte) error {
	if len(code) < 2 {
		return ErrorCodeTypeModelMeta
	}
	if !(code[0] == 0x0 && code[1] == 0x1) {
		return ErrorCodeTypeModelMeta
	}
	err := rlp.DecodeBytes(code[2:], mm)
	if err != nil {
		return err
	}
	return nil
}

func (im *InputMeta) DecodeRLP(code []byte) error {
	if len(code) < 2 {
		return ErrorCodeTypeInputMeta
	}
	if !(code[0] == 0x0 && code[1] == 0x2) {
		return ErrorCodeTypeInputMeta
	}
	err := rlp.DecodeBytes(code[2:], im)
	if err != nil {
		return err
	}
	return nil
}

func ParseInputMeta(code []byte) (*InputMeta, error) {
	if len(code) < 2 {
		return nil, ErrorCodeTypeInputMeta
	}
	if !(code[0] == 0x0 && code[1] == 0x2) {
		return nil, ErrorCodeTypeInputMeta
	}
	var inputMeta InputMeta
	err := rlp.DecodeBytes(code[2:], &inputMeta)
	if err != nil {
		return nil, err
	}

	return &inputMeta, nil
}

func ParseMeta(code []byte) (*Meta, error) {
	if len(code) < 2 {
		return nil, ErrorCodeTypeMeta
	}
	if !(code[0] == 0x0 && (code[1] == 0x1 || code[1] == 0x2)) {
		return nil, ErrorCodeTypeMeta
	}
	var meta Meta
	err := rlp.DecodeBytes(code[2:], &meta)
	if err != nil {
		return nil, err
	}
	return &meta, nil
}
