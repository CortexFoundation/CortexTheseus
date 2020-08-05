package inference

import (
	"encoding/json"

	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
)

// infer send types
type InferType uint32

const (
	INFER_UNKNOWN  = InferType(0)
	INFER_BY_IH    = InferType(1) // Infer By Input Hash
	INFER_BY_IC    = InferType(2) // Infer By Input Content
	GAS_BY_H       = InferType(3) // Gas By Model Hash
	AVAILABLE_BY_H = InferType(4) // Available by info hash
)

// Infer by input info hash
//go:generate gencodec -type IHWork -out gen_ih_json.go
type IHWork struct {
	Type         InferType `json:"type"  gencodec:"required"`
	Model        string    `json:"model" gencodec:"required"`
	Input        string    `json:"input" gencodec:"required"`
	ModelSize    uint64    `json:"modelSize"`
	InputSize    uint64    `json:"inputSize"`
	CvmVersion   int       `json:"cvm_version"`
	CvmNetworkId int64     `json:"cvm_networkid"`
}

// Infer by input content
//go:generate gencodec -type ICWork -out gen_ic_json.go
type ICWork struct {
	Type         InferType     `json:"type" gencodec:"required"`
	Model        string        `json:"model" gencodec:"required"`
	Input        hexutil.Bytes `json:"input" gencodec:"required"`
	ModelSize    uint64        `json:"modelSize"`
	CvmVersion   int           `json:"cvm_version"`
	CvmNetworkId int64         `json:"cvm_networkid"`
}

// Infer gas
//go:generate gencodec -type GasWork -out gen_gas_json.go
type GasWork struct {
	Type         InferType `json:"type" gencodec:"required"`
	Model        string    `json:"model" gencodec:"required"`
	ModelSize    uint64    `json:"modelSize"`
	CvmNetworkId int64     `json:"cvm_networkid"`
}

// check Available
//go:generate gencodec -type AvailableWork -out gen_avaiable_json.go
type AvailableWork struct {
	Type         InferType `json:"type" gencodec:"required"`
	InfoHash     string    `json:"infohash" gencodec:"required"`
	RawSize      uint64    `json:"rawSize" gencodec:"required"`
	CvmNetworkId int64     `json:"cvm_networkid"`
}

//go:generate gencodec -type Work -out gen_work_json.go
type Work struct {
	Type InferType `json:"type" gencodec:"required"`
}

func RetriveType(input []byte) InferType {
	if len(input) == 0 {
		return INFER_UNKNOWN
	}

	var dec Work
	if err := json.Unmarshal(input, &dec); err != nil {
		return INFER_UNKNOWN
	}

	return dec.Type
}

// infer response types
const (
	RES_OK    = "ok"
	RES_ERROR = "error"
)

type InferResult struct {
	Data hexutil.Bytes `json:"data"`
	Info string        `json:"info"`
}
