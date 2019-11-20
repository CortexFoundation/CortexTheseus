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
type IHWork struct {
	Type  InferType `json:"type"`
	Model string    `json:"model"`
	Input string    `json:"input"`
}

// Infer by input content
type ICWork struct {
	Type  InferType     `json:"type"`
	Model string        `json:"model"`
	Input hexutil.Bytes `json:"input"`
}

// Infer gas
type GasWork struct {
	Type  InferType `json:"type"`
	Model string    `json:"model"`
}

// check Available
type AvailableWork struct {
	Type     InferType `json:"type"`
	InfoHash string    `json:"infohash"`
	RawSize  int64     `json:"rawSize"`
}

type Work struct {
	Type InferType `json:"type"`
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
