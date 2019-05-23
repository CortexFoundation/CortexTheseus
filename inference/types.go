package inference

import (
	"encoding/json"

	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
)

// infer send types
type InferType uint32

const (
	INFER_UNKNOWN = InferType(0)
	INFER_BY_IH   = InferType(1) // Infer By Input Hash
	INFER_BY_IC   = InferType(2) // Infer By Input Content
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

func RetriveType(input []byte) InferType {
	type Work struct {
		Type *InferType `json:"type"`
	}

	var dec Work
	if err := json.Unmarshal(input, &dec); err != nil {
		return INFER_UNKNOWN
	}

	return *dec.Type
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
