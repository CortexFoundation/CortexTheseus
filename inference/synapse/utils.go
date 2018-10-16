package synapse

import (
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/crypto/sha3"
	"github.com/ethereum/go-ethereum/rlp"
)

func ReadImage(inputFilePath string) ([]byte, error) {
	r, rerr := NewFileReader(inputFilePath)
	if rerr != nil {
		return nil, rerr
	}

	data, derr := r.GetBytes()
	if derr != nil {
		return nil, derr
	}

	// Tmp Code
	// DumpToFile("tmp.dump", data)

	return data, nil
}

func ProcessImage(data []byte) error {
	// Infer data must between [0, 127)
	for i, v := range data {
		data[i] = uint8(v) / 2
	}

	return nil
}

func RLPHashString(x interface{}) string {
	var h common.Hash
	hw := sha3.NewKeccak256()
	rlp.Encode(hw, x)
	return hexutil.Encode(hw.Sum(h[:0]))
}
