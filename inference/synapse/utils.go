package synapse

import (
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/crypto/sha3"
	"github.com/ethereum/go-ethereum/inference"
	"github.com/ethereum/go-ethereum/rlp"
)

func ArgMax(res []byte) uint64 {
	if res == nil {
		return 0
	}

	var (
		max    = int8(res[0])
		label  = uint64(0)
		resLen = len(res)
	)

	// If result length large than 1, find the index of max value;
	// Else the question is two-classify model, and value of result[0] is the prediction.
	if resLen > 1 {
		for idx := 1; idx < resLen; idx++ {
			if int8(res[idx]) > max {
				max = int8(res[idx])
				label = uint64(idx)
			}
		}
	} else {
		if max > 0 {
			label = 1
		} else {
			label = 0
		}
	}

	return label
}

func ReadImage(inputFilePath string) ([]byte, error) {
	r, rerr := inference.NewFileReader(inputFilePath)
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
