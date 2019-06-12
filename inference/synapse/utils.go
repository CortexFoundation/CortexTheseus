package synapse

import (
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/crypto/sha3"
	"github.com/CortexFoundation/CortexTheseus/inference"
	"github.com/CortexFoundation/CortexTheseus/rlp"
	"math/big"
)

func GetFirstObject(byte_res []byte) *big.Int {
	if byte_res == nil || len(byte_res) < 6 {
		return big.NewInt(0)
	}

	res := big.NewInt(0)
	res_tmp := make([]byte, 32)
	
	// for i := 0; i < len(res_tmp) / 4; i++ {
	// 	res_tmp[i] = 255;
	// 	if i == 0  {
	// 		res_tmp[i] = 0
	// 	}
	// }
	res.SetBytes(res_tmp)
	return res 
}

func ArgMax(res []byte) *big.Int {
	if res == nil {
		return big.NewInt(0)
	}
	ret := big.NewInt(0)
	var (
		max    = int8(res[0])
		label  = uint64(0)
		resLen = len(res)
	)

	for idx := 1; idx < resLen; idx++ {
		if int8(res[idx]) > max {
			max = int8(res[idx])
			label = uint64(idx)
		}
	}
	ret.SetInt64(int64(label))
	return ret
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

// func ProcessImage(data []byte) error {
// 	return nil
// 	// Infer data must between [0, 127)
// 	for i, v := range data {
// 		data[i] = uint8(v) / 2
// 	}
// 
// 	return nil
// }

func RLPHashString(x interface{}) string {
	var h common.Hash
	hw := sha3.NewKeccak256()
	rlp.Encode(hw, x)
	return hexutil.Encode(hw.Sum(h[:0]))
}
