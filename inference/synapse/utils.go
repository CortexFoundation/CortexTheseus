package synapse

import (
	"encoding/binary"
	"errors"

	"bytes"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/inference"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/rlp"
	"golang.org/x/crypto/sha3"
)

func ReadData(r *inference.NpyReader) ([]byte, error) {
	log.Debug("ReadData", "r.Dtype", r.Dtype)
	if r.Dtype == "i1" {
		data, err := r.GetBytes()
		if err != nil {
			return nil, err
		}
		return data, nil
	} else if r.Dtype == "i4" {
		i4_data, err := r.GetInt32()

		if err != nil {
			return nil, err
		}
		//for i := 0; i < len(i4_data); i++ {
		//	binary.LittleEndian.PutUint32(data[i:i+4], uint32(i4_data[i]))
		//}
		buf := new(bytes.Buffer)
		//for i := 0; i < len(i4_data); i++ {
		for _, d := range i4_data {
			binary.Write(buf, binary.LittleEndian, d)
		}
		return buf.Bytes(), nil
	} else {
		return nil, errors.New("not support dtype for " + r.Dtype)
	}
}

func RLPHashString(x interface{}) string {
	var h common.Hash
	hw := sha3.NewLegacyKeccak256()
	rlp.Encode(hw, x)
	return hexutil.Encode(hw.Sum(h[:0]))
}
