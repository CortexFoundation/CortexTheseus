package synapse

import (
	"encoding/binary"
	"errors"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/crypto/sha3"
	"github.com/CortexFoundation/CortexTheseus/inference"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/rlp"
)

func ReadData(r *inference.NpyReader) ([]byte, error) {
	var (
		data []byte
		derr error
	)
	log.Debug("ReadData", "r.Dtype", r.Dtype)
	if r.Dtype == "i1" {
		data, derr = r.GetBytes()
	} else if r.Dtype == "i4" {
		i4_data, i4_derr := r.GetInt32()
		data = make([]byte, len(i4_data)*4)
		if i4_derr != nil {
			return nil, derr
		}
		//TODO(tian) assume input is uint31! not int32
		for i := 0; i < len(i4_data); i++ {
			//tmp := make([]byte, 8)
			//binary.PutUvarint(tmp[:], uint64(i4_data[idx]))
			//copy(data[idx*4:idx*4+4], tmp[:4])
			binary.LittleEndian.PutUint32(data[i:i+4], uint32(i4_data[i]))
		}
	} else {
		return nil, errors.New("not support dtype for " + r.Dtype)
	}
	if derr != nil {
		return nil, derr
	}

	return data, nil
}

func RLPHashString(x interface{}) string {
	var h common.Hash
	hw := sha3.NewKeccak256()
	rlp.Encode(hw, x)
	return hexutil.Encode(hw.Sum(h[:0]))
}
