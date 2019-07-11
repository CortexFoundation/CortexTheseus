package synapse

import (
	"encoding/binary"
	"errors"
	"fmt"
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
		for idx := 0; idx < len(i4_data); idx++ {
			tmp := make([]byte, 8)
			binary.PutUvarint(tmp[:], uint64(i4_data[idx]))
			// fmt.Println(uint64(i4_data[idx]))
			copy(data[idx*4:idx*4+4], tmp[:4])
			// fmt.Println(data[idx * 4: idx * 4 + 4], tmp[:4])
		}
		// fmt.Println("data = ", data)
	} else {
		return nil, errors.New("not support dtype for " + r.Dtype)
	}
	// fmt.Println("read image", data, "len: ", len(data), "rdtype", r.Dtype)
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

func SwitchEndian(data []byte, bytes int) ([]byte, error) {
	if len(data)%bytes != 0 {
		return nil, errors.New(fmt.Sprintf("data is not aligned with %d", bytes))
	}
	ret := make([]byte, len(data))
	for i := 0; i < len(data); i += bytes {
		for j := 0; j < bytes; j++ {
			ret[i+bytes-j-1] = data[i+j]
		}
	}
	return ret, nil
}

func ToAlignedData(data []byte, bytes int) ([]byte, error) {
	data_aligned := make([]byte, len(data))
	if bytes > 1 {
		tmp_res, input_conv_err := SwitchEndian(data, int(bytes))
		if input_conv_err != nil {
			return nil, input_conv_err
		}
		copy(data_aligned[:], tmp_res)
	} else {
		copy(data_aligned[:], data)
	}
	return data_aligned, nil

}
