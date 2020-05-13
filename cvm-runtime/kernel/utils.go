package kernel

import (
	"errors"
	"fmt"
)

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
