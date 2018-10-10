package infernet

/*
#cgo LDFLAGS: -L./ -Wl,-rpath,./bin/ -lcortexnet
#cgo CFLAGS: -I./int_mnist_model

#include "interface.h"
*/
import "C"
import (
	"errors"
	"unsafe"
)

func InferCore(modelCfg, modelBin string, imageData []byte) (uint64, error) {

	net := C.load_model(
		C.CString(modelCfg),
		C.CString(modelBin))

	// TODO net may be a C.NULL, dose it equal Go.nil?
	if net == nil {
		return 0, errors.New("Model load error")
	}

	defer C.free_model(net)

	resLen := int(C.get_output_length(net))
	if resLen == 0 {
		return 0, errors.New("Model result len is 0")
	}

	res := make([]byte, resLen)

	flag := C.predict(
		net,
		(*C.char)(unsafe.Pointer(&imageData[0])),
		(*C.char)(unsafe.Pointer(&res[0])))

	if flag != 0 {
		return 0, errors.New("Predict Error")
	}

	max := int8(res[0])
	label := uint64(0)

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

	return label, nil
}
