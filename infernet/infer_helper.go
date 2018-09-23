package infernet

/*
#cgo LDFLAGS: -L./ -Wl,-rpath,./bin/ -lcortexnet
#cgo CFLAGS: -I./int_mnist_model

#include "interface.h"
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

func readImg(input string) ([]byte, error) {
	r, rerr := NewFileReader(input)
	if rerr != nil {
		fmt.Println("Error: ", rerr)
		return nil, rerr
	}

	// Infer data must between [0, 127)
	data, derr := r.GetBytes()
	if derr != nil {
		return nil, derr
	}

	for i, v := range data {
		data[i] = uint8(v) / 2
	}

	// Tmp Code
	// DumpToFile("tmp.dump", data)

	return data, nil
}

func InferCore(modelCfg, modelBin, image string) (uint64, error) {

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

	imageData, rerr := readImg(image)
	if rerr != nil {
		return 0, rerr
	}

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

func main() {
	label, err := InferCore("./infer_data/model/param", "./infer_data/model/symbol", "./infer_data/image/data")

	fmt.Println(label, err)

	// readImg("./img.8b")
	// fmt.Println("Result: " + res)
	// fmt.Println("Result: " + res + " Length: " + string(len(res)))
}
