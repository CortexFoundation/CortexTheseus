package infernet

/*
#cgo LDFLAGS: -L./ -lcortexnet -lstdc++
#cgo CFLAGS: -I./src

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

	data, derr := r.GetBytes()
	if derr != nil {
		return nil, derr
	}

	return data, nil
}

func inferCore(modelDir, inputDir string) (uint64, error) {
	net := C.load_model(
		C.CString(modelDir+"/data/params"),
		C.CString(modelDir+"/data/symbol"))

	resLen := int(C.get_output_length(net))
	if resLen == 0 {
		return 0, errors.New("Model result len is 0")
	}

	res := make([]byte, resLen)

	imageData, rerr := readImg(inputDir + "/data")
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
	for idx := 1; idx < resLen; idx++ {
		if int8(res[idx]) > max {
			max = int8(res[idx])
			label = uint64(idx)
		}
	}

	C.free_model(net)

	return label, nil
}

func Infer(modelDir, inputDir string, resultCh chan uint64, errCh chan error) {
	label, err := inferCore(modelDir, inputDir)
	if err != nil {
		errCh <- err
		return
	}

	resultCh <- label
}

func main() {
	label, err := inferCore("./infer_data/model", "./infer_data/image")

	fmt.Println(label, err)

	// readImg("./img.8b")
	// fmt.Println("Result: " + res)
	// fmt.Println("Result: " + res + " Length: " + string(len(res)))
}
