package main

/*
#cgo LDFLAGS: -L./ -lcortexnet -lstdc++
#cgo CFLAGS: -I./int_mnist_model

#include "interface.h"
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

func infer_core(modelDir, inputDir string) (uint64, error) {
	fmt.Println(modelDir, inputDir)
	net := C.load_model(
		C.CString(modelDir+"/data/params"),
		C.CString(modelDir+"/data/symbol"))

	resLen := C.getOutputLength(net)
	if resLen == 0 {
		return 0, errors.New("Model result len is 0")
	}

	pred := make([]byte, resLen)

	flag := C.predict(
		net,
		C.CString(inputDir+"/data"),
		pred)

	res := C.GoBytes(unsafe.Pointer(pred), resLen)
	max := uint64(res[0])
	label := 0
	for idx := 0; idx < resLen; idx++ {
		if uint64(res[idx]) > maxLabel {
			maxLabel = uint64(res[idx])
			label = idx
		}
		fmt.Printf("%d ", res[idx])
	}

	C.free_model(net)

	return label, nil
}

func Infer(modelDir, inputDir string, resultCh chan uint64, errCh chan error) {
	label, err := infer_core(modelDir, inputDir)
	if err != nil {
		errCh <- err
		return
	}

	resultCh <- label
}

func main() {
	infer_core("./infer_data/model", "./infer_data/input")
	// fmt.Println("Result: " + res)
	// fmt.Println("Result: " + res + " Length: " + string(len(res)))
}
