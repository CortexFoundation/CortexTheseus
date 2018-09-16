package main

/*
#cgo LDFLAGS: -L./ -lcortexnet -lstdc++
#cgo CFLAGS: -I./int_mnist_model 

#include "interface.h"
 */
import "C"
import (
	"fmt"
	"unsafe"
)

func Infer(modelDir, inputDir string) string {
	fmt.Println(modelDir, inputDir)
	net := C.load_model(
		C.CString(modelDir + "/data/params"),
		C.CString(modelDir + "/data/symbol"))

	pred := C.predict(
		net,
		C.CString(inputDir + "/data"))

	tmp := (*[10]byte)(unsafe.Pointer(pred))
	fmt.Println(tmp)
	fmt.Print("Result: ")
	for idx := 0; idx < 10; idx++ {
		fmt.Printf("%d ", tmp[idx])
	}
	fmt.Println("")

	res := C.GoBytes(unsafe.Pointer(pred), 10)

	fmt.Print("Result: ")
	for idx := 0; idx < len(res); idx++ {
		fmt.Printf("%d ", res[idx])
	}
	fmt.Println("")

	C.free_model(net)

	return C.GoString(pred)
}


func main() {
	Infer("./infer_data/model", "./infer_data/input")
	// fmt.Println("Result: " + res)
	// fmt.Println("Result: " + res + " Length: " + string(len(res)))
}
