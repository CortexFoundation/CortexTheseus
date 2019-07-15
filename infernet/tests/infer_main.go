package main

/*
#include <stdio.h>
#include <stdlib.h>

enum CVMStatus {
	SUCCEED = 0,
	ERROR_LOGIC = -1,
	ERROR_RUNTIME = -2
};

void myprint_int(long long *num) {
	printf("%lld\n", *num);
	*num = 4;
}
void myprint(char *s) {
	printf("%s\n", s);
}

void new_arr(void **arr) {
	void *mal = malloc(sizeof(int) * 1);
	int *nums = (int*)mal;
	*nums = 1000;
	printf("%p %d\n", mal, *nums);
	*arr = mal;
}

void myprint_void(void *arr) {
	int *nums = (int*)arr;
	printf("%p %d\n", arr, *nums);
}

*/
import "C"
import (
	"fmt"
	"io/ioutil"
	_ "io/ioutil"
	"os"
	_ "reflect"
	_ "runtime"
	"unsafe"

	"github.com/CortexFoundation/CortexTheseus/inference/synapse/kernel"
	"github.com/CortexFoundation/CortexTheseus/log"
)

func test() {
	cs := C.CString("Hello from stdio")
	C.myprint(cs)
	C.free(unsafe.Pointer(cs))

	var num C.longlong
	num = 3
	C.myprint_int(&num)
	fmt.Println(int64(num))

	var s1 C.enum_CVMStatus
	s1 = C.ERROR_LOGIC
	s2 := C.ERROR_LOGIC
	fmt.Println(int(s1) == int(s2))

	var arr unsafe.Pointer
	C.new_arr(&arr)
	C.myprint_void(arr)
}

func main() {
	// Set log
	log.Root().SetHandler(log.LvlFilterHandler(log.Lvl(5), log.StreamHandler(os.Stdout, log.TerminalFormat(true))))

	var (
		lib    *kernel.LibCVM
		net    *kernel.Model
		res    []byte
		status int
	)
	// lib, status = kernel.LibOpen("./libcvm_runtime_cuda.so")
	lib, status = kernel.LibOpen("./libcvm_runtime_cpu.so")
	if status != kernel.SUCCEED {
		fmt.Printf("open library error: %d\n", status)
		return
	}

	root := "/data/std_out/log2"
	modelCfg, sErr := ioutil.ReadFile(root + "/symbol")
	if sErr != nil {
		fmt.Println(sErr)
		return
	}
	modelBin, pErr := ioutil.ReadFile(root + "/params")
	if pErr != nil {
		fmt.Println(pErr)
		return
	}
	// modelCfg := []byte("{}")
	// modelBin := []byte("dkjflsiejflsdkj")
	net, status = kernel.New(lib, modelCfg, modelBin, 0, 0)
	if status != kernel.SUCCEED {
		fmt.Printf("CVMAPILoadModel failed: %d\n", status)
		return
	}
	input_size := net.GetInputLength()
	fmt.Printf("CVMAPILoadModel succeed: %p ops=%s size=%s input_size=%s\n",
		&net, net.Ops(), net.Size(), input_size)

	var data []byte = make([]byte, input_size)
	res, status = net.Predict(data)
	if status != kernel.SUCCEED {
		fmt.Printf("Predict failed: %d\n", status)
		return
	}
	fmt.Printf("Predict succeed: %v\n", res[:100])

	status = net.Free()
	if status != kernel.SUCCEED {
		fmt.Printf("Free model failed: %d\n", status)
		return
	}
	fmt.Printf("Free model succeed\n")

	var gas uint64
	gas, status = kernel.GetModelGasFromGraphFile(lib, modelCfg)
	if status != kernel.SUCCEED {
		fmt.Printf("Get model gas from file failed: %s\n", status)
		return
	}
	fmt.Printf("Get model gas from file succeed: %s\n", int(gas))
}
