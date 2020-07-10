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

	"github.com/CortexFoundation/CortexTheseus/cvm-runtime/kernel"
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

	device := "cpu"
	deviceType := 0
	if device == "cuda" {
		deviceType = 1
	}
	lib, status = kernel.LibOpen("./libcvm_runtime_" + device + ".so")
	// lib, status = kernel.LibOpen("./libcvm_runtime_cpu.so")
	if status != kernel.SUCCEED {
		fmt.Printf("open library error: %d\n", status)
		return
	}

	root := "/data/std_out/log2"
	// root := "/home/serving/ctxc_data/cpu/3145ad19228c1cd2d051314e72f26c1ce77b7f02/data"
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
	net, status = kernel.New(lib, modelCfg, modelBin, deviceType, 0)
	if status != kernel.SUCCEED {
		fmt.Printf("Failed LoadModel: %d\n", status)
		return
	}
	input_size := net.GetInputLength()
	fmt.Printf("Succeed LoadModel: %p ops=%s size=%s input_size=%s\n",
		&net, net.Ops(), net.Size(), input_size)

	var data []byte = make([]byte, input_size)
	res, status = net.Predict(data)
	if status != kernel.SUCCEED {
		fmt.Printf("Failed Predict: %d\n", status)
		return
	}
	fmt.Printf("Succeed Predict: %v\n", res)

	status = net.Free()
	if status != kernel.SUCCEED {
		fmt.Printf("Failed Free model: %d\n", status)
		return
	}
	fmt.Printf("Succeed Free model\n")

	var gas uint64
	gas, status = kernel.GetModelGasFromGraphFile(lib, modelCfg)
	if status != kernel.SUCCEED {
		fmt.Printf("Failed get model gas from file: %s\n", status)
		return
	}
	fmt.Printf("Succeed get model gas from file: %s\n", int(gas))
}
