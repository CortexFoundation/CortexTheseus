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
	_ "reflect"
	_ "runtime"
	"unsafe"

	"github.com/CortexFoundation/CortexTheseus/infernet/kernel"
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
	var (
		lib    *kernel.Library
		net    unsafe.Pointer
		status int
	)
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
	net, status = lib.LoadModel(modelCfg, modelBin, 0)
	if status != kernel.SUCCEED {
		fmt.Printf("CVMAPILoadModel failed: %d\n", status)
		return
	}
	fmt.Printf("CVMAPILoadModel succeed: %p\n", &net)
}
