package main

// #include "test.h"
import "C"

import "fmt"

var function func(str *C.char)

//export InternalFunc
func InternalFunc(str *C.char) {
    function(str)
}

func Register(fnct func(str *C.char)) {
    function = fnct
    cs := C.CString("aaaaaaaaaaa")
    C.SetFunc(cs)
}

func test(str *C.char) {
    fmt.Println("How should I do it", C.GoString(str))
}

func main() {
    Register(test)
}