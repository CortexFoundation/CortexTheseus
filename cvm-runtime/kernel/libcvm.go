package kernel

/*
#cgo LDFLAGS: -ldl -lstdc++
#cgo CFLAGS: -I../../../infernet/include -O2
#cgo CFLAGS: -Wall -Wno-unused-result -Wno-unknown-pragmas -Wno-unused-variable

#include "dlopen.h"
*/
import "C"
import (
	"path/filepath"
	"unsafe"

	"github.com/CortexFoundation/CortexTheseus/log"
)

var (
	SUCCEED       = int(C.SUCCEED)
	ERROR_LOGIC   = int(C.ERROR_LOGIC)
	ERROR_RUNTIME = int(C.ERROR_RUNTIME)
)

type LibCVM struct {
	path string
	lib  unsafe.Pointer
}

func LibOpen(libpath string) (*LibCVM, int) {
	cPath := make([]byte, C.PATH_MAX+1)
	if fullpath, err := filepath.Abs(libpath); err == nil {
		if len(fullpath) >= C.PATH_MAX {
			log.Error("library path exceed MAX_PATH_LEN",
				"path", libpath, "real path", fullpath, "max_len", C.PATH_MAX)
			return nil, ERROR_RUNTIME
		}
		copy(cPath, fullpath)
	} else {
		log.Error("library path extend to absolute path failed",
			"libpath", libpath, "error", err)
		return nil, ERROR_RUNTIME
	}

	var cErr *C.char
	lib := C.plugin_open((*C.char)(unsafe.Pointer(&cPath[0])), &cErr)
	if lib == nil {
		log.Error("library open failed", "path", libpath, "error", C.GoString(cErr))
		return nil, ERROR_RUNTIME
	}

	return &LibCVM{
		path: string(cPath),
		lib:  lib,
	}, SUCCEED
}

func (l *LibCVM) LoadModel(modelCfg, modelBin []byte,
	deviceType, deviceId int) (unsafe.Pointer, int) {
	jptr := (*C.char)(unsafe.Pointer(&(modelCfg[0])))
	pptr := (*C.char)(unsafe.Pointer(&(modelBin[0])))
	j_len := C.int(len(modelCfg))
	p_len := C.int(len(modelBin))
	var net unsafe.Pointer
	status := int(C.dl_CVMAPILoadModel(l.lib,
		jptr, j_len,
		pptr, p_len,
		&net,
		C.int(deviceType), C.int(deviceId)))
	return net, status
}
func (l *LibCVM) FreeModel(net unsafe.Pointer) int {
	return int(C.dl_CVMAPIFreeModel(l.lib, net))
}
func (l *LibCVM) Inference(net unsafe.Pointer, input []byte) ([]byte, int) {
	out_size, status := l.GetOutputLength(net)
	if status != SUCCEED {
		return nil, status
	}

	in_size := C.int(len(input))
	data := (*C.char)(unsafe.Pointer(&input[0]))
	tmp := make([]byte, out_size)
	output := (*C.char)(unsafe.Pointer(&tmp[0]))

	status = int(C.dl_CVMAPIInference(l.lib, net, data, in_size, output))
	if status != SUCCEED {
		return nil, status
	}
	return tmp, status
}

// version and method string length not greater than 32
func (l *LibCVM) GetVersion(net unsafe.Pointer) ([34]byte, int) {
	var version [34]byte
	sptr := (*C.char)(unsafe.Pointer(&version[0]))
	status := int(C.dl_CVMAPIGetVersion(l.lib, net, sptr))
	return version, status
}
func (l *LibCVM) GetPreprocessMethod(net unsafe.Pointer) ([34]byte, int) {
	var method [34]byte
	sptr := (*C.char)(unsafe.Pointer(&method[0]))
	status := int(C.dl_CVMAPIGetPreprocessMethod(l.lib, net, sptr))
	return method, status
}

func (l *LibCVM) GetInputLength(net unsafe.Pointer) (uint64, int) {
	var tmp C.ulonglong
	status := int(C.dl_CVMAPIGetInputLength(l.lib, net, &tmp))
	return uint64(tmp), status
}
func (l *LibCVM) GetOutputLength(net unsafe.Pointer) (uint64, int) {
	var tmp C.ulonglong
	status := int(C.dl_CVMAPIGetOutputLength(l.lib, net, &tmp))
	return uint64(tmp), status
}
func (l *LibCVM) GetInputTypeSize(net unsafe.Pointer) (uint64, int) {
	var tmp C.ulonglong
	status := int(C.dl_CVMAPIGetInputTypeSize(l.lib, net, &tmp))
	return uint64(tmp), status
}
func (l *LibCVM) GetOutputTypeSize(net unsafe.Pointer) (uint64, int) {
	var tmp C.ulonglong
	status := int(C.dl_CVMAPIGetOutputTypeSize(l.lib, net, &tmp))
	return uint64(tmp), status
}

func (l *LibCVM) GetStorageSize(net unsafe.Pointer) (uint64, int) {
	var tmp C.ulonglong
	status := int(C.dl_CVMAPIGetStorageSize(l.lib, net, &tmp))
	return uint64(tmp), status
}
func (l *LibCVM) GetGasFromModel(net unsafe.Pointer) (uint64, int) {
	var tmp C.ulonglong
	status := int(C.dl_CVMAPIGetGasFromModel(l.lib, net, &tmp))
	return uint64(tmp), status
}
func (l *LibCVM) GetGasFromGraphFile(json []byte) (uint64, int) {
	jptr := (*C.char)(unsafe.Pointer(&json[0]))
	var tmp C.ulonglong
	status := int(C.dl_CVMAPIGetGasFromGraphFile(l.lib, jptr, &tmp))
	return uint64(tmp), status
}
