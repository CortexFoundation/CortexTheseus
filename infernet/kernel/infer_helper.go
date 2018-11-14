package kernel

/* #cgo LDFLAGS: activation_kernels.o     blas_kernels.o    convolutional_kernels.o  deconvolutional_kernels.o  gemm_kernels.o    int_convolutional_kernels.o  maxpool_layer_kernels.o  trivial_mul_kernels.o
#cgo LDFLAGS: avgpool_layer_kernels.o  col2im_kernels.o  crop_layer_kernels.o     dropout_layer_kernels.o    im2col_kernels.o  int_maxpool_layer_kernels.o  scale_kernels.o */

/*
#cgo LDFLAGS: -lm -pthread
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
#cgo LDFLAGS: -lstdc++

#cgo CFLAGS: -I/usr/local/cuda/include/
#cgo CFLAGS: -DGPU
#cgo CFLAGS: -Wall -Wno-unused-result -Wno-unknown-pragmas

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
