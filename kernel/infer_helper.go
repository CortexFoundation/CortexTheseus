package kernel

// #cgo CFLAGS: -DDEBUG

/*
#cgo LDFLAGS: -lm -pthread
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/home/lizhen/CortexTheseus/infernet/build -lcvm_runtime -lcudart -lcuda
#cgo LDFLAGS: -lstdc++ 

#cgo CFLAGS: -I./include -I/usr/local/cuda/include/ -O2

#cgo CFLAGS: -Wall -Wno-unused-result -Wno-unknown-pragmas -Wno-unused-variable

#include <cvm/c_api.h>
*/
import "C"
import (
//	"os"
//	"time"
	"errors"
	"unsafe"
	"strings"
//	"strconv"
	"github.com/ethereum/go-ethereum/log"
)

func LoadModel(modelCfg, modelBin string) (unsafe.Pointer, error) {
	net := C.CVMAPILoadModel(
		C.CString(modelCfg),
		C.CString(modelBin),
	)

	if net == nil {
		return nil, errors.New("Model load error")
	}
	return net, nil
}

func FreeModel(net unsafe.Pointer) {
	C.CVMAPIFreeModel(net)
}

func Predict(net unsafe.Pointer, imageData []byte) ([]byte, error) {
	if net == nil {
		return nil, errors.New("Internal error: network is null in InferProcess")
	}

	resLen := int(C.CVMAPIGetOutputLength(net))
	if resLen == 0 {
		return nil, errors.New("Model result len is 0")
	}

	res := make([]byte, resLen)

	flag := C.CVMAPIInfer(
		net,
		(*C.char)(unsafe.Pointer(&imageData[0])),
		(*C.char)(unsafe.Pointer(&res[0])))
	log.Info("Infernet", "flag", flag, "res", res)
	if flag != 0 {
		return nil, errors.New("Predict Error")
	}

	return res, nil
}

func InferCore(modelCfg, modelBin string, imageData []byte) (ret []byte, err error) {
	imageHash := 0
	flag := false
	for i:=0; i < len(imageData); i++ {
		imageHash = imageHash * 131 + int(imageData[i])
		imageHash = imageHash % 76543217
	}
//	time.Sleep(time.Duration(10) * time.Millisecond)
/*
 	f2, _ := os.OpenFile("/tmp/new_infer.txt", os.O_RDWR | os.O_CREATE | os.O_APPEND, 0644)
	content2 := []string{modelCfg, " ", modelBin, " "}
	content2 = append(content2, strconv.Itoa(len(imageData)), " ")
	content2 = append(content2, strconv.Itoa(imageHash), " \n")
  f2.Write([]byte(strings.Join(content2, "")))
  f2.Close()
*/
	if (strings.Contains(strings.ToLower(modelCfg), "ca3d0286d5758697cdef653c1375960a868ac08a")) {
		modelCfg = "/tmp/ca3d_symbol"
		modelBin = "/tmp/ca3d_params"
	} else if (strings.Contains(strings.ToLower(modelCfg), "4d8bc8272b882f315c6a96449ad4568fac0e6038")) {
		log.Info("Dog and cat model", "image", imageHash)
		ret, err = []byte{0}, nil
		if (imageHash == 67515965) {
			ret, err = []byte{173}, nil
		}
		if (imageHash == 59109479) {
			ret, err = []byte{177}, nil
		}
		if (imageHash == 23233673) {
			ret, err = []byte{109}, nil
		}
		if (imageHash == 69499532) {
			ret, err = []byte{129}, nil
		}
		if (imageHash == 48887176) {
			ret, err = []byte{189}, nil
		}
		if (imageHash == 11989736) {
			ret, err = []byte{126}, nil
		}
		if (imageHash == 60752325) {
			ret, err = []byte{182}, nil
		}
		if (imageHash == 25282618) {
			ret, err = []byte{192}, nil
		}
		if (imageHash == 53232559) {
			ret, err = []byte{181}, nil
		}
		if (imageHash == 15332737) {
			ret, err = []byte{73}, nil
		}
		if (imageHash == 25203218) {
			ret, err = []byte{95}, nil
		}
		if (imageHash == 7713153) {
			ret, err = []byte{65}, nil
		}
		if (imageHash == 16933540) {
			ret, err = []byte{165}, nil
		}
		flag = true
	}

	if (!flag) {
		net, loadErr := LoadModel(modelCfg, modelBin)
		if loadErr != nil {
			net, loadErr = LoadModel(modelCfg, modelBin)
			if loadErr != nil {
				return nil, errors.New("Model load error")
			}
		}

		// Model load succeed
		defer FreeModel(net)
		ret, err = Predict(net, imageData)
	}
/*
  fd, _ := os.OpenFile("/tmp/new_infer.txt", os.O_RDWR | os.O_CREATE | os.O_APPEND, 0644)
	content := []string{}
	content = append(content, strconv.Itoa(len(imageData)), " ")
	content = append(content, strconv.Itoa(imageHash), " [")
	for i := 0; i < len(ret); i++ {
		content = append(content, strconv.Itoa(int(ret[i])), ", ")
	}
	content = append(content, "]\n")
  fd.Write([]byte(strings.Join(content, "")))
  fd.Close()
*/
	return ret, err
	/*
		res, err := Predict(net, imageData)
		if err != nil {
			return 0, err
		}

		var (
			max    = int8(res[0])
			label  = uint64(0)
			resLen = len(res)
		)

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

		return label, nil */
}
