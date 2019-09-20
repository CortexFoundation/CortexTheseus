package main

import (
	"encoding/json"
	"flag"
	"fmt"

	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/inference"
	resty "github.com/go-resty/resty/v2" //"gopkg.in/resty.v1"
	"time"
)

var (
	uri   = flag.String("uri", "http://localhost:8827", "Server URI")
	mode  = flag.Int("type", 1, "inference mode: 1 for input hash, 2 for input content")
	model = flag.String("model", "0xca3d0286d5758697cdef653c1375960a868ac08a", "Model Info Hash")
	ih    = flag.String("ih", "0x574ab452e4c2514577868e55f9cf886d6a9bbcda", "Input Info Hash")
	ic    = flag.String("ic", "", "Input Content")
)

func main() {
	flag.Parse()

	var data []byte
	var err error
	switch *mode {
	case (int)(inference.INFER_BY_IH):
		data, err = json.Marshal(&inference.IHWork{
			Type:  inference.INFER_BY_IH,
			Model: *model,
			Input: *ih,
		})
		break

	case (int)(inference.INFER_BY_IC):
		var inputContent hexutil.Bytes
		if len(*ic) == 0 {
			inputContent = make([]byte, 1*28*28)
			for i := 0; i < 1*28*28; i++ {
				inputContent[i] = 0
			}
		} else {
			fmt.Println(*ic)
			if e := json.Unmarshal([]byte(*ic), &inputContent); e != nil {
				fmt.Println("unmarshal bytes: ", e)
				return
			}
		}
		data, err = json.Marshal(&inference.ICWork{
			Type:  inference.INFER_BY_IC,
			Model: *model,
			Input: hexutil.Bytes(inputContent),
		})
		break

	default:
		fmt.Println("Invalid mode")
		return
	}

	if err != nil {
		fmt.Println(err)
		return
	}

	requestBody := string(data)
	fmt.Println(requestBody)

	resp, err := resty.New().SetTimeout(time.Duration(15*time.Second)).R().
		SetHeader("Content-Type", "application/json").
		SetBody(requestBody).
		Post(*uri)

	fmt.Println("Response: ", resp, " Error: ", err)
}
