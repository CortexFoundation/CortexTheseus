package main

import (
	"flag"
	"fmt"

	resty "gopkg.in/resty.v1"
)

var (
	uri   = flag.String("uri", "http://localhost:8827", "Server URI")
	model = flag.String("model", "0x2bff3a8bf63c88e6f435af18f4b1a12bc09d4d1c", "Model Info Hash")
	input = flag.String("input", "0x393c5fcbaf56fd516cd5076da1acf37e173aeed1", "Input Info Hash")
)

func main() {
	requestBody := fmt.Sprintf(`{"ModelHash":"%v", "InputHash": "%v"}`, *model, *input)
	fmt.Println(requestBody)
	resp, err := resty.R().
		SetHeader("Content-Type", "application/json").
		SetBody(requestBody).
		Post(*uri)

	fmt.Println("Response: ", resp, " Error: ", err)
}
