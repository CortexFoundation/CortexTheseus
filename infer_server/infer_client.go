package main

import (
	"fmt"
	resty "gopkg.in/resty.v1"
)

func main() {
	requestBody := `{"Model":"0xb2bbbde78f465ffb7bd5ef091e93da5d86096dc6", "InputHash": "0xd18f903e316d9915c848d48862544913df2b4907"}`
	uri := "http://localhost:8827"
	fmt.Println(requestBody)
	resp, err := resty.R().
		SetHeader("Content-Type", "application/json").
		SetBody(requestBody).
		Post(uri)

	fmt.Println(resp, err)
}
