package vm

import (
	"errors"
	"fmt"
	"strconv"
	"sync"

	simplejson "github.com/bitly/go-simplejson"
	"github.com/ethereum/go-ethereum/infernet"
	"github.com/ethereum/go-ethereum/log"
	resty "gopkg.in/resty.v1"
)

func LocalInfer(modelDir, inputDir string) (uint64, error) {
	log.Trace(fmt.Sprintf("Model&Input Dir : %v | %v", modelDir, inputDir))
	resultCh := make(chan uint64)
	errCh := make(chan error)

	var pend sync.WaitGroup
	pend.Add(1)
	go func(modelDir, inputDir string) {
		defer pend.Done()
		infernet.Infer(modelDir, inputDir, resultCh, errCh)
	}(modelDir, inputDir)

	select {
	case result := <-resultCh:
		log.Trace(fmt.Sprintf("Local Infer Result : %v", result))
		return result, nil
	case err := <-errCh:
		log.Trace(fmt.Sprintf("Local Infer Error : %v", err))
		return 0, err
	}
	pend.Wait()

	return 0, nil
}

func RemoteInfer(requestBody, uri string) (uint64, error) {
	log.Trace(fmt.Sprintf("%v", requestBody))
	resp, err := resty.R().
		SetHeader("Content-Type", "application/json").
		SetBody(requestBody).
		Post(uri)
	if err != nil || resp.StatusCode() != 200 {
		return 0, errors.New(fmt.Sprintf("%s | %s | %s | %s | %v", "evm.Infer: External Call Error: ", requestBody, resp, uri, err))
	}
	log.Trace(fmt.Sprintf("%v", resp.String()))
	js, js_err := simplejson.NewJson([]byte(resp.String()))
	if js_err != nil {
		return 0, errors.New(fmt.Sprintf("evm.Infer: External Call Error | %v ", js_err))
	}
	int_output_tmp, out_err := js.Get("info").String()
	if out_err != nil {
		return 0, errors.New(fmt.Sprintf("evm.Infer: External Call Error | %v ", out_err))
	}
	uint64_output, err := strconv.ParseUint(int_output_tmp, 10, 64)
	if err != nil {
		return 0, errors.New("evm.Infer: Type Conversion Error")
	}
	return uint64_output, nil
}
