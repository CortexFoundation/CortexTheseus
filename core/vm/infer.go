package vm

import (
	"errors"
	"fmt"
	"strconv"

	simplejson "github.com/bitly/go-simplejson"
	infer "github.com/ethereum/go-ethereum/infer_server"
	"github.com/ethereum/go-ethereum/log"
	resty "gopkg.in/resty.v1"
)

func LocalInfer(modelHash, inputHash string, fakeVM bool) (uint64, error) {
	var (
		resultCh = make(chan uint64, 1)
		errCh    = make(chan error, 1)
	)

	err := infer.SubmitInferWork(
		modelHash,
		inputHash,
		!fakeVM,
		resultCh,
		errCh)

	if err != nil {
		return 0, err
	}

	select {
	case result := <-resultCh:
		return result, nil
	case err := <-errCh:
		return 0, err
	}

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
