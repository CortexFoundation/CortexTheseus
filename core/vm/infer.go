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

var (
	ErrInvalidInferFlag = "infer while verifying block error"
)

func CreateVerifyBlockInferError(err error) error {
	return errors.New(fmt.Sprintf(ErrInvalidInferFlag+": %v", err.Error()))
}

func ParseVerifyBlockInferError(vbErr error) error {
	errLen := len(vbErr.Error())
	flagLen := len(ErrInvalidInferFlag)
	if errLen >= flagLen && vbErr.Error()[0:flagLen] == ErrInvalidInferFlag {
		return errors.New(vbErr.Error()[flagLen+2:])
	}

	return nil
}

/**
 * Infer progress should waiting for file download
 * while verifying block fetched from other peer.
 * Other case return infer result or error.
 */
func LocalInfer(modelHash, inputHash string) (uint64, error) {
	var (
		resultCh = make(chan uint64, 1)
		errCh    = make(chan error, 1)
	)

	err := infer.SubmitInferWork(
		modelHash,
		inputHash,
		resultCh,
		errCh,
	)

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

	log.Debug("Remote Infer", "Response", resp.String())

	js, js_err := simplejson.NewJson([]byte(resp.String()))
	if js_err != nil {
		return 0, errors.New(fmt.Sprintf("evm.Infer: External Call Error | %v ", js_err))
	}

	msg, msgErr := js.Get("msg").String()
	if msgErr != nil {
		return 0, errors.New(fmt.Sprintf("evm.Infer: External Call Error | %v ", msgErr))
	}

	int_output_tmp, out_err := js.Get("info").String()
	if out_err != nil {
		return 0, errors.New(fmt.Sprintf("evm.Infer: External Call Error | %v ", out_err))
	}

	if msg != "ok" {
		return 0, errors.New(fmt.Sprintf("evm.Infer: External Response not OK | %v ", int_output_tmp))
	}

	uint64_output, err := strconv.ParseUint(int_output_tmp, 10, 64)
	if err != nil {
		return 0, errors.New("evm.Infer: Type Conversion Error")
	}
	return uint64_output, nil
}
