package synapse

import (
	"errors"
	"fmt"
	"strconv"

	simplejson "github.com/bitly/go-simplejson"
	"github.com/ethereum/go-ethereum/log"
	resty "gopkg.in/resty.v1"
)

func (s *Synapse) RemoteInferByInfoHash(modelInfoHash, inputInfoHash, uri string) (uint64, error) {
	requestBody := fmt.Sprintf(`{"ModelHash":"%s", "InputHash":"%s"}`, modelInfoHash, inputInfoHash)
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

func (s *Synapse) RemoteInferByInputContent(modelInfoHash, uri string, inputContent []byte) (uint64, error) {
	return 0, errors.New("RemoteInferByInputContent not implemented")
}
