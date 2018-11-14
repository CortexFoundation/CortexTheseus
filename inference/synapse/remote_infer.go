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
	requestBody := fmt.Sprintf(`{"Type": 1, "ModelHash":"%s", "InputHash":"%s"}`, modelInfoHash, inputInfoHash)
	log.Debug("Remote Inference", "request", requestBody)

	return s.sendRequest(requestBody, uri)
}

func (s *Synapse) RemoteInferByInputContent(modelInfoHash, uri string, addr, slot, blockNumber string, txIndex int) (uint64, error) {
	requestBody := fmt.Sprintf(`{"Type": 2, "ModelHash":"%s", "InputAddress":"%s", "InputSlot":"%s", "InputBlockNumber":"%s", "InputTxIndex": "%d"}`, modelInfoHash, addr, slot, blockNumber, txIndex)
	log.Debug("Remote Inference", "request", requestBody)

	return s.sendRequest(requestBody, uri)
}

func (s *Synapse) sendRequest(requestBody, uri string) (uint64, error) {
	cacheKey := RLPHashString(requestBody)
	if v, ok := s.simpleCache.Load(cacheKey); ok && !s.config.IsNotCache {
		log.Debug("Infer Succeed via Cache", "result", v.(uint64))
		return v.(uint64), nil
	}

	resp, err := resty.R().
		SetHeader("Content-Type", "application/json").
		SetBody(requestBody).
		Post(uri)
	if err != nil || resp.StatusCode() != 200 {
		return 0, errors.New(fmt.Sprintf("%s | %s | %s | %s | %v", "evm.Infer: External Call Error: ", requestBody, resp, uri, err))
	}

	log.Debug("Remote Inference", "response", resp.String())

	js, js_err := simplejson.NewJson([]byte(resp.String()))
	if js_err != nil {
		return 0, errors.New(fmt.Sprintf("Remote Infer: resonse json parse error | %v ", js_err))
	}

	msg, msgErr := js.Get("msg").String()
	if msgErr != nil {
		return 0, errors.New(fmt.Sprintf("Remote Infer: response `msg` parse error | %v ", msgErr))
	}

	int_output_tmp, out_err := js.Get("info").String()
	if out_err != nil {
		return 0, errors.New(fmt.Sprintf("Remote Infer: response `info` parse error | %v ", out_err))
	}

	if msg != "ok" {
		return 0, errors.New(int_output_tmp)
	}

	uint64_output, err := strconv.ParseUint(int_output_tmp, 10, 64)
	if err != nil {
		return 0, errors.New("Remote Infer: result conversion error")
	}

	if !s.config.IsNotCache {
		s.simpleCache.Store(cacheKey, uint64_output)
	}

	return uint64_output, nil
}
