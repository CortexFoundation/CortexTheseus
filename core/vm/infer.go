package vm

import (
	"errors"
	"fmt"
	"strconv"
	"sync"
	"time"

	simplejson "github.com/bitly/go-simplejson"
	"github.com/ethereum/go-ethereum/common"
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
		localInfer(modelDir, inputDir, resultCh, errCh)
	}(modelDir, inputDir)

	select {
	case result := <-resultCh:
		return result, nil
	case err := <-errCh:
		return 0, err
	}
	pend.Wait()

	return 0, nil
}

func localInfer(modelDir, inputDir string, resultCh chan uint64, errCh chan error) {
	startTime := time.Now()
	timeout := float64(10) // ten minutes timeout

	// Check File Exists
	modelCfg := modelDir + "/data/params"
	for !common.FileExist(modelCfg) {
		if time.Since(startTime).Minutes() > timeout {
			errCh <- errors.New("Infer pending time too long")
			return
		}
		log.Warn(fmt.Sprintf("Waiting for model config file %v sync", modelCfg))
		time.Sleep(5 * time.Second)
	}

	modelBin := modelDir + "/data/symbol"
	for !common.FileExist(modelBin) {
		if time.Since(startTime).Minutes() > timeout {
			errCh <- errors.New("Infer pending time too long")
			return
		}
		log.Warn(fmt.Sprintf("Waiting for model bin file %v sync", modelBin))
		time.Sleep(5 * time.Second)
	}

	image := inputDir + "/data"
	for !common.FileExist(image) {
		if time.Since(startTime).Minutes() > timeout {
			errCh <- errors.New("Infer pending time too long")
			return
		}
		log.Warn(fmt.Sprintf("Waiting for input data %v sync", image))
		time.Sleep(5 * time.Second)
	}

	label, err := infernet.InferCore(modelCfg, modelBin, image)
	if err != nil {
		errCh <- err
		return
	}

	resultCh <- label
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
