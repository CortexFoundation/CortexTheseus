// +build !remote

package synapse

import (
	"errors"
	"fmt"
	"strings"

	"github.com/CortexFoundation/CortexTheseus/common/lru"
	"github.com/CortexFoundation/CortexTheseus/inference/synapse/kernel"
	"github.com/CortexFoundation/CortexTheseus/log"
//	"github.com/CortexFoundation/CortexTheseus/torrentfs"
)

func (s *Synapse) InferByInfoHash(modelInfoHash, inputInfoHash string) ([]byte, error) {
	var (
		resCh = make(chan []byte)
		errCh = make(chan error)
	)

	go func() {
		s.inferByInfoHash(modelInfoHash, inputInfoHash, resCh, errCh)
	}()

	select {
	case result := <-resCh:
		return result, nil
	case err := <-errCh:
		return nil, err
	case <-s.exitCh:
		return nil, errors.New("Synapse Engine is closed")
	}
}

func (s *Synapse) InferByInputContent(modelInfoHash string, inputContent []byte) ([]byte, error) {
	var (
		resCh = make(chan []byte)
		errCh = make(chan error)
	)

	inputInfoHash := RLPHashString(inputContent)

	go func() {
		s.inferByInputContent(modelInfoHash, inputInfoHash, inputContent, resCh, errCh)
	}()

	select {
	case result := <-resCh:
		return result, nil
	case err := <-errCh:
		return nil, err
	case <-s.exitCh:
		return nil, errors.New("Synapse Engine is closed")
	}
	return nil, nil
}

func (s *Synapse) GetGasByInfoHash(modelInfoHash string) (gas uint64, err error) {
	fmt.Println("synapse: ", s)
	var (
		modelHash = strings.ToLower(modelInfoHash[2:])
		// modelDir  = s.config.StorageDir + "/" + modelHash

		modelJson []byte
		modelJson_err error
	)
	modelJson, modelJson_err = s.config.Storagefs.GetFile(modelHash, "/data/symbol")
	if modelJson_err != nil || modelJson == nil{
		return 0,  modelJson_err
	}
	// fmt.Println("modelCfg =" , modelCfg, "modelBin = ", modelBin)
	// Inference Cache
	cacheKey := RLPHashString("estimate_ops_" + modelHash)
	if v, ok := s.simpleCache.Load(cacheKey); ok && !s.config.IsNotCache {
		log.Debug("Infer Success via Cache", "result", v.(uint64))
		return v.(uint64), nil
	}
	// if s.config.Debug {
	// 	fmt.Println("modelCfg =", modelCfg, "modelBin = ", modelBin)
	// }

	gas, err = kernel.GetModelOps(s.lib, modelJson)
	if err != nil {
		return 0, err
	}

	if !s.config.IsNotCache {
		s.simpleCache.Store(cacheKey, gas)
	}
	return gas, err
}

func (s *Synapse) inferByInfoHash(modelInfoHash, inputInfoHash string, resCh chan []byte, errCh chan error) {
	var (
		modelHash = strings.ToLower(modelInfoHash[2:])
		inputHash = strings.ToLower(string(inputInfoHash[2:]))
	)

	// Inference Cache
	cacheKey := RLPHashString(modelHash + inputHash)
	if v, ok := s.simpleCache.Load(cacheKey); ok && !s.config.IsNotCache {
		log.Debug("Infer Success via Cache", "result", v.([]byte))
		resCh <- v.([]byte)
		return
	}

	// Image Path Check
	// inputDir := s.config.StorageDir + "/" + inputHash
	// inputFilePath := inputDir + "/data"
	// log.Debug("Inference Core", "Input Data File", inputFilePath)
	// if _, fsErr := os.Stat(inputFilePath); os.IsNotExist(fsErr) {
	// 	errCh <- ErrInputFileNotExist
	// 	return
	// }

	inputBytes, dataErr := s.config.Storagefs.GetFile(inputHash, "/data")
	if dataErr != nil {
		errCh <- dataErr
		return
	}

	s.inferByInputContent(modelInfoHash, inputInfoHash, inputBytes, resCh, errCh)
}

func (s *Synapse) infer(modelCfg, modelBin []byte, inputContent []byte) ([]byte, error) {
	var model *kernel.Model
	if _, ok := s.caches[s.config.DeviceId]; !ok {
		s.caches[s.config.DeviceId] = lru.New(4000000)
	}

	// ret, ok := s.caches[s.config.DeviceId].Get(modelCfg)

	model = kernel.New(s.lib, s.config.DeviceId, modelCfg, modelBin)
	if model == nil {
		return nil, errors.New("create model error")
	}
	//TODO(tian) replace it with gas per KB
	// s.caches[s.config.DeviceId].Add(modelCfg, model, model.Size()/1000)

	return model.Predict(inputContent)
}

func (s *Synapse) inferByInputContent(modelInfoHash, inputInfoHash string, inputContent []byte, resCh chan []byte, errCh chan error) {
	var (
		modelHash = strings.ToLower(modelInfoHash[2:])
		inputHash = strings.ToLower(inputInfoHash[2:])
		// modelDir  = s.config.StorageDir + "/" + modelHash
	)

	if checkErr := CheckMetaHash(Model_V1, modelHash); checkErr != nil {
		errCh <- checkErr
		return
	}

	// Input process
	// if procErr := ProcessImage(inputContent); procErr != nil {
	// 	errCh <- procErr
	// 	return
	// }

	// Inference Cache
	cacheKey := RLPHashString(modelHash + inputHash)
	if v, ok := s.simpleCache.Load(cacheKey); ok && !s.config.IsNotCache {
		log.Debug("Infer Succeed via Cache", "result", v.([]byte))
		resCh <- v.([]byte)
		return
	}

	// Model Path Check
	// modelCfg := modelDir + "/data/symbol"
	// modelBin := modelDir + "/data/params"
	modelJson, modelJson_err := s.config.Storagefs.GetFile(modelHash, "/data/symbol")
	if modelJson_err != nil || modelJson == nil {
		errCh <- modelJson_err
	}
	modelParams, modelParams_err := s.config.Storagefs.GetFile(modelHash, "/data/params")
	if modelParams_err != nil || modelParams == nil {
		errCh <- ErrModelFileNotExist
	}

	label, inferErr := s.infer(modelJson, modelParams, inputContent)

	if inferErr != nil {
		errCh <- inferErr
		return
	}

	if !s.config.IsNotCache {
		s.simpleCache.Store(cacheKey, label)
	}

	resCh <- label
	return

}

