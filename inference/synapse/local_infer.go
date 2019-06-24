// +build !remote

package synapse

import (
	"errors"
	"strings"

	"github.com/CortexFoundation/CortexTheseus/common/lru"
	"github.com/CortexFoundation/CortexTheseus/inference/synapse/kernel"
	"github.com/CortexFoundation/CortexTheseus/log"
)

func (s *Synapse) InferByInfoHash(modelInfoHash, inputInfoHash string) ([]byte, error) {
	if s.config.IsRemoteInfer {
		inferRes, errRes := s.remoteInferByInfoHash(
			modelInfoHash,
			inputInfoHash,
			s.config.InferURI)
		return inferRes, errRes
	}
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

	if s.config.IsRemoteInfer {
		inferRes, errRes := s.remoteInferByInputContent(
			modelInfoHash,
			s.config.InferURI,
			inputContent,
		)
		return inferRes, errRes
	}

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
	// fmt.Println("synapse: ", s)
	if s.config.IsRemoteInfer {
		opsRes, errRes := s.remoteGasByModelHash(
			modelInfoHash,
			s.config.InferURI)
		return opsRes, errRes
	}

	var (
		modelHash = strings.ToLower(modelInfoHash[2:])
		modelJson []byte
		modelJson_err error
	)
	modelJson, modelJson_err = s.config.Storagefs.GetFile(modelHash, "/data/symbol")
	if modelJson_err != nil || modelJson == nil{
		return 0,  modelJson_err
	}

	cacheKey := RLPHashString("estimate_ops_" + modelHash)
	if v, ok := s.simpleCache.Load(cacheKey); ok && !s.config.IsNotCache {
		log.Debug("Infer Success via Cache", "result", v.(uint64))
		return v.(uint64), nil
	}

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

	inputBytes, dataErr := s.config.Storagefs.GetFile(inputHash, "/data")
	if dataErr != nil {
		errCh <- dataErr
		return
	}

	s.inferByInputContent(modelInfoHash, inputInfoHash, inputBytes, resCh, errCh)
}

func (s *Synapse) inferByInputContent(modelInfoHash, inputInfoHash string, inputContent []byte, resCh chan []byte, errCh chan error) {
	var (
		modelHash = strings.ToLower(modelInfoHash[2:])
		inputHash = strings.ToLower(inputInfoHash[2:])
		// modelDir  = s.config.StorageDir + "/" + modelHash
	)

	// Inference Cache
	ModelInputKey := RLPHashString(modelHash + "_" + inputHash)
	if v, ok := s.simpleCache.Load(ModelInputKey); ok && !s.config.IsNotCache {
		log.Debug("Infer Succeed via Cache", "result", v.([]byte))
		resCh <- v.([]byte)
		return
	}

	// lazy initialization of model cache
	if _, ok := s.caches[s.config.DeviceId]; !ok {
		s.caches[s.config.DeviceId] = lru.New(s.config.MaxMemoryUsage)
	}

	var (
		inferErr error
		result []byte
		model *kernel.Model
	)

	model_tmp, has_model := s.caches[s.config.DeviceId].Get(modelInfoHash)

	if !has_model{
		modelJson, modelJson_err := s.config.Storagefs.GetFile(modelHash, "/data/symbol")
		if modelJson_err != nil || modelJson == nil {
			errCh <- modelJson_err
		}
		modelParams, modelParams_err := s.config.Storagefs.GetFile(modelHash, "/data/params")
		if modelParams_err != nil || modelParams == nil {
			errCh <- ErrModelFileNotExist
		}
		model = kernel.New(s.lib, s.config.DeviceId, modelJson, modelParams)
		if model == nil {
			errCh <- errors.New("create model error " + modelHash)
			return
		}
		s.caches[s.config.DeviceId].Add(modelHash, model, model.Size())

	} else {
		model = model_tmp.(*kernel.Model)
	}

	result, inferErr = model.Predict(inputContent)
	if inferErr != nil {
		errCh <- inferErr
		return
	}

	if !s.config.IsNotCache {
		s.simpleCache.Store(ModelInputKey, result)
	}

	resCh <- result
	return

}
