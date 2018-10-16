// +build !remote

package synapse

import (
	"errors"
	"strings"

	"github.com/ethereum/go-ethereum/infernet"
	"github.com/ethereum/go-ethereum/infernet/parser"
	"github.com/ethereum/go-ethereum/log"
)

func (s *Synapse) InferByInfoHash(modelInfoHash, inputInfoHash string) (uint64, error) {
	var (
		resCh = make(chan uint64)
		errCh = make(chan error)
	)

	go func() {
		s.inferByInfoHash(modelInfoHash, inputInfoHash, resCh, errCh)
	}()

	select {
	case result := <-resCh:
		return result, nil
	case err := <-errCh:
		return 0, err
	case <-s.exitCh:
		return 0, errors.New("Synapse Engine is closed")
	}
}

func (s *Synapse) inferByInfoHash(modelInfoHash, inputInfoHash string, resCh chan uint64, errCh chan error) {
	var (
		modelHash = strings.ToLower(modelInfoHash[2:])
		inputHash = strings.ToLower(string(inputInfoHash[2:]))
	)

	// Inference Cache
	cacheKey := modelHash + inputHash
	if v, ok := s.simpleCache.Load(cacheKey); ok && !s.config.IsNotCache {
		log.Debug("Infer Success via Cache", "result", v.(uint64))
		resCh <- v.(uint64)
		return
	}

	inputDir := s.config.StorageDir + "/" + inputHash
	inputFilePath := inputDir + "/data"
	log.Debug("Inference Core", "Read Image From File", inputFilePath)
	inputContent, imageErr := ReadImage(inputFilePath)
	if imageErr != nil {
		errCh <- imageErr
		return
	}

	s.inferByInputContent(modelInfoHash, inputInfoHash, inputContent, resCh, errCh)
}

func (s *Synapse) inferByInputContent(modelInfoHash, inputInfoHash string, inputContent []byte, resCh chan uint64, errCh chan error) {
	var (
		modelHash = strings.ToLower(modelInfoHash[2:])
		inputHash = strings.ToLower(inputInfoHash[2:])
		modelDir  = s.config.StorageDir + "/" + modelHash
	)

	if checkErr := CheckMetaHash(Model_V1, modelHash); checkErr != nil {
		errCh <- checkErr
		return
	}

	// Inference Cache
	cacheKey := modelHash + inputHash
	if v, ok := s.simpleCache.Load(cacheKey); ok && !s.config.IsNotCache {
		log.Debug("Infer Success via Cache", "result", v.(uint64))
		resCh <- v.(uint64)
		return
	}

	// Model Check
	modelCfg := modelDir + "/data/symbol"
	modelBin := modelDir + "/data/params"
	if parseErr := parser.CheckModel(modelCfg, modelBin); parseErr != nil {
		errCh <- parseErr
		return
	}

	log.Debug("Inference Core", "Model Config File", modelCfg, "Model Binary File", modelBin, "InputInfoHash", inputInfoHash)
	label, inferErr := infernet.InferCore(modelCfg, modelBin, inputContent)
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

func (s *Synapse) InferByInputContent(modelInfoHash string, inputContent []byte) (uint64, error) {
	var (
		resCh = make(chan uint64)
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
		return 0, err
	case <-s.exitCh:
		return 0, errors.New("Synapse Engine is closed")
	}
	return 0, nil
}
