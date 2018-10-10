// +build !remote

package synapse

import (
	"errors"
	"strings"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/crypto/sha3"
	"github.com/ethereum/go-ethereum/infernet"
	"github.com/ethereum/go-ethereum/infernet/parser"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/rlp"
)

func (s Synapse) InferByInfoHash(modelInfoHash, inputInfoHash string) (uint64, error) {
	var (
		resCh <-chan uint64
		errCh <-chan error
	)

	go func() {
		resCh, errCh = s.inferByInfoHash(modelInfoHash, inputInfoHash)
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

func (s Synapse) inferByInfoHash(modelInfoHash, inputInfoHash string) (<-chan uint64, <-chan error) {
	var (
		resCh = make(chan uint64)
		errCh = make(chan error)

		inputHash = strings.ToLower(string(inputInfoHash[2:]))
		inputDir  = s.config.StorageDir + "/" + inputHash
	)

	inputFilePath := inputDir + "/data"
	log.Debug("Inference Core", "Read Image From File", inputFilePath)
	inputContent, imageErr := ReadImage(inputFilePath)
	if imageErr != nil {
		errCh <- imageErr
		return resCh, errCh
	}

	return s.inferByInputContent(modelInfoHash, inputInfoHash, inputContent)
}

func (s Synapse) inferByInputContent(modelInfoHash, inputInfoHash string, inputContent []byte) (<-chan uint64, <-chan error) {
	var (
		resCh = make(chan uint64)
		errCh = make(chan error)

		modelHash = strings.ToLower(modelInfoHash[2:])
		inputHash = strings.ToLower(inputInfoHash[2:])
		modelDir  = s.config.StorageDir + "/" + modelHash
	)

	if checkErr := CheckMetaHash(Model_V1, modelHash); checkErr != nil {
		errCh <- checkErr
		return resCh, errCh
	}

	// Inference Cache
	cacheKey := modelHash + inputHash
	if v, ok := s.simpleCache.Load(cacheKey); ok && !s.config.IsNotCache {
		resCh <- v.(uint64)
		return resCh, errCh
	}

	// Model Check
	modelCfg := modelDir + "/data/symbol"
	modelBin := modelDir + "/data/params"
	if parseErr := parser.CheckModel(modelCfg, modelBin); parseErr != nil {
		errCh <- parseErr
		return resCh, errCh
	}

	log.Debug("Inference Core", "Model Config File", modelCfg, "Model Binary File", modelBin, "InputInfoHash", inputInfoHash)
	label, inferErr := infernet.InferCore(modelCfg, modelBin, inputContent)
	if inferErr != nil {
		errCh <- inferErr
		return resCh, errCh
	}

	if !s.config.IsNotCache {
		s.simpleCache.Store(cacheKey, label)
	}

	resCh <- label
	return resCh, errCh

}

func (s Synapse) InferByInputContent(modelInfoHash string, inputContent []byte) (uint64, error) {
	var (
		resCh <-chan uint64
		errCh <-chan error

		hash common.Hash
	)

	hw := sha3.NewKeccak256()
	rlp.Encode(hw, inputContent)
	inputInfoHash := hexutil.Encode(hw.Sum(hash[:0]))

	go func() {
		resCh, errCh = s.inferByInputContent(modelInfoHash, inputInfoHash, inputContent)
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
