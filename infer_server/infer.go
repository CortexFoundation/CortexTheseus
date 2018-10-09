// +build !remote

package infer_server

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/ethereum/go-ethereum/infernet"
	"github.com/ethereum/go-ethereum/infernet/parser"
	"github.com/ethereum/go-ethereum/log"
)

var globalInferServer *InferenceServer = nil

type InferWork struct {
	modelInfoHash string
	inputInfoHash string

	res chan uint64
	err chan error
}

type Config struct {
	StorageDir string
	IsNotCache bool
}

type InferenceServer struct {
	config Config

	inferSimpleCache sync.Map

	inferWorkCh chan *InferWork

	exitCh    chan struct{}
	stopInfer int32
}

func New(config Config) *InferenceServer {
	if globalInferServer != nil {
		log.Warn("Inference server has been initialized")
		return globalInferServer
	}

	globalInferServer = &InferenceServer{
		config:      config,
		inferWorkCh: make(chan *InferWork),
		exitCh:      make(chan struct{}),
		stopInfer:   0,
	}

	go globalInferServer.fetchWork()

	log.Info("Initializing Inference Server", "Storage Dir", config.StorageDir, "Cache Disabled", config.IsNotCache)
	return globalInferServer
}

func SubmitInferWork(modelHash, inputHash string, resCh chan uint64, errCh chan error) error {
	if globalInferServer == nil {
		return errors.New("Inference Server State Invalid")
	}

	return globalInferServer.submitInferWork(&InferWork{
		modelInfoHash: modelHash,
		inputInfoHash: inputHash,
		res:           resCh,
		err:           errCh,
	})
}

func (is *InferenceServer) submitInferWork(iw *InferWork) error {
	if stopSubmit := atomic.LoadInt32(&is.stopInfer) == 1; stopSubmit {
		return errors.New("Inference Server is closed")
	}

	is.inferWorkCh <- iw
	return nil
}

func (is *InferenceServer) Close() {
	atomic.StoreInt32(&is.stopInfer, 1)
	close(is.exitCh)
	log.Info("Global Inference Server Closed")
}

func (is *InferenceServer) fetchWork() {
	for {
		select {
		case inferWork := <-is.inferWorkCh:
			go func() {
				is.localInfer(inferWork)
			}()
		case <-is.exitCh:
			return
		}
	}
}

func (is *InferenceServer) localInfer(inferWork *InferWork) {
	modelHash := strings.ToLower(string(inferWork.modelInfoHash[2:]))
	inputHash := strings.ToLower(string(inferWork.inputInfoHash[2:]))

	modelDir := is.config.StorageDir + "/" + modelHash
	inputDir := is.config.StorageDir + "/" + inputHash

	if checkErr := CheckMetaHash(Model_V1, modelHash); checkErr != nil {
		inferWork.err <- checkErr
		return
	}

	// Inference Cache
	cacheKey := modelHash + inputHash
	log.Debug(fmt.Sprintf("InferWork: %v", inferWork))
	if v, ok := is.inferSimpleCache.Load(cacheKey); ok && !is.config.IsNotCache {
		inferWork.res <- v.(uint64)
		return
	}

	// Model Check
	modelCfg := modelDir + "/data/symbol"
	modelBin := modelDir + "/data/params"
	if parseErr := parser.CheckModel(modelCfg, modelBin); parseErr != nil {
		inferWork.err <- parseErr
		return
	}

	// Input Check
	image := inputDir + "/data"
	if _, imageErr := os.Stat(image); imageErr != nil {
		inferWork.err <- imageErr
		return
	}

	log.Debug("Inference Core", "Model Config File", modelCfg, "Model Binary File", modelBin, "Image", image)
	label, err := infernet.InferCore(modelCfg, modelBin, image)
	if err != nil {
		inferWork.err <- err
		return
	}

	inferWork.res <- label
	if !is.config.IsNotCache {
		is.inferSimpleCache.Store(cacheKey, label)
	}
	return
}
