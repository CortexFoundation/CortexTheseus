package infernet

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ethereum/go-ethereum/log"
)

var globalInferServer *InferenceServer = nil

type InferWork struct {
	modelInfoHash string
	inputInfoHash string

	forcePending bool

	res chan uint64
	err chan error
}

type Config struct {
	StorageDir string
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
		return globalInferServer
	}

	globalInferServer = &InferenceServer{
		config:      config,
		inferWorkCh: make(chan *InferWork),
		exitCh:      make(chan struct{}),
		stopInfer:   0,
	}

	go globalInferServer.fetchWork()

	log.Info("Initialising Inference Server", "Storage Dir", config.StorageDir, "Global Inference Server", globalInferServer)
	return globalInferServer
}

func SubmitInferWork(modelHash, inputHash string, force bool, resCh chan uint64, errCh chan error) error {
	if globalInferServer == nil {
		return errors.New("Inference Server State Invalid")
	}

	return globalInferServer.submitInferWork(&InferWork{
		modelInfoHash: modelHash,
		inputInfoHash: inputHash,
		forcePending:  force,
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
	forcePending := inferWork.forcePending

	modelDir := is.config.StorageDir + "/" + modelHash
	inputDir := is.config.StorageDir + "/" + inputHash

	cacheKey := modelHash + inputHash

	// Inference Cache
	log.Debug(fmt.Sprintf("InferWork: %v", inferWork))
	if v, ok := is.inferSimpleCache.Load(cacheKey); ok {
		inferWork.res <- v.(uint64)
		return
	}

	// File Exists Check
	modelCfg := modelDir + "/data/params"
	if cfgError := is.checkFileExists(modelCfg, forcePending); cfgError != nil {
		inferWork.err <- cfgError
		return
	}

	modelBin := modelDir + "/data/symbol"
	if binError := is.checkFileExists(modelBin, forcePending); binError != nil {
		inferWork.err <- binError
		return
	}

	image := inputDir + "/data"
	if imageError := is.checkFileExists(image, forcePending); imageError != nil {
		inferWork.err <- imageError
		return
	}

	log.Debug("Infer Core", "Model Config File", modelCfg, "Model Binary File", modelBin, "Image", image)
	label, err := InferCore(modelCfg, modelBin, image)
	if err != nil {
		inferWork.err <- err
		return
	}

	inferWork.res <- label
	is.inferSimpleCache.Store(cacheKey, label)
	return
}

// blockIO with waiting for file sync done
func (is *InferenceServer) checkFileExists(fpath string, forcePending bool) error {
	for !FileExist(fpath) {
		if !forcePending {
			return errors.New(fmt.Sprintf("File %v does not exists", fpath))
		}

		stopPending := atomic.LoadInt32(&is.stopInfer) == 1
		if stopPending {
			return errors.New("Atomic stop pending")
		}

		log.Warn("Wait for file sync done", "File Name", fpath)
		time.Sleep(5 * time.Second)
	}

	return nil
}

// FileExist checks if a file exists at filePath.
func FileExist(filePath string) bool {
	_, err := os.Stat(filePath)
	if err != nil && os.IsNotExist(err) {
		return false
	}

	return true
}
