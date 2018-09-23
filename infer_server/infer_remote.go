// +build remote

package infer_server

import (
	"errors"
	"sync"
	"sync/atomic"

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
	inferWork.err <- errors.New("localInfer not implemented")
	return
}
