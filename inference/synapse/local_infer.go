package synapse

import (
	"strings"
	//"sync"

	"github.com/CortexFoundation/CortexTheseus/common/lru"
	"github.com/CortexFoundation/CortexTheseus/cvm-runtime/kernel"
	"github.com/CortexFoundation/CortexTheseus/inference"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/metrics"
)

const (
	DATA_PATH   string = "/data"
	SYMBOL_PATH string = "/data/symbol"
	PARAM_PATH  string = "/data/params"
)

var (
	gasCacheHitMeter  = metrics.NewRegisteredMeter("synapse/gascache/hit", nil)
	gasCacheMissMeter = metrics.NewRegisteredMeter("synapse/gascache/miss", nil)

	simpleCacheHitMeter  = metrics.NewRegisteredMeter("synapse/simplecache/hit", nil)
	simpleCacheMissMeter = metrics.NewRegisteredMeter("synapse/simplecache/miss", nil)
)

func getReturnByStatusCode(ret interface{}, status int) (interface{}, error) {
	switch status {
	case kernel.ERROR_RUNTIME:
		return nil, KERNEL_RUNTIME_ERROR
	case kernel.ERROR_LOGIC:
		return nil, KERNEL_LOGIC_ERROR
	case kernel.SUCCEED:
		return ret, nil
	}
	log.Warn("status code invalid", "code", status)
	return nil, KERNEL_RUNTIME_ERROR
}

func (s *Synapse) getGasByInfoHash(modelInfoHash string) (uint64, error) {

	if len(modelInfoHash) < 2 || !strings.HasPrefix(modelInfoHash, "0x") {
		return 0, KERNEL_RUNTIME_ERROR
	}

	modelHash := strings.ToLower(modelInfoHash[2:])

	cacheKey := RLPHashString("estimate_ops_" + modelHash)
	if v, ok := s.gasCache.Load(cacheKey); ok && !s.config.IsNotCache {
		log.Debug("Infer Success via Cache", "result", v.(uint64))
		gasCacheHitMeter.Mark(1)
		return v.(uint64), nil
	}

	modelJson, modelJson_err := s.config.Storagefs.GetFile(s.ctx, modelHash, SYMBOL_PATH)
	if modelJson_err != nil || modelJson == nil {
		log.Warn("GetGasByInfoHash: get file failed", "error", modelJson_err, "hash", modelInfoHash)
		return 0, KERNEL_RUNTIME_ERROR
	}

	//var status int
	gas, status := kernel.GetModelGasFromGraphFile(s.lib, modelJson)
	_, err := getReturnByStatusCode(gas, status)
	if err != nil {
		return 0, err
	}

	if !s.config.IsNotCache {
		gasCacheMissMeter.Mark(1)
		s.gasCache.Store(cacheKey, gas)
	}
	return gas, err
}

func (s *Synapse) inferByInfoHash(modelInfoHash, inputInfoHash string) ([]byte, error) {
	return s.infer(modelInfoHash, inputInfoHash, nil)
}

func (s *Synapse) inferByInputContent(modelInfoHash string, inputContent []byte) ([]byte, error) {
	return s.infer(modelInfoHash, "", inputContent)
}

func (s *Synapse) infer(modelInfoHash, inputInfoHash string, inputContent []byte) ([]byte, error) {
	if inputInfoHash == "" {
		inputInfoHash = RLPHashString(inputContent)
	}

	if len(modelInfoHash) < 2 || len(inputInfoHash) < 2 || !strings.HasPrefix(modelInfoHash, "0x") || !strings.HasPrefix(inputInfoHash, "0x") {
		return nil, KERNEL_RUNTIME_ERROR
	}

	var (
		modelHash = strings.ToLower(modelInfoHash[2:])
		inputHash = strings.ToLower(inputInfoHash[2:])
	)
	// Inference Cache
	cacheKey := RLPHashString(modelHash + "_" + inputHash)

	if hash, ok := CvmFixHashes[cacheKey]; ok {
		return hash, nil
	}

	if v, ok := s.simpleCache.Load(cacheKey); ok && !s.config.IsNotCache {
		log.Debug("Infer Succeed via Cache", "result", v.([]byte))
		simpleCacheHitMeter.Mark(1)
		return v.([]byte), nil
	}

	if inputContent == nil {
		inputBytes, dataErr := s.config.Storagefs.GetFile(s.ctx, inputHash, DATA_PATH)
		if dataErr != nil {
			return nil, KERNEL_RUNTIME_ERROR
		}
		reader, reader_err := inference.NewBytesReader(inputBytes)
		if reader_err != nil {
			return nil, KERNEL_LOGIC_ERROR
		}
		var read_data_err error
		inputContent, read_data_err = ReadData(reader)
		if read_data_err != nil {
			return nil, KERNEL_LOGIC_ERROR
		}
	}

	s.mutex.Lock()
	defer s.mutex.Unlock()
	// lazy initialization of model cache
	if _, ok := s.caches[s.config.DeviceId]; !ok {
		memoryUsage := s.config.MaxMemoryUsage
		if memoryUsage < MinMemoryUsage {
			memoryUsage = MinMemoryUsage
		}
		memoryUsage -= ReservedMemoryUsage
		log.Info("Memory alloc", "size", memoryUsage)
		s.caches[s.config.DeviceId] = lru.New(memoryUsage)
		s.caches[s.config.DeviceId].OnEvicted = func(key lru.Key, value interface{}) {
			log.Warn("C FREE On Evicted", "k", key, "size", value.(*kernel.Model).Size(), "max", s.config.MaxMemoryUsage, "min", MinMemoryUsage)
			value.(*kernel.Model).Free()
		}
	}

	var (
		result []byte
		model  *kernel.Model
		status int
	)

	//v, _ := s.modelLock.LoadOrStore(modelHash, sync.Mutex{})
	//mutex := v.(sync.Mutex)

	model_tmp, has_model := s.caches[s.config.DeviceId].Get(modelHash)
	if !has_model {
		modelJson, modelJson_err := s.config.Storagefs.GetFile(s.ctx, modelHash, SYMBOL_PATH)
		if modelJson_err != nil || modelJson == nil {
			log.Warn("inferByInputContent: model loaded failed", "model hash", modelHash, "error", modelJson_err)
			return nil, KERNEL_RUNTIME_ERROR
		}
		modelParams, modelParams_err := s.config.Storagefs.GetFile(s.ctx, modelHash, PARAM_PATH)
		if modelParams_err != nil || modelParams == nil {
			log.Warn("inferByInputContent: params loaded failed", "model hash", modelHash, "error", modelParams_err)
			return nil, KERNEL_RUNTIME_ERROR
		}
		var deviceType = 0
		if s.config.DeviceType == "cuda" {
			deviceType = 1
		}
		model, status = kernel.New(s.lib, modelJson, modelParams, deviceType, s.config.DeviceId)
		// TODO(wlt): all returned runtime_error
		if _, err := getReturnByStatusCode(model, status); err != nil {
			return nil, KERNEL_RUNTIME_ERROR
		}
		s.caches[s.config.DeviceId].Add(modelHash, model, int64(model.Size()))
	} else {
		model = model_tmp.(*kernel.Model)
	}
	log.Trace("iput content", "input", inputContent, "len", len(inputContent))
	result, status = model.Predict(inputContent)
	// TODO(wlt): all returned runtime_error
	if _, err := getReturnByStatusCode(result, status); err != nil {
		return nil, KERNEL_RUNTIME_ERROR
	}

	if !s.config.IsNotCache {
		simpleCacheMissMeter.Mark(1)
		s.simpleCache.Store(cacheKey, result)
	}

	return result, nil
}

func (s *Synapse) Available(infoHash string, rawSize int64) error {
	if s.config.IsRemoteInfer {
		errRes := s.remoteAvailable(
			infoHash,
			rawSize)
		//s.config.InferURI)
		return errRes
	}
	if len(infoHash) < 2 || !strings.HasPrefix(infoHash, "0x") {
		return KERNEL_RUNTIME_ERROR
	}
	ih := strings.ToLower(infoHash[2:])
	is_ok, err := s.config.Storagefs.Available(s.ctx, ih, rawSize)
	if err != nil {
		log.Debug("File verification failed", "infoHash", infoHash, "error", err)
		return KERNEL_RUNTIME_ERROR
	} else if !is_ok {
		log.Warn("File is unavailable",
			"info hash", infoHash, "error", KERNEL_LOGIC_ERROR)
		return KERNEL_LOGIC_ERROR
	}
	log.Debug("File available", "info hash", infoHash)
	return nil
}
