package synapse

import (
	// "io/ioutil"
	// "os"
	// "path/filepath"
	"strings"
	//"sync"
	"context"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/lru/legacy"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/metrics"
	"github.com/CortexFoundation/cvm-runtime/kernel"
	"github.com/CortexFoundation/inference"
	"time"

	gopsutil "github.com/shirou/gopsutil/mem"
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

func fixTorrentHash(ih string, cvmNetworkID int64) string {
	if cvmNetworkID == 43 {
		if ch, ok := CvmDolFixTorrHashes[ih]; ok {
			log.Debug("start hacking hash", "ih", ih, "ch", ch, "cvmNetworkID", cvmNetworkID)
			return ch
		}
	}
	return ih
}

func (s *Synapse) getGasByInfoHashWithSize(modelInfoHash string, modelSize uint64, cvmNetworkID int64) (uint64, error) {
	if !common.IsHexAddress(modelInfoHash) {
		return 0, KERNEL_RUNTIME_ERROR
	}

	modelHash := strings.ToLower(strings.TrimPrefix(modelInfoHash, common.Prefix))
	modelHash = fixTorrentHash(modelHash, cvmNetworkID)

	cacheKey := RLPHashString("estimate_ops_" + modelHash)
	if v, ok := s.gasCache.Load(cacheKey); ok && !s.config.IsNotCache {
		log.Debug("Infer Success via Cache", "result", v.(uint64))
		gasCacheHitMeter.Mark(1)
		return v.(uint64), nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	modelJson, modelJson_err := s.config.Storagefs.GetFileWithSize(ctx, modelHash, modelSize, SYMBOL_PATH)
	if modelJson_err != nil || modelJson == nil {
		log.Debug("Searching file for gas", "ih", modelInfoHash, "error", modelJson_err)
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

func (s *Synapse) inferByInfoHashWithSize(modelInfoHash, inputInfoHash string, modelSize uint64, inputSize uint64, cvmVersion int, cvmNetworkID int64) ([]byte, error) {
	return s.infer(modelInfoHash, inputInfoHash, nil, modelSize, inputSize, cvmVersion, cvmNetworkID)
}

func (s *Synapse) inferByInputContentWithSize(modelInfoHash string, inputContent []byte, modelSize uint64, cvmVersion int, cvmNetworkID int64) ([]byte, error) {
	return s.infer(modelInfoHash, "", inputContent, modelSize, 0, cvmVersion, cvmNetworkID)
}

func (s *Synapse) infer(modelInfoHash, inputInfoHash string, inputContent []byte, modelSize uint64, inputSize uint64, cvmVersion int, cvmNetworkID int64) ([]byte, error) {
	if inputInfoHash == "" {
		inputInfoHash = RLPHashString(inputContent)
	} else {
		if !common.IsHexAddress(inputInfoHash) {
			return nil, KERNEL_RUNTIME_ERROR
		}
	}

	if !common.IsHexAddress(modelInfoHash) {
		return nil, KERNEL_RUNTIME_ERROR
	}

	var (
		modelHash = strings.ToLower(strings.TrimPrefix(modelInfoHash, common.Prefix))
		inputHash = strings.ToLower(strings.TrimPrefix(inputInfoHash, common.Prefix))
	)
	modelHash = fixTorrentHash(modelHash, cvmNetworkID)
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

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if inputContent == nil {
		inputBytes, dataErr := s.config.Storagefs.GetFileWithSize(ctx, inputHash, inputSize, DATA_PATH)
		if dataErr != nil || inputBytes == nil {
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
		if mem, err := gopsutil.VirtualMemory(); err == nil {
			allowance := int64(mem.Total / 2)
			log.Warn("Memory detected", "allowance", common.StorageSize(allowance))
			if memoryUsage > allowance {
				memoryUsage = allowance
			}
		}

		if memoryUsage < MinMemoryUsage {
			log.Warn("8G+ Memory is suggested to run the full node fluently")
			memoryUsage = MinMemoryUsage
		}

		memoryUsage -= ReservedMemoryUsage
		log.Info("Memory alloc", "size", common.StorageSize(memoryUsage))
		s.caches[s.config.DeviceId] = legacy.New(memoryUsage)
		s.caches[s.config.DeviceId].OnEvicted = func(key legacy.Key, value interface{}) {
			log.Warn("C FREE On Evicted", "k", key, "size", common.StorageSize(value.(*kernel.Model).Size()), "free", common.StorageSize(memoryUsage), "max", common.StorageSize(s.config.MaxMemoryUsage), "min", common.StorageSize(MinMemoryUsage))
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
		modelJson, modelJson_err := s.config.Storagefs.GetFileWithSize(ctx, modelHash, modelSize, SYMBOL_PATH)
		if modelJson_err != nil || modelJson == nil {
			log.Debug("Searching symbol", "ih", modelHash, "error", modelJson_err)
			return nil, KERNEL_RUNTIME_ERROR
		}
		modelParams, modelParams_err := s.config.Storagefs.GetFileWithSize(ctx, modelHash, modelSize, PARAM_PATH)
		if modelParams_err != nil || modelParams == nil {
			log.Debug("Searching params", "ih", modelHash, "error", modelParams_err)
			return nil, KERNEL_RUNTIME_ERROR
		}
		var deviceType = 0
		if s.config.DeviceType == "cuda" {
			deviceType = 1
		}
		model, status = kernel.New(s.lib, modelJson, modelParams, deviceType, s.config.DeviceId)
		if _, err := getReturnByStatusCode(model, status); err != nil {
			return nil, KERNEL_RUNTIME_ERROR
		}
		log.Info("AI memory used", "size", common.StorageSize(int64(model.Size())))
		s.caches[s.config.DeviceId].Add(modelHash, model, int64(model.Size()))
	} else {
		model = model_tmp.(*kernel.Model)
	}

	result, status = model.Predict(inputContent, cvmVersion)
	if _, err := getReturnByStatusCode(result, status); err != nil {
		return nil, KERNEL_RUNTIME_ERROR
	}

	if !s.config.IsNotCache {
		simpleCacheMissMeter.Mark(1)
		s.simpleCache.Store(cacheKey, result)
	}

	return result, nil
}

/*func (s *Synapse) available(infoHash string, rawSize uint64, cvmNetworkID int64) error {
	if !common.IsHexAddress(infoHash) {
		return KERNEL_RUNTIME_ERROR
	}
	ih := strings.ToLower(strings.TrimPrefix(infoHash, common.Prefix))
	if cvmNetworkID == 43 {
		if _, ok := CvmDolFixTorrHashes[ih]; ok {
			log.Debug("Available: start hacking...", "ih", ih, "cvmNetworkID", cvmNetworkID)
			return nil
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	isOK, err := s.config.Storagefs.Available(ctx, ih, rawSize)
	if err != nil {
		log.Debug("File verification failed", "infoHash", infoHash, "error", err)
		return KERNEL_RUNTIME_ERROR
	} else if !isOK {
		log.Debug("File is unavailable", "ih", infoHash, "error", KERNEL_LOGIC_ERROR)
		return KERNEL_LOGIC_ERROR
	}
	log.Debug("File available", "info hash", infoHash)
	return nil
}*/

func (s *Synapse) download(infohash string, request uint64) error {
	if !common.IsHexAddress(infohash) {
		return KERNEL_RUNTIME_ERROR //errors.New("Invalid infohash format")
	}
	ih := strings.TrimPrefix(strings.ToLower(infohash), common.Prefix)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	err := s.config.Storagefs.Download(ctx, ih, request)
	if err != nil {
		return KERNEL_RUNTIME_ERROR
	}

	return nil
}

func (s *Synapse) seedingLocal(filePath string, isLinkMode bool) (ih string, err error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	ih, err = s.config.Storagefs.SeedingLocal(ctx, filePath, isLinkMode)
	if err != nil {
		log.Error("synapse", "SeedingLocal", err.Error())
		return ih, KERNEL_RUNTIME_ERROR
	}

	return
}

func (s *Synapse) pauseLocalSeedFile(ih string) (err error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	err = s.config.Storagefs.PauseLocalSeed(ctx, ih)
	if err != nil {
		log.Error("synapse", "PauseLocalSeed", err.Error())
		return KERNEL_RUNTIME_ERROR
	}

	return
}

func (s *Synapse) resumeLocalSeedFile(ih string) (err error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	err = s.config.Storagefs.ResumeLocalSeed(ctx, ih)
	if err != nil {
		log.Error("synapse", "ResumeLocalSeed", err.Error())
		return KERNEL_RUNTIME_ERROR
	}

	return
}

func (s *Synapse) listAllTorrents() map[string]map[string]int {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return s.config.Storagefs.ListAllTorrents(ctx)
}
