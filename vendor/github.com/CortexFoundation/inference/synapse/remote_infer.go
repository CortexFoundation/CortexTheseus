package synapse

import (
	"encoding/binary"
	//"encoding/json"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/inference"
)

func (s *Synapse) remoteGasByModelHashWithSize(modelInfoHash string, modelSize uint64, cvmNetworkID int64) (uint64, error) {
	inferWork := inference.GasWork{
		Type:         inference.GAS_BY_H,
		Model:        modelInfoHash,
		ModelSize:    modelSize,
		CvmNetworkId: cvmNetworkID,
	}

	requestBody, errMarshal := inferWork.MarshalJSON() //json.Marshal(inferWork)
	if errMarshal != nil {
		log.Warn("remote infer: marshal json failed", "body", inferWork, "error", errMarshal)
		return 0, KERNEL_RUNTIME_ERROR
	}
	log.Debug("remoteGasByModelHash", "request", string(requestBody))

	retArray, err := s.sendRequest(requestBody)
	if err != nil {
		return 0, err
	}
	return binary.BigEndian.Uint64(retArray), nil
}

// func (s *Synapse) remoteAvailable(infoHash string, rawSize int64, uri string) error {
func (s *Synapse) remoteAvailable(infoHash string, rawSize uint64, cvmNetworkID int64) error {
	inferWork := inference.AvailableWork{
		Type:         inference.AVAILABLE_BY_H,
		InfoHash:     infoHash,
		RawSize:      rawSize,
		CvmNetworkId: cvmNetworkID,
	}

	requestBody, errMarshal := inferWork.MarshalJSON() //json.Marshal(inferWork)
	if errMarshal != nil {
		log.Warn("remote infer: marshal json failed", "error", errMarshal)
		return KERNEL_RUNTIME_ERROR
	}
	log.Debug("remoteAvailable", "request", string(requestBody))

	_, err := s.sendRequest(requestBody)
	return err
}

func (s *Synapse) remoteInferByInfoHashWithSize(modelInfoHash, inputInfoHash string, modelSize uint64, inputSize uint64, cvmVersion int, cvmNetworkID int64) ([]byte, error) {
	inferWork := inference.IHWork{
		Type:         inference.INFER_BY_IH,
		Model:        modelInfoHash,
		Input:        inputInfoHash,
		ModelSize:    modelSize,
		InputSize:    inputSize,
		CvmVersion:   cvmVersion,
		CvmNetworkId: cvmNetworkID,
	}

	requestBody, err := inferWork.MarshalJSON() //json.Marshal(inferWork)
	if err != nil {
		return nil, KERNEL_RUNTIME_ERROR
	}
	log.Debug("remoteInferByInfoHash", "request", string(requestBody))

	return s.sendRequest(requestBody)
}

func (s *Synapse) remoteInferByInputContentWithSize(modelInfoHash string, inputContent []byte, modelSize uint64, cvmVersion int, cvmNetworkID int64) ([]byte, error) {
	inferWork := inference.ICWork{
		Type:         inference.INFER_BY_IC,
		Model:        modelInfoHash,
		Input:        hexutil.Bytes(inputContent),
		ModelSize:    modelSize,
		CvmVersion:   cvmVersion,
		CvmNetworkId: cvmNetworkID,
	}

	requestBody, err := inferWork.MarshalJSON() //json.Marshal(inferWork)
	if err != nil {
		log.Warn("remote infer: marshal json failed", "body", inferWork, "err", err)
		return nil, KERNEL_RUNTIME_ERROR
	}
	//log.Debug("remoteInferByInputContent", "request", string(requestBody)[:20])

	return s.sendRequest(requestBody)
}

func (s *Synapse) sendRequest(requestBody []byte) ([]byte, error) {
	/*cacheKey := RLPHashString(requestBody)
	if v, ok := s.simpleCache.Load(cacheKey); ok && !s.config.IsNotCache {
		log.Debug("Infer Succeed via Cache", "result", v.([]byte))
		return v.([]byte), nil
	}*/

	resp, err := s.client.SetTimeout(time.Duration(15*time.Second)).R().
		SetHeader("Content-Type", "application/json; charset=utf-8").
		SetHeader("Accept", "application/json; charset=utf-8").
		SetBody(requestBody).
		Post(s.config.InferURI)
	if err != nil || resp == nil {
		log.Warn("remote infer: request response failed", "error", err, "body", requestBody)
		return nil, KERNEL_RUNTIME_ERROR
	} else if resp.StatusCode() != 200 {
		log.Warn("remote infer: request response failed", "status code", resp.StatusCode())
		return nil, KERNEL_RUNTIME_ERROR
	}

	log.Debug("Remote Inference", "response", resp.String())

	var res inference.InferResult
	if jsErr := res.UnmarshalJSON(resp.Body()); jsErr != nil {
		//if jsErr := json.Unmarshal(resp.Body(), &res); jsErr != nil {
		log.Warn("remote infer: response json parsed failed", "error", jsErr)
		return nil, KERNEL_RUNTIME_ERROR
	}

	if res.Info == inference.RES_OK {
		var data = []byte(res.Data)
		/*if !s.config.IsNotCache {
			s.simpleCache.Store(cacheKey, data)
		}*/
		return data, nil
	}
	// res.Info == inference.RES_ERROR
	errStr := string(res.Data)

	if errStr == KERNEL_LOGIC_ERROR.Error() {
		return nil, KERNEL_LOGIC_ERROR
	}

	log.Debug("VM runtime error", "err", errStr, "req", requestBody)

	return nil, KERNEL_RUNTIME_ERROR
}
