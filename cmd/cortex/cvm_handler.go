package main

import (
	"encoding/binary"
	"net/http"

	//"fmt"

	"github.com/CortexFoundation/inference"
	"github.com/CortexFoundation/inference/synapse"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
)

var IsNotCache bool = false

func Uint64ToBytes(i uint64) []byte {
	var buf = make([]byte, 8)
	binary.BigEndian.PutUint64(buf, uint64(i))
	return buf
}

/*func AvailableHandler(w http.ResponseWriter, inferWork *inference.AvailableWork) {
	log.Debug("Available", "Model Hash", inferWork.InfoHash, "rawSize", inferWork.RawSize)
	if inferWork.InfoHash == "" {
		RespErrorText(w, synapse.KERNEL_RUNTIME_ERROR)
		return
	}
	info := common.StorageEntry{
		Hash: inferWork.InfoHash,
		Size: inferWork.RawSize,
	}
	if err := synapse.Engine().Available(info, inferWork.CvmNetworkId); err != nil {
		RespErrorText(w, err)
	} else {
		retArr := Uint64ToBytes(1)
		log.Debug("File avaiable", "hash", inferWork.InfoHash)
		RespInfoText(w, retArr)
	}
}*/

func gasHandler(w http.ResponseWriter, inferWork *inference.GasWork) {
	log.Debug("Gas Task", "Model Hash", inferWork.Model)
	if inferWork.Model == "" {
		RespErrorText(w, synapse.KERNEL_RUNTIME_ERROR)
		return
	}
	model := common.StorageEntry{
		Hash: inferWork.Model,
		Size: inferWork.ModelSize,
	}
	ret, err := synapse.Engine().GetGasByInfoHashWithSize(model, inferWork.CvmNetworkId)
	if err != nil {
		RespErrorText(w, err)
		return
	}

	log.Debug("Gas calculate Succeed", "result", ret)
	retArr := Uint64ToBytes(ret)
	RespInfoText(w, retArr)
}

func infoHashHandler(w http.ResponseWriter, inferWork *inference.IHWork) {
	if inferWork.Model == "" {
		RespErrorText(w, synapse.KERNEL_RUNTIME_ERROR)
		return
	}
	if inferWork.Input == "" {
		RespErrorText(w, synapse.KERNEL_RUNTIME_ERROR)
		return
	}

	log.Debug("Infer Task", "Model Hash", inferWork.Model, "Input Hash", inferWork.Input)
	model := common.StorageEntry{
		Hash: inferWork.Model,
		Size: inferWork.ModelSize,
	}
	input := common.StorageEntry{
		Hash: inferWork.Input,
		Size: inferWork.InputSize,
	}
	label, err := synapse.Engine().InferByInfoHashWithSize(model, input, inferWork.CvmVersion, inferWork.CvmNetworkId)

	if err != nil {
		RespErrorText(w, err)
		return
	}
	RespInfoText(w, label)
}

func inputContentHandler(w http.ResponseWriter, inferWork *inference.ICWork) {
	if inferWork.Model == "" {
		RespErrorText(w, synapse.KERNEL_RUNTIME_ERROR)
		return
	}

	/*var cacheKey = synapse.RLPHashString(fmt.Sprintf("%s:%x", model, input))
	if v, ok := simpleCache.Load(cacheKey); ok && !(IsNotCache) {
		log.Debug("Infer succeed via cache", "cache key", cacheKey, "label", v.([]byte))
		RespInfoText(w, v.([]byte))
		return
	}*/

	// Fixed bugs, ctx_getSolidityBytes returns 0x which stands for state invalid
	// if len(input) == 0 {
	// 	log.Warn("Input content state invalid", "error", "bytes length is zero")
	// 	RespErrorText(w, synapse.KERNEL_RUNTIME_ERROR)
	// 	return
	// }

	model := common.StorageEntry{
		Hash: inferWork.Model,
		Size: inferWork.ModelSize,
	}

	label, err := synapse.Engine().InferByInputContentWithSize(
		model, inferWork.Input, inferWork.CvmVersion, inferWork.CvmNetworkId)
	if err != nil {
		RespErrorText(w, err)
		return
	}

	/*if !(IsNotCache) {
		simpleCache.Store(cacheKey, label)
	}*/

	RespInfoText(w, label)
}
