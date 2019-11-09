package main

import (
	"encoding/binary"
	"fmt"
	"net/http"

	"github.com/CortexFoundation/CortexTheseus/inference"
	"github.com/CortexFoundation/CortexTheseus/inference/synapse"
	"github.com/CortexFoundation/CortexTheseus/log"
)

var IsNotCache bool = false

func Uint64ToBytes(i uint64) []byte {
	var buf = make([]byte, 8)
	binary.BigEndian.PutUint64(buf, uint64(i))
	return buf
}

func AvailableHandler(w http.ResponseWriter, inferWork *inference.AvailableWork) {
	log.Debug("Available", "Model Hash", inferWork.InfoHash, "rawSize", inferWork.RawSize)
	if inferWork.InfoHash == "" {
		log.Warn("info hash is empty")
		RespErrorText(w, synapse.KERNEL_RUNTIME_ERROR)
		return
	}

	if err := synapse.Engine().Available(inferWork.InfoHash, inferWork.RawSize); err != nil {
		RespErrorText(w, err)
	} else {
		ret_arr := Uint64ToBytes(1)
		log.Debug("File avaiable", "hash", inferWork.InfoHash)
		RespInfoText(w, ret_arr)
	}
}

func gasHandler(w http.ResponseWriter, inferWork *inference.GasWork) {
	log.Debug("Gas Task", "Model Hash", inferWork.Model)
	if inferWork.Model == "" {
		log.Warn("model info hash is empty")
		RespErrorText(w, synapse.KERNEL_RUNTIME_ERROR)
		return
	}

	ret, err := synapse.Engine().GetGasByInfoHash(inferWork.Model)
	if err != nil {
		log.Warn("Gas calculate Failed", "error", err)
		RespErrorText(w, err)
		return
	}

	log.Debug("Gas calculate Succeed", "result", ret)
	ret_arr := Uint64ToBytes(ret)
	RespInfoText(w, ret_arr)
}

func infoHashHandler(w http.ResponseWriter, inferWork *inference.IHWork) {
	if inferWork.Model == "" {
		log.Warn("model info hash is empty")
		RespErrorText(w, synapse.KERNEL_RUNTIME_ERROR)
		return
	}
	if inferWork.Input == "" {
		log.Warn("input info hash is empty")
		RespErrorText(w, synapse.KERNEL_RUNTIME_ERROR)
		return
	}

	log.Debug("Infer Task", "Model Hash", inferWork.Model, "Input Hash", inferWork.Input)
	label, err := synapse.Engine().InferByInfoHash(inferWork.Model, inferWork.Input)

	if err != nil {
		RespErrorText(w, err)
		return
	}
	RespInfoText(w, label)
}

func inputContentHandler(w http.ResponseWriter, inferWork *inference.ICWork) {
	if inferWork.Model == "" {
		log.Warn("model info hash is empty")
		RespErrorText(w, synapse.KERNEL_RUNTIME_ERROR)
		return
	}

	model, input := inferWork.Model, inferWork.Input

	log.Debug("Infer Work", "Model Hash", model)
	var cacheKey = synapse.RLPHashString(fmt.Sprintf("%s:%x", model, input))
	if v, ok := simpleCache.Load(cacheKey); ok && !(IsNotCache) {
		log.Debug("Infer succeed via cache", "cache key", cacheKey, "label", v.([]byte))
		RespInfoText(w, v.([]byte))
		return
	}

	// Fixed bugs, ctx_getSolidityBytes returns 0x which stands for state invalid
	// if len(input) == 0 {
	// 	log.Warn("Input content state invalid", "error", "bytes length is zero")
	// 	RespErrorText(w, synapse.KERNEL_RUNTIME_ERROR)
	// 	return
	// }

	label, err := synapse.Engine().InferByInputContent(model, input)
	if err != nil {
		log.Warn("Infer Failed", "error", err)
		RespErrorText(w, err)
		return
	}

	if !(IsNotCache) {
		simpleCache.Store(cacheKey, label)
	}

	RespInfoText(w, label)
}
