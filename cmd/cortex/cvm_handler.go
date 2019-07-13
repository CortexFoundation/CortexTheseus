package main

import (
	"errors"
	"fmt"
	"net/http"
	"encoding/binary"

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
	log.Info("Available", "Model Hash", inferWork.InfoHash, "rawSize", inferWork.RawSize)
	if inferWork.InfoHash == "" {
		RespErrorText(w, ErrModelEmpty)
		return
	}

	isAvailable := synapse.Engine().Available(inferWork.InfoHash, inferWork.RawSize)
	var ret uint64 = 1
	if !isAvailable {
		RespErrorText(w, errors.New(inferWork.InfoHash + " Not Available"))
	} else {
		ret_arr := Uint64ToBytes(ret)
		log.Info("Get Operators Succeed", "result", ret)
		RespInfoText(w, ret_arr)
	}
}

func gasHandler(w http.ResponseWriter, inferWork *inference.GasWork) {
	log.Info("Gas Task", "Model Hash", inferWork.Model)
	if inferWork.Model == "" {
		RespErrorText(w, ErrModelEmpty)
		return
	}

	ret, err := synapse.Engine().GetGasByInfoHash(inferWork.Model)
	ret_arr := Uint64ToBytes(ret)

	if err == nil {
		log.Info("Get Operators Succeed", "result", ret)
		RespInfoText(w, ret_arr)
	} else {
		log.Warn("Get Operators Failed", "error", err)
		RespErrorText(w, err)
	}
}


func infoHashHandler(w http.ResponseWriter, inferWork *inference.IHWork) {
	if inferWork.Model == "" {
		RespErrorText(w, ErrModelEmpty)
		return
	}
	if inferWork.Input == "" {
		RespErrorText(w, ErrInputEmpty)
		return
	}

	log.Debug("Infer Task", "Model Hash", inferWork.Model, "Input Hash", inferWork.Input)
	label, err := synapse.Engine().InferByInfoHash(inferWork.Model, inferWork.Input)

	if err == nil {
	//	log.Info("Infer Succeed", "result", label)
		RespInfoText(w, label)
	} else {
		log.Warn("Infer Failed", "error", err)
		RespErrorText(w, err)
	}
}

func inputContentHandler(w http.ResponseWriter, inferWork *inference.ICWork) {
	if inferWork.Model == "" {
		RespErrorText(w, ErrModelEmpty)
		return
	}

	model, input := inferWork.Model, inferWork.Input

	log.Info("Infer Work", "Model Hash", model)
	var cacheKey = synapse.RLPHashString(fmt.Sprintf("%s:%x", model, input))
	if v, ok := simpleCache.Load(cacheKey); ok && !(IsNotCache) {
	//	log.Info("Infer succeed via cache", "cache key", cacheKey, "label", v.([]byte))
		RespInfoText(w, v.([]byte))
		return
	}

	// Fixed bugs, ctx_getSolidityBytes returns 0x which stands for state invalid
	if len(input) == 0 {
		log.Warn("Input content state invalid", "error", "bytes length is zero")
		RespErrorText(w, "input bytes length is zero")
		return
	}

	label, err := synapse.Engine().InferByInputContent(model, input)

	if err != nil {
		log.Warn("Infer Failed", "error", err)
		RespErrorText(w, err)
		return
	}

	// log.Info("Infer Succeed", "result", label)
	if !(IsNotCache) {
		simpleCache.Store(cacheKey, label)
	}

	RespInfoText(w, label)
}
