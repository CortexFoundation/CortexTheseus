package main

import (
	"context"
	"net/http"

	"github.com/ethereum/go-ethereum/common/hexutil"
	infer "github.com/ethereum/go-ethereum/inference/synapse"
	"github.com/ethereum/go-ethereum/log"
)

func infoHashHandler(w http.ResponseWriter, inferWork *InferWork) {
	if inferWork.ModelHash == "" {
		RespErrorText(w, ErrModelEmpty)
		return
	}
	if inferWork.InputHash == "" {
		RespErrorText(w, ErrInputEmpty)
		return
	}

	log.Info("Infer Task", "Model Hash", inferWork.ModelHash, "Input Hash", inferWork.InputHash)
	label, err := infer.Engine().InferByInfoHash(inferWork.ModelHash, inferWork.InputHash)

	if err == nil {
		log.Info("Infer Succeed", "result", label)
	} else {
		log.Warn("Infer Failed", "error", err)
		RespErrorText(w, err)
		return
	}

	RespInfoText(w, label)
}

func inputContentHandler(w http.ResponseWriter, inferWork *InferWork) {
	if inferWork.ModelHash == "" {
		RespErrorText(w, ErrModelEmpty)
		return
	}

	addr, slot, number := inferWork.InputAddress, inferWork.InputSlot, inferWork.InputBlockNumber
	log.Info("Infer Work", "Model Hash", inferWork.ModelHash, "Input Address", addr, "Input Slot", slot, "Input Block Number", number)
	var cacheKey string
	if len(number) >= 2 && number[:2] == "0x" {
		cacheKey = infer.RLPHashString(inferWork.ModelHash + addr + slot + number)
		if v, ok := simpleCache.Load(cacheKey); ok && !(*IsNotCache) {
			RespInfoText(w, v.(uint64))
			return
		}
	}

	log.Debug("JSON-RPC request | ctx_getSolidityBytes", "address", addr, "slot", slot, "block number", number)
	var inputArray hexutil.Bytes
	if rpcErr := rpcClient.CallContext(context.Background(), &inputArray, "ctx_getSolidityBytes", addr, slot, number); rpcErr != nil {
		log.Warn("JSON-RPC request failed", "error", rpcErr)
		RespErrorText(w, "JSON-RPC invoke ctx_getSolidityBytes", "error", rpcErr)
		return
	}

	log.Debug("Infer Detail", "Input Content", inputArray.String())
	label, err := infer.Engine().InferByInputContent(inferWork.ModelHash, inputArray)

	if err == nil {
		log.Info("Infer Succeed", "result", label)
	} else {
		log.Warn("Infer Failed", "error", err)
		RespErrorText(w, err)
		return
	}

	if !(*IsNotCache) {
		simpleCache.Store(cacheKey, label)
	}

	RespInfoText(w, label)
}

func defaultHandler(w http.ResponseWriter, inferWork *InferWork) {
	RespErrorText(w, ErrInvalidInferTaskType, "type", inferWork.Type)
}
