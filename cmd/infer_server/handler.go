package main

import (
	"context"
	"fmt"
	"net/http"

	infer "github.com/ethereum/go-ethereum/inference/synapse"
	"github.com/ethereum/go-ethereum/log"
)

func infoHashHandler(w http.ResponseWriter, inferWork *InferWork) {
	if inferWork.ModelHash == "" || inferWork.InputHash == "" {
		fmt.Fprintf(w, `{"msg": "error", "info": "Data is empty"}`)
		return
	}

	log.Info("Infer Work", "Model Hash", inferWork.ModelHash, "Input Hash", inferWork.InputHash)

	label, err := infer.Engine().InferByInfoHash(inferWork.ModelHash, inferWork.InputHash)
	log.Info("Infer Result", "label", label, "error", err)

	if err != nil {
		fmt.Fprintf(w, fmt.Sprintf(`{"msg": "error", "info": "%v"}`, err))
		return
	}

	fmt.Fprintf(w, fmt.Sprintf(`{"msg": "ok", "info": "%v"}`, label))
}

func inputContentHandler(w http.ResponseWriter, inferWork *InferWork) {
	if inferWork.ModelHash == "" {
		fmt.Fprintf(w, `{"msg": "error", "info": "ModelHash is empty"}`)
		return
	}

	log.Info("Infer Work", "Model Hash", inferWork.ModelHash, "Input Address", inferWork.InputAddress, "Input Slot", inferWork.InputSlot)

	addr, slot := inferWork.InputAddress, inferWork.InputSlot
	if len(addr) != 2*20+2 {
		fmt.Fprintf(w, `{"msg": "error", "info": "Invalid InputAddress Length"}`)
		return
	}
	if len(slot) != 2*32+2 {
		fmt.Fprintf(w, `{"msg": "error", "info": "Invalid InputSlot Length"}`)
		return
	}

	var inputArray []byte
	if rpcErr := rpcClient.CallContext(context.Background(), &inputArray, "ctx_getSolidityBytes", addr, slot, "latest"); rpcErr != nil {
		fmt.Fprintf(w, fmt.Sprintf(`{"msg": "error", "info": "JSONRPC ctx_getSoildityBytes error: %v"}`, rpcErr))
		return
	}

	label, err := infer.Engine().InferByInputContent(inferWork.ModelHash, inputArray)
	log.Info("Infer Result", "label", label, "error", err)

	if err != nil {
		fmt.Fprintf(w, fmt.Sprintf(`{"msg": "error", "info": "%v"}`, err))
		return
	}

	fmt.Fprintf(w, fmt.Sprintf(`{"msg": "ok", "info": "%v"}`, label))
}

func defaultHandler(w http.ResponseWriter, inferWork *InferWork) {
	fmt.Fprintf(w, fmt.Sprintf(`{"msg": "error", "info": "unknown request property Type %v"}`), inferWork.Type)
}
