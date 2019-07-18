package main

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"sync"

	"github.com/CortexFoundation/CortexTheseus/inference"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/rpc"
)

var rpcClient *rpc.Client
var simpleCache sync.Map

func handler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		RespErrorText(w, ErrRequestMethodNotPost)
		return
	}

	body, rerr := ioutil.ReadAll(r.Body)
	if rerr != nil {
		RespErrorText(w, ErrRequestBodyRead)
		return
	}

	log.Trace("Handler Info", "request", r, "inference.RetriveType(body)", inference.RetriveType(body))

	switch inference.RetriveType(body) {
	case inference.INFER_BY_IH:
		var iw inference.IHWork
		if err := json.Unmarshal(body, &iw); err != nil {
			RespErrorText(w, ErrDataParse)
		}
		infoHashHandler(w, &iw)
		break

	case inference.INFER_BY_IC:
		var iw inference.ICWork
		if err := json.Unmarshal(body, &iw); err != nil {
			RespErrorText(w, ErrDataParse)
		}
		inputContentHandler(w, &iw)
		break

	case inference.GAS_BY_H:
		var iw inference.GasWork
		if err := json.Unmarshal(body, &iw); err != nil {
			RespErrorText(w, ErrDataParse)
		}
		gasHandler(w, &iw)
		break

	case inference.AVAILABLE_BY_H:
		var iw inference.AvailableWork
		if err := json.Unmarshal(body, &iw); err != nil {
			RespErrorText(w, ErrDataParse)
		}
		AvailableHandler(w, &iw)
		break

	default:
		RespErrorText(w, ErrInvalidInferTaskType)
		break
	}
}
