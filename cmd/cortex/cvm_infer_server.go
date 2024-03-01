package main

import (
	"io"
	"net/http"

	"github.com/CortexFoundation/inference"
	// "sync"
	// "github.com/CortexFoundation/CortexTheseus/rpc"
)

//var rpcClient *rpc.Client
//var simpleCache sync.Map

func handler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		RespErrorText(w, ErrRequestMethodNotPost)
		return
	}

	body, rerr := io.ReadAll(r.Body)
	if rerr != nil {
		RespErrorText(w, ErrRequestBodyRead)
		return
	}

	switch inference.RetriveType(body) {
	case inference.INFER_BY_IH:
		var iw inference.IHWork
		if err := iw.UnmarshalJSON(body); err != nil {
			RespErrorText(w, ErrDataParse)
		}
		infoHashHandler(w, &iw)

	case inference.INFER_BY_IC:
		var iw inference.ICWork
		if err := iw.UnmarshalJSON(body); err != nil {
			RespErrorText(w, ErrDataParse)
		}
		inputContentHandler(w, &iw)

	case inference.GAS_BY_H:
		var iw inference.GasWork
		if err := iw.UnmarshalJSON(body); err != nil {
			RespErrorText(w, ErrDataParse)
		}
		gasHandler(w, &iw)

	default:
		RespErrorText(w, ErrInvalidInferTaskType)
	}
}
