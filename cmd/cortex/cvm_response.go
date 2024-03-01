package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	//"io"

	"github.com/CortexFoundation/inference"

	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/log"
)

const (
	MsgCorrect = "ok"
	MsgError   = "error"
)

var (
	ErrRequestMethodNotPost = errors.New("request method not POST")
	ErrRequestBodyRead      = errors.New("request body read error")

	ErrDataParse = errors.New("data parse error")

	ErrModelHashParse = errors.New("model info hash parse error")
	ErrModelEmpty     = errors.New("model info hash is empty")

	ErrInputHashParse    = errors.New("input info hash parse error")
	ErrInputEmpty        = errors.New("input info hash is empty")
	ErrInputAddressParse = errors.New("input content > contract address parse error")
	ErrInputSlotParse    = errors.New("input content > storage slot error")

	ErrInvalidInferTaskType = errors.New("unknown request property type")
)

func RespErrorText(w http.ResponseWriter, ctx ...any) {
	var info = ""
	if len(ctx)%2 != 0 {
		info = fmt.Sprintf("%v", ctx[0])
		ctx = ctx[1:]
	}

	for i := 0; i+1 < len(ctx); i += 2 {
		info += " %v=%v,"
		info = fmt.Sprintf(info, ctx[i], ctx[i+1])
	}

	var res = &inference.InferResult{
		Info: inference.RES_ERROR,
		Data: hexutil.Bytes(info),
	}

	data, err := json.Marshal(res)
	if err != nil {
		log.Error("Json marshal invalid", "err", err, "res", res)
		return
	}

	//fmt.Fprintf(w, string(data))
	//io.WriteString(w, string(data))
	w.Write(data)
}

func RespInfoText(w http.ResponseWriter, result []byte) {
	var res = &inference.InferResult{
		Info: inference.RES_OK,
		Data: hexutil.Bytes(result),
	}

	data, err := json.Marshal(res)
	if err != nil {
		log.Error("Json marshal invalid", "err", err, "res", res)
		return
	}

	//fmt.Fprintf(w, string(data))
	//io.WriteString(w, string(data))
	w.Write(data)
}
