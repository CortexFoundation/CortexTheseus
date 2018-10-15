package main

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
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

const (
	RespPattern = `{"msg": "%v", "info": "%v"}`
)

func RespErrorText(w http.ResponseWriter, ctx ...interface{}) {
	var info = ""
	if len(ctx)%2 != 0 {
		info = fmt.Sprintf("%v | ", ctx[0])
		ctx = ctx[1:]
	}

	for i := 0; i+1 < len(ctx); i += 2 {
		info += "%v=%v, "
		info = fmt.Sprintf(info, ctx[i], ctx[i+1])
	}

	fmt.Fprintf(w, fmt.Sprintf(RespPattern, MsgError, strings.TrimSuffix(strings.TrimSuffix(info, ", "), " | ")))
}

func RespInfoText(w http.ResponseWriter, result interface{}) {
	fmt.Fprintf(w, fmt.Sprintf(RespPattern, MsgCorrect, result))
}
