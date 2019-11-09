package log

import (
	"runtime"

	"github.com/anacrolix/missinggo/iter"
)

type MsgImpl interface {
	Text() string
	Callers(skip int, pc []uintptr) int
	Values(callback iter.Callback)
}

// maybe implement finalizer to ensure msgs are sunk
type rootMsgImpl struct {
	text string
}

func (m rootMsgImpl) Text() string {
	return m.text
}

func (m rootMsgImpl) Callers(skip int, pc []uintptr) int {
	return runtime.Callers(skip+2, pc)
}

func (m rootMsgImpl) Values(iter.Callback) {}
