package log

import (
	"fmt"
	"io"
	"time"
)

type Handler interface {
	Emit(Msg)
}

type StreamHandler struct {
	W   io.Writer
	Fmt ByteFormatter
}

func (me StreamHandler) Emit(msg Msg) {
	me.W.Write(me.Fmt(msg))
}

type ByteFormatter func(Msg) []byte

func LineFormatter(msg Msg) []byte {
	ret := []byte(fmt.Sprintf(
		"%s %s: %s%s",
		time.Now().Format("2006-01-02 15:04:05"),
		humanPc(msg.callers[0]),
		msg.text,
		func() string {
			extras := groupExtras(msg.values, msg.fields)
			if len(extras) == 0 {
				return ""
			} else {
				return fmt.Sprintf(", %v", sortExtras(extras))
			}
		}(),
	))
	if ret[len(ret)-1] != '\n' {
		ret = append(ret, '\n')
	}
	return ret
}
