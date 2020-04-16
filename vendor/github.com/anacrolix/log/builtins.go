package log

import (
	"fmt"
	"io"
	"path/filepath"
	"runtime"
	"time"
)

type StreamLogger struct {
	W   io.Writer
	Fmt ByteFormatter
}

func (me StreamLogger) Log(msg Msg) {
	me.W.Write(me.Fmt(msg.Skip(1)))
}

type ByteFormatter func(Msg) []byte

func LineFormatter(msg Msg) []byte {
	var pc [1]uintptr
	msg.Callers(1, pc[:])
	ret := []byte(fmt.Sprintf(
		"%s %-5s %s: %s",
		time.Now().Format("2006-01-02T15:04:05-0700"),
		func() string {
			if level, ok := msg.GetLevel(); ok {
				return level.LogString()
			}
			return "NONE"
		}(),
		humanPc(pc[0]),
		msg.Text(),
	))
	if ret[len(ret)-1] != '\n' {
		ret = append(ret, '\n')
	}
	return ret
}

func humanPc(pc uintptr) string {
	if pc == 0 {
		panic(pc)
	}
	f, _ := runtime.CallersFrames([]uintptr{pc}).Next()
	return fmt.Sprintf("%s:%d", filepath.Base(f.File), f.Line)
}
