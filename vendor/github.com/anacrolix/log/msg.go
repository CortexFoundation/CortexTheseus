package log

import (
	"fmt"

	"github.com/anacrolix/missinggo/iter"
)

type Msg struct {
	MsgImpl
}

func newMsg(text string) Msg {
	return Msg{rootMsgImpl{text}}
}

func Fmsg(format string, a ...interface{}) Msg {
	return newMsg(fmt.Sprintf(format, a...))
}

var Fstr = Fmsg

func Str(s string) (m Msg) {
	return newMsg(s)
}

type msgSkipCaller struct {
	MsgImpl
	skip int
}

func (me msgSkipCaller) Callers(skip int, pc []uintptr) int {
	return me.MsgImpl.Callers(skip+1+me.skip, pc)
}

func (m Msg) Skip(skip int) Msg {
	return Msg{msgSkipCaller{m.MsgImpl, skip}}
}

type item struct {
	key, value interface{}
}

// rename sink
func (msg Msg) Log(l Logger) Msg {
	l.Log(msg.Skip(1))
	return msg
}

type msgWithValues struct {
	MsgImpl
	values []interface{}
}

func (me msgWithValues) Values(cb iter.Callback) {
	for _, v := range me.values {
		if !cb(v) {
			return
		}
	}
	me.MsgImpl.Values(cb)
}

func (me Msg) WithValues(v ...interface{}) Msg {
	return Msg{msgWithValues{me.MsgImpl, v}}
}

func (me Msg) AddValues(v ...interface{}) Msg {
	return me.WithValues(v...)
}

func (me Msg) With(key, value interface{}) Msg {
	return me.WithValues(item{key, value})
}

func (me Msg) Add(key, value interface{}) Msg {
	return me.With(key, value)
}

func (me Msg) HasValue(v interface{}) (has bool) {
	me.Values(func(i interface{}) bool {
		if i == v {
			has = true
		}
		return !has
	})
	return
}

func (me Msg) AddValue(v interface{}) Msg {
	return me.AddValues(v)
}
