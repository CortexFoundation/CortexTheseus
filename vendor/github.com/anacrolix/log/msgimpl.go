package log

import (
	g "github.com/anacrolix/generics"
	"log/slog"
	"runtime"
)

type valueIterCallback func(value interface{}) (more bool)

// The minimal interface required for the Msg helper/wrapper to operate on.
type MsgImpl interface {
	// Returns the message text. Allows for lazy evaluation/prefixing etc.
	Text() string
	// Sets the program counters in pc. Having it in the interface may allow us to cache/freeze them
	// for serialization etc.
	Callers(skip int, pc []uintptr) int
	// Iterates over the values as added LIFO.
	Values(callback valueIterCallback)
	// Returns Some(slog.Record) if the Msg supports it.
	SlogRecord() g.Option[slog.Record]
}

// maybe implement finalizer to ensure msgs are sunk
type rootMsgImpl struct {
	text func() string
}

func (m rootMsgImpl) Text() string {
	return m.text()
}

func (m rootMsgImpl) Callers(skip int, pc []uintptr) int {
	return runtime.Callers(skip+2, pc)
}

func (m rootMsgImpl) Values(valueIterCallback) {}

func (m rootMsgImpl) SlogRecord() g.Option[slog.Record] {
	return g.Some(slog.Record{Message: m.text()})
}
