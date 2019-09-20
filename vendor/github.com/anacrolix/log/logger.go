package log

import (
	"fmt"
)

type LoggerImpl interface {
	Log(Msg)
}

type LoggerFunc func(Msg)

func (me LoggerFunc) Log(m Msg) {
	me(m.Skip(2))
}

type Logger struct {
	LoggerImpl
}

func (l Logger) WithValues(v ...interface{}) Logger {
	return Logger{LoggerFunc(func(m Msg) {
		l.Log(m.WithValues(v...))
	})}
}

func (l Logger) WithFilter(f func(Msg) bool) Logger {
	return Logger{LoggerFunc(func(m Msg) {
		if f(m) {
			l.Log(m)
		}
	})}
}

// Helper for compatibility with "log".Logger.
func (l Logger) Printf(format string, a ...interface{}) {
	l.Log(Fmsg(format, a...).Skip(1))
}

// Helper for compatibility with "log".Logger.
func (l Logger) Print(v ...interface{}) {
	l.Log(Str(fmt.Sprint(v...)).Skip(1))
}
