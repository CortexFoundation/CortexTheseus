package log

import (
	"fmt"
)

// LoggerImpl is the minimal interface for Logger.
type LoggerImpl interface {
	Log(Msg)
}

// LoggerFunc is a helper type that implements LoggerImpl from just a logging function.
type LoggerFunc func(Msg)

func (me LoggerFunc) Log(m Msg) {
	// Skip 1 for this function, and 1 for me.
	me(m.Skip(2))
}

// Logger is a helper wrapping LoggerImpl.
type Logger struct {
	LoggerImpl
}

// Returns a logger that adds the given values to logged messages.
func (l Logger) WithValues(v ...interface{}) Logger {
	return Logger{LoggerFunc(func(m Msg) {
		l.Log(m.WithValues(v...))
	})}
}

// Returns a new logger that suppresses further propagation for messages if `f` returns false.
func (l Logger) WithFilter(f func(Msg) bool) Logger {
	return Logger{LoggerFunc(func(m Msg) {
		if f(m) {
			l.Log(m)
		}
	})}
}

// Returns a logger that for a given message propagates the result of `f` instead.
func (l Logger) WithMap(f func(Msg) Msg) Logger {
	return Logger{LoggerFunc(func(m Msg) {
		l.Log(f(m))
	})}
}

func (l Logger) WithText(f func(Msg) string) Logger {
	return Logger{LoggerFunc(func(m Msg) {
		l.Log(m.WithText(f))
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

func (l Logger) WithDefaultLevel(level Level) Logger {
	return l.WithMap(func(m Msg) Msg {
		_, ok := m.GetLevel()
		if !ok {
			m = m.SetLevel(level)
		}
		return m
	})
}

func (l Logger) FilterLevel(minLevel Level) Logger {
	return l.WithFilter(func(m Msg) bool {
		level, ok := m.GetLevel()
		return !ok || !level.LessThan(minLevel)
	})
}
