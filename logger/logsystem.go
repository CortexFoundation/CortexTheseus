package logger

import (
	"io"
	"log"
	"sync/atomic"
)

// LogSystem is implemented by log output devices.
// All methods can be called concurrently from multiple goroutines.
type LogSystem interface {
	GetLogLevel() LogLevel
	SetLogLevel(i LogLevel)
	LogPrint(LogLevel, string)
}

// NewStdLogSystem creates a LogSystem that prints to the given writer.
// The flag values are defined package log.
func NewStdLogSystem(writer io.Writer, flags int, level LogLevel) LogSystem {
	logger := log.New(writer, "", flags)
	return &stdLogSystem{logger, uint32(level)}
}

type stdLogSystem struct {
	logger *log.Logger
	level  uint32
}

func (t *stdLogSystem) LogPrint(level LogLevel, msg string) {
	t.logger.Print(msg)
}

func (t *stdLogSystem) SetLogLevel(i LogLevel) {
	atomic.StoreUint32(&t.level, uint32(i))
}

func (t *stdLogSystem) GetLogLevel() LogLevel {
	return LogLevel(atomic.LoadUint32(&t.level))
}

// NewRawLogSystem creates a LogSystem that prints to the given writer without
// adding extra information. Suitable for preformatted output
func NewRawLogSystem(writer io.Writer, flags int, level LogLevel) LogSystem {
	logger := log.New(writer, "", 0)
	return &rawLogSystem{logger, uint32(level)}
}

type rawLogSystem struct {
	logger *log.Logger
	level  uint32
}

func (t *rawLogSystem) LogPrint(level LogLevel, msg string) {
	t.logger.Print(msg)
}

func (t *rawLogSystem) SetLogLevel(i LogLevel) {
	atomic.StoreUint32(&t.level, uint32(i))
}

func (t *rawLogSystem) GetLogLevel() LogLevel {
	return LogLevel(atomic.LoadUint32(&t.level))
}

// NewRawLogSystem creates a LogSystem that prints to the given writer without
// adding extra information. Suitable for preformatted output
func NewJsonLogSystem(writer io.Writer, flags int, level LogLevel) LogSystem {
	logger := log.New(writer, "", 0)
	return &jsonLogSystem{logger, uint32(level)}
}

type jsonLogSystem struct {
	logger *log.Logger
	level  uint32
}

func (t *jsonLogSystem) LogPrint(level LogLevel, msg string) {
	t.logger.Print(msg)
}

func (t *jsonLogSystem) SetLogLevel(i LogLevel) {
	atomic.StoreUint32(&t.level, uint32(i))
}

func (t *jsonLogSystem) GetLogLevel() LogLevel {
	return LogLevel(atomic.LoadUint32(&t.level))
}
