package log

import (
	"fmt"
	"io/ioutil"
	"os"
	"sync"
)

var (
	mu            sync.RWMutex
	defaultLogger = Logger{StreamLogger{
		W:   os.Stderr,
		Fmt: LineFormatter,
	}}
	Discard = Logger{StreamLogger{
		W:   ioutil.Discard,
		Fmt: func(Msg) []byte { return nil },
	}}
)

func Default() Logger {
	mu.RLock()
	defer mu.RUnlock()
	return defaultLogger
}

func SetDefault(l Logger) {
	mu.Lock()
	defer mu.Unlock()
	defaultLogger = l
}

// Prints the formatted arguments to the Default Logger.
func Printf(format string, a ...interface{}) {
	Default().Log(Fmsg(format, a...).Skip(1))
}

// Prints the arguments to the Default Logger.
func Print(a ...interface{}) {
	Str(fmt.Sprint(a...)).Skip(1).Log(Default())
}
