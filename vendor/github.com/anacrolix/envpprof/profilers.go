package envpprof

import (
	"io"
	"os"
	"runtime"
	"runtime/pprof"
	"runtime/trace"

	"github.com/anacrolix/log"
)

type profiler interface {
	// Requiring name might be a vestige.
	start(name string)
	stop()
}

var profilers = map[string]profiler{
	"trace": newContinuousWriter(func(w io.Writer) (func() error, error) {
		return func() error { trace.Stop(); return nil }, trace.Start(w)
	}),
	"mutex": &pprofWrite{
		startFunc: func() {
			// Taken from Safe Rate at
			// https://github.com/DataDog/go-profiler-notes/blob/main/guide/README.md#go-profilers.
			runtime.SetMutexProfileFraction(100)
		},
	},
	"heap": &pprofWrite{},
	"block": &pprofWrite{
		startFunc: func() {
			// Taken from Safe Rate at
			// https://github.com/DataDog/go-profiler-notes/blob/main/guide/README.md#go-profilers.
			runtime.SetBlockProfileRate(10000)
		},
	},
	"cpu": newContinuousWriter(func(w io.Writer) (func() error, error) {
		return func() error { pprof.StopCPUProfile(); return nil }, pprof.StartCPUProfile(w)
	}),
}

func newContinuousWriter(start func(w io.Writer) (func() error, error)) profiler {
	return &continuousWriter{
		startWriter: start,
	}
}

// Continuous writers must be started at the beginning and be given a writer. They are told when to
// stop and (hopefully) flush.
type continuousWriter struct {
	profileName string
	startWriter func(io.Writer) (func() error, error)
	stopFunc    func() error
	// If not nil, the profiler is active.
	file *os.File
}

func (me *continuousWriter) start(name string) {
	if me.file != nil {
		return
	}
	me.file = newPprofFileOrLog(name)
	if me.file == nil {
		return
	}
	me.profileName = name
	var err error
	me.stopFunc, err = me.startWriter(me.file)
	if err != nil {
		log.Printf("error starting %v profiling: %v", name, err)
		me.file.Close()
		me.file = nil
		return
	}
	log.Printf("%v profiling to %q", name, me.file.Name())
}

func (me *continuousWriter) stop() {
	if me.file == nil {
		return
	}
	err := me.stopFunc()
	if err != nil {
		log.Printf("error stopping %v profiling: %v", me.file.Name(), err)
	}
	me.file.Close()
	logWroteProfile(me.file, me.profileName)
	me.file = nil
}

// These are builtin runtime/pprof profiles that need to be given reasonable configuration at the
// start, and their internal state written out when stopped.
type pprofWrite struct {
	// This is not "" if the profile type is enabled (and requested).
	pprofProfileName string
	startFunc        func()
}

func (me *pprofWrite) start(name string) {
	me.pprofProfileName = name
	if me.startFunc != nil {
		me.startFunc()
	}
}

func (me *pprofWrite) stop() {
	if me.pprofProfileName == "" {
		return
	}
	f := newPprofFileOrLog(me.pprofProfileName)
	if f == nil {
		return
	}
	defer f.Close()
	pprof.Lookup(me.pprofProfileName).WriteTo(f, 0)
	logWroteProfile(f, me.pprofProfileName)
}
