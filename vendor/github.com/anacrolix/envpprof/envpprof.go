package envpprof

import (
	"expvar"
	"fmt"
	"net"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"weak"

	"github.com/anacrolix/log"
)

var (
	pprofDir = filepath.Join(os.Getenv("HOME"), "pprof")
)

// Stop ends CPU profiling, waiting for writes to complete. If heap profiling is enabled, it also
// writes the heap profile to a file. Stop should be deferred from main if cpu or heap profiling
// are to be used through envpprof.
func Stop() {
	stop()
}

// Replaced to track forgetting to Stop-by-GC.
var stop = func() {
	stopProfilers()
}

func stopProfilers() {
	for _, profiler := range profilers {
		profiler.stop()
	}
}

func startHTTP(value string) {
	var l net.Listener
	if value == "" {
		for port := uint16(6061); port != 6060; port++ {
			var err error
			l, err = net.Listen("tcp", fmt.Sprintf("localhost:%d", port))
			if err == nil {
				break
			}
		}
		if l == nil {
			log.Print("unable to create envpprof listener for http")
			return
		}
	} else {
		var addr string
		_, _, err := net.SplitHostPort(value)
		if err == nil {
			addr = value
		} else {
			addr = "localhost:" + value
		}
		l, err = net.Listen("tcp", addr)
		if err != nil {
			panic(err)
		}
	}
	log.Printf("(pid=%d) envpprof serving http://%s", os.Getpid(), l.Addr())
	go func() {
		defer l.Close()
		log.Printf("error serving http on envpprof listener: %s", http.Serve(l, nil))
	}()
}

var (
	forgotStopIfGCed  weak.Pointer[forgotStopValueType]
	cleanupForgotStop runtime.Cleanup
)

func init() {
	expvar.Publish("numGoroutine", expvar.Func(func() interface{} { return runtime.NumGoroutine() }))

	envValue := os.Getenv("GOPPROF")
	if envValue == "" {
		return
	}
	needStop := false
	for _, item := range strings.Split(envValue, ",") {
		equalsPos := strings.IndexByte(item, '=')
		var key, value string
		if equalsPos < 0 {
			key = item
		} else {
			key = item[:equalsPos]
			value = item[equalsPos+1:]
		}
		switch key {
		case "http":
			startHTTP(value)
		default:
			profiler, ok := profilers[key]
			if ok {
				profiler.start(key)
				needStop = true
			} else {
				log.Printf("unexpected GOPPROF key %q", key)
			}
		}
	}
	// This only installs the warning if profiling is enabled. But it could be for any consumer of
	// envpprof...
	if needStop {
		strong, cleanup := makeForget()
		forgotStopIfGCed = weak.Make(strong)
		cleanupForgotStop = cleanup
		stop = makeStopFunc(strong)
	}
}

func makeStopFunc(strong *forgotStopValueType) func() {
	return func() {
		cleanupForgotStop.Stop()
		stopProfilers()
		_ = strong
	}
}

// I suspect struct{} silently fails (or succeeds) as it might not be on the heap.
type forgotStopValueType = int

func makeForget() (*forgotStopValueType, runtime.Cleanup) {
	forgot := new(forgotStopValueType)
	return forgot, runtime.AddCleanup(
		forgot,
		func(struct{}) {
			log.Printf("envpprof: forgot to Stop()")
		},
		struct{}{},
	)
}

// Synchronous init that returns the cleanup function directly with no risk. Future proofing for a
// safer way to do it.
func Init() (stop func()) {
	// Extract the strong pointer created by the package init. Eventually we want to make it
	// synchronously here.
	strong := forgotStopIfGCed.Value()
	// Allocate a new func since that seems to work with GC better than a global variable.
	return makeStopFunc(strong)
}

// Runs main test suite with clean handled for you.
func TestMain(m *testing.M) {
	stop := Init()
	code := m.Run()
	stop()
	os.Exit(code)
}
