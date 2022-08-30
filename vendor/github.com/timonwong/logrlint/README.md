# logrlint

A linter checks the odd number of key and value pairs for common logger libraries:
- [logr](https://github.com/go-logr/logr)
- [klog](https://github.com/kubernetes/klog)

## Install

```shel
go install github.com/timonwong/logrlint/cmd/logrlint
```

## Usage

```shell
logrlint: Checks key valur pairs for common logger libraries (logr,klog,zap).

Usage: logrlint [-flag] [package]


Flags:
  -V    print version and exit
  -all
        no effect (deprecated)
  -c int
        display offending line with this many lines of context (default -1)
  -cpuprofile string
        write CPU profile to this file
  -debug string
        debug flags, any subset of "fpstv"
  -disable value
        comma-separated list of disabled logger checker (klog,logr,zap)
  -disableall
        disable all logger checkers
  -enable value
        comma-separated list of enabled logger checker (klog,logr,zap) (default klog,logr,zap)
  -fix
        apply all suggested fixes
  -flags
        print analyzer flags in JSON
  -json
        emit JSON output
  -memprofile string
        write memory profile to this file
  -source
        no effect (deprecated)
  -tags string
        no effect (deprecated)
  -test
        indicates whether test files should be analyzed, too (default true)
  -trace string
        write trace log to this file
  -v    no effect (deprecated)
```

## Example

```go
package a

import (
	"context"
	"fmt"

	"github.com/go-logr/logr"
)

func Example() {
	log := logr.Discard()
	log = log.WithValues("key")                              
	log.Info("message", "key1", "value1", "key2", "value2", "key3") 
	log.Error(fmt.Errorf("error"), "message", "key1", "value1", "key2")
	log.Error(fmt.Errorf("error"), "message", "key1", "value1", "key2", "value2")

	var log2 logr.Logger
	log2 = log
	log2.Info("message", "key1") 

	log3 := logr.FromContextOrDiscard(context.TODO())
	log3.Error(fmt.Errorf("error"), "message", "key1")
}
```

```
a.go:12:23: odd number of arguments passed as key-value pairs for logging
a.go:13:22: odd number of arguments passed as key-value pairs for logging
a.go:14:44: odd number of arguments passed as key-value pairs for logging
a.go:19:23: odd number of arguments passed as key-value pairs for logging
a.go:22:45: odd number of arguments passed as key-value pairs for logging
```