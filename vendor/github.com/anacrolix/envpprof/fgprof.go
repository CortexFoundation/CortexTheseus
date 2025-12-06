package envpprof

import (
	"io"

	g "github.com/anacrolix/generics"
	"github.com/anacrolix/missinggo/v2/panicif"
	"github.com/felixge/fgprof"
)

// We have to do this to reliably run before init(). Plus as a bonus you can check from other code
// if it ran when you expected.
var registeredFgprof = func() bool {
	g.MapMustAssignNew(
		profilers,
		"fgprof",
		newContinuousWriter(func(w io.Writer) (func() error, error) {
			stop := fgprof.Start(w, fgprof.FormatPprof)
			return func() error {
				err := stop()
				println("stopped fgprof")
				panicif.Err(err)
				return err
			}, nil
		}))
	return true
}()
