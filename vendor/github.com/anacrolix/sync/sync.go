package sync

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"runtime/pprof"
	"strings"
	"text/tabwriter"

	"github.com/anacrolix/missinggo/v2"
)

var (
	contentionOn = false
	lockTimesOn  = false
	// Current lock holders.
	lockHolders *pprof.Profile
	// Those blocked on acquiring a lock.
	lockBlockers *pprof.Profile
)

// Writes out the longest time a Mutex remains locked for each stack trace
// that locks a Mutex.
func PrintLockTimes(w io.Writer) {
	lockTimes := sortedLockTimes()
	tw := tabwriter.NewWriter(w, 1, 8, 1, '\t', 0)
	defer tw.Flush()
	w = tw
	for _, elem := range lockTimes {
		fmt.Fprintf(w, "%s (%s * %d [%s, %s])\n", elem.Total, elem.MeanTime(), elem.Count, elem.Min, elem.Max)
		missinggo.WriteStack(w, elem.stack[:])
	}
}

func Enable() {
	EnableContention()
	EnableLockTimes()
}

func EnableContention() {
	lockHolders = pprof.NewProfile("lockHolders")
	lockBlockers = pprof.NewProfile("lockBlockers")
	contentionOn = true
}

func EnableLockTimes() {
	lockStatsByStack = make(map[lockStackKey]lockStats)
	http.DefaultServeMux.HandleFunc("/debug/lockTimes", func(w http.ResponseWriter, r *http.Request) {
		PrintLockTimes(w)
	})
	lockTimesOn = true
}

func init() {
	env := os.Getenv("PPROF_SYNC")
	all := true
	if strings.Contains(env, "times") {
		EnableLockTimes()
		all = false
	}
	if strings.Contains(env, "contention") {
		EnableContention()
		all = false
	}
	if all && env != "" {
		Enable()
	}
}
