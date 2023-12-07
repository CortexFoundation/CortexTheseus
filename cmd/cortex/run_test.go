// Copyright 2018 The go-ethereum Authors
// This file is part of CortexFoundation.
//
// CortexFoundation is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// CortexFoundation is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with CortexFoundation. If not, see <http://www.gnu.org/licenses/>.

package main

import (
	"fmt"
	"os"
	"testing"

	"github.com/CortexFoundation/CortexTheseus/internal/cmdtest"
	"github.com/CortexFoundation/CortexTheseus/internal/reexec"
)

func tmpdir(t *testing.T) string {
	dir, err := os.MkdirTemp("", "cortex-test")
	if err != nil {
		t.Fatal(err)
	}
	return dir
}

type testcortex struct {
	*cmdtest.TestCmd

	// template variables for expect
	Datadir  string
	Coinbase string
}

func init() {
	// Run the app if we've been exec'd as "cortex-test" in runCtxc.
	reexec.Register("cortex-test", func() {
		if err := app.Run(os.Args); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	})
}

func TestMain(m *testing.M) {
	// check if we have been reexec'd
	if reexec.Init() {
		return
	}
	os.Exit(m.Run())
}

// spawns cortex with the given command line args. If the args don't set --datadir, the
// child g gets a temporary data directory.
func runCtxc(t *testing.T, args ...string) *testcortex {
	tt := &testcortex{}
	tt.TestCmd = cmdtest.NewTestCmd(t, tt)
	for i, arg := range args {
		switch {
		case arg == "-datadir" || arg == "--datadir":
			if i < len(args)-1 {
				tt.Datadir = args[i+1]
			}
		case arg == "-coinbase" || arg == "--coinbase":
			if i < len(args)-1 {
				tt.Coinbase = args[i+1]
			}
		}
	}
	if tt.Datadir == "" {
		tt.Datadir = tmpdir(t)
		tt.Cleanup = func() { os.RemoveAll(tt.Datadir) }
		args = append([]string{"-datadir", tt.Datadir}, args...)
		// Remove the temporary datadir if something fails below.
		defer func() {
			if t.Failed() {
				tt.Cleanup()
			}
		}()
	}

	// Boot "cortex". This actually runs the test binary but the TestMain
	// function will prevent any tests from running.
	tt.Run("cortex-test", args...)

	return tt
}
