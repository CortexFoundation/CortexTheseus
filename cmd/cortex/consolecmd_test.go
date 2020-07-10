// Copyright 2018 The CortexTheseus Authors
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
	"crypto/rand"
	"math/big"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/CortexFoundation/CortexTheseus/params"
)

const (
	ipcAPIs  = "admin:1.0 debug:1.0 ctxc:1.0 ctxcash:1.0 miner:1.0 net:1.0 personal:1.0 rpc:1.0 shh:1.0 txpool:1.0 web3:1.0"
	httpAPIs = "cxtc:1.0 net:1.0 rpc:1.0 web3:1.0"
)

func TestConsoleWelcome(t *testing.T) {
	coinbase := "0x8605cdbbdb6d264aa742e77020dcbc58fcdce182"

	// Start a cortex console, make sure it's cleaned up and terminate the console
	cortex := runCtxc(t,
		"--port", "0", "--maxpeers", "0", "--nodiscover", "--nat", "none",
		"--coinbase", coinbase, "--shh",
		"console")

	// Gather all the infos the welcome message needs to contain
	cortex.SetTemplateFunc("goos", func() string { return runtime.GOOS })
	cortex.SetTemplateFunc("goarch", func() string { return runtime.GOARCH })
	cortex.SetTemplateFunc("gover", runtime.Version)
	cortex.SetTemplateFunc("cortexver", func() string { return params.VersionWithCommit("", "") })
	cortex.SetTemplateFunc("niltime", func() string {
		return time.Unix(0, 0).Format("Mon Jan 02 2006 15:04:05 GMT-0700 (MST)")
	})
	cortex.SetTemplateFunc("apis", func() string { return ipcAPIs })

	// Verify the actual welcome message to the required template
	cortex.Expect(`
Welcome to the cortex JavaScript console!

instance: cortex/v{{cortexver}}/{{goos}}-{{goarch}}/{{gover}}
coinbase: {{.Coinbase}}
at block: 0 ({{niltime}})
 datadir: {{.Datadir}}
 modules: {{apis}}

> {{.InputLine "exit"}}
`)
	cortex.ExpectExit()
}

// Tests that a console can be attached to a running node via various means.
func TestIPCAttachWelcome(t *testing.T) {
	// Configure the instance for IPC attachment
	coinbase := "0x8605cdbbdb6d264aa742e77020dcbc58fcdce182"
	var ipc string
	if runtime.GOOS == "windows" {
		ipc = `\\.\pipe\cortex` + strconv.Itoa(trulyRandInt(100000, 999999))
	} else {
		ws := tmpdir(t)
		defer os.RemoveAll(ws)
		ipc = filepath.Join(ws, "cortex.ipc")
	}
	// Note: we need --shh because testAttachWelcome checks for default
	// list of ipc modules and shh is included there.
	cortex := runCtxc(t,
		"--port", "0", "--maxpeers", "0", "--nodiscover", "--nat", "none",
		"--coinbase", coinbase, "--shh", "--ipcpath", ipc)

	defer func() {
		cortex.Interrupt()
		cortex.ExpectExit()
	}()

	waitForEndpoint(t, ipc, 3*time.Second)
	testAttachWelcome(t, cortex, "ipc:"+ipc, ipcAPIs)

}

func TestHTTPAttachWelcome(t *testing.T) {
	coinbase := "0x8605cdbbdb6d264aa742e77020dcbc58fcdce182"
	port := strconv.Itoa(trulyRandInt(1024, 65536)) // Yeah, sometimes this will fail, sorry :P
	cortex := runCtxc(t,
		"--port", "0", "--maxpeers", "0", "--nodiscover", "--nat", "none",
		"--coinbase", coinbase, "--rpc", "--rpcport", port)
	defer func() {
		cortex.Interrupt()
		cortex.ExpectExit()
	}()

	endpoint := "http://127.0.0.1:" + port
	waitForEndpoint(t, endpoint, 3*time.Second)
	testAttachWelcome(t, cortex, endpoint, httpAPIs)
}

func TestWSAttachWelcome(t *testing.T) {
	coinbase := "0x8605cdbbdb6d264aa742e77020dcbc58fcdce182"
	port := strconv.Itoa(trulyRandInt(1024, 65536)) // Yeah, sometimes this will fail, sorry :P

	cortex := runCtxc(t,
		"--port", "0", "--maxpeers", "0", "--nodiscover", "--nat", "none",
		"--coinbase", coinbase, "--ws", "--wsport", port)
	defer func() {
		cortex.Interrupt()
		cortex.ExpectExit()
	}()

	endpoint := "ws://127.0.0.1:" + port
	waitForEndpoint(t, endpoint, 3*time.Second)
	testAttachWelcome(t, cortex, endpoint, httpAPIs)
}

func testAttachWelcome(t *testing.T, cortex *testcortex, endpoint, apis string) {
	// Attach to a running cortex note and terminate immediately
	attach := runCtxc(t, "attach", endpoint)
	defer attach.ExpectExit()
	attach.CloseStdin()

	// Gather all the infos the welcome message needs to contain
	attach.SetTemplateFunc("goos", func() string { return runtime.GOOS })
	attach.SetTemplateFunc("goarch", func() string { return runtime.GOARCH })
	attach.SetTemplateFunc("gover", runtime.Version)
	attach.SetTemplateFunc("cortexver", func() string { return params.VersionWithMeta })
	attach.SetTemplateFunc("coinbase", func() string { return cortex.Coinbase })
	attach.SetTemplateFunc("niltime", func() string { return time.Unix(0, 0).Format(time.RFC1123) })
	attach.SetTemplateFunc("ipc", func() bool { return strings.HasPrefix(endpoint, "ipc") })
	attach.SetTemplateFunc("datadir", func() string { return cortex.Datadir })
	attach.SetTemplateFunc("apis", func() string { return apis })

	// Verify the actual welcome message to the required template
	attach.Expect(`
Welcome to the Ctxc JavaScript console!

instance: Ctxc/v{{cortexver}}/{{goos}}-{{goarch}}/{{gover}}
coinbase: {{coinbase}}
at block: 0 ({{niltime}}){{if ipc}}
 datadir: {{datadir}}{{end}}
 modules: {{apis}}

> {{.InputLine "exit" }}
`)
	attach.ExpectExit()
}

// trulyRandInt generates a crypto random integer used by the console tests to
// not clash network ports with other tests running cocurrently.
func trulyRandInt(lo, hi int) int {
	num, _ := rand.Int(rand.Reader, big.NewInt(int64(hi-lo)))
	return int(num.Int64()) + lo
}
