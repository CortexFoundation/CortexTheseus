// Copyright 2016 The go-ethereum Authors
// This file is part of The go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with The go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package node

import (
	"os"
	"os/user"
	"path/filepath"
	"runtime"

	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/CortexFoundation/CortexTheseus/p2p/nat"
	"github.com/CortexFoundation/CortexTheseus/rpc"
)

const (
	DefaultHTTPHost    = "localhost" // Default host interface for the HTTP RPC server
	DefaultHTTPPort    = 8545        // Default TCP port for the HTTP RPC server
	DefaultWSHost      = "localhost" // Default host interface for the websocket RPC server
	DefaultWSPort      = 8546        // Default TCP port for the websocket RPC server
	DefaultAuthHost    = "localhost" // Default host interface for the authenticated apis
	DefaultAuthPort    = 8551
	DefaultGraphQLHost = "localhost" // Default host interface for the GraphQL server
	DefaultGraphQLPort = 8547        // Default TCP port for the GraphQL server
)

var (
	DefaultAuthCors    = []string{"localhost"} // Default cors domain for the authenticated apis
	DefaultAuthVhosts  = []string{"localhost"} // Default virtual hosts for the authenticated apis
	DefaultAuthOrigins = []string{"localhost"} // Default origins for the authenticated apis
	DefaultAuthPrefix  = ""                    // Default prefix for the authenticated apis
	DefaultAuthModules = []string{"ctxc", "engine"}
)

// DefaultConfig contains reasonable default settings.
var DefaultConfig = Config{
	DataDir:             DefaultDataDir(),
	HTTPPort:            DefaultHTTPPort,
	AuthAddr:            DefaultAuthHost,
	AuthPort:            DefaultAuthPort,
	AuthVirtualHosts:    DefaultAuthVhosts,
	HTTPModules:         []string{"net", "web3"},
	HTTPVirtualHosts:    []string{"localhost"},
	HTTPTimeouts:        rpc.DefaultHTTPTimeouts,
	WSPort:              DefaultWSPort,
	WSModules:           []string{"net", "web3"},
	GraphQLPort:         DefaultGraphQLPort,
	GraphQLVirtualHosts: []string{"localhost"},
	P2P: p2p.Config{
		ListenAddr: ":40404",
		MaxPeers:   50,
		NAT:        nat.Any(),
	},
	DBEngine: "",
}

// DefaultDataDir is the default data directory to use for the databases and other
// persistence requirements.
func DefaultDataDir() string {
	// Try to place the data folder in the user's home dir
	home := homeDir()
	if home != "" {
		switch runtime.GOOS {
		case "darwin":
			return filepath.Join(home, "Library", "Cortex")
		case "windows":
			// We used to put everything in %HOME%\AppData\Roaming, but this caused
			// problems with non-typical setups. If this fallback location exists and
			// is non-empty, use it, otherwise DTRT and check %LOCALAPPDATA%.
			fallback := filepath.Join(home, "AppData", "Roaming", "Cortex")
			appdata := windowsAppData()
			if appdata == "" || isNonEmptyDir(fallback) {
				return fallback
			}
			return filepath.Join(appdata, "Cortex")
		default:
			return filepath.Join(home, ".cortex")
		}
	}
	// As we cannot guess a stable location, return empty and handle later
	return ""
}

// DefaultStorageDir is the default storage directory to use for the models and inputs.
func DefaultStorageDir(DataDir string) string {
	if DataDir == "" {
		DataDir = DefaultDataDir()
	}
	// Try to place the data folder in the user's home dir
	if DataDir != "" {
		return filepath.Join(DataDir, "storage")
	}
	// As we cannot guess a stable location, return empty and handle later
	return ""
}

func windowsAppData() string {
	v := os.Getenv("LOCALAPPDATA")
	if v == "" {
		// Windows XP and below don't have LocalAppData. Crash here because
		// we don't support Windows XP and undefining the variable will cause
		// other issues.
		panic("environment variable LocalAppData is undefined")
	}
	return v
}

func isNonEmptyDir(dir string) bool {
	f, err := os.Open(dir)
	if err != nil {
		return false
	}
	names, _ := f.Readdir(1)
	f.Close()
	return len(names) > 0
}

func homeDir() string {
	if home := os.Getenv("HOME"); home != "" {
		return home
	}
	if usr, err := user.Current(); err == nil {
		return usr.HomeDir
	}
	return ""
}
