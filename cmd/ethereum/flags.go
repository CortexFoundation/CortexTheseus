/*
	This file is part of go-ethereum

	go-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	go-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with go-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/**
 * @authors
 * 	Jeffrey Wilcke <i@jev.io>
 */
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/user"
	"path"

	"github.com/ethereum/go-ethereum/logger"
	"github.com/ethereum/go-ethereum/vm"
)

var (
	Identifier      string
	KeyRing         string
	DiffTool        bool
	DiffType        string
	KeyStore        string
	StartRpc        bool
	StartWebSockets bool
	RpcPort         int
	WsPort          int
	NatType         string
	PMPGateway      string
	OutboundPort    string
	ShowGenesis     bool
	AddPeer         string
	MaxPeer         int
	GenAddr         bool
	SeedNode        string
	SecretFile      string
	ExportDir       string
	NonInteractive  bool
	Datadir         string
	LogFile         string
	ConfigFile      string
	DebugFile       string
	LogLevel        int
	Dump            bool
	DumpHash        string
	DumpNumber      int
	VmType          int
	ImportChain     string
	SHH             bool
	Dial            bool
	PrintVersion    bool
)

// flags specific to cli client
var (
	StartMining    bool
	StartJsConsole bool
	InputFile      string
)

func defaultDataDir() string {
	usr, _ := user.Current()
	return path.Join(usr.HomeDir, ".ethereum")
}

var defaultConfigFile = path.Join(defaultDataDir(), "conf.ini")

func Init() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "%s [options] [filename]:\noptions precedence: default < config file < environment variables < command line\n", os.Args[0])
		flag.PrintDefaults()
	}

	flag.IntVar(&VmType, "vm", 0, "Virtual Machine type: 0-1: standard, debug")
	flag.StringVar(&Identifier, "id", "", "Custom client identifier")
	flag.StringVar(&KeyRing, "keyring", "", "identifier for keyring to use")
	flag.StringVar(&KeyStore, "keystore", "db", "system to store keyrings: db|file (db)")
	flag.StringVar(&OutboundPort, "port", "30303", "listening port")
	flag.StringVar(&NatType, "nat", "", "NAT support (UPNP|PMP) (none)")
	flag.StringVar(&PMPGateway, "pmp", "", "Gateway IP for PMP")
	flag.IntVar(&MaxPeer, "maxpeer", 30, "maximum desired peers")
	flag.IntVar(&RpcPort, "rpcport", 8545, "port to start json-rpc server on")
	flag.IntVar(&WsPort, "wsport", 40404, "port to start websocket rpc server on")
	flag.BoolVar(&StartRpc, "rpc", false, "start rpc server")
	flag.BoolVar(&StartWebSockets, "ws", false, "start websocket server")
	flag.BoolVar(&NonInteractive, "y", false, "non-interactive mode (say yes to confirmations)")
	flag.StringVar(&SeedNode, "seednode", "poc-8.ethdev.com:30303", "ip:port of seed node to connect to. Set to blank for skip")
	flag.BoolVar(&SHH, "shh", true, "whisper protocol (on)")
	flag.BoolVar(&Dial, "dial", true, "dial out connections (on)")
	flag.BoolVar(&GenAddr, "genaddr", false, "create a new priv/pub key")
	flag.StringVar(&SecretFile, "import", "", "imports the file given (hex or mnemonic formats)")
	flag.StringVar(&ExportDir, "export", "", "exports the session keyring to files in the directory given")
	flag.StringVar(&LogFile, "logfile", "", "log file (defaults to standard output)")
	flag.StringVar(&Datadir, "datadir", defaultDataDir(), "specifies the datadir to use")
	flag.StringVar(&ConfigFile, "conf", defaultConfigFile, "config file")
	flag.StringVar(&DebugFile, "debug", "", "debug file (no debugging if not set)")
	flag.IntVar(&LogLevel, "loglevel", int(logger.InfoLevel), "loglevel: 0-5: silent,error,warn,info,debug,debug detail)")
	flag.BoolVar(&DiffTool, "difftool", false, "creates output for diff'ing. Sets LogLevel=0")
	flag.StringVar(&DiffType, "diff", "all", "sets the level of diff output [vm, all]. Has no effect if difftool=false")
	flag.BoolVar(&ShowGenesis, "genesis", false, "Dump the genesis block")
	flag.StringVar(&ImportChain, "chain", "", "Imports given chain")

	flag.BoolVar(&Dump, "dump", false, "output the ethereum state in JSON format. Sub args [number, hash]")
	flag.StringVar(&DumpHash, "hash", "", "specify arg in hex")
	flag.IntVar(&DumpNumber, "number", -1, "specify arg in number")

	flag.BoolVar(&StartMining, "mine", false, "start dagger mining")
	flag.BoolVar(&StartJsConsole, "js", false, "launches javascript console")
	flag.BoolVar(&PrintVersion, "version", false, "prints version number")

	flag.Parse()

	if VmType >= int(vm.MaxVmTy) {
		log.Fatal("Invalid VM type ", VmType)
	}

	InputFile = flag.Arg(0)
}
