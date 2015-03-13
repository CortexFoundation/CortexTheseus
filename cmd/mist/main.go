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
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/codegangsta/cli"
	"github.com/ethereum/go-ethereum/cmd/utils"
	"github.com/ethereum/go-ethereum/eth"
	"github.com/ethereum/go-ethereum/ethutil"
	"github.com/ethereum/go-ethereum/logger"
	"github.com/ethereum/go-ethereum/ui/qt/webengine"
	"github.com/obscuren/qml"
)

const (
	ClientIdentifier = "Mist"
	Version          = "0.9.0"
)

var (
	app           = utils.NewApp(Version, "the ether browser")
	assetPathFlag = cli.StringFlag{
		Name:  "asset_path",
		Usage: "absolute path to GUI assets directory",
		Value: ethutil.DefaultAssetPath(),
	}
)

func init() {
	app.Action = run
	app.Flags = []cli.Flag{
		assetPathFlag,

		utils.BootnodesFlag,
		utils.DataDirFlag,
		utils.ListenPortFlag,
		utils.LogFileFlag,
		utils.LogLevelFlag,
		utils.MaxPeersFlag,
		utils.MinerThreadsFlag,
		utils.NATFlag,
		utils.NodeKeyFileFlag,
		utils.RPCListenAddrFlag,
		utils.RPCPortFlag,
	}
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())

	// This is a bit of a cheat, but ey!
	os.Setenv("QTWEBKIT_INSPECTOR_SERVER", "127.0.0.1:99999")

	var interrupted = false
	utils.RegisterInterrupt(func(os.Signal) {
		interrupted = true
	})
	utils.HandleInterrupt()

	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, "Error: ", err)
	}

	// we need to run the interrupt callbacks in case gui is closed
	// this skips if we got here by actual interrupt stopping the GUI
	if !interrupted {
		utils.RunInterruptCallbacks(os.Interrupt)
	}
	logger.Flush()
}

func run(ctx *cli.Context) {
	tstart := time.Now()

	// TODO: show qml popup instead of exiting if initialization fails.
	cfg := utils.MakeEthConfig(ClientIdentifier, Version, ctx)
	ethereum, err := eth.New(cfg)
	if err != nil {
		utils.Fatalf("%v", err)
	}
	utils.StartRPC(ethereum, ctx)
	go utils.StartEthereum(ethereum)
	fmt.Println("initializing eth stack took", time.Since(tstart))

	// Open the window
	qml.Run(func() error {
		webengine.Initialize()
		gui := NewWindow(ethereum)
		utils.RegisterInterrupt(func(os.Signal) { gui.Stop() })
		// gui blocks the main thread
		gui.Start(ctx.GlobalString(assetPathFlag.Name))
		return nil
	})
}
