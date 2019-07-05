// Copyright 2016 The CortexFoundation Authors
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
	// "os"
	"fmt"

	"net/http"
	"gopkg.in/urfave/cli.v1"
	"github.com/CortexFoundation/CortexTheseus/cmd/utils"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/torrentfs"
	"github.com/CortexFoundation/CortexTheseus/inference/synapse"
)

var (
	StorageDirFlag = utils.DirectoryFlag{
		Name:  "cvm.dir",
		Usage: "P2P storage directory",
		Value: utils.DirectoryString{"~/.cortex/storage/"},
	}

	CVMPortFlag = cli.IntFlag{
		Name:  "cvm.port",
		Usage: "4321",
		Value: 4321,
	}

	CVMDeviceType = cli.StringFlag{
		Name:  "cvm.devicetype",
		Usage: "gpu or cpu",
		Value: "gpu",
	}

	CVMDeviceId = cli.IntFlag{
		Name:  "cvm.deviceid",
		Usage: "gpu id",
		Value: 0,
	}
	cvmFlags = []cli.Flag{
		StorageDirFlag,
		CVMPortFlag,
		CVMDeviceType,
		CVMDeviceId,
	}

	cvmCommand = cli.Command{
		Action:   utils.MigrateFlags(cvmServer),
		Name:     "cvm",
		Usage:    "CVM",
		Flags:    cvmFlags,
		Category: "CVMSERVER COMMANDS",
		Description: ``,
	}

)

// localConsole starts a new cortex node, attaching a JavaScript console to it at the
// same time.
func cvmServer(ctx *cli.Context) error {
	// flag.Parse()

	// Set log
	// log.Root().SetHandler(log.LvlFilterHandler(log.Lvl(*logLevel), log.StreamHandler(os.Stdout, log.TerminalFormat(true))))

	log.Info("Inference Server", "Help Command", "./infer_server -h")
	// torrentfs.New()
	storagefs := torrentfs.CreateStorage("simple", torrentfs.Config{
		DataDir:  ctx.GlobalString(StorageDirFlag.Name),
	})

	port := ctx.GlobalInt(CVMPortFlag.Name)
	DeviceType := ctx.GlobalString(CVMDeviceType.Name)
	DeviceId := ctx.GlobalInt(CVMDeviceId.Name)

	DeviceName := "cpu"

	if DeviceType == "gpu" {
		DeviceName = "cuda"
	}

	inferServer := synapse.New(&synapse.Config{
		// StorageDir: *storageDir,
		IsNotCache: false,
		DeviceType:  DeviceName,
		DeviceId: DeviceId,
		MaxMemoryUsage: synapse.DefaultConfig.MaxMemoryUsage,
		IsRemoteInfer: false,
		InferURI: "",
		Storagefs:				 storagefs,
	})
	log.Info("Initilized inference server with synapse engine")

	http.HandleFunc("/", handler)

	log.Info(fmt.Sprintf("Http Server Listen on 0.0.0.0:%d", port))
	err := http.ListenAndServe(fmt.Sprintf(":%d", port), nil)

	log.Error(fmt.Sprintf("Server Closed with Error %v", err))
	inferServer.Close()

	return nil
}

