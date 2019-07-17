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
	"os"
	"os/user"
	"fmt"
	"path/filepath"

	"net/http"
	"gopkg.in/urfave/cli.v1"
	"github.com/CortexFoundation/CortexTheseus/cmd/utils"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/torrentfs"
	"github.com/CortexFoundation/CortexTheseus/inference/synapse"
)

func homeDir() string {
	if home := os.Getenv("HOME"); home != "" {
		return home
	}
	if usr, err := user.Current(); err == nil {
		return usr.HomeDir
	}
	return ""
}
var (
	// StorageDirFlag = utils.DirectoryFlag{
	// 	Name:  "cvm.dir",
	// 	Usage: "P2P storage directory",
	// 	Value: utils.DirectoryString{"~/.cortex/storage/"},
	// }

	CVMPortFlag = cli.IntFlag{
		Name:  "cvm.port",
		Usage: "4321",
		Value: 4321,
	}

	CVMVerbosity = cli.IntFlag{
		Name:  "cvm.verbosity",
		Usage: "verbose level",
		Value: 3,
	}

	//value,_ := utils.DirectoryString("~/.cortex/cortex.ipc")
	CVMCortexDir = utils.DirectoryFlag{
		Name:  "cvm.datadir",
		Usage: "cortex fulllnode dir",
		//Value: utils.DirectoryString("~/.cortex/" + "cortex.ipc"),
		Value: utils.DirectoryString{homeDir() + "/.cortex/"},
	}

	cvmFlags = []cli.Flag{
		// StorageDirFlag,
		CVMPortFlag,
		// CVMDeviceType,
		// CVMDeviceId,
		CVMVerbosity,
		CVMCortexDir,
	}

	cvmCommand = cli.Command{
		Action:   utils.MigrateFlags(cvmServer),
		Name:     "cvm",
		Usage:    "CVM",
		Flags:    append(append(cvmFlags, storageFlags...), inferFlags...),
		Category: "CVMSERVER COMMANDS",
		Description: ``,
	}

)

// localConsole starts a new cortex node, attaching a JavaScript console to it at the
// same time.
func cvmServer(ctx *cli.Context) error {
	// flag.Parse()

	// Set log
	log.Root().SetHandler(log.LvlFilterHandler(log.Lvl(ctx.GlobalInt(CVMVerbosity.Name)), log.StreamHandler(os.Stdout, log.TerminalFormat(true))))

	fsCfg := torrentfs.DefaultConfig
	utils.SetTorrentFsConfig(ctx, &fsCfg)
	fsCfg.DataDir = ctx.GlobalString(utils.StorageDirFlag.Name)
	fsCfg.IpcPath = filepath.Join(ctx.GlobalString(CVMCortexDir.Name), "cortex.ipc")
	log.Info("cvmServer", "torrentfs.Config", fsCfg, "StorageDirFlag.Name", ctx.GlobalString(utils.StorageDirFlag.Name), "ipc path", fsCfg.IpcPath)
	storagefs, fs_err := torrentfs.New(&fsCfg, "")
	storagefs.Start(nil)
	if fs_err != nil {
		panic (fs_err)
	}
	port := ctx.GlobalInt(CVMPortFlag.Name)
	DeviceType := ctx.GlobalString(utils.InferDeviceTypeFlag.Name)
	DeviceId := ctx.GlobalInt(utils.InferDeviceIdFlag.Name)

	DeviceName := "cpu"
	if DeviceType == "gpu" {
		DeviceName = "cuda"
	}
	synpapseConfig := synapse.Config{
		IsNotCache: false,
		DeviceType:  DeviceName,
		DeviceId: DeviceId,
		MaxMemoryUsage: synapse.DefaultConfig.MaxMemoryUsage,
		IsRemoteInfer: false,
		InferURI: "",
		Storagefs:				 storagefs,
	}
	inferServer := synapse.New(&synpapseConfig)
	log.Info("Initilized inference server with synapse engine", "config", synpapseConfig)

	http.HandleFunc("/", handler)

	log.Info(fmt.Sprintf("Http Server Listen on 0.0.0.0:%d", port))
	err := http.ListenAndServe(fmt.Sprintf(":%d", port), nil)

	log.Error(fmt.Sprintf("Server Closed with Error %v", err))
	inferServer.Close()

	return nil
}

