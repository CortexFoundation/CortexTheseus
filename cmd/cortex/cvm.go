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
	"os"
	"os/user"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"errors"
	"github.com/CortexFoundation/CortexTheseus/cmd/utils"
	"github.com/CortexFoundation/CortexTheseus/inference/synapse"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/CortexFoundation/CortexTheseus/torrentfs"
	"gopkg.in/urfave/cli.v1"
	"net/http"
	"os/signal"
	"sync"
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
	StorageMaxSeedingFlag = cli.IntFlag{
		Name:  "cvm.max_seeding",
		Usage: "The maximum number of seeding tasks in the same time",
		Value: torrentfs.DefaultConfig.MaxSeedingNum,
	}
	StorageMaxActiveFlag = cli.IntFlag{
		Name:  "cvm.max_active",
		Usage: "The maximum number of active tasks in the same time",
		Value: torrentfs.DefaultConfig.MaxActiveNum,
	}
	StorageBoostNodesFlag = cli.StringFlag{
		Name:  "cvm.boostnodes",
		Usage: "p2p storage boostnodes",
		Value: strings.Join(torrentfs.DefaultConfig.BoostNodes, ","),
	}
	StorageTrackerFlag = cli.StringFlag{
		Name:  "cvm.tracker",
		Usage: "P2P storage tracker list",
		Value: strings.Join(torrentfs.DefaultConfig.DefaultTrackers, ","),
	}
	StorageDisableDHTFlag = cli.BoolFlag{
		Name:  "cvm.disable_dht",
		Usage: "disable DHT network in TorrentFS",
	}
	cvmFlags = []cli.Flag{
		// StorageDirFlag,
		CVMPortFlag,
		// CVMDeviceType,
		// CVMDeviceId,
		CVMVerbosity,
		CVMCortexDir,
		StorageMaxSeedingFlag,
		StorageMaxActiveFlag,
		StorageBoostNodesFlag,
		StorageTrackerFlag,
		StorageDisableDHTFlag,
	}

	cvmCommand = cli.Command{
		Action:      utils.MigrateFlags(cvmServer),
		Name:        "cvm",
		Usage:       "CVM",
		Flags:       append(append(cvmFlags, storageFlags...), inferFlags...),
		Category:    "CVMSERVER COMMANDS",
		Description: ``,
	}
)
var (
	wg sync.WaitGroup
	c  chan os.Signal
)

// localConsole starts a new cortex node, attaching a JavaScript console to it at the
// same time.
func cvmServer(ctx *cli.Context) error {
	c = make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, os.Kill)
	log.Root().SetHandler(log.LvlFilterHandler(log.Lvl(ctx.GlobalInt(CVMVerbosity.Name)), log.StreamHandler(os.Stdout, log.TerminalFormat(true))))

	fsCfg := torrentfs.DefaultConfig
	utils.SetTorrentFsConfig(ctx, &fsCfg)
	trackers := ctx.GlobalString(StorageTrackerFlag.Name)
	boostnodes := ctx.GlobalString(StorageBoostNodesFlag.Name)
	fsCfg.DefaultTrackers = strings.Split(trackers, ",")
	fsCfg.BoostNodes = strings.Split(boostnodes, ",")
	fsCfg.MaxSeedingNum = ctx.GlobalInt(StorageMaxSeedingFlag.Name)
	fsCfg.MaxActiveNum = ctx.GlobalInt(StorageMaxActiveFlag.Name)
	fsCfg.DataDir = ctx.GlobalString(utils.StorageDirFlag.Name)
	fsCfg.DisableDHT = ctx.GlobalBool(utils.StorageDisableDHTFlag.Name)
	fsCfg.IpcPath = filepath.Join(ctx.GlobalString(CVMCortexDir.Name), "cortex.ipc")
	log.Debug("Cvm Server", "fs", fsCfg, "storage", ctx.GlobalString(utils.StorageDirFlag.Name), "ipc path", fsCfg.IpcPath)
	storagefs, fs_err := torrentfs.New(&fsCfg, "")
	if fs_err != nil {
		return errors.New("fs start failed")
	}

	err := storagefs.Start(&p2p.Server{})
	if err != nil {
		return err
	}

	port := ctx.GlobalInt(CVMPortFlag.Name)
	DeviceType := ctx.GlobalString(utils.InferDeviceTypeFlag.Name)
	DeviceId := ctx.GlobalInt(utils.InferDeviceIdFlag.Name)

	DeviceName := "cpu"
	if DeviceType == "gpu" {
		DeviceName = "cuda"
	}
	synpapseConfig := synapse.Config{
		IsNotCache:     false,
		DeviceType:     DeviceName,
		DeviceId:       DeviceId,
		MaxMemoryUsage: synapse.DefaultConfig.MaxMemoryUsage,
		IsRemoteInfer:  false,
		InferURI:       "",
		Storagefs:      storagefs,
	}
	inferServer := synapse.New(&synpapseConfig)
	log.Info("Initilized inference server with synapse engine", "config", synpapseConfig)

	wg.Add(1)
	go func(port int, inferServer *synapse.Synapse) {
		defer wg.Done()
		log.Info("CVM http server listen on 0.0.0.0", "port", port, "uri", "/infer")
		mux := http.NewServeMux()
		mux.HandleFunc("/infer", handler)
		server := &http.Server{
			Addr:         ":" + strconv.Itoa(port),
			WriteTimeout: 15 * time.Second,
			ReadTimeout:  15 * time.Second,
			Handler:      mux,
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			server.ListenAndServe()
		}()
		select {
		case <-c:
			if err := server.Close(); err != nil {
				log.Info("Close http server failed", "err", err)
			} else {
				log.Info("CVM http server closed")
			}
			inferServer.Close()
		}
	}(port, inferServer)

	wg.Wait()

	return nil
}
