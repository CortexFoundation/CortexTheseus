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
	"errors"
	"math"
	"net/http"
	_ "net/http/pprof"
	"os"
	"os/signal"
	"os/user"
	"path/filepath"
	"runtime"
	godebug "runtime/debug"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/CortexFoundation/inference/synapse"
	"github.com/CortexFoundation/torrentfs"
	"github.com/CortexFoundation/torrentfs/params"
	gopsutil "github.com/shirou/gopsutil/mem"
	"gopkg.in/urfave/cli.v1"

	"github.com/CortexFoundation/CortexTheseus/cmd/utils"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/p2p"
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
		Value: utils.DirectoryString{Value: homeDir() + "/.cortex/"},
	}
	StorageMaxSeedingFlag = cli.IntFlag{
		Name:  "cvm.max_seeding",
		Usage: "The maximum number of seeding tasks in the same time",
		Value: params.DefaultConfig.MaxSeedingNum,
	}
	StorageMaxActiveFlag = cli.IntFlag{
		Name:  "cvm.max_active",
		Usage: "The maximum number of active tasks in the same time",
		Value: params.DefaultConfig.MaxActiveNum,
	}
	StorageBoostNodesFlag = cli.StringFlag{
		Name:  "cvm.boostnodes",
		Usage: "p2p storage boostnodes",
		Value: strings.Join(params.DefaultConfig.BoostNodes, ","),
	}
	StorageTrackerFlag = cli.StringFlag{
		Name:  "cvm.tracker",
		Usage: "P2P storage tracker list",
		Value: strings.Join(params.DefaultConfig.DefaultTrackers, ","),
	}
	StorageBoostFlag = cli.BoolFlag{
		Name:  "cvm.boost",
		Usage: "boost fs network",
	}
	StorageDisableTCPFlag = cli.BoolFlag{
		Name:  "cvm.disable_tcp",
		Usage: "disable TCP network",
	}
	StorageFullFlag = cli.BoolFlag{
		Name:  "cvm.full",
		Usage: "full file download",
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
		//StorageBoostNodesFlag,
		StorageTrackerFlag,
		//StorageDisableDHTFlag,
		//StorageFullFlag,
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
	c chan os.Signal
)

// localConsole starts a new cortex node, attaching a JavaScript console to it at the
// same time.
func cvmServer(ctx *cli.Context) error {
	if !ctx.GlobalIsSet(utils.CacheFlag.Name) && !ctx.GlobalIsSet(utils.NetworkIdFlag.Name) {
		// Make sure we're not on any supported preconfigured testnet either
		// Nope, we're really on mainnet. Bump that cache up!
		log.Info("Bumping default cache on mainnet", "provided", ctx.GlobalInt(utils.CacheFlag.Name), "updated", 4096)
		ctx.GlobalSet(utils.CacheFlag.Name, strconv.Itoa(4096))
	}
	mem, err := gopsutil.VirtualMemory()
	// Workaround until OpenBSD support lands into gosigar
	// Check https://github.com/elastic/gosigar#supported-platforms
	if err == nil {
		if 32<<(^uintptr(0)>>63) == 32 && mem.Total > 2*1024*1024*1024 {
			log.Warn("Lowering memory allowance on 32bit arch", "available", mem.Total/1024/1024, "addressable", 2*1024)
			mem.Total = 2 * 1024 * 1024 * 1024
		}
		allowance := int(mem.Total / 1024 / 1024 / 3)
		if cache := ctx.GlobalInt(utils.CacheFlag.Name); cache > allowance {
			log.Warn("Sanitizing cache to Go's GC limits", "provided", cache, "updated", allowance)
			ctx.GlobalSet(utils.CacheFlag.Name, strconv.Itoa(allowance))
		}
	} else {
		log.Warn("Memory total get failed", "err", err)
	}
	// Ensure Go's GC ignores the database cache for trigger percentage
	cache := ctx.GlobalInt(utils.CacheFlag.Name)
	gogc := math.Max(20, math.Min(100, 100/(float64(cache)/1024)))

	log.Info("Sanitizing Go's GC trigger", "percent", int(gogc), "cache", cache, "os", runtime.GOOS)
	godebug.SetGCPercent(int(gogc))

	c = make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	log.SetDefault(log.NewLogger(log.NewTerminalHandlerWithLevel(os.Stderr, log.LevelInfo, true)))

	fsCfg := params.DefaultConfig
	utils.SetTorrentFsConfig(ctx, &fsCfg)
	trackers := ctx.GlobalString(StorageTrackerFlag.Name)
	boostnodes := ctx.GlobalString(StorageBoostNodesFlag.Name)
	fsCfg.DefaultTrackers = strings.Split(trackers, ",")
	fsCfg.BoostNodes = strings.Split(boostnodes, ",")
	fsCfg.MaxSeedingNum = ctx.GlobalInt(StorageMaxSeedingFlag.Name)
	fsCfg.MaxActiveNum = ctx.GlobalInt(StorageMaxActiveFlag.Name)
	fsCfg.DataDir = ctx.GlobalString(utils.StorageDirFlag.Name)
	fsCfg.RpcURI = ctx.GlobalString(utils.StorageRpcFlag.Name)
	//fsCfg.DisableDHT = ctx.GlobalBool(utils.StorageDisableDHTFlag.Name)
	//fsCfg.DisableTCP = ctx.GlobalBool(utils.StorageDisableTCPFlag.Name)
	fsCfg.Mode = ctx.GlobalString(utils.StorageModeFlag.Name)
	//fsCfg.FullSeed = ctx.GlobalBool(utils.StorageFullFlag.Name)
	//if fsCfg.Mode == "full" {
	//	fsCfg.FullSeed = true
	//}
	fsCfg.Boost = ctx.GlobalBool(utils.StorageBoostFlag.Name)
	log.Warn("fsCfg.DataDir", "fsCfg.DataDir", fsCfg.DataDir)
	fsCfg.IpcPath = filepath.Join(ctx.GlobalString(CVMCortexDir.Name), "cortex.ipc")
	log.Debug("Cvm Server", "fs", fsCfg, "storage", ctx.GlobalString(utils.StorageDirFlag.Name), "ipc path", fsCfg.IpcPath)
	storagefs, fsErr := torrentfs.New(&fsCfg, false, false, true)
	if fsErr != nil {
		return errors.New("fs start failed")
	}

	err = storagefs.Start(&p2p.Server{})
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
	var wg sync.WaitGroup
	wg.Add(1)
	host := "127.0.0.1"
	go func(port int, inferServer *synapse.Synapse) {
		defer wg.Done()
		log.Info("CVM http server listen on "+host, "port", port, "uri", "/infer")
		mux := http.NewServeMux()
		mux.HandleFunc("/infer", handler)
		server := &http.Server{
			Addr:         host + ":" + strconv.Itoa(port),
			WriteTimeout: 15 * time.Second,
			ReadTimeout:  15 * time.Second,
			Handler:      mux,
		}
		//s1 := &http.Server{
		//	Addr:    ":6060",
		//	Handler: nil,
		//}
		//if fsCfg.FullSeed {
		//	wg.Add(1)
		//	go func() {
		//		defer wg.Done()
		//		runtime.SetMutexProfileFraction(1)
		//		runtime.SetBlockProfileRate(1)
		//		s1.ListenAndServe()
		//	}()
		//}

		wg.Add(1)
		go func() {
			defer wg.Done()
			server.ListenAndServe()
		}()
		//		select {
		//		case <-c:
		<-c
		//if fsCfg.FullSeed {
		//	if err := s1.Close(); err != nil {
		//		log.Info("Close resource server failed", "err", err)
		//	} else {
		//		log.Info("CVM resource server closed")
		//	}
		//}
		storagefs.Stop()
		if err := server.Close(); err != nil {
			log.Info("Close http server failed", "err", err)
		} else {
			log.Info("CVM http server closed")
		}
		inferServer.Close()
		//		}
	}(port, inferServer)

	wg.Wait()
	log.Info("CVM finally stop")

	return nil
}
