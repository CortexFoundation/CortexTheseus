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
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"reflect"
	"strconv"
	"strings"
	"sync"
	//"time"
	"unicode"

	cli "gopkg.in/urfave/cli.v1"

	"github.com/CortexFoundation/CortexTheseus/cmd/utils"
	"github.com/CortexFoundation/CortexTheseus/ctxc"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/node"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/CortexTheseus/torrentfs"
	"github.com/naoina/toml"
)

var (
	dumpConfigCommand = cli.Command{
		Action:      utils.MigrateFlags(dumpConfig),
		Name:        "dumpconfig",
		Usage:       "Show configuration values",
		ArgsUsage:   "",
		Flags:       append(append(nodeFlags, rpcFlags...), whisperFlags...),
		Category:    "MISCELLANEOUS COMMANDS",
		Description: `The dumpconfig command shows configuration values.`,
	}

	configFileFlag = cli.StringFlag{
		Name:  "config",
		Usage: "TOML configuration file",
	}
)

// These settings ensure that TOML keys use the same names as Go struct fields.
var tomlSettings = toml.Config{
	NormFieldName: func(rt reflect.Type, key string) string {
		return key
	},
	FieldToKey: func(rt reflect.Type, field string) string {
		return field
	},
	MissingField: func(rt reflect.Type, field string) error {
		link := ""
		if unicode.IsUpper(rune(rt.Name()[0])) && rt.PkgPath() != "main" {
			link = fmt.Sprintf(", see https://godoc.org/%s#%s for available fields", rt.PkgPath(), rt.Name())
		}
		return fmt.Errorf("field '%s' is not defined in %s%s", field, rt.String(), link)
	},
}

type ctxcstatsConfig struct {
	URL string `toml:",omitempty"`
}

type cortexConfig struct {
	Cortex      ctxc.Config
	Node        node.Config
	Cortexstats ctxcstatsConfig
	// Dashboard dashboard.Config
	TorrentFs torrentfs.Config
}

func loadConfig(file string, cfg *cortexConfig) error {
	f, err := os.Open(file)
	if err != nil {
		return err
	}
	defer f.Close()

	err = tomlSettings.NewDecoder(bufio.NewReader(f)).Decode(cfg)
	// Add file name to errors that have a line number.
	if _, ok := err.(*toml.LineError); ok {
		err = errors.New(file + ", " + err.Error())
	}
	return err
}

func defaultNodeConfig() node.Config {
	cfg := node.DefaultConfig
	cfg.Name = clientIdentifier
	cfg.Version = params.VersionWithCommit(gitCommit)
	cfg.HTTPModules = append(cfg.HTTPModules, "ctxc", "shh")
	cfg.WSModules = append(cfg.WSModules, "ctxc", "shh")
	cfg.IPCPath = "cortex.ipc"
	return cfg
}

func makeConfigNode(ctx *cli.Context) (*node.Node, cortexConfig) {
	// Load defaults.
	cfg := cortexConfig{
		Cortex: ctxc.DefaultConfig,
		Node:   defaultNodeConfig(),
		// Dashboard: dashboard.DefaultConfig,
		TorrentFs: torrentfs.DefaultConfig,
	}

	// Load config file.
	if file := ctx.GlobalString(configFileFlag.Name); file != "" {
		if err := loadConfig(file, &cfg); err != nil {
			utils.Fatalf("%v", err)
		}
	}

	// Apply flags.
	utils.SetNodeConfig(ctx, &cfg.Node)
	stack, err := node.New(&cfg.Node)
	if err != nil {
		utils.Fatalf("Failed to create the protocol stack: %v", err)
	}
	utils.SetCortexConfig(ctx, stack, &cfg.Cortex)
	// if ctx.GlobalIsSet(utils.CortexStatsURLFlag.Name) {
	// 	cfg.Cortexstats.URL = ctx.GlobalString(utils.CortexStatsURLFlag.Name)
	// }

	//utils.SetShhConfig(ctx, stack, &cfg.Shh)
	// utils.SetDashboardConfig(ctx, &cfg.Dashboard)
	utils.SetTorrentFsConfig(ctx, &cfg.TorrentFs)

	return stack, cfg
}

// enableWhisper returns true in case one of the whisper flags is set.
func enableWhisper(ctx *cli.Context) bool {
	for _, flag := range whisperFlags {
		if ctx.GlobalIsSet(flag.GetName()) {
			return true
		}
	}
	return false
}

func makeFullNode(ctx *cli.Context) *node.Node {
	stack, cfg := makeConfigNode(ctx)

	storageEnabled := ctx.GlobalBool(utils.StorageEnabledFlag.Name) || !strings.HasPrefix(ctx.GlobalString(utils.InferDeviceTypeFlag.Name), "remote")
	if utils.IsCVMIPC(ctx.GlobalString(utils.InferDeviceTypeFlag.Name)) != "" {
		storageEnabled = false
	}
	if storageEnabled {
		log.Info("FullNode", "storageEnabled", storageEnabled)
		utils.RegisterStorageService(stack, &cfg.TorrentFs, gitCommit)
	}
	if deviceType := utils.IsCVMIPC(ctx.GlobalString(utils.InferDeviceTypeFlag.Name)); deviceType != "" {
		go func() {
			cmd := os.Args[0]
			log.Info("RegisterCVMService", "cmd", cmd)
			args := []string{"cvm",
				"--cvm.port", strconv.Itoa(ctx.GlobalInt(utils.InferPortFlag.Name)),
				"--storage.dir", utils.MakeStorageDir(ctx),
				"--cvm.datadir", utils.MakeDataDir(ctx),
				"--infer.devicetype", deviceType,
				"--cvm.max_seeding", strconv.Itoa(ctx.GlobalInt(utils.StorageMaxSeedingFlag.Name)),
				"--cvm.max_active", strconv.Itoa(ctx.GlobalInt(utils.StorageMaxActiveFlag.Name)),
				"--cvm.boostnodes", ctx.GlobalString(utils.StorageBoostNodesFlag.Name),
				"--cvm.tracker", ctx.GlobalString(utils.StorageTrackerFlag.Name),
			}
			if ctx.GlobalBool(utils.StorageDisableDHTFlag.Name) {
				args = append(args, "--cvm.disable_dht")
			}
			log.Debug("RegisterCVMService", "cmd", cmd, "args", args)
			prg := exec.Command(cmd, args...)
			var stdoutBuf, stderrBuf bytes.Buffer
			stdoutIn, _ := prg.StdoutPipe()
			stderrIn, _ := prg.StderrPipe()
			stdout := io.MultiWriter(os.Stdout, &stdoutBuf)
			stderr := io.MultiWriter(os.Stderr, &stderrBuf)
			if err := prg.Start(); err != nil {
				panic(err)
			}
			log.Debug("Cvm service register success", "config", cfg)
			run_result := prg.Start()
			var wg sync.WaitGroup
			wg.Add(1)
			go func() {
				defer wg.Done()
				_, _ = io.Copy(stdout, stdoutIn)
			}()
			wg.Add(1)
			go func() {
				defer wg.Done()
				_, _ = io.Copy(stderr, stderrIn)
			}()
			wg.Wait()
			if err := prg.Wait(); err != nil {
				log.Error("RegisterCVMService", "err", err)
			}
			log.Debug("RegisterCVMService", "deviceType", deviceType, "Exited", run_result)
			// outStr, errStr := string(stdoutBuf.Bytes()), string(stderrBuf.Bytes())
			// log.Debug("RegisterCVMService", "out", outStr, "err", errStr)
		}()
	}

	utils.RegisterCortexService(stack, &cfg.Cortex)

	// if ctx.GlobalBool(utils.DashboardEnabledFlag.Name) {
	// 	utils.RegisterDashboardService(stack, &cfg.Dashboard, gitCommit)
	// }
	// Whisper must be explicitly enabled by specifying at least 1 whisper flag or in dev mode
	//shhEnabled := enableWhisper(ctx)
	// shhAutoEnabled := !ctx.GlobalIsSet(utils.WhisperEnabledFlag.Name) && ctx.GlobalIsSet(utils.DeveloperFlag.Name)
	// if shhEnabled || shhAutoEnabled {
	// 	if ctx.GlobalIsSet(utils.WhisperMaxMessageSizeFlag.Name) {
	// 		cfg.Shh.MaxMessageSize = uint32(ctx.Int(utils.WhisperMaxMessageSizeFlag.Name))
	// 	}
	// 	if ctx.GlobalIsSet(utils.WhisperMinPOWFlag.Name) {
	// 		cfg.Shh.MinimumAcceptedPOW = ctx.Float64(utils.WhisperMinPOWFlag.Name)
	// 	}
	// 	if ctx.GlobalIsSet(utils.WhisperRestrictConnectionBetweenLightClientsFlag.Name) {
	// 		cfg.Shh.RestrictConnectionBetweenLightClients = true
	// 	}
	// 	utils.RegisterShhService(stack, &cfg.Shh)
	// }

	// Add the Cortex Stats daemon if requested.
	// if cfg.Cortexstats.URL != "" {
	// 	utils.RegisterCortexStatsService(stack, cfg.Cortexstats.URL)
	// }
	//storageEnabled := ctx.GlobalBool(utils.StorageEnabledFlag.Name) && ctx.GlobalString(utils.SyncModeFlag.Name) == "full"
	return stack
}

// dumpConfig is the dumpconfig command.
func dumpConfig(ctx *cli.Context) error {
	_, cfg := makeConfigNode(ctx)
	comment := ""

	if cfg.Cortex.Genesis != nil {
		cfg.Cortex.Genesis = nil
		comment += "# Note: this config doesn't contain the genesis block.\n\n"
	}

	out, err := tomlSettings.Marshal(&cfg)
	if err != nil {
		return err
	}
	io.WriteString(os.Stdout, comment)
	os.Stdout.Write(out)
	return nil
}
