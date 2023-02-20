// Copyright 2019 The CortexTheseus Authors
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

// Contains the cortex command usage template and generator.

package main

import (
	"io"
	"sort"
	"strings"

	"github.com/CortexFoundation/CortexTheseus/cmd/utils"
	// "github.com/CortexFoundation/CortexTheseus/internal/debug"
	"gopkg.in/urfave/cli.v1"
)

// AppHelpTemplate is the test template for the default, global app help topic.
var AppHelpTemplate = `NAME:
   {{.App.Name}} - {{.App.Usage}}

   Copyright 2018-2019 The CortexTheseus Authors

USAGE:
   {{.App.HelpName}} [options]{{if .App.Commands}} command [command options]{{end}} {{if .App.ArgsUsage}}{{.App.ArgsUsage}}{{else}}[arguments...]{{end}}
   {{if .App.Version}}
VERSION:
   {{.App.Version}}
   {{end}}{{if len .App.Authors}}
AUTHOR(S):
   {{range .App.Authors}}{{ . }}{{end}}
   {{end}}{{if .App.Commands}}
COMMANDS:
   {{range .App.Commands}}{{join .Names ", "}}{{ "\t" }}{{.Usage}}
   {{end}}{{end}}{{if .FlagGroups}}
{{range .FlagGroups}}{{.Name}} OPTIONS:
  {{range .Flags}}{{.}}
  {{end}}
{{end}}{{end}}{{if .App.Copyright }}
COPYRIGHT:
   {{.App.Copyright}}
   {{end}}
`

// flagGroup is a collection of flags belonging to a single topic.
type flagGroup struct {
	Name  string
	Flags []cli.Flag
}

// AppHelpFlagGroups is the application flags, grouped by functionality.
var AppHelpFlagGroups = []flagGroup{
	{
		Name: "CORTEX",
		Flags: []cli.Flag{
			configFileFlag,
			utils.DataDirFlag,
			utils.DBEngineFlag,
			utils.AncientFlag,
			utils.MinFreeDiskSpaceFlag,
			utils.KeyStoreDirFlag,
			// utils.NoUSBFlag,
			utils.NetworkIdFlag,
			// utils.TestnetFlag,
			// utils.LazynetFlag,
			utils.SyncModeFlag,
			utils.GCModeFlag,
			utils.TxLookupLimitFlag,
			utils.WhitelistFlag,
			// utils.CortexStatsURLFlag,
			utils.IdentityFlag,
			utils.ViperFlag,
		},
	},
	// {
	// 	Name: "DEVELOPER CHAIN",
	// 	Flags: []cli.Flag{
	// 		utils.DeveloperFlag,
	// 		utils.DeveloperPeriodFlag,
	// 	},
	// },
	//{
	//	Name: "DASHBOARD",
	//	Flags: []cli.Flag{
	//		utils.DashboardEnabledFlag,
	//		utils.DashboardAddrFlag,
	//		utils.DashboardPortFlag,
	//		utils.DashboardRefreshFlag,
	//		utils.DashboardAssetsFlag,
	//	},
	//},
	{
		Name: "TRANSACTION POOL",
		Flags: []cli.Flag{
			utils.TxPoolLocalsFlag,
			utils.TxPoolNoLocalsFlag,
			//			utils.TxPoolNoInfersFlag,
			utils.TxPoolJournalFlag,
			utils.TxPoolRejournalFlag,
			utils.TxPoolPriceLimitFlag,
			utils.TxPoolPriceBumpFlag,
			utils.TxPoolAccountSlotsFlag,
			utils.TxPoolGlobalSlotsFlag,
			utils.TxPoolAccountQueueFlag,
			utils.TxPoolGlobalQueueFlag,
			utils.TxPoolLifetimeFlag,
		},
	},
	{
		Name: "PERFORMANCE TUNING",
		Flags: []cli.Flag{
			utils.CacheFlag,
			utils.CacheDatabaseFlag,
			utils.CacheTrieFlag,
			utils.CacheTrieJournalFlag,
			utils.CacheTrieRejournalFlag,
			utils.CacheGCFlag,
			utils.CacheSnapshotFlag,
			utils.CacheNoPrefetchFlag,
			utils.CachePreimagesFlag,
			utils.FDLimitFlag,
			utils.TrieCacheGenFlag,
		},
	},
	{
		Name: "ACCOUNT",
		Flags: []cli.Flag{
			utils.UnlockedAccountFlag,
			utils.PasswordFileFlag,
			utils.ExternalSignerFlag,
			utils.InsecureUnlockAllowedFlag,
		},
	},
	{
		Name: "API AND CONSOLE",
		Flags: []cli.Flag{
			utils.RPCEnabledFlag,
			utils.RPCListenAddrFlag,
			utils.RPCPortFlag,
			utils.RPCApiFlag,
			utils.WSEnabledFlag,
			utils.WSListenAddrFlag,
			utils.WSPortFlag,
			utils.WSApiFlag,
			utils.WSAllowedOriginsFlag,
			utils.IPCDisabledFlag,
			utils.IPCPathFlag,
			utils.RPCCORSDomainFlag,
			utils.RPCVirtualHostsFlag,
			utils.RPCGlobalGasCapFlag,
			utils.RPCGlobalTxFeeCapFlag,
			//utils.AllowUnprotectedTxs,
			utils.JSpathFlag,
			utils.ExecFlag,
			utils.PreloadJSFlag,
		},
	},
	{
		Name: "NETWORKING",
		Flags: []cli.Flag{
			utils.BootnodesFlag,
			//utils.BootnodesV4Flag,
			//		utils.BootnodesV5Flag,
			utils.ListenPortFlag,
			utils.MaxPeersFlag,
			utils.MaxPendingPeersFlag,
			utils.NATFlag,
			utils.NoDiscoverFlag,
			utils.DiscoveryV5Flag,
			utils.NetrestrictFlag,
			utils.NodeKeyFileFlag,
			utils.NodeKeyHexFlag,
		},
	},
	{
		Name: "MINER",
		Flags: []cli.Flag{
			utils.MiningEnabledFlag,
			utils.MinerThreadsFlag,
			//utils.MinerNotifyFlag,
			utils.MinerGasPriceFlag,
			utils.MinerGasTargetFlag,
			utils.MinerGasLimitFlag,
			utils.MinerCoinbaseFlag,
			utils.MinerExtraDataFlag,
			utils.MinerRecommitIntervalFlag,
			//utils.MinerNoVerfiyFlag,
			utils.MinerCudaFlag,
			//utils.MinerOpenCLFlag,
			utils.MinerDevicesFlag,
			//utils.MinerAlgorithmFlag,
		},
	},
	{
		Name: "INFER",
		Flags: []cli.Flag{
			utils.InferDeviceTypeFlag,
			utils.InferDeviceIdFlag,
			utils.InferPortFlag,
			utils.InferMemoryFlag,
		},
	},
	{
		Name: "STORAGE",
		Flags: []cli.Flag{
			//utils.StorageEnabledFlag,
			utils.StorageDirFlag,
			utils.StorageRpcFlag,
			utils.StoragePortFlag,
			utils.StorageMaxSeedingFlag,
			utils.StorageMaxActiveFlag,
			//utils.StorageBoostNodesFlag,
			utils.StorageTrackerFlag,
			//			utils.StorageDisableDHTFlag,
			utils.StorageDisableTCPFlag,
			utils.StorageEnableUTPFlag,
			utils.StorageEnableWormholeFlag,
			utils.StorageModeFlag,
			utils.StorageBoostFlag,
		},
	},
	{
		Name: "GAS PRICE ORACLE",
		Flags: []cli.Flag{
			utils.GpoBlocksFlag,
			utils.GpoPercentileFlag,
			utils.GpoMaxGasPriceFlag,
		},
	},
	{
		Name: "VIRTUAL MACHINE",
		Flags: []cli.Flag{
			// utils.ModelCallInterfaceFlag,
			utils.VMEnableDebugFlag,
		},
	},
	// {
	// 	Name: "LOGGING AND DEBUGGING",
	// 	Flags: append([]cli.Flag{
	// 		utils.FakePoWFlag,
	// 		utils.NoCompactionFlag,
	// 	}, debug.threadsFlags...),
	// },
	{
		Name:  "METRICS AND STATS",
		Flags: metricsFlags,
	},
	{
		Name:  "WHISPER (EXPERIMENTAL)",
		Flags: whisperFlags,
	},
	// {
	// 	Name: "DEPRECATED",
	// 	Flags: []cli.Flag{
	// 		utils.MinerLegacyThreadsFlag,
	// 		utils.MinerLegacyGasTargetFlag,
	// 		utils.MinerLegacyGasPriceFlag,
	// 		utils.MinerLegacyCoinbaseFlag,
	// 		utils.MinerLegacyExtraDataFlag,
	// 	},
	// },
	{
		Name: "MISC",
	},
}

// byCategory sorts an array of flagGroup by Name in the order
// defined in AppHelpFlagGroups.
type byCategory []flagGroup

func (a byCategory) Len() int      { return len(a) }
func (a byCategory) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a byCategory) Less(i, j int) bool {
	iCat, jCat := a[i].Name, a[j].Name
	iIdx, jIdx := len(AppHelpFlagGroups), len(AppHelpFlagGroups) // ensure non categorized flags come last

	for i, group := range AppHelpFlagGroups {
		if iCat == group.Name {
			iIdx = i
		}
		if jCat == group.Name {
			jIdx = i
		}
	}

	return iIdx < jIdx
}

func flagCategory(flag cli.Flag) string {
	for _, category := range AppHelpFlagGroups {
		for _, flg := range category.Flags {
			if flg.GetName() == flag.GetName() {
				return category.Name
			}
		}
	}
	return "MISC"
}

func init() {
	// Override the default app help template
	cli.AppHelpTemplate = AppHelpTemplate

	// Define a one shot struct to pass to the usage template
	type helpData struct {
		App        interface{}
		FlagGroups []flagGroup
	}

	// Override the default app help printer, but only for the global app help
	originalHelpPrinter := cli.HelpPrinter
	cli.HelpPrinter = func(w io.Writer, tmpl string, data interface{}) {
		if tmpl == AppHelpTemplate {
			// Iterate over all the flags and add any uncategorized ones
			categorized := make(map[string]struct{})
			for _, group := range AppHelpFlagGroups {
				for _, flag := range group.Flags {
					categorized[flag.String()] = struct{}{}
				}
			}
			uncategorized := []cli.Flag{}
			for _, flag := range data.(*cli.App).Flags {
				if _, ok := categorized[flag.String()]; !ok {
					if strings.HasPrefix(flag.GetName(), "dashboard") {
						continue
					}
					uncategorized = append(uncategorized, flag)
				}
			}
			if len(uncategorized) > 0 {
				// Append all ungategorized options to the misc group
				miscs := len(AppHelpFlagGroups[len(AppHelpFlagGroups)-1].Flags)
				AppHelpFlagGroups[len(AppHelpFlagGroups)-1].Flags = append(AppHelpFlagGroups[len(AppHelpFlagGroups)-1].Flags, uncategorized...)

				// Make sure they are removed afterwards
				defer func() {
					AppHelpFlagGroups[len(AppHelpFlagGroups)-1].Flags = AppHelpFlagGroups[len(AppHelpFlagGroups)-1].Flags[:miscs]
				}()
			}
			// Render out custom usage screen
			originalHelpPrinter(w, tmpl, helpData{data, AppHelpFlagGroups})
		} else if tmpl == utils.CommandHelpTemplate {
			// Iterate over all command specific flags and categorize them
			categorized := make(map[string][]cli.Flag)
			for _, flag := range data.(cli.Command).Flags {
				if _, ok := categorized[flag.String()]; !ok {
					categorized[flagCategory(flag)] = append(categorized[flagCategory(flag)], flag)
				}
			}

			// sort to get a stable ordering
			sorted := make([]flagGroup, 0, len(categorized))
			for cat, flgs := range categorized {
				sorted = append(sorted, flagGroup{cat, flgs})
			}
			sort.Sort(byCategory(sorted))

			// add sorted array to data and render with default printer
			originalHelpPrinter(w, tmpl, map[string]interface{}{
				"cmd":              data,
				"categorizedFlags": sorted,
			})
		} else {
			originalHelpPrinter(w, tmpl, data)
		}
	}
}
