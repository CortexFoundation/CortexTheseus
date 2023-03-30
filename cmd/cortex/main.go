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

// cortex is the official command-line client for Cortex.
package main

import (
	"fmt"
	"math/big"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/CortexFoundation/CortexTheseus/accounts"
	"github.com/CortexFoundation/CortexTheseus/accounts/keystore"
	"github.com/CortexFoundation/CortexTheseus/client"
	"github.com/CortexFoundation/CortexTheseus/cmd/utils"
	"github.com/CortexFoundation/CortexTheseus/console"
	"github.com/CortexFoundation/CortexTheseus/ctxc"
	"github.com/CortexFoundation/CortexTheseus/internal/debug"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/metrics"
	"github.com/CortexFoundation/CortexTheseus/node"
	_ "github.com/CortexFoundation/statik"
	"github.com/arsham/figurine/figurine"
	"gopkg.in/urfave/cli.v1"

	// Force-load the tracer engines to trigger registration
	_ "github.com/CortexFoundation/CortexTheseus/ctxc/tracers/js"
	_ "github.com/CortexFoundation/CortexTheseus/ctxc/tracers/native"
)

const (
	clientIdentifier = "cortex" // Client identifier to advertise over the network
)

var (
	// Git SHA1 commit hash of the release (set via linker flags)
	gitCommit = ""
	// The app that holds all commands and flags.
	app = utils.NewApp(gitCommit, "the cortex golang command line interface")
	// flags that configure the node
	nodeFlags = []cli.Flag{
		utils.IdentityFlag,
		utils.UnlockedAccountFlag,
		utils.PasswordFileFlag,
		utils.BootnodesFlag,
		utils.BootnodesV4Flag,
		// utils.BootnodesV5Flag,
		utils.DataDirFlag,
		utils.DBEngineFlag,
		utils.AncientFlag,
		utils.MinFreeDiskSpaceFlag,
		utils.KeyStoreDirFlag,
		utils.ExternalSignerFlag,
		// utils.NoUSBFlag,
		utils.TxPoolLocalsFlag,
		utils.TxPoolNoLocalsFlag,
		//utils.TxPoolNoInfersFlag,
		utils.TxPoolJournalFlag,
		utils.TxPoolRejournalFlag,
		utils.TxPoolPriceLimitFlag,
		utils.TxPoolPriceBumpFlag,
		utils.TxPoolAccountSlotsFlag,
		utils.TxPoolGlobalSlotsFlag,
		utils.TxPoolAccountQueueFlag,
		utils.TxPoolGlobalQueueFlag,
		utils.TxPoolLifetimeFlag,
		utils.SyncModeFlag,
		utils.GCModeFlag,
		utils.SnapshotFlag,
		utils.TxLookupLimitFlag,
		utils.WhitelistFlag,
		utils.CacheFlag,
		utils.CacheDatabaseFlag,
		utils.CacheTrieFlag,
		utils.CacheTrieJournalFlag,
		utils.CacheTrieRejournalFlag,
		utils.CacheGCFlag,
		utils.CacheSnapshotFlag,
		utils.CacheNoPrefetchFlag,
		utils.CachePreimagesFlag,
		utils.TrieCacheGenFlag,
		utils.FDLimitFlag,
		utils.ListenPortFlag,
		utils.DiscoveryPortFlag,
		utils.MaxPeersFlag,
		utils.MaxPendingPeersFlag,
		utils.ViperFlag,
		utils.MiningEnabledFlag,
		utils.MinerThreadsFlag,
		//utils.MinerNotifyFlag,
		utils.MinerGasTargetFlag,
		// utils.MinerLegacyGasTargetFlag,
		utils.MinerGasLimitFlag,
		utils.MinerGasPriceFlag,
		// utils.MinerLegacyGasPriceFlag,
		utils.MinerCoinbaseFlag,
		utils.MinerLegacyCoinbaseFlag,
		utils.MinerExtraDataFlag,
		// utils.MinerLegacyExtraDataFlag,
		utils.MinerRecommitIntervalFlag,
		//utils.MinerNoVerfiyFlag,
		utils.MinerCudaFlag,
		//utils.MinerOpenCLFlag,
		utils.MinerDevicesFlag,
		//utils.MinerAlgorithmFlag,
		utils.NATFlag,
		utils.NoDiscoverFlag,
		utils.DiscoveryV5Flag,
		utils.NetrestrictFlag,
		utils.NodeKeyFileFlag,
		utils.NodeKeyHexFlag,
		// utils.DeveloperFlag,
		// utils.DeveloperPeriodFlag,
		utils.BernardFlag,
		utils.DoloresFlag,
		// utils.TestnetFlag,
		// utils.LazynetFlag,
		// utils.VMEnableDebugFlag,
		utils.NetworkIdFlag,
		utils.RPCCORSDomainFlag,
		utils.RPCVirtualHostsFlag,
		// utils.CortexStatsURLFlag,
		// utils.MetricsEnabledFlag,
		// utils.FakePoWFlag,
		// utils.NoCompactionFlag,
		utils.GpoBlocksFlag,
		utils.GpoPercentileFlag,
		configFileFlag,
		// utils.ModelCallInterfaceFlag,
		utils.GpoMaxGasPriceFlag,
	}

	inferFlags = []cli.Flag{
		utils.InferDeviceTypeFlag,
		utils.InferDeviceIdFlag,
		utils.InferPortFlag,
		utils.InferMemoryFlag,
	}

	storageFlags = []cli.Flag{
		utils.StorageDirFlag,
		utils.StorageRpcFlag,
		utils.StorageEngineFlag,
		utils.StoragePortFlag,
		//utils.StorageEnabledFlag,
		utils.StorageMaxSeedingFlag,
		utils.StorageMaxActiveFlag,
		//utils.StorageBoostNodesFlag,
		utils.StorageTrackerFlag,
		//utils.StorageDisableDHTFlag,
		utils.StorageDisableTCPFlag,
		utils.StorageEnableUTPFlag,
		utils.StorageEnableWormholeFlag,
		utils.StorageModeFlag,
		utils.StorageBoostFlag,
	}

	rpcFlags = []cli.Flag{
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
		utils.InsecureUnlockAllowedFlag,
		utils.RPCGlobalGasCapFlag,
		utils.RPCGlobalTxFeeCapFlag,
		utils.AllowUnprotectedTxs,
	}

	whisperFlags = []cli.Flag{
		utils.WhisperEnabledFlag,
		utils.WhisperMaxMessageSizeFlag,
		utils.WhisperMinPOWFlag,
	}

	metricsFlags = []cli.Flag{
		utils.MetricsEnabledFlag,
		utils.MetricsEnabledExpensiveFlag,
		utils.MetricsHTTPFlag,
		utils.MetricsPortFlag,
		utils.MetricsEnableInfluxDBFlag,
		utils.MetricsInfluxDBEndpointFlag,
		utils.MetricsInfluxDBDatabaseFlag,
		utils.MetricsInfluxDBUsernameFlag,
		utils.MetricsInfluxDBPasswordFlag,
		utils.MetricsInfluxDBTagsFlag,
	}
)

func init() {
	// Initialize the CLI app and start Ctxc
	app.Action = cortex
	app.HideVersion = true // we have a command to print the version
	app.Copyright = "Copyright 2018-2019 The CortexTheseus Authors"
	app.Commands = []cli.Command{
		// See chaincmd.go:
		initCommand,
		// importCommand,
		// exportCommand,
		// importPreimagesCommand,
		// exportPreimagesCommand,
		// copydbCommand,
		removedbCommand,
		// dumpCommand,
		dumpGenesisCommand,
		inspectCommand,
		// See monitorcmd.go:
		// monitorCommand,
		// See accountcmd.go:
		accountCommand,

		// walletCommand,
		// See consolecmd.go:
		consoleCommand,
		attachCommand,
		// javascriptCommand,
		// See misccmd.go:
		// makecacheCommand,
		// makedagCommand,
		versionCommand,
		cvmCommand,
		// bugCommand,
		// licenseCommand,
		// See config.go
		// dumpConfigCommand,
		verkleCommand,
	}
	sort.Sort(cli.CommandsByName(app.Commands))

	app.Flags = append(app.Flags, nodeFlags...)
	app.Flags = append(app.Flags, rpcFlags...)
	app.Flags = append(app.Flags, consoleFlags...)
	app.Flags = append(app.Flags, debug.Flags...)
	app.Flags = append(app.Flags, whisperFlags...)
	app.Flags = append(app.Flags, metricsFlags...)
	app.Flags = append(app.Flags, cvmFlags...)
	app.Flags = append(app.Flags, storageFlags...)
	app.Flags = append(app.Flags, inferFlags...)

	app.Before = func(ctx *cli.Context) error {
		return debug.Setup(ctx)
	}

	app.After = func(ctx *cli.Context) error {
		debug.Exit()
		console.Stdin.Close() // Resets terminal mode.
		return nil
	}
}

func main() {
	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// cortex is the main entry point into the system if no special subcommand is ran.
// It creates a default node based on the command line arguments and runs it in
// blocking mode, waiting for it to be shut down.

func cortex(ctx *cli.Context) error {
	if args := ctx.Args(); len(args) > 0 {
		return fmt.Errorf("invalid command: %q", args[0])
	}

	err := figurine.Write(os.Stdout, "CORTEX", "3d.flf")
	if err != nil {
		log.Error("", "err", err)
	}
	prepare(ctx)
	node := makeFullNode(ctx)
	defer node.Close()
	startNode(ctx, node)
	node.Wait()
	return nil
}

// prepare manipulates memory cache allowance and setups metric system.
// This function should be called before launching devp2p stack.
func prepare(ctx *cli.Context) {
	// If we're a full node on mainnet without --cache specified, bump default cache allowance
	if !ctx.GlobalIsSet(utils.CacheFlag.Name) && !ctx.GlobalIsSet(utils.NetworkIdFlag.Name) {
		// Make sure we're not on any supported preconfigured testnet either
		// Nope, we're really on mainnet. Bump that cache up!
		log.Info("Bumping default cache on mainnet", "provided", ctx.GlobalInt(utils.CacheFlag.Name), "updated", 4096)
		ctx.GlobalSet(utils.CacheFlag.Name, strconv.Itoa(4096))
	}
	// Start metrics export if enabled
	utils.SetupMetrics(ctx)

	// Start system runtime metrics collection
	go func() {
		metrics.CollectProcessMetrics(3 * time.Second)
	}()
}

// startNode boots up the system node and all registered protocols, after which
// it unlocks any requested accounts, and starts the RPC/IPC interfaces and the
// miner.
func startNode(ctx *cli.Context, stack *node.Node) {
	debug.Memsize.Add("node", stack)

	// Start up the node itself
	utils.StartNode(ctx, stack)
	var unlocks []string
	inputs := strings.Split(ctx.GlobalString(utils.UnlockedAccountFlag.Name), ",")
	for _, input := range inputs {
		if trimmed := strings.TrimSpace(input); trimmed != "" {
			unlocks = append(unlocks, trimmed)
		}
	}
	if len(unlocks) > 0 {

		if !stack.Config().InsecureUnlockAllowed {
			utils.Fatalf("Account unlock with HTTP access is forbidden! account=%v", len(unlocks))
		}

		// Unlock any account specifically requested
		ks := stack.AccountManager().Backends(keystore.KeyStoreType)[0].(*keystore.KeyStore)

		passwords := utils.MakePasswordList(ctx)
		//unlocks := strings.Split(ctx.GlobalString(utils.UnlockedAccountFlag.Name), ",")
		for i, account := range unlocks {
			if trimmed := strings.TrimSpace(account); trimmed != "" {
				unlockAccount(ctx, ks, trimmed, i, passwords)
			}
		}
	}
	// Register wallet event handlers to open and auto-derive wallets
	events := make(chan accounts.WalletEvent, 16)
	stack.AccountManager().Subscribe(events)

	go func() {
		// Create a chain state reader for self-derivation
		rpcClient, err := stack.Attach()
		if err != nil {
			utils.Fatalf("Failed to attach to self: %v", err)
		}
		stateReader := ctxcclient.NewClient(rpcClient)

		// Open any wallets already attached
		for _, wallet := range stack.AccountManager().Wallets() {
			if err := wallet.Open(""); err != nil {
				log.Warn("Failed to open wallet", "url", wallet.URL(), "err", err)
			}
		}
		// Listen for wallet event till termination
		for event := range events {
			switch event.Kind {
			case accounts.WalletArrived:
				if err := event.Wallet.Open(""); err != nil {
					log.Warn("New wallet appeared, failed to open", "url", event.Wallet.URL(), "err", err)
				}
			case accounts.WalletOpened:
				status, _ := event.Wallet.Status()
				log.Info("New wallet appeared", "url", event.Wallet.URL(), "status", status)

				var derivationPaths []accounts.DerivationPath
				if event.Wallet.URL().Scheme == "ledger" {
					derivationPaths = append(derivationPaths, accounts.LegacyLedgerBaseDerivationPath)
				}
				derivationPaths = append(derivationPaths, accounts.DefaultBaseDerivationPath)
				event.Wallet.SelfDerive(derivationPaths, stateReader)

			case accounts.WalletDropped:
				log.Info("Old wallet dropped", "url", event.Wallet.URL())
				event.Wallet.Close()
			}
		}
	}()
	// Start auxiliary services if enabled
	// if ctx.GlobalBool(utils.MiningEnabledFlag.Name) || ctx.GlobalBool(utils.DeveloperFlag.Name) {
	if ctx.GlobalBool(utils.MiningEnabledFlag.Name) {
		// Mining only makes sense if a full Cortex node is running
		var cortex *ctxc.Cortex
		if err := stack.Service(&cortex); err != nil {
			utils.Fatalf("Cortex service not running: %v", err)
		}
		// Set the gas price to the limits from the CLI and start mining
		// gasprice := utils.GlobalBig(ctx, utils.MinerLegacyGasPriceFlag.Name)
		var gasprice *big.Int = nil
		if ctx.IsSet(utils.MinerGasPriceFlag.Name) {
			gasprice = utils.GlobalBig(ctx, utils.MinerGasPriceFlag.Name)
		}
		if gasprice == nil {
			gasprice = big.NewInt(0)
		}
		cortex.TxPool().SetGasPrice(gasprice)

		threads := ctx.GlobalInt(utils.MinerThreadsFlag.Name)
		if ctx.GlobalIsSet(utils.MinerThreadsFlag.Name) {
			threads = ctx.GlobalInt(utils.MinerThreadsFlag.Name)
		}
		if err := cortex.StartMining(threads); err != nil {
			utils.Fatalf("Failed to start mining: %v", err)
		}
	}
}
