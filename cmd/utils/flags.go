// Copyright 2019 The go-ethereum Authors
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

// Package utils contains internal helper functions for CortexFoundation commands.
package utils

import (
	"crypto/ecdsa"
	"encoding/hex"
	"fmt"
	"math"
	"math/big"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	godebug "runtime/debug"
	"strconv"
	"strings"
	"time"

	"github.com/CortexFoundation/inference/synapse"
	"github.com/CortexFoundation/torrentfs"
	params1 "github.com/CortexFoundation/torrentfs/params"
	gopsutil "github.com/shirou/gopsutil/mem"
	"gopkg.in/urfave/cli.v1"

	"github.com/CortexFoundation/CortexTheseus/accounts"
	"github.com/CortexFoundation/CortexTheseus/accounts/keystore"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/fdlimit"
	"github.com/CortexFoundation/CortexTheseus/consensus"
	"github.com/CortexFoundation/CortexTheseus/consensus/clique"
	"github.com/CortexFoundation/CortexTheseus/consensus/cuckoo"
	"github.com/CortexFoundation/CortexTheseus/core"

	//"github.com/CortexFoundation/CortexTheseus/ctxc/tracers"
	"github.com/CortexFoundation/CortexTheseus/core/txpool"
	"github.com/CortexFoundation/CortexTheseus/crypto/kzg4844"
	whisper "github.com/CortexFoundation/CortexTheseus/whisper/whisperv6"

	// "github.com/CortexFoundation/CortexTheseus/core/state"
	"github.com/CortexFoundation/CortexTheseus/core/rawdb"
	"github.com/CortexFoundation/CortexTheseus/core/vm"
	"github.com/CortexFoundation/CortexTheseus/crypto"
	"github.com/CortexFoundation/CortexTheseus/ctxc"
	"github.com/CortexFoundation/CortexTheseus/ctxc/downloader"
	"github.com/CortexFoundation/CortexTheseus/ctxc/gasprice"
	"github.com/CortexFoundation/CortexTheseus/ctxcdb"
	"github.com/CortexFoundation/CortexTheseus/log"

	// "github.com/CortexFoundation/CortexTheseus/stats"
	"github.com/CortexFoundation/CortexTheseus/metrics"
	"github.com/CortexFoundation/CortexTheseus/metrics/exp"
	"github.com/CortexFoundation/CortexTheseus/metrics/influxdb"
	"github.com/CortexFoundation/CortexTheseus/node"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/CortexFoundation/CortexTheseus/p2p/enode"
	"github.com/CortexFoundation/CortexTheseus/p2p/nat"
	"github.com/CortexFoundation/CortexTheseus/p2p/netutil"
	"github.com/CortexFoundation/CortexTheseus/params"
)

var (
	CommandHelpTemplate = `{{.cmd.Name}}{{if .cmd.Subcommands}} command{{end}}{{if .cmd.Flags}} [command options]{{end}} [arguments...]
{{if .cmd.Description}}{{.cmd.Description}}
{{end}}{{if .cmd.Subcommands}}
SUBCOMMANDS:
	{{range .cmd.Subcommands}}{{.Name}}{{with .ShortName}}, {{.}}{{end}}{{ "\t" }}{{.Usage}}
	{{end}}{{end}}{{if .categorizedFlags}}
{{range $idx, $categorized := .categorizedFlags}}{{$categorized.Name}} OPTIONS:
{{range $categorized.Flags}}{{"\t"}}{{.}}
{{end}}
{{end}}{{end}}`
	OriginCommandHelpTemplate = `{{.Name}}{{if .Subcommands}} command{{end}}{{if .Flags}} [command options]{{end}} [arguments...]
{{if .Description}}{{.Description}}
{{end}}{{if .Subcommands}}
SUBCOMMANDS:
        {{range .Subcommands}}{{.Name}}{{with .ShortName}}, {{.}}{{end}}{{ "\t" }}{{.Usage}}
        {{end}}{{end}}{{if .Flags}}
OPTIONS:
{{range $.Flags}}{{"\t"}}{{.}}
{{end}}
{{end}}`
)

func init() {
	cli.AppHelpTemplate = `{{.Name}} {{if .Flags}}[global options] {{end}}command{{if .Flags}} [command options]{{end}} [arguments...]

VERSION:
   {{.Version}}

COMMANDS:
   {{range .Commands}}{{.Name}}{{with .ShortName}}, {{.}}{{end}}{{ "\t" }}{{.Usage}}
   {{end}}{{if .Flags}}
GLOBAL OPTIONS:
   {{range .Flags}}{{.}}
   {{end}}{{end}}
`

	cli.CommandHelpTemplate = CommandHelpTemplate
}

// NewApp creates an app with sane defaults.
func NewApp(gitCommit, usage string) *cli.App {
	app := cli.NewApp()
	app.Name = filepath.Base(os.Args[0])
	app.HelpName = "cortex"
	app.Author = "Cortex Labs"
	app.Email = "support@cortexlabs.ai"
	app.Version = params.VersionWithMeta
	if len(gitCommit) >= 8 {
		app.Version += "-" + gitCommit[:8]
	}
	app.Usage = usage
	return app
}

// These are all the command line flags we support.
// If you add to this list, please remember to include the
// flag in the appropriate command definition.
//
// The flags are defined here so their names and help texts
// are the same for all commands.

var (
	// General settings
	DataDirFlag = DirectoryFlag{
		Name:  "datadir",
		Usage: "Data directory for the databases and keystore",
		Value: DirectoryString{node.DefaultDataDir()},
	}
	DBEngineFlag = &cli.StringFlag{
		Name:  "db.engine",
		Usage: "Backing database implementation to use ('pebble' or 'leveldb')",
		Value: node.DefaultConfig.DBEngine,
	}
	AncientFlag = DirectoryFlag{
		Name:  "datadir.ancient",
		Usage: "Data directory for ancient chain segments (default = inside chaindata)",
	}
	MinFreeDiskSpaceFlag = DirectoryFlag{
		Name:  "datadir.minfreedisk",
		Usage: "Minimum free disk space in MB, once reached triggers auto shut down (default = --cache.gc converted to MB, 0 = disabled)",
	}
	KeyStoreDirFlag = DirectoryFlag{
		Name:  "keystore",
		Usage: "Directory for the keystore (default = inside the datadir)",
	}
	// NoUSBFlag = cli.BoolFlag{
	// 	Name:  "nousb",
	// 	Usage: "Disables monitoring for and managing USB hardware wallets",
	// }

	// Deprecated November 2023
	LogBacktraceAtFlag = &cli.StringFlag{
		Name:  "log.backtrace",
		Usage: "Request a stack trace at a specific logging statement (deprecated)",
		Value: "",
	}
	LogDebugFlag = &cli.BoolFlag{
		Name:  "log.debug",
		Usage: "Prepends log messages with call-site location (deprecated)",
	}
	NetworkIdFlag = cli.Uint64Flag{
		Name:  "networkid",
		Usage: "Network identifier (integer, 21=Mainnet, 42=Bernard, 43=Dolores)",
		Value: ctxc.DefaultConfig.NetworkId,
	}
	BernardFlag = cli.BoolFlag{
		Name:  "bernard",
		Usage: "Bernard network: pre-configured cortex test network with POA",
	}
	DoloresFlag = cli.BoolFlag{
		Name:  "dolores",
		Usage: "Dolores network: pre-configured cortex test network",
	}

	ViperFlag = cli.BoolFlag{
		Name:  "viper",
		Usage: "Sync block with 180s local sprout time",
	}
	// DeveloperFlag = cli.BoolFlag{
	// 	Name:  "dev",
	// 	Usage: "Ephemeral proof-of-authority network with a pre-funded developer account, mining enabled",
	// }
	// DeveloperPeriodFlag = cli.IntFlag{
	// 	Name:  "dev.period",
	// 	Usage: "Block period to use in developer mode (0 = mine only if transaction pending)",
	// }
	IdentityFlag = cli.StringFlag{
		Name:  "identity",
		Usage: "Custom node name",
	}
	DocRootFlag = DirectoryFlag{
		Name:  "docroot",
		Usage: "Document Root for HTTPClient file scheme",
		Value: DirectoryString{homeDir()},
	}
	defaultSyncMode = ctxc.DefaultConfig.SyncMode
	SyncModeFlag    = TextMarshalerFlag{
		Name:  "syncmode",
		Usage: `Blockchain sync mode ("full")`,
		Value: &defaultSyncMode,
	}
	GCModeFlag = cli.StringFlag{
		Name:  "gcmode",
		Usage: `Blockchain garbage collection mode ("full", "archive")`,
		Value: "full",
	}

	SnapshotFlag = cli.BoolFlag{
		Name:  "snapshot",
		Usage: `Enables snapshot-database mode -- experimental work in progress feature`,
	}
	TxLookupLimitFlag = cli.Int64Flag{
		Name:  "txlookuplimit",
		Usage: "Number of recent blocks to maintain transactions index by-hash for (default = index all blocks)",
		Value: 0,
	}

	WhitelistFlag = cli.StringFlag{
		Name:  "whitelist",
		Usage: "Comma separated block number-to-hash mappings to enforce (<number>=<hash>)",
	}

	// P2P storage settings
	StorageEnabledFlag = cli.BoolFlag{
		Name:  "storage",
		Usage: "Enable P2P storage",
	}
	StorageDirFlag = DirectoryFlag{
		Name:  "storage.dir",
		Usage: "P2P storage directory",
		Value: DirectoryString{node.DefaultStorageDir("")},
	}

	StorageRpcFlag = cli.StringFlag{
		Name:  "storage.rpc",
		Usage: "P2P storage status sync from blockchain rpc link",
		Value: "http://127.0.0.1:8545",
	}

	StoragePortFlag = cli.IntFlag{
		Name:  "storage.port",
		Usage: "p2p storage listening port",
		Value: params1.DefaultConfig.Port,
	}

	StorageEngineFlag = cli.StringFlag{
		Name:  "storage.engine",
		Usage: "Torrent storage engine (badger, pebble, leveldb, bolt)",
		Value: params1.DefaultConfig.Engine,
	}

	StorageMaxSeedingFlag = cli.IntFlag{
		Name:  "storage.max_seeding",
		Usage: "The maximum number of seeding tasks in the same time",
		Value: params1.DefaultConfig.MaxSeedingNum,
	}
	StorageMaxActiveFlag = cli.IntFlag{
		Name:  "storage.max_active",
		Usage: "The maximum number of active tasks in the same time",
		Value: params1.DefaultConfig.MaxActiveNum,
	}
	StorageBoostNodesFlag = cli.StringFlag{
		Name:  "storage.boostnodes",
		Usage: "p2p storage boostnodes (EXPERIMENTAL)",
		Value: strings.Join(params1.DefaultConfig.BoostNodes, ","),
	}
	StorageTrackerFlag = cli.StringFlag{
		Name:  "storage.tracker",
		Usage: "P2P storage tracker list",
		Value: strings.Join(params1.DefaultConfig.DefaultTrackers, ","),
	}
	StorageDHTFlag = cli.BoolFlag{
		Name:  "storage.dht",
		Usage: "enable DHT network in FS",
	}
	StorageBoostFlag = cli.BoolFlag{
		Name:  "storage.boost",
		Usage: "Boost fs (EXPERIMENTAL)",
	}
	StorageDisableTCPFlag = cli.BoolFlag{
		Name:  "storage.disable_tcp",
		Usage: "disable TCP network in FS (EXPERIMENTAL)",
	}
	StorageEnableUTPFlag = cli.BoolFlag{
		Name:  "storage.utp",
		Usage: "enable UTP network in FS (EXPERIMENTAL)",
	}
	StorageEnableWormholeFlag = cli.BoolFlag{
		Name:  "storage.wormhole",
		Usage: "enable wormhole network in FS (EXPERIMENTAL)",
	}
	StorageModeFlag = cli.StringFlag{
		Name:  "storage.mode",
		Usage: "P2P storage running mode",
		Value: "normal",
	}
	StorageDebugFlag = cli.BoolFlag{
		Name:  "storage.debug",
		Usage: "debug mod for nas",
	}
	// Dashboard settings
	// DashboardEnabledFlag = cli.BoolFlag{
	// 	Name:  metrics.DashboardEnabledFlag,
	// 	Usage: "Enable the dashboard",
	// }
	// DashboardAddrFlag = cli.StringFlag{
	// 	Name:  "dashboard.addr",
	// 	Usage: "Dashboard listening interface",
	// 	Value: dashboard.DefaultConfig.Host,
	// }
	// DashboardPortFlag = cli.IntFlag{
	// 	Name:  "dashboard.host",
	// 	Usage: "Dashboard listening port",
	// 	Value: dashboard.DefaultConfig.Port,
	// }
	// DashboardRefreshFlag = cli.DurationFlag{
	// 	Name:  "dashboard.refresh",
	// 	Usage: "Dashboard metrics collection refresh rate",
	// 	Value: dashboard.DefaultConfig.Refresh,
	// }
	// Transaction pool settings
	TxPoolLocalsFlag = cli.StringFlag{
		Name:  "txpool.locals",
		Usage: "Comma separated accounts to treat as locals (no flush, priority inclusion)",
	}
	TxPoolNoLocalsFlag = cli.BoolFlag{
		Name:  "txpool.nolocals",
		Usage: "Disables price exemptions for locally submitted transactions",
	}
	//TxPoolNoInfersFlag = cli.BoolFlag{
	//	Name:  "txpool.noinfers",
	//	Usage: "Disables infer transactions in this node",
	//}
	TxPoolJournalFlag = cli.StringFlag{
		Name:  "txpool.journal",
		Usage: "Disk journal for local transaction to survive node restarts",
		Value: txpool.DefaultConfig.Journal,
	}
	TxPoolRejournalFlag = cli.DurationFlag{
		Name:  "txpool.rejournal",
		Usage: "Time interval to regenerate the local transaction journal",
		Value: txpool.DefaultConfig.Rejournal,
	}
	TxPoolPriceLimitFlag = cli.Uint64Flag{
		Name:  "txpool.pricelimit",
		Usage: "Minimum gas price limit to enforce for acceptance into the pool",
		Value: ctxc.DefaultConfig.TxPool.PriceLimit,
	}
	TxPoolPriceBumpFlag = cli.Uint64Flag{
		Name:  "txpool.pricebump",
		Usage: "Price bump percentage to replace an already existing transaction",
		Value: ctxc.DefaultConfig.TxPool.PriceBump,
	}
	TxPoolAccountSlotsFlag = cli.Uint64Flag{
		Name:  "txpool.accountslots",
		Usage: "Minimum number of executable transaction slots guaranteed per account",
		Value: ctxc.DefaultConfig.TxPool.AccountSlots,
	}
	TxPoolGlobalSlotsFlag = cli.Uint64Flag{
		Name:  "txpool.globalslots",
		Usage: "Maximum number of executable transaction slots for all accounts",
		Value: ctxc.DefaultConfig.TxPool.GlobalSlots,
	}
	TxPoolAccountQueueFlag = cli.Uint64Flag{
		Name:  "txpool.accountqueue",
		Usage: "Maximum number of non-executable transaction slots permitted per account",
		Value: ctxc.DefaultConfig.TxPool.AccountQueue,
	}
	TxPoolGlobalQueueFlag = cli.Uint64Flag{
		Name:  "txpool.globalqueue",
		Usage: "Maximum number of non-executable transaction slots for all accounts",
		Value: ctxc.DefaultConfig.TxPool.GlobalQueue,
	}
	TxPoolLifetimeFlag = cli.DurationFlag{
		Name:  "txpool.lifetime",
		Usage: "Maximum amount of time non-executable transaction are queued",
		Value: ctxc.DefaultConfig.TxPool.Lifetime,
	}
	// Performance tuning settings
	CacheFlag = cli.IntFlag{
		Name:  "cache",
		Usage: "Megabytes of memory allocated to internal caching",
		Value: 1024,
	}
	CacheDatabaseFlag = cli.IntFlag{
		Name:  "cache.database",
		Usage: "Percentage of cache memory allowance to use for database io",
		Value: 50,
	}
	CacheTrieFlag = cli.IntFlag{
		Name:  "cache.trie",
		Usage: "Percentage of cache memory allowance to use for trie caching (default = 15% full mode, 30% archive mode)",
		Value: 15,
	}
	CacheTrieJournalFlag = cli.StringFlag{
		Name:  "cache.trie.journal",
		Usage: "Disk journal directory for trie cache to survive node restarts",
		Value: ctxc.DefaultConfig.TrieCleanCacheJournal,
	}
	CacheTrieRejournalFlag = cli.DurationFlag{
		Name:  "cache.trie.rejournal",
		Usage: "Time interval to regenerate the trie cache journal",
		Value: ctxc.DefaultConfig.TrieCleanCacheRejournal,
	}
	CacheGCFlag = cli.IntFlag{
		Name:  "cache.gc",
		Usage: "Percentage of cache memory allowance to use for trie pruning 25%",
		Value: 25,
	}
	CacheSnapshotFlag = cli.IntFlag{
		Name:  "cache.snapshot",
		Usage: "Percentage of cache memory allowance to use for snapshot caching (default = 10% full mode, 20% archive mode)",
		Value: 10,
	}
	CacheNoPrefetchFlag = cli.BoolFlag{
		Name:  "cache.noprefetch",
		Usage: "Disable heuristic state prefetch during block import (less CPU and disk IO, more time waiting for data)",
	}
	CachePreimagesFlag = cli.BoolTFlag{
		Name:  "cache.preimages",
		Usage: "Enable recording the SHA3/keccak preimages of trie keys (default: true)",
	}
	FDLimitFlag = cli.IntFlag{
		Name:  "fdlimit",
		Usage: "Raise the open file descriptor resource limit (default = system fd limit)",
	}
	TrieCacheGenFlag = cli.IntFlag{
		Name:  "trie-cache-gens",
		Usage: "Number of trie node generations to keep in memory",
		//Value: int(state.MaxTrieCacheGen),
		Value: 0,
	}
	CryptoKZGFlag = &cli.StringFlag{
		Name:  "crypto.kzg",
		Usage: "KZG library implementation to use; gokzg (recommended) or ckzg",
		Value: "gokzg",
	}
	// Miner settings
	MiningEnabledFlag = cli.BoolFlag{
		Name:  "mine",
		Usage: "Enable mining",
	}
	MinerThreadsFlag = cli.IntFlag{
		Name:  "miner.threads",
		Usage: "Number of CPU threads to use for mining",
		Value: 0,
	}
	//MinerLegacyThreadsFlag = cli.IntFlag{
	// 	Name:  "minerthreads",
	// 	Usage: "Number of CPU threads to use for mining (deprecated, use --miner.threads)",
	// 	Value: 0,
	// }
	MinerNotifyFlag = cli.StringFlag{
		Name:  "miner.notify",
		Usage: "Comma separated HTTP URL list to notify of new work packages",
	}
	MinerGasTargetFlag = cli.Uint64Flag{
		Name:  "miner.gastarget",
		Usage: "Target gas floor for mined blocks",
		Value: ctxc.DefaultConfig.Miner.GasFloor,
	}
	// MinerLegacyGasTargetFlag = cli.Uint64Flag{
	// 	Name:  "targetgaslimit",
	// 	Usage: "Target gas floor for mined blocks (deprecated, use --miner.gastarget)",
	// 	Value: ctxc.DefaultConfig.MinerGasFloor,
	// }
	MinerGasLimitFlag = cli.Uint64Flag{
		Name:  "miner.gaslimit",
		Usage: "Target gas ceiling for mined blocks",
		Value: ctxc.DefaultConfig.Miner.GasCeil,
	}
	MinerGasPriceFlag = BigFlag{
		Name:  "miner.gasprice",
		Usage: "Minimum gas price for mining a transaction",
		Value: ctxc.DefaultConfig.Miner.GasPrice,
	}
	MinerLegacyGasPriceFlag = BigFlag{
		Name:  "gasprice",
		Usage: "Minimum gas price for mining a transaction (deprecated, use --miner.gasprice)",
		Value: ctxc.DefaultConfig.Miner.GasPrice,
	}
	MinerCoinbaseFlag = cli.StringFlag{
		Name:  "miner.coinbase",
		Usage: "Public address for block mining rewards (default = first account)",
		Value: "0",
	}
	MinerLegacyCoinbaseFlag = cli.StringFlag{
		Name:  "coinbase",
		Usage: "Public address for block mining rewards (default = first account, deprecated, use --miner.coinbase)",
		Value: "0",
	}
	MinerExtraDataFlag = cli.StringFlag{
		Name:  "miner.extradata",
		Usage: "Block extra data set by the miner (default = client version)",
	}
	MinerLegacyExtraDataFlag = cli.StringFlag{
		Name:  "extradata",
		Usage: "Block extra data set by the miner (default = client version, deprecated, use --miner.extradata)",
	}
	MinerRecommitIntervalFlag = cli.DurationFlag{
		Name:  "miner.recommit",
		Usage: "Time interval to recreate the block being mined",
		Value: ctxc.DefaultConfig.Miner.Recommit,
	}
	MinerNoVerfiyFlag = cli.BoolFlag{
		Name:  "miner.noverify",
		Usage: "Disable remote sealing verification",
	}
	MinerCudaFlag = cli.BoolFlag{
		Name:  "miner.cuda",
		Usage: "use cuda miner plugin",
	}
	//	MinerOpenCLFlag = cli.BoolFlag{
	//		Name:  "miner.opencl",
	//		Usage: "use opencl miner plugin",
	//	}
	MinerDevicesFlag = cli.StringFlag{
		Name:  "miner.devices",
		Usage: "the devices used mining, use --miner.devices=0,1",
	}
	//	MinerAlgorithmFlag = cli.StringFlag{
	//		Name:  "miner.algorithm",
	//		Usage: "use mining algorithm, --miner.algorithm=cuckoo/cuckaroo",
	//              Value: "cuckaroo"
	//	}
	InferDeviceTypeFlag = cli.StringFlag{
		Name:  "infer.devicetype",
		Usage: "infer device type : cpu or gpu",
		Value: "cpu",
	}
	InferDeviceIdFlag = cli.IntFlag{
		Name:  "infer.device",
		Usage: "the device used infering, use --infer.device=2, not available on cpu",
		Value: 0,
	}
	InferPortFlag = cli.IntFlag{
		Name:  "infer.port",
		Usage: "local infer port",
		Value: 4321,
	}
	InferMemoryFlag = cli.IntFlag{
		Name:  "infer.memory",
		Usage: "the maximum memory usage of infer engine, use --infer.memory=4096. shoule at least be 2048 (MiB)",
		Value: int(synapse.DefaultConfig.MaxMemoryUsage >> 20),
	}

	// Account settings
	UnlockedAccountFlag = cli.StringFlag{
		Name:  "unlock",
		Usage: "Comma separated list of accounts to unlock",
		Value: "",
	}
	PasswordFileFlag = cli.StringFlag{
		Name:  "password",
		Usage: "Password file to use for non-interactive password input",
		Value: "",
	}
	ExternalSignerFlag = cli.StringFlag{
		Name:  "signer",
		Usage: "External signer (url or path to ipc file)",
		Value: "",
	}
	InsecureUnlockAllowedFlag = cli.BoolFlag{
		Name:  "allow-insecure-unlock",
		Usage: "Allow insecure account unlocking when account-related RPCs are exposed by http",
	}
	RPCGlobalGasCapFlag = cli.Uint64Flag{
		Name:  "rpc.gascap",
		Usage: "Sets a cap on gas that can be used in ctxc_call/estimateGas",
		Value: ctxc.DefaultConfig.RPCGasCap,
	}
	RPCGlobalTxFeeCapFlag = cli.Float64Flag{
		Name:  "rpc.txfeecap",
		Usage: "Sets a cap on transaction fee (in ctxc) that can be sent via the RPC APIs (0 = no cap)",
		Value: ctxc.DefaultConfig.RPCTxFeeCap,
	}

	VMEnableDebugFlag = cli.BoolFlag{
		Name:  "vmdebug",
		Usage: "Record information useful for VM and contract debugging",
	}
	// Logging and debug settings
	// CortexStatsURLFlag = cli.StringFlag{
	// 	Name:  "stats",
	// 	Usage: "Reporting URL of a ctxcstats service (nodename:secret@host:port)",
	// }
	FakePoWFlag = cli.BoolFlag{
		Name:  "fakepow",
		Usage: "Disables proof-of-work verification",
	}
	// NoCompactionFlag = cli.BoolFlag{
	// 	Name:  "nocompaction",
	// 	Usage: "Disables db compaction after import",
	// }
	// RPC settings
	RPCEnabledFlag = cli.BoolFlag{
		Name:  "rpc",
		Usage: "Enable the HTTP-RPC server",
	}
	RPCListenAddrFlag = cli.StringFlag{
		Name:  "rpcaddr",
		Usage: "HTTP-RPC server listening interface",
		Value: node.DefaultHTTPHost,
	}
	RPCPortFlag = cli.IntFlag{
		Name:  "rpcport",
		Usage: "HTTP-RPC server listening port",
		Value: node.DefaultHTTPPort,
	}
	RPCCORSDomainFlag = cli.StringFlag{
		Name:  "rpccorsdomain",
		Usage: "Comma separated list of domains from which to accept cross origin requests (browser enforced)",
		Value: "",
	}
	RPCVirtualHostsFlag = cli.StringFlag{
		Name:  "rpcvhosts",
		Usage: "Comma separated list of virtual hostnames from which to accept requests (server enforced). Accepts '*' wildcard.",
		Value: strings.Join(node.DefaultConfig.HTTPVirtualHosts, ","),
	}
	RPCApiFlag = cli.StringFlag{
		Name:  "rpcapi",
		Usage: "API's offered over the HTTP-RPC interface",
		Value: "",
	}
	IPCDisabledFlag = cli.BoolFlag{
		Name:  "ipcdisable",
		Usage: "Disable the IPC-RPC server",
	}
	IPCPathFlag = DirectoryFlag{
		Name:  "ipcpath",
		Usage: "Filename for IPC socket/pipe within the datadir (explicit paths escape it)",
		Value: DirectoryString{"cortex.ipc"},
	}
	WSEnabledFlag = cli.BoolFlag{
		Name:  "ws",
		Usage: "Enable the WS-RPC server",
	}
	WSListenAddrFlag = cli.StringFlag{
		Name:  "wsaddr",
		Usage: "WS-RPC server listening interface",
		Value: node.DefaultWSHost,
	}
	WSPortFlag = cli.IntFlag{
		Name:  "wsport",
		Usage: "WS-RPC server listening port",
		Value: node.DefaultWSPort,
	}
	WSApiFlag = cli.StringFlag{
		Name:  "wsapi",
		Usage: "API's offered over the WS-RPC interface",
		Value: "",
	}
	WSAllowedOriginsFlag = cli.StringFlag{
		Name:  "wsorigins",
		Usage: "Origins from which to accept websockets requests",
		Value: "",
	}
	ExecFlag = cli.StringFlag{
		Name:  "exec",
		Usage: "Execute JavaScript statement",
	}
	PreloadJSFlag = cli.StringFlag{
		Name:  "preload",
		Usage: "Comma separated list of JavaScript files to preload into the console",
	}
	AllowUnprotectedTxs = cli.BoolFlag{
		Name:  "rpc.allow-unprotected-txs",
		Usage: "Allow for unprotected (non EIP155 signed) transactions to be submitted via RPC",
	}

	// Network Settings
	MaxPeersFlag = cli.IntFlag{
		Name:  "maxpeers",
		Usage: "Maximum number of network peers (network disabled if set to 0)",
		Value: node.DefaultConfig.P2P.MaxPeers,
	}
	MaxPendingPeersFlag = cli.IntFlag{
		Name:  "maxpendpeers",
		Usage: "Maximum number of pending connection attempts (defaults used if set to 0)",
		Value: 0,
	}
	ListenPortFlag = cli.IntFlag{
		Name:  "port",
		Usage: "Network listening port (mainnet: '40404' dolores: '40405' bernard: '40406')",
		Value: 40404,
	}
	BootnodesFlag = cli.StringFlag{
		Name:  "bootnodes",
		Usage: "Comma separated enode URLs for P2P discovery bootstrap",
		Value: "",
	}
	BootnodesV4Flag = cli.StringFlag{
		Name:  "bootnodesv4",
		Usage: "Comma separated enode URLs for P2P v4 discovery bootstrap",
		Value: "",
	}
	BootnodesV5Flag = cli.StringFlag{
		Name:  "bootnodesv5",
		Usage: "Comma separated enode URLs for P2P v5 discovery bootstrap",
		Value: "",
	}
	NodeKeyFileFlag = cli.StringFlag{
		Name:  "nodekey",
		Usage: "P2P node key file",
	}
	NodeKeyHexFlag = cli.StringFlag{
		Name:  "nodekeyhex",
		Usage: "P2P node key as hex (for testing)",
	}
	NATFlag = cli.StringFlag{
		Name:  "nat",
		Usage: "NAT port mapping mechanism (any|none|upnp|pmp|extip:<IP>)",
		Value: "any",
	}
	NoDiscoverFlag = cli.BoolFlag{
		Name:  "nodiscover",
		Usage: "Disables the peer discovery mechanism (manual peer addition)",
	}
	DiscoveryV4Flag = cli.BoolFlag{
		Name:  "discovery.v4",
		Usage: "Enables the V4 discovery mechanism",
	}
	DiscoveryV5Flag = cli.BoolFlag{
		Name:  "v5disc",
		Usage: "Enables the experimental RLPx V5 (Topic Discovery) mechanism",
	}
	DNSDiscoveryFlag = cli.StringFlag{
		Name:  "discovery.dns",
		Usage: "Sets DNS discovery entry points (use \"\" to disable DNS)",
	}
	NetrestrictFlag = cli.StringFlag{
		Name:  "netrestrict",
		Usage: "Restricts network communication to the given IP networks (CIDR masks)",
	}
	DiscoveryPortFlag = &cli.IntFlag{
		Name:  "discovery.port",
		Usage: "Use a custom UDP port for P2P discovery",
		Value: 40404,
	}

	// ATM the url is left to the user and deployment to
	JSpathFlag = DirectoryFlag{
		Name:  "jspath",
		Usage: "JavaScript root path for `loadScript`",
		Value: DirectoryString{"."},
	}

	// Gas price oracle settings
	GpoBlocksFlag = cli.IntFlag{
		Name:  "gpoblocks",
		Usage: "Number of recent blocks to check for gas prices",
		Value: ctxc.DefaultConfig.GPO.Blocks,
	}

	GpoPercentileFlag = cli.IntFlag{
		Name:  "gpopercentile",
		Usage: "Suggested gas price is the given percentile of a set of recent transaction gas prices",
		Value: ctxc.DefaultConfig.GPO.Percentile,
	}
	GpoMaxGasPriceFlag = cli.Int64Flag{
		Name:  "gpo.maxprice",
		Usage: "Maximum gas price will be recommended by gpo",
		Value: ctxc.DefaultConfig.GPO.MaxPrice.Int64(),
	}
	WhisperEnabledFlag = cli.BoolFlag{
		Name:  "shh",
		Usage: "Enable Whisper",
	}
	WhisperMaxMessageSizeFlag = cli.IntFlag{
		Name:  "shh.maxmessagesize",
		Usage: "Max message size accepted",
		Value: int(whisper.DefaultMaxMessageSize),
	}
	WhisperMinPOWFlag = cli.Float64Flag{
		Name:  "shh.pow",
		Usage: "Minimum POW accepted",
		Value: whisper.DefaultMinimumPoW,
	}
	WhisperRestrictConnectionBetweenLightClientsFlag = cli.BoolFlag{
		Name:  "shh.restrict-light",
		Usage: "Restrict connection between two whisper light clients",
	}

	// Metrics flags
	MetricsEnabledFlag = cli.BoolFlag{
		Name:  "metrics",
		Usage: "Enable metrics collection and reporting",
	}
	MetricsEnabledExpensiveFlag = cli.BoolFlag{
		Name:  "metrics.expensive",
		Usage: "Enable expensive metrics collection and reporting",
	}
	// MetricsHTTPFlag defines the endpoint for a stand-alone metrics HTTP endpoint.
	// Since the pprof service enables sensitive/vulnerable behavior, this allows a user
	// to enable a public-OK metrics endpoint without having to worry about ALSO exposing
	// other profiling behavior or information.
	MetricsHTTPFlag = cli.StringFlag{
		Name:  "metrics.addr",
		Usage: "Enable stand-alone metrics HTTP server listening interface",
		Value: "127.0.0.1",
	}
	MetricsPortFlag = cli.IntFlag{
		Name:  "metrics.port",
		Usage: "Metrics HTTP server listening port",
		Value: 6060,
	}
	MetricsEnableInfluxDBFlag = cli.BoolFlag{
		Name:  "metrics.influxdb",
		Usage: "Enable metrics export/push to an external InfluxDB database",
	}
	MetricsInfluxDBEndpointFlag = cli.StringFlag{
		Name:  "metrics.influxdb.endpoint",
		Usage: "InfluxDB API endpoint to report metrics to",
		Value: "http://localhost:8086",
	}
	MetricsInfluxDBDatabaseFlag = cli.StringFlag{
		Name:  "metrics.influxdb.database",
		Usage: "InfluxDB database name to push reported metrics to",
		Value: "cortex",
	}
	MetricsInfluxDBUsernameFlag = cli.StringFlag{
		Name:  "metrics.influxdb.username",
		Usage: "Username to authorize access to the database",
		Value: "test",
	}
	MetricsInfluxDBPasswordFlag = cli.StringFlag{
		Name:  "metrics.influxdb.password",
		Usage: "Password to authorize access to the database",
		Value: "test",
	}
	// The `host` tag is part of every measurement sent to InfluxDB. Queries on tags are faster in InfluxDB.
	// It is used so that we can group all nodes and average a measurement across all of them, but also so
	// that we can select a specific node and inspect its measurements.
	// https://docs.influxdata.com/influxdb/v1.4/concepts/key_concepts/#tag-key
	//MetricsInfluxDBHostTagFlag = cli.StringFlag{
	//	Name:  "metrics.influxdb.host.tag",
	//	Usage: "InfluxDB `host` tag attached to all measurements",
	//	Value: "localhost",
	//}
	MetricsInfluxDBTagsFlag = cli.StringFlag{
		Name:  "metrics.influxdb.tags",
		Usage: "Comma-separated InfluxDB tags (key/values) attached to all measurements",
		Value: "host=localhost",
	}

	// DatabaseFlags is the flag group of all database flags.
	DatabaseFlags = []cli.Flag{
		DataDirFlag,
		AncientFlag,
		DBEngineFlag,
	}
)

// MakeDataDir retrieves the currently requested data directory, terminating
// if none (or the empty string) is specified. If the node is starting a testnet,
// the a subdirectory of the specified datadir will be used.
func MakeDataDir(ctx *cli.Context) string {
	switch {
	case ctx.GlobalIsSet(DataDirFlag.Name):
		return ctx.GlobalString(DataDirFlag.Name)
	case ctx.GlobalBool(BernardFlag.Name):
		return filepath.Join(node.DefaultDataDir(), BernardFlag.Name)
	case ctx.GlobalBool(DoloresFlag.Name):
		return filepath.Join(node.DefaultDataDir(), DoloresFlag.Name)
	}

	return node.DefaultDataDir()
}

// MakeStorageDir retrieves the currently requested data directory, terminating
// if none (or the empty string) is specified.
func MakeStorageDir(ctx *cli.Context) string {
	switch {
	case ctx.GlobalIsSet(StorageDirFlag.Name):
		return ctx.GlobalString(StorageDirFlag.Name)
	}

	return filepath.Join(MakeDataDir(ctx), "storage")
}

// setNodeKey creates a node key from set command line flags, either loading it
// from a file or as a specified hex value. If neither flags were provided, this
// method returns nil and an emphemeral key is to be generated.
func setNodeKey(ctx *cli.Context, cfg *p2p.Config) {
	var (
		hex  = ctx.GlobalString(NodeKeyHexFlag.Name)
		file = ctx.GlobalString(NodeKeyFileFlag.Name)
		key  *ecdsa.PrivateKey
		err  error
	)
	switch {
	case file != "" && hex != "":
		Fatalf("Options %q and %q are mutually exclusive", NodeKeyFileFlag.Name, NodeKeyHexFlag.Name)
	case file != "":
		if key, err = crypto.LoadECDSA(file); err != nil {
			Fatalf("Option %q: %v", NodeKeyFileFlag.Name, err)
		}
		cfg.PrivateKey = key
	case hex != "":
		if key, err = crypto.HexToECDSA(hex); err != nil {
			Fatalf("Option %q: %v", NodeKeyHexFlag.Name, err)
		}
		cfg.PrivateKey = key
	}
}

// setNodeUserIdent creates the user identifier from CLI flags.
func setNodeUserIdent(ctx *cli.Context, cfg *node.Config) {
	if identity := ctx.GlobalString(IdentityFlag.Name); len(identity) > 0 {
		cfg.UserIdent = identity
	}
}

// setBootstrapNodes creates a list of bootstrap nodes from the command line
// flags, reverting to pre-configured ones if none have been specified.
func setBootstrapNodes(ctx *cli.Context, cfg *p2p.Config) {
	urls := params.MainnetBootnodes
	switch {
	case ctx.GlobalIsSet(BootnodesFlag.Name) || ctx.GlobalIsSet(BootnodesV4Flag.Name):
		if ctx.GlobalIsSet(BootnodesV4Flag.Name) {
			urls = strings.Split(ctx.GlobalString(BootnodesV4Flag.Name), ",")
		} else {
			urls = strings.Split(ctx.GlobalString(BootnodesFlag.Name), ",")
		}
	case ctx.GlobalBool(BernardFlag.Name):
		urls = params.BernardBootnodes
	case ctx.GlobalBool(DoloresFlag.Name):
		urls = params.DoloresBootnodes
	case cfg.BootstrapNodes != nil:
		return // already set, don't apply defaults.
	}

	cfg.BootstrapNodes = make([]*enode.Node, 0, len(urls))
	for _, url := range urls {
		if url != "" {
			node, err := enode.Parse(enode.ValidSchemes, url)
			if err != nil {
				log.Crit("Bootstrap URL invalid", "enode", url, "err", err)
				continue
			}
			cfg.BootstrapNodes = append(cfg.BootstrapNodes, node)
		}
	}
}

// setBootstrapNodesV5 creates a list of bootstrap nodes from the command line
// flags, reverting to pre-configured ones if none have been specified.
func setBootstrapNodesV5(ctx *cli.Context, cfg *p2p.Config) {
	urls := params.V5Bootnodes
	switch {
	case ctx.GlobalIsSet(BootnodesFlag.Name) || ctx.GlobalIsSet(BootnodesV5Flag.Name):
		if ctx.GlobalIsSet(BootnodesV5Flag.Name) {
			urls = splitAndTrim(ctx.GlobalString(BootnodesV5Flag.Name))
		} else {
			urls = splitAndTrim(ctx.GlobalString(BootnodesFlag.Name))
		}
	//case ctx.GlobalBool(BernardFlag.Name):
	//	urls = params.BernardBootnodes
	//case ctx.GlobalBool(DoloresFlag.Name):
	//	urls = params.BernardBootnodes
	case cfg.BootstrapNodesV5 != nil:
		return // already set, don't apply defaults.
	}

	cfg.BootstrapNodesV5 = make([]*enode.Node, 0, len(urls))
	for _, url := range urls {
		if url != "" {
			node, err := enode.Parse(enode.ValidSchemes, url)
			if err != nil {
				log.Error("Bootstrap URL invalid", "enode", url, "err", err)
				continue
			}
			cfg.BootstrapNodesV5 = append(cfg.BootstrapNodesV5, node)
		}
	}
}

// setListenAddress creates a TCP listening address string from set command
// line flags.
func setListenAddress(ctx *cli.Context, cfg *p2p.Config) {
	if ctx.GlobalIsSet(ListenPortFlag.Name) {
		cfg.ListenAddr = fmt.Sprintf(":%d", ctx.GlobalInt(ListenPortFlag.Name))
	} else if ctx.GlobalBool(DoloresFlag.Name) {
		cfg.ListenAddr = fmt.Sprintf(":%d", 40405)
	} else if ctx.GlobalBool(BernardFlag.Name) {
		cfg.ListenAddr = fmt.Sprintf(":%d", 40406)
	}

	if ctx.GlobalIsSet(DiscoveryPortFlag.Name) {
		cfg.DiscAddr = fmt.Sprintf(":%d", ctx.GlobalInt(DiscoveryPortFlag.Name))
	}
}

// setNAT creates a port mapper from command line flags.
func setNAT(ctx *cli.Context, cfg *p2p.Config) {
	if ctx.GlobalIsSet(NATFlag.Name) {
		natif, err := nat.Parse(ctx.GlobalString(NATFlag.Name))
		if err != nil {
			Fatalf("Option %s: %v", NATFlag.Name, err)
		}
		cfg.NAT = natif
	}
}

// splitAndTrim splits input separated by a comma
// and trims excessive white space from the substrings.
func splitAndTrim(input string) (ret []string) {
	l := strings.Split(input, ",")
	for _, r := range l {
		r = strings.TrimSpace(r)
		if len(r) > 0 {
			ret = append(ret, r)
		}
	}
	return ret
}

// setHTTP creates the HTTP RPC listener interface string from the set
// command line flags, returning empty if the HTTP endpoint is disabled.
func setHTTP(ctx *cli.Context, cfg *node.Config) {
	if ctx.GlobalBool(RPCEnabledFlag.Name) && cfg.HTTPHost == "" {
		cfg.HTTPHost = "127.0.0.1"
		if ctx.GlobalIsSet(RPCListenAddrFlag.Name) {
			cfg.HTTPHost = ctx.GlobalString(RPCListenAddrFlag.Name)
		}
	}

	if ctx.GlobalIsSet(RPCPortFlag.Name) {
		cfg.HTTPPort = ctx.GlobalInt(RPCPortFlag.Name)
	}
	if ctx.GlobalIsSet(RPCCORSDomainFlag.Name) {
		cfg.HTTPCors = splitAndTrim(ctx.GlobalString(RPCCORSDomainFlag.Name))
	}
	if ctx.GlobalIsSet(RPCApiFlag.Name) {
		cfg.HTTPModules = splitAndTrim(ctx.GlobalString(RPCApiFlag.Name))
	}
	if ctx.GlobalIsSet(RPCVirtualHostsFlag.Name) {
		cfg.HTTPVirtualHosts = splitAndTrim(ctx.GlobalString(RPCVirtualHostsFlag.Name))
	}
	if ctx.GlobalIsSet(AllowUnprotectedTxs.Name) {
		cfg.AllowUnprotectedTxs = ctx.GlobalBool(AllowUnprotectedTxs.Name)
	}
}

// setWS creates the WebSocket RPC listener interface string from the set
// command line flags, returning empty if the HTTP endpoint is disabled.
func setWS(ctx *cli.Context, cfg *node.Config) {
	if ctx.GlobalBool(WSEnabledFlag.Name) && cfg.WSHost == "" {
		cfg.WSHost = "127.0.0.1"
		if ctx.GlobalIsSet(WSListenAddrFlag.Name) {
			cfg.WSHost = ctx.GlobalString(WSListenAddrFlag.Name)
		}
	}

	if ctx.GlobalIsSet(WSPortFlag.Name) {
		cfg.WSPort = ctx.GlobalInt(WSPortFlag.Name)
	}
	if ctx.GlobalIsSet(WSAllowedOriginsFlag.Name) {
		cfg.WSOrigins = splitAndTrim(ctx.GlobalString(WSAllowedOriginsFlag.Name))
	}
	if ctx.GlobalIsSet(WSApiFlag.Name) {
		cfg.WSModules = splitAndTrim(ctx.GlobalString(WSApiFlag.Name))
	}
}

// setIPC creates an IPC path configuration from the set command line flags,
// returning an empty string if IPC was explicitly disabled, or the set path.
func setIPC(ctx *cli.Context, cfg *node.Config) {
	checkExclusive(ctx, IPCDisabledFlag, IPCPathFlag)
	switch {
	case ctx.GlobalBool(IPCDisabledFlag.Name):
		cfg.IPCPath = ""
	case ctx.GlobalIsSet(IPCPathFlag.Name):
		cfg.IPCPath = ctx.GlobalString(IPCPathFlag.Name)
	}
}

// makeDatabaseHandles raises out the number of allowed file handles per process
// for Ctxc and returns half of the allowance to assign to the database.
func makeDatabaseHandles(max int) int {
	limit, err := fdlimit.Maximum()
	if err != nil {
		Fatalf("Failed to retrieve file descriptor allowance: %v", err)
	}
	switch {
	case max == 0:
		// User didn't specify a meaningful value, use system limits
	case max < 128:
		// User specified something unhealthy, just use system defaults
		log.Error("File descriptor limit invalid (<128)", "had", max, "updated", limit)
	case max > limit:
		// User requested more than the OS allows, notify that we can't allocate it
		log.Warn("Requested file descriptors denied by OS", "req", max, "limit", limit)
	default:
		// User limit is meaningful and within allowed range, use that
		limit = max
	}
	raised, err := fdlimit.Raise(uint64(limit))
	if err != nil {
		Fatalf("Failed to raise file descriptor allowance: %v", err)
	}
	return int(raised / 2) // Leave half for networking and other stuff
}

// MakeAddress converts an account specified directly as a hex encoded string or
// a key index in the key store to an internal account representation.
func MakeAddress(ks *keystore.KeyStore, account string) (accounts.Account, error) {
	// If the specified account is a valid address, return it
	if common.IsHexAddress(account) {
		return accounts.Account{Address: common.HexToAddress(account)}, nil
	}
	// Otherwise try to interpret the account as a keystore index
	index, err := strconv.Atoi(account)
	if err != nil || index < 0 {
		return accounts.Account{}, fmt.Errorf("invalid account address or index %q", account)
	}
	log.Warn("-------------------------------------------------------------------")
	log.Warn("Referring to accounts by order in the keystore folder is dangerous!")
	log.Warn("This functionality is deprecated and will be removed in the future!")
	log.Warn("Please use explicit addresses! (can search via `cortex account list`)")
	log.Warn("-------------------------------------------------------------------")

	accs := ks.Accounts()
	if len(accs) <= index {
		return accounts.Account{}, fmt.Errorf("index %d higher than number of accounts %d", index, len(accs))
	}
	return accs[index], nil
}

// setCoinbase retrieves the coinbase either from the directly specified
// command line flags or from the keystore if CLI indexed.
func setCoinbase(ctx *cli.Context, ks *keystore.KeyStore, cfg *ctxc.Config) {
	// Extract the current coinbase, new flag overriding legacy one
	var coinbase string
	if ctx.GlobalIsSet(MinerLegacyCoinbaseFlag.Name) {
		coinbase = ctx.GlobalString(MinerLegacyCoinbaseFlag.Name)
	}
	if ctx.GlobalIsSet(MinerCoinbaseFlag.Name) {
		coinbase = ctx.GlobalString(MinerCoinbaseFlag.Name)
	}
	// Convert the coinbase into an address and configure it
	if coinbase != "" {
		if strings.HasPrefix(coinbase, "0x") || strings.HasPrefix(coinbase, "0X") {
			coinbase = coinbase[2:]
		}
		b, err := hex.DecodeString(coinbase)
		if err != nil || len(b) != common.AddressLength {
			Fatalf("-%s: invalid coinbase address %q", MinerCoinbaseFlag.Name, coinbase)
			return
		}
		account, err := MakeAddress(ks, coinbase)
		if err != nil {
			Fatalf("Invalid miner coinbase: %v", err)
		}
		cfg.Coinbase = account.Address
	}
}

// MakePasswordList reads password lines from the file specified by the global --password flag.
func MakePasswordList(ctx *cli.Context) []string {
	path := ctx.GlobalString(PasswordFileFlag.Name)
	if path == "" {
		return nil
	}
	text, err := os.ReadFile(path)
	if err != nil {
		Fatalf("Failed to read password file: %v", err)
	}
	lines := strings.Split(string(text), "\n")
	// Sanitise DOS line endings.
	for i := range lines {
		lines[i] = strings.TrimRight(lines[i], "\r")
	}
	return lines
}

func SetP2PConfig(ctx *cli.Context, cfg *p2p.Config) {
	setNodeKey(ctx, cfg)
	setNAT(ctx, cfg)
	setListenAddress(ctx, cfg)
	setBootstrapNodes(ctx, cfg)
	setBootstrapNodesV5(ctx, cfg)

	if ctx.GlobalIsSet(MaxPeersFlag.Name) {
		cfg.MaxPeers = ctx.GlobalInt(MaxPeersFlag.Name)
	}
	ctxcPeers := cfg.MaxPeers
	log.Info("Maximum peer count", "Cortex", ctxcPeers, "total", cfg.MaxPeers)

	if ctx.GlobalIsSet(MaxPendingPeersFlag.Name) {
		cfg.MaxPendingPeers = ctx.GlobalInt(MaxPendingPeersFlag.Name)
	}
	if ctx.GlobalIsSet(NoDiscoverFlag.Name) {
		cfg.NoDiscovery = true
	}

	// if we're running a light client or server, force enable the v5 peer discovery
	// unless it is explicitly disabled with --nodiscover note that explicitly specifying
	// --v5disc overrides --nodiscover, in which case the later only disables v4 discovery
	//forceV5Discovery := (lightClient || lightServer) && !ctx.GlobalBool(NoDiscoverFlag.Name)
	cfg.DiscoveryV4 = !cfg.NoDiscovery //ctx.GlobalBool(DiscoveryV4Flag.Name)
	if ctx.GlobalIsSet(DiscoveryV5Flag.Name) {
		cfg.DiscoveryV5 = ctx.GlobalBool(DiscoveryV5Flag.Name)
		//} else if forceV5Discovery {
		//        cfg.DiscoveryV5 = true
	}

	if cfg.DiscoveryV5 {
		cfg.NoDiscovery = false
	}

	if netrestrict := ctx.GlobalString(NetrestrictFlag.Name); netrestrict != "" {
		list, err := netutil.ParseNetlist(netrestrict)
		if err != nil {
			Fatalf("Option %q: %v", NetrestrictFlag.Name, err)
		}
		cfg.NetRestrict = list
	}
}

// SetNodeConfig applies node-related command line flags to the config.
func SetNodeConfig(ctx *cli.Context, cfg *node.Config) {
	SetP2PConfig(ctx, &cfg.P2P)
	setIPC(ctx, cfg)
	setHTTP(ctx, cfg)
	setWS(ctx, cfg)
	setNodeUserIdent(ctx, cfg)

	cfg.DataDir = MakeDataDir(ctx)

	// switch {
	// case ctx.GlobalIsSet(DataDirFlag.Name):
	// 	cfg.DataDir = ctx.GlobalString(DataDirFlag.Name)
	// case ctx.GlobalBool(BernardFlag.Name):
	// 	cfg.DataDir = filepath.Join(node.DefaultDataDir(), "cerebro")
	// case ctx.GlobalBool(LazynetFlag.Name):
	// 	cfg.DataDir = filepath.Join(node.DefaultDataDir(), "lazynet")
	// }

	if ctx.GlobalIsSet(ExternalSignerFlag.Name) {
		cfg.ExternalSigner = ctx.GlobalString(ExternalSignerFlag.Name)
	}

	if ctx.GlobalIsSet(KeyStoreDirFlag.Name) {
		cfg.KeyStoreDir = ctx.GlobalString(KeyStoreDirFlag.Name)
	}
	// if ctx.GlobalIsSet(LightKDFFlag.Name) {
	// 	cfg.UseLightweightKDF = ctx.GlobalBool(LightKDFFlag.Name)
	// }
	// if ctx.GlobalIsSet(NoUSBFlag.Name) {
	// 	cfg.NoUSB = ctx.GlobalBool(NoUSBFlag.Name)
	// }
	if ctx.GlobalIsSet(InsecureUnlockAllowedFlag.Name) {
		cfg.InsecureUnlockAllowed = ctx.GlobalBool(InsecureUnlockAllowedFlag.Name)
	}
	if ctx.IsSet(DBEngineFlag.Name) {
		dbEngine := ctx.String(DBEngineFlag.Name)
		if dbEngine != "leveldb" && dbEngine != "pebble" {
			Fatalf("Invalid choice for db.engine '%s', allowed 'leveldb' or 'pebble'", dbEngine)
		}
		log.Info(fmt.Sprintf("Using %s as db engine", dbEngine))
		cfg.DBEngine = dbEngine
	}
	// deprecation notice for log debug flags (TODO: find a more appropriate place to put these?)
	if ctx.IsSet(LogBacktraceAtFlag.Name) {
		log.Warn("log.backtrace flag is deprecated")
	}
	if ctx.IsSet(LogDebugFlag.Name) {
		log.Warn("log.debug flag is deprecated")
	}
}

func setGPO(ctx *cli.Context, cfg *gasprice.Config) {
	if ctx.GlobalIsSet(GpoBlocksFlag.Name) {
		cfg.Blocks = ctx.GlobalInt(GpoBlocksFlag.Name)
	}
	if ctx.GlobalIsSet(GpoPercentileFlag.Name) {
		cfg.Percentile = ctx.GlobalInt(GpoPercentileFlag.Name)
	}
	if ctx.GlobalIsSet(GpoMaxGasPriceFlag.Name) {
		cfg.MaxPrice = big.NewInt(ctx.GlobalInt64(GpoMaxGasPriceFlag.Name))
	}
}

func setTxPool(ctx *cli.Context, cfg *txpool.Config) {
	if ctx.GlobalIsSet(TxPoolLocalsFlag.Name) {
		locals := strings.Split(ctx.GlobalString(TxPoolLocalsFlag.Name), ",")
		for _, account := range locals {
			if trimmed := strings.TrimSpace(account); !common.IsHexAddress(trimmed) {
				Fatalf("Invalid account in --txpool.locals: %s", trimmed)
			} else {
				cfg.Locals = append(cfg.Locals, common.HexToAddress(account))
			}
		}
	}
	if ctx.GlobalIsSet(TxPoolNoLocalsFlag.Name) {
		cfg.NoLocals = ctx.GlobalBool(TxPoolNoLocalsFlag.Name)
	}
	if ctx.GlobalIsSet(TxPoolJournalFlag.Name) {
		cfg.Journal = ctx.GlobalString(TxPoolJournalFlag.Name)
	}
	if ctx.GlobalIsSet(TxPoolRejournalFlag.Name) {
		cfg.Rejournal = ctx.GlobalDuration(TxPoolRejournalFlag.Name)
	}
	if ctx.GlobalIsSet(TxPoolPriceLimitFlag.Name) {
		cfg.PriceLimit = ctx.GlobalUint64(TxPoolPriceLimitFlag.Name)
	}
	if ctx.GlobalIsSet(TxPoolPriceBumpFlag.Name) {
		cfg.PriceBump = ctx.GlobalUint64(TxPoolPriceBumpFlag.Name)
	}
	if ctx.GlobalIsSet(TxPoolAccountSlotsFlag.Name) {
		cfg.AccountSlots = ctx.GlobalUint64(TxPoolAccountSlotsFlag.Name)
	}
	if ctx.GlobalIsSet(TxPoolGlobalSlotsFlag.Name) {
		cfg.GlobalSlots = ctx.GlobalUint64(TxPoolGlobalSlotsFlag.Name)
	}
	if ctx.GlobalIsSet(TxPoolAccountQueueFlag.Name) {
		cfg.AccountQueue = ctx.GlobalUint64(TxPoolAccountQueueFlag.Name)
	}
	if ctx.GlobalIsSet(TxPoolGlobalQueueFlag.Name) {
		cfg.GlobalQueue = ctx.GlobalUint64(TxPoolGlobalQueueFlag.Name)
	}
	if ctx.GlobalIsSet(TxPoolLifetimeFlag.Name) {
		cfg.Lifetime = ctx.GlobalDuration(TxPoolLifetimeFlag.Name)
	}
}

// checkExclusive verifies that only a single instance of the provided flags was
// set by the user. Each flag might optionally be followed by a string type to
// specialize it further.
func checkExclusive(ctx *cli.Context, args ...any) {
	set := make([]string, 0, 1)
	for i := 0; i < len(args); i++ {
		// Make sure the next argument is a flag and skip if not set
		flag, ok := args[i].(cli.Flag)
		if !ok {
			panic(fmt.Sprintf("invalid argument, not cli.Flag type: %T", args[i]))
		}
		// Check if next arg extends current and expand its name if so
		name := flag.GetName()

		if i+1 < len(args) {
			switch option := args[i+1].(type) {
			case string:
				// Extended flag, expand the name and shift the arguments
				if ctx.GlobalString(flag.GetName()) == option {
					name += "=" + option
				}
				i++

			case cli.Flag:
			default:
				panic(fmt.Sprintf("invalid argument, not cli.Flag or string extension: %T", args[i+1]))
			}
		}
		// Mark the flag if it's set
		if ctx.GlobalIsSet(flag.GetName()) {
			set = append(set, "--"+name)
		}
	}
	if len(set) > 1 {
		Fatalf("Flags %v can't be used at the same time", strings.Join(set, ", "))
	}
}

func setWhitelist(ctx *cli.Context, cfg *ctxc.Config) {
	whitelist := ctx.GlobalString(WhitelistFlag.Name)
	if whitelist == "" {
		return
	}
	cfg.Whitelist = make(map[uint64]common.Hash)
	for _, entry := range strings.Split(whitelist, ",") {
		parts := strings.Split(entry, "=")
		if len(parts) != 2 {
			Fatalf("Invalid whitelist entry: %s", entry)
		}
		number, err := strconv.ParseUint(parts[0], 0, 64)
		if err != nil {
			Fatalf("Invalid whitelist block number %s: %v", parts[0], err)
		}
		var hash common.Hash
		if err = hash.UnmarshalText([]byte(parts[1])); err != nil {
			Fatalf("Invalid whitelist hash %s: %v", parts[1], err)
		}
		cfg.Whitelist[number] = hash
	}
}

func CheckExclusive(ctx *cli.Context, args ...any) {
	set := make([]string, 0, 1)
	for i := 0; i < len(args); i++ {
		// Make sure the next argument is a flag and skip if not set
		flag, ok := args[i].(cli.Flag)
		if !ok {
			panic(fmt.Sprintf("invalid argument, not cli.Flag type: %T", args[i]))
		}
		// Check if next arg extends current and expand its name if so
		name := flag.GetName()

		if i+1 < len(args) {
			switch option := args[i+1].(type) {
			case string:
				// Extended flag check, make sure value set doesn't conflict with passed in option
				if ctx.GlobalString(flag.GetName()) == option {
					name += "=" + option
					set = append(set, "--"+name)
				}
				// shift arguments and continue
				i++
				continue

			case cli.Flag:
			default:
				panic(fmt.Sprintf("invalid argument, not cli.Flag or string extension: %T", args[i+1]))
			}
		}
		// Mark the flag if it's set
		if ctx.GlobalIsSet(flag.GetName()) {
			set = append(set, "--"+name)
		}
	}
	if len(set) > 1 {
		Fatalf("Flags %v can't be used at the same time", strings.Join(set, ", "))
	}
}

// SetShhConfig applies shh-related command line flags to the config.
func SetShhConfig(ctx *cli.Context, stack *node.Node, cfg *whisper.Config) {
	if ctx.GlobalIsSet(WhisperMaxMessageSizeFlag.Name) {
		cfg.MaxMessageSize = uint32(ctx.GlobalUint(WhisperMaxMessageSizeFlag.Name))
	}
	if ctx.GlobalIsSet(WhisperMinPOWFlag.Name) {
		cfg.MinimumAcceptedPOW = ctx.GlobalFloat64(WhisperMinPOWFlag.Name)
	}
	if ctx.GlobalIsSet(WhisperRestrictConnectionBetweenLightClientsFlag.Name) {
		cfg.RestrictConnectionBetweenLightClients = true
	}
}

// SetCortexConfig applies ctxc-related command line flags to the config.
func SetCortexConfig(ctx *cli.Context, stack *node.Node, cfg *ctxc.Config) {
	// Avoid conflicting network flags
	// checkExclusive(ctx, DeveloperFlag, BernardFlag, LazynetFlag)
	//CheckExclusive(ctx, DeveloperFlag, ExternalSignerFlag) // Can't use both ephemeral unlocked and external signer
	CheckExclusive(ctx, GCModeFlag, "archive", TxLookupLimitFlag)

	if ctx.GlobalString(GCModeFlag.Name) == "archive" && ctx.GlobalUint64(TxLookupLimitFlag.Name) != 0 {
		ctx.GlobalSet(TxLookupLimitFlag.Name, "0")
		log.Warn("Disable transaction unindexing for archive node")
	}
	var ks *keystore.KeyStore
	if keystores := stack.AccountManager().Backends(keystore.KeyStoreType); len(keystores) > 0 {
		ks = keystores[0].(*keystore.KeyStore)
	}
	setCoinbase(ctx, ks, cfg)
	setGPO(ctx, &cfg.GPO)
	setTxPool(ctx, &cfg.TxPool)
	setWhitelist(ctx, cfg)
	// Cap the cache allowance and tune the garbage collector
	mem, err := gopsutil.VirtualMemory()
	// Workaround until OpenBSD support lands into gosigar
	// Check https://github.com/elastic/gosigar#supported-platforms
	if err == nil {
		if 32<<(^uintptr(0)>>63) == 32 && mem.Total > 2*1024*1024*1024 {
			log.Warn("Lowering memory allowance on 32bit arch", "available", mem.Total/1024/1024, "addressable", 2*1024)
			mem.Total = 2 * 1024 * 1024 * 1024
		}
		allowance := int(mem.Total / 1024 / 1024 / 3)
		if cache := ctx.GlobalInt(CacheFlag.Name); cache > allowance {
			log.Warn("Sanitizing cache to Go's GC limits", "provided", cache, "updated", allowance)
			ctx.GlobalSet(CacheFlag.Name, strconv.Itoa(allowance))
		}
	}
	// Ensure Go's GC ignores the database cache for trigger percentage
	cache := ctx.GlobalInt(CacheFlag.Name)
	gogc := math.Max(20, math.Min(100, 100/(float64(cache)/1024)))

	log.Info("Sanitizing Go's GC trigger", "percent", int(gogc), "cache", cache)
	godebug.SetGCPercent(int(gogc))

	if ctx.GlobalIsSet(SyncModeFlag.Name) {
		cfg.SyncMode = *GlobalTextMarshaler(ctx, SyncModeFlag.Name).(*downloader.SyncMode)
	}
	if ctx.GlobalIsSet(NetworkIdFlag.Name) {
		cfg.NetworkId = ctx.GlobalUint64(NetworkIdFlag.Name)
	}
	if ctx.GlobalIsSet(CacheFlag.Name) || ctx.GlobalIsSet(CacheDatabaseFlag.Name) {
		cfg.DatabaseCache = ctx.GlobalInt(CacheFlag.Name) * ctx.GlobalInt(CacheDatabaseFlag.Name) / 100
	}
	cfg.DatabaseHandles = makeDatabaseHandles(ctx.GlobalInt(FDLimitFlag.Name))
	if ctx.GlobalIsSet(AncientFlag.Name) {
		cfg.DatabaseFreezer = ctx.GlobalString(AncientFlag.Name)
	}

	if gcmode := ctx.GlobalString(GCModeFlag.Name); gcmode != "full" && gcmode != "archive" {
		Fatalf("--%s must be either 'full' or 'archive'", GCModeFlag.Name)
	}
	if ctx.GlobalIsSet(GCModeFlag.Name) {
		cfg.NoPruning = ctx.GlobalString(GCModeFlag.Name) == "archive"
	}
	if ctx.GlobalIsSet(CacheNoPrefetchFlag.Name) {
		cfg.NoPrefetch = ctx.GlobalBool(CacheNoPrefetchFlag.Name)
	}
	// Read the value from the flag no matter if it's set or not.
	cfg.Preimages = ctx.GlobalBool(CachePreimagesFlag.Name)
	if cfg.NoPruning && !cfg.Preimages {
		cfg.Preimages = true
		log.Info("Enabling recording of key preimages since archive mode is used")
	}
	if ctx.GlobalIsSet(TxLookupLimitFlag.Name) {
		cfg.TxLookupLimit = ctx.GlobalUint64(TxLookupLimitFlag.Name)
	}
	if ctx.GlobalIsSet(CacheFlag.Name) || ctx.GlobalIsSet(CacheTrieFlag.Name) {
		cfg.TrieCleanCache = ctx.GlobalInt(CacheFlag.Name) * ctx.GlobalInt(CacheTrieFlag.Name) / 100
	}
	if ctx.GlobalIsSet(CacheTrieJournalFlag.Name) {
		cfg.TrieCleanCacheJournal = ctx.GlobalString(CacheTrieJournalFlag.Name)
	}
	if ctx.GlobalIsSet(CacheTrieRejournalFlag.Name) {
		cfg.TrieCleanCacheRejournal = ctx.GlobalDuration(CacheTrieRejournalFlag.Name)
	}
	if ctx.GlobalIsSet(CacheFlag.Name) || ctx.GlobalIsSet(CacheGCFlag.Name) {
		cfg.TrieDirtyCache = ctx.GlobalInt(CacheFlag.Name) * ctx.GlobalInt(CacheGCFlag.Name) / 100
	}
	if ctx.GlobalIsSet(CacheFlag.Name) || ctx.GlobalIsSet(CacheSnapshotFlag.Name) {
		cfg.SnapshotCache = ctx.GlobalInt(CacheFlag.Name) * ctx.GlobalInt(CacheSnapshotFlag.Name) / 100
	}
	if !ctx.GlobalIsSet(SnapshotFlag.Name) {
		cfg.TrieCleanCache += cfg.SnapshotCache
		cfg.SnapshotCache = 0 // Disabled
	}
	if ctx.GlobalIsSet(DocRootFlag.Name) {
		cfg.DocRoot = ctx.GlobalString(DocRootFlag.Name)
	}
	if ctx.GlobalIsSet(VMEnableDebugFlag.Name) {
		cfg.EnablePreimageRecording = ctx.GlobalBool(VMEnableDebugFlag.Name)
	}
	if ctx.GlobalIsSet(MinerLegacyExtraDataFlag.Name) {
		cfg.Miner.ExtraData = []byte(ctx.GlobalString(MinerLegacyExtraDataFlag.Name))
	}
	if ctx.GlobalIsSet(MinerExtraDataFlag.Name) {
		cfg.Miner.ExtraData = []byte(ctx.GlobalString(MinerExtraDataFlag.Name))
	}
	// if ctx.GlobalIsSet(MinerLegacyGasTargetFlag.Name) {
	//	cfg.MinerGasFloor = ctx.GlobalUint64(MinerLegacyGasTargetFlag.Name)
	// }
	if ctx.GlobalIsSet(MinerGasTargetFlag.Name) {
		cfg.Miner.GasFloor = ctx.GlobalUint64(MinerGasTargetFlag.Name)
	}
	if ctx.GlobalIsSet(MinerGasLimitFlag.Name) {
		cfg.Miner.GasCeil = ctx.GlobalUint64(MinerGasLimitFlag.Name)
	}
	if ctx.GlobalIsSet(MinerLegacyGasPriceFlag.Name) {
		cfg.Miner.GasPrice = GlobalBig(ctx, MinerLegacyGasPriceFlag.Name)
	}
	if ctx.GlobalIsSet(MinerGasPriceFlag.Name) {
		cfg.Miner.GasPrice = GlobalBig(ctx, MinerGasPriceFlag.Name)
	}
	if ctx.GlobalIsSet(MinerRecommitIntervalFlag.Name) {
		cfg.Miner.Recommit = ctx.GlobalDuration(MinerRecommitIntervalFlag.Name)
	}
	if ctx.GlobalIsSet(MinerNoVerfiyFlag.Name) {
		cfg.Miner.Noverify = ctx.GlobalBool(MinerNoVerfiyFlag.Name)
	}

	if ctx.GlobalIsSet(MiningEnabledFlag.Name) {
		//cfg.Cuckoo.Mine = true
	}

	if ctx.GlobalIsSet(ViperFlag.Name) {
		log.Warn("Viper mode is ON", "sprout", "180s")
		cfg.Viper = true
	}

	if ctx.GlobalIsSet(MinerCudaFlag.Name) {
		cfg.Miner.Cuda = ctx.Bool(MinerCudaFlag.Name)
		cfg.Cuckoo.UseCuda = cfg.Miner.Cuda
	}
	//	if ctx.GlobalIsSet(MinerOpenCLFlag.Name) {
	//		cfg.MinerOpenCL = ctx.Bool(MinerOpenCLFlag.Name)
	//		cfg.Cuckoo.UseOpenCL = cfg.MinerOpenCL
	//	}
	if ctx.GlobalIsSet(RPCGlobalGasCapFlag.Name) {
		cfg.RPCGasCap = ctx.GlobalUint64(RPCGlobalGasCapFlag.Name)
	}
	if cfg.RPCGasCap != 0 {
		log.Info("Set global gas cap", "cap", cfg.RPCGasCap)
	} else {
		log.Info("Global gas cap disabled")
	}
	if ctx.GlobalIsSet(RPCGlobalTxFeeCapFlag.Name) {
		cfg.RPCTxFeeCap = ctx.GlobalFloat64(RPCGlobalTxFeeCapFlag.Name)
	}
	if ctx.GlobalIsSet(NoDiscoverFlag.Name) {
		cfg.DiscoveryURLs = []string{}
	} else if ctx.GlobalIsSet(DNSDiscoveryFlag.Name) {
		urls := ctx.GlobalString(DNSDiscoveryFlag.Name)
		if urls == "" {
			cfg.DiscoveryURLs = []string{}
		} else {
			cfg.DiscoveryURLs = splitAndTrim(urls)
		}
	}
	cfg.Miner.Devices = ctx.GlobalString(MinerDevicesFlag.Name)
	cfg.Cuckoo.StrDeviceIds = cfg.Miner.Devices
	cfg.Cuckoo.Threads = ctx.GlobalInt(MinerThreadsFlag.Name)
	cfg.Cuckoo.Algorithm = "cuckaroo" //ctx.GlobalString(MinerAlgorithmFlag.Name)
	// cfg.InferURI = ctx.GlobalString(ModelCallInterfaceFlag.Name)
	cfg.StorageDir = MakeStorageDir(ctx)
	//cfg.RpcURI = ctx.GlobalString(StorageRpcFlag.Name)
	cfg.InferDeviceType = ctx.GlobalString(InferDeviceTypeFlag.Name)
	if cfg.InferDeviceType == "cpu" {
	} else if cfg.InferDeviceType == "gpu" {
		cfg.InferDeviceType = "cuda"
	} else if strings.HasPrefix(cfg.InferDeviceType, "remote") {
		u, err := url.Parse(cfg.InferDeviceType)
		if err == nil && u.Scheme == "remote" && len(u.Hostname()) > 0 && len(u.Port()) > 0 {
			cfg.InferURI = "http://" + u.Hostname() + ":" + u.Port() + "/infer"
			log.Info("Cortex", "inferUri", cfg.InferURI)
		} else {
			panic(fmt.Sprintf("invalid device: %s", cfg.InferDeviceType))
		}
	} else {
		panic(fmt.Sprintf("invalid device: %s", cfg.InferDeviceType))
	}

	cfg.InferDeviceId = ctx.GlobalInt(InferDeviceIdFlag.Name)
	cfg.SynapseTimeout = 10 // TODO flags
	mem, err = gopsutil.VirtualMemory()
	if err == nil {
		if 32<<(^uintptr(0)>>63) == 32 && mem.Total > 2*1024*1024*1024 {
			log.Warn("Lowering memory allowance on 32bit arch", "available", mem.Total/1024/1024, "addressable", 2*1024)
			mem.Total = 2 * 1024 * 1024 * 1024
		}
		allowance := int(mem.Total / 1024 / 1024 / 2)
		if cache := ctx.GlobalInt(InferMemoryFlag.Name); cache > allowance {
			log.Warn("Sanitizing cache to C's GC limits", "provided", cache, "updated", allowance)
			ctx.GlobalSet(InferMemoryFlag.Name, strconv.Itoa(allowance))
		}
	}

	cfg.InferMemoryUsage = int64(ctx.GlobalInt(InferMemoryFlag.Name))
	cfg.InferMemoryUsage = cfg.InferMemoryUsage << 20
	//log.Warn("C MEMORY FOR CVM", "cache", cfg.InferMemoryUsage)
	// Override any default configs for hard coded networks.
	switch {
	case ctx.GlobalBool(BernardFlag.Name):
		if !ctx.GlobalIsSet(NetworkIdFlag.Name) {
			cfg.NetworkId = 42
		}
		cfg.Genesis = core.DefaultBernardGenesisBlock()
	case ctx.GlobalBool(DoloresFlag.Name):
		if !ctx.GlobalIsSet(NetworkIdFlag.Name) {
			cfg.NetworkId = 43
		}
		cfg.Genesis = core.DefaultDoloresGenesisBlock()
		//case ctx.GlobalBool(TestnetFlag.Name):
		//	if !ctx.GlobalIsSet(NetworkIdFlag.Name) {
		//		cfg.NetworkId = 28
		//	}
		//	cfg.Genesis = core.DefaultTestnetGenesisBlock()
		//case ctx.GlobalBool(LazynetFlag.Name):
		//	if !ctx.GlobalIsSet(NetworkIdFlag.Name) {
		//		cfg.NetworkId = 4
		//	}
		//	cfg.Genesis = core.DefaultRinkebyGenesisBlock()
		// case ctx.GlobalBool(DeveloperFlag.Name):
		// 	if !ctx.GlobalIsSet(NetworkIdFlag.Name) {
		// 		cfg.NetworkId = 1337
		// 	}
		// 	// Create new developer account or reuse existing one
		// 	var (
		// 		developer accounts.Account
		// 		err       error
		// 	)
		// 	if accs := ks.Accounts(); len(accs) > 0 {
		// 		developer = ks.Accounts()[0]
		// 	} else {
		// 		developer, err = ks.NewAccount("")
		// 		if err != nil {
		// 			Fatalf("Failed to create developer account: %v", err)
		// 		}
		// 	}
		// 	if err := ks.Unlock(developer, ""); err != nil {
		// 		Fatalf("Failed to unlock developer account: %v", err)
		// 	}
		// 	log.Info("Using developer account", "address", developer.Address)

		// 	cfg.Genesis = core.DeveloperGenesisBlock(uint64(ctx.GlobalInt(DeveloperPeriodFlag.Name)), developer.Address)
		// 	if !ctx.GlobalIsSet(MinerGasPriceFlag.Name) && !ctx.GlobalIsSet(MinerLegacyGasPriceFlag.Name) {
		// 		cfg.MinerGasPrice = big.NewInt(1)
		// 	}
	default:
		if cfg.NetworkId == 21 {
			setDNSDiscoveryDefaults(cfg, params.MainnetGenesisHash)
		}
	}
	if gen := ctx.GlobalInt(TrieCacheGenFlag.Name); gen > 0 {
		//state.MaxTrieCacheGen = uint16(gen)
	}

	// Set any dangling config values
	if ctx.String(CryptoKZGFlag.Name) != "gokzg" && ctx.String(CryptoKZGFlag.Name) != "ckzg" {
		Fatalf("--%s flag must be 'gokzg' or 'ckzg'", CryptoKZGFlag.Name)
	}
	log.Info("Initializing the KZG library", "backend", ctx.String(CryptoKZGFlag.Name))
	if err := kzg4844.UseCKZG(ctx.String(CryptoKZGFlag.Name) == "ckzg"); err != nil {
		Fatalf("Failed to set KZG library implementation to %s: %v", ctx.String(CryptoKZGFlag.Name), err)
	}
}

// setDNSDiscoveryDefaults configures DNS discovery with the given URL if
// no URLs are set.
func setDNSDiscoveryDefaults(cfg *ctxc.Config, genesis common.Hash) {
	if cfg.DiscoveryURLs != nil {
		return
	}

	protocol := "all"
	if url := params.KnownDNSNetwork(genesis, protocol); url != "" {
		//log.Info("Dns found", "url", url)
		cfg.DiscoveryURLs = []string{url}
	}
}

// SetDashboardConfig applies dashboard related command line flags to the config.
// func SetDashboardConfig(ctx *cli.Context, cfg *dashboard.Config) {
//	cfg.Host = ctx.GlobalString(DashboardAddrFlag.Name)
//	cfg.Port = ctx.GlobalInt(DashboardPortFlag.Name)
//	cfg.Refresh = ctx.GlobalDuration(DashboardRefreshFlag.Name)
// }

// SetTorrentFsConfig applies torrentFs related command line flags to the config.
func SetTorrentFsConfig(ctx *cli.Context, cfg *params1.Config) {
	//	cfg.Host = ctx.GlobalString(StorageAddrFlag.Name)
	cfg.Port = ctx.GlobalInt(StoragePortFlag.Name)
	IPCDisabled := ctx.GlobalBool(IPCDisabledFlag.Name)
	if runtime.GOOS == "windows" || IPCDisabled {
		cfg.IpcPath = ""
		//cfg.RpcURI = ctx.GlobalString(StorageRpcFlag.Name)//"http://" + ctx.GlobalString(RPCListenAddrFlag.Name) + ":" + string(ctx.GlobalInt(RPCPortFlag.Name))
	} else {
		path := MakeDataDir(ctx)
		IPCPath := ctx.GlobalString(IPCPathFlag.Name)
		cfg.IpcPath = filepath.Join(path, IPCPath)
	}
	cfg.Engine = ctx.GlobalString(StorageEngineFlag.Name)
	cfg.RpcURI = ctx.GlobalString(StorageRpcFlag.Name)

	trackers := ctx.GlobalString(StorageTrackerFlag.Name)
	//boostnodes := ctx.GlobalString(StorageBoostNodesFlag.Name)
	cfg.DefaultTrackers = strings.Split(trackers, ",")
	//cfg.BoostNodes = strings.Split(boostnodes, ",")
	cfg.MaxSeedingNum = ctx.GlobalInt(StorageMaxSeedingFlag.Name)
	log.Debug("FsConfig", "MaxSeedingNum", ctx.GlobalInt(StorageMaxSeedingFlag.Name),
		"MaxActiveNum", ctx.GlobalInt(StorageMaxActiveFlag.Name))
	cfg.MaxActiveNum = ctx.GlobalInt(StorageMaxActiveFlag.Name)
	cfg.Mode = ctx.GlobalString(StorageModeFlag.Name)
	cfg.DisableDHT = !ctx.GlobalBool(StorageDHTFlag.Name)
	cfg.DisableTCP = ctx.GlobalBool(StorageDisableTCPFlag.Name)
	cfg.DisableUTP = !ctx.GlobalBool(StorageEnableUTPFlag.Name)
	cfg.Wormhole = ctx.GlobalBool(StorageEnableWormholeFlag.Name)
	//cfg.FullSeed = ctx.GlobalBool(StorageFullFlag.Name)
	//if cfg.Mode == "full" {
	//	cfg.FullSeed = true
	//}
	cfg.Boost = ctx.GlobalBool(StorageBoostFlag.Name)
	cfg.DataDir = MakeStorageDir(ctx)
}

// RegisterCortexService adds an Cortex client to the stack.
func RegisterCortexService(stack *node.Node, cfg *ctxc.Config) {
	var err error
	err = stack.Register(func(ctx *node.ServiceContext) (node.Service, error) {
		fullNode, err := ctxc.New(stack, cfg)
		return fullNode, err
	})

	if err != nil {
		Fatalf("Failed to register the Cortex service: %v", err)
	}
}

// RegisterShhService configures Whisper and adds it to the given node.
func RegisterShhService(stack *node.Node, cfg *whisper.Config) {
	if err := stack.Register(func(n *node.ServiceContext) (node.Service, error) {
		return whisper.New(cfg), nil
	}); err != nil {
		Fatalf("Failed to register the Whisper service: %v", err)
	}
}

// RegisterStorageService adds a torrent file system to the stack.
// func RegisterStorageService(stack *node.Node, cfg *torrentfs.Config, mode downloader.SyncMode) {
func RegisterStorageService(stack *node.Node, cfg *params1.Config) {
	if err := stack.Register(func(ctx *node.ServiceContext) (node.Service, error) {
		//return torrentfs.New(cfg, true, false, downloader.FastSync == mode)
		return torrentfs.New(cfg, true, false, false)
	}); err != nil {
		Fatalf("Failed to register the storage service: %v", err)
	}
}

// RegisterDashboardService adds a dashboard to the stack.
// func RegisterDashboardService(stack *node.Node, cfg *dashboard.Config, commit string) {
// 	stack.Register(func(ctx *node.ServiceContext) (node.Service, error) {
// 		return dashboard.New(cfg, commit, ctx.ResolvePath("logs")), nil
// 	})
// }

// RegisterCortexStatsService configures the Cortex Stats daemon and adds it to
// the given node.
// func RegisterCortexStatsService(stack *node.Node, url string) {
// 	if err := stack.Register(func(ctx *node.ServiceContext) (node.Service, error) {
// 		// Retrieve both ctxc and les services
// 		var ctxcServ *ctxc.Cortex
// 		ctx.Service(&ctxcServ)
//
// 		return ctxcstats.New(url, ctxcServ)
// 	}); err != nil {
// 		Fatalf("Failed to register the Cortex Stats service: %v", err)
// 	}
// }

func SetupMetrics(ctx *cli.Context) {
	if metrics.Enabled {
		log.Info("Enabling metrics collection")
		var (
			enableExport = ctx.GlobalBool(MetricsEnableInfluxDBFlag.Name)
			endpoint     = ctx.GlobalString(MetricsInfluxDBEndpointFlag.Name)
			database     = ctx.GlobalString(MetricsInfluxDBDatabaseFlag.Name)
			username     = ctx.GlobalString(MetricsInfluxDBUsernameFlag.Name)
			password     = ctx.GlobalString(MetricsInfluxDBPasswordFlag.Name)
		)

		if enableExport {
			tagsMap := SplitTagsFlag(ctx.GlobalString(MetricsInfluxDBTagsFlag.Name))
			log.Info("Enabling metrics export to InfluxDB", "tags", tagsMap)
			go influxdb.InfluxDBWithTags(metrics.DefaultRegistry, 10*time.Second, endpoint, database, username, password, "cortex.", tagsMap)
		}

		if ctx.IsSet(MetricsHTTPFlag.Name) {
			address := net.JoinHostPort(ctx.String(MetricsHTTPFlag.Name), fmt.Sprintf("%d", ctx.Int(MetricsPortFlag.Name)))
			log.Info("Enabling stand-alone metrics HTTP endpoint", "address", address)
			exp.Setup(address)
		}
	}
}

func SplitTagsFlag(tagsFlag string) map[string]string {
	tags := strings.Split(tagsFlag, ",")
	tagsMap := map[string]string{}

	for _, t := range tags {
		if t != "" {
			kv := strings.Split(t, "=")

			if len(kv) == 2 {
				tagsMap[kv[0]] = kv[1]
			}
		}
	}

	return tagsMap
}

// MakeChainDatabase open an LevelDB using the flags passed to the client and will hard crash if it fails.
func MakeChainDatabase(ctx *cli.Context, stack *node.Node, readonly bool) ctxcdb.Database {
	var (
		cache   = ctx.GlobalInt(CacheFlag.Name) * ctx.GlobalInt(CacheDatabaseFlag.Name) / 100
		handles = makeDatabaseHandles(ctx.GlobalInt(FDLimitFlag.Name))
	)
	name := "chaindata"
	//chainDb, err := stack.OpenDatabase(name, cache, handles)
	chainDb, err := stack.OpenDatabaseWithFreezer(name, cache, handles, ctx.GlobalString(AncientFlag.Name), "", readonly)
	if err != nil {
		Fatalf("Could not open database: %v", err)
	}
	return chainDb
}

// tryMakeReadOnlyDatabase try to open the chain database in read-only mode,
// or fallback to write mode if the database is not initialized.
func tryMakeReadOnlyDatabase(ctx *cli.Context, stack *node.Node) ctxcdb.Database {
	// If the database doesn't exist we need to open it in write-mode to allow
	// the engine to create files.
	readonly := true
	if rawdb.PreexistingDatabase(stack.ResolvePath("chaindata")) == "" {
		readonly = false
	}
	return MakeChainDatabase(ctx, stack, readonly)
}

func MakeGenesis(ctx *cli.Context) *core.Genesis {
	var genesis *core.Genesis
	switch {
	case ctx.GlobalBool(BernardFlag.Name):
		genesis = core.DefaultBernardGenesisBlock()
		// case ctx.GlobalBool(TestnetFlag.Name):
		// 	genesis = core.DefaultTestnetGenesisBlock()
		// case ctx.GlobalBool(LazynetFlag.Name):
		// 	genesis = core.DefaultRinkebyGenesisBlock()
		// case ctx.GlobalBool(DeveloperFlag.Name):
		//	Fatalf("Developer chains are ephemeral")
	}
	return genesis
}

// MakeChain creates a chain manager from set command line flags.
func MakeChain(ctx *cli.Context, stack *node.Node, readonly bool) (chain *core.BlockChain, chainDb ctxcdb.Database) {
	var err error
	chainDb = MakeChainDatabase(ctx, stack, readonly)

	config, _, err := core.SetupGenesisBlock(chainDb, MakeGenesis(ctx))
	if err != nil {
		Fatalf("%v", err)
	}
	var engine consensus.Engine
	if config.Clique != nil {
		engine = clique.New(config.Clique, chainDb)
	} else {
		engine = cuckoo.NewFaker()
		if !ctx.GlobalBool(FakePoWFlag.Name) {
			engine = cuckoo.New(cuckoo.Config{})
		}
	}
	if gcmode := ctx.GlobalString(GCModeFlag.Name); gcmode != "full" && gcmode != "archive" {
		Fatalf("--%s must be either 'full' or 'archive'", GCModeFlag.Name)
	}
	cache := &core.CacheConfig{
		TrieCleanLimit:      ctxc.DefaultConfig.TrieCleanCache,
		TrieCleanNoPrefetch: ctx.GlobalBool(CacheNoPrefetchFlag.Name),
		TrieDirtyLimit:      ctxc.DefaultConfig.TrieDirtyCache,
		TrieDirtyDisabled:   ctx.GlobalString(GCModeFlag.Name) == "archive",
		TrieTimeLimit:       ctxc.DefaultConfig.TrieTimeout,
		SnapshotLimit:       ctxc.DefaultConfig.SnapshotCache,
		Preimages:           ctx.GlobalBool(CachePreimagesFlag.Name),
	}
	if cache.TrieDirtyDisabled && !cache.Preimages {
		cache.Preimages = true
		log.Info("Enabling recording of key preimages since archive mode is used")
	}
	if !ctx.GlobalIsSet(SnapshotFlag.Name) {
		cache.SnapshotLimit = 0 // Disabled
	}
	// If we're in readonly, do not bother generating snapshot data.
	if readonly {
		cache.SnapshotNoBuild = true
	}
	if ctx.GlobalIsSet(CacheFlag.Name) || ctx.GlobalIsSet(CacheTrieFlag.Name) {
		cache.TrieCleanLimit = ctx.GlobalInt(CacheFlag.Name) * ctx.GlobalInt(CacheTrieFlag.Name) / 100
	}
	if ctx.GlobalIsSet(CacheFlag.Name) || ctx.GlobalIsSet(CacheGCFlag.Name) {
		cache.TrieDirtyLimit = ctx.GlobalInt(CacheFlag.Name) * ctx.GlobalInt(CacheGCFlag.Name) / 100
	}
	vmcfg := vm.Config{
		EnablePreimageRecording: ctx.GlobalBool(VMEnableDebugFlag.Name),
	}
	var limit *uint64
	if ctx.GlobalIsSet(TxLookupLimitFlag.Name) && !readonly {
		l := ctx.GlobalUint64(TxLookupLimitFlag.Name)
		limit = &l
	}
	chain, err = core.NewBlockChain(chainDb, cache, config, engine, vmcfg, nil, limit)
	if err != nil {
		Fatalf("Can't create BlockChain: %v", err)
	}
	return chain, chainDb
}

// MakeConsolePreloads retrieves the absolute paths for the console JavaScript
// scripts to preload before starting.
func MakeConsolePreloads(ctx *cli.Context) []string {
	// Skip preloading if there's nothing to preload
	if ctx.GlobalString(PreloadJSFlag.Name) == "" {
		return nil
	}
	// Otherwise resolve absolute paths and return them
	preloads := []string{}

	// assets := ctx.GlobalString(JSpathFlag.Name)
	// for _, file := range strings.Split(ctx.GlobalString(PreloadJSFlag.Name), ",") {
	// 	preloads = append(preloads, common.AbsolutePath(assets, strings.TrimSpace(file)))
	// }
	return preloads
}

// MigrateFlags sets the global flag from a local flag when it's set.
// This is a temporary function used for migrating old command/flags to the
// new format.
//
// is equivalent after calling this method with:
//
// cortex --keystore /tmp/mykeystore --lightkdf account new
//
// This allows the use of the existing configuration functionality.
// When all flags are migrated this function can be removed and the existing
// configuration functionality must be changed that is uses local flags
func MigrateFlags(action func(ctx *cli.Context) error) func(*cli.Context) error {
	return func(ctx *cli.Context) error {
		for _, name := range ctx.FlagNames() {
			if ctx.IsSet(name) {
				ctx.GlobalSet(name, ctx.String(name))
			}
		}
		return action(ctx)
	}
}
