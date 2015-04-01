package utils

import (
	"crypto/ecdsa"
	"os"
	"path"
	"runtime"

	"github.com/codegangsta/cli"
	"github.com/ethereum/go-ethereum/accounts"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/eth"
	"github.com/ethereum/go-ethereum/ethdb"
	"github.com/ethereum/go-ethereum/event"
	"github.com/ethereum/go-ethereum/logger"
	"github.com/ethereum/go-ethereum/p2p/nat"
	"github.com/ethereum/go-ethereum/rpc"
	"github.com/ethereum/go-ethereum/xeth"
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

	cli.CommandHelpTemplate = `{{.Name}}{{if .Subcommands}} command{{end}}{{if .Flags}} [command options]{{end}} [arguments...]
{{if .Description}}{{.Description}}
{{end}}{{if .Subcommands}}
SUBCOMMANDS:
	{{range .Subcommands}}{{.Name}}{{with .ShortName}}, {{.}}{{end}}{{ "\t" }}{{.Usage}}
	{{end}}{{end}}{{if .Flags}}
OPTIONS:
	{{range .Flags}}{{.}}
	{{end}}{{end}}
`
}

// NewApp creates an app with sane defaults.
func NewApp(version, usage string) *cli.App {
	app := cli.NewApp()
	app.Name = path.Base(os.Args[0])
	app.Author = ""
	//app.Authors = nil
	app.Email = ""
	app.Version = version
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
	DataDirFlag = cli.StringFlag{
		Name:  "datadir",
		Usage: "Data directory to be used",
		Value: common.DefaultDataDir(),
	}
	ProtocolVersionFlag = cli.IntFlag{
		Name:  "protocolversion",
		Usage: "ETH protocol version",
		Value: eth.ProtocolVersion,
	}
	NetworkIdFlag = cli.IntFlag{
		Name:  "networkid",
		Usage: "Network Id",
		Value: eth.NetworkId,
	}

	// miner settings
	MinerThreadsFlag = cli.IntFlag{
		Name:  "minerthreads",
		Usage: "Number of miner threads",
		Value: runtime.NumCPU(),
	}
	MiningEnabledFlag = cli.BoolFlag{
		Name:  "mine",
		Usage: "Enable mining",
	}
	EtherbaseFlag = cli.StringFlag{
		Name:  "etherbase",
		Usage: "public address for block mining rewards. By default the address of your primary account is used",
		Value: "primary",
	}

	UnlockedAccountFlag = cli.StringFlag{
		Name:  "unlock",
		Usage: "unlock the account given until this program exits (prompts for password). '--unlock primary' unlocks the primary account",
		Value: "",
	}
	PasswordFileFlag = cli.StringFlag{
		Name:  "password",
		Usage: "Path to password file for (un)locking an existing account.",
		Value: "",
	}

	// logging and debug settings
	LogFileFlag = cli.StringFlag{
		Name:  "logfile",
		Usage: "Send log output to a file",
	}
	LogLevelFlag = cli.IntFlag{
		Name:  "loglevel",
		Usage: "0-5 (silent, error, warn, info, debug, debug detail)",
		Value: int(logger.InfoLevel),
	}
	LogJSONFlag = cli.StringFlag{
		Name:  "logjson",
		Usage: "Send json structured log output to a file or '-' for standard output (default: no json output)",
		Value: "",
	}
	VMDebugFlag = cli.BoolFlag{
		Name:  "vmdebug",
		Usage: "Virtual Machine debug output",
	}

	// RPC settings
	RPCEnabledFlag = cli.BoolFlag{
		Name:  "rpc",
		Usage: "Whether RPC server is enabled",
	}
	RPCListenAddrFlag = cli.StringFlag{
		Name:  "rpcaddr",
		Usage: "Listening address for the JSON-RPC server",
		Value: "127.0.0.1",
	}
	RPCPortFlag = cli.IntFlag{
		Name:  "rpcport",
		Usage: "Port on which the JSON-RPC server should listen",
		Value: 8545,
	}
	RPCCORSDomainFlag = cli.StringFlag{
		Name:  "rpccorsdomain",
		Usage: "Domain on which to send Access-Control-Allow-Origin header",
		Value: "",
	}
	// Network Settings
	MaxPeersFlag = cli.IntFlag{
		Name:  "maxpeers",
		Usage: "Maximum number of network peers",
		Value: 16,
	}
	ListenPortFlag = cli.IntFlag{
		Name:  "port",
		Usage: "Network listening port",
		Value: 30303,
	}
	BootnodesFlag = cli.StringFlag{
		Name:  "bootnodes",
		Usage: "Space-separated enode URLs for discovery bootstrap",
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
		Usage: "Port mapping mechanism (any|none|upnp|pmp|extip:<IP>)",
		Value: "any",
	}
	JSpathFlag = cli.StringFlag{
		Name:  "jspath",
		Usage: "JS library path to be used with console and js subcommands",
		Value: ".",
	}
)

func GetNAT(ctx *cli.Context) nat.Interface {
	natif, err := nat.Parse(ctx.GlobalString(NATFlag.Name))
	if err != nil {
		Fatalf("Option %s: %v", NATFlag.Name, err)
	}
	return natif
}

func GetNodeKey(ctx *cli.Context) (key *ecdsa.PrivateKey) {
	hex, file := ctx.GlobalString(NodeKeyHexFlag.Name), ctx.GlobalString(NodeKeyFileFlag.Name)
	var err error
	switch {
	case file != "" && hex != "":
		Fatalf("Options %q and %q are mutually exclusive", NodeKeyFileFlag.Name, NodeKeyHexFlag.Name)
	case file != "":
		if key, err = crypto.LoadECDSA(file); err != nil {
			Fatalf("Option %q: %v", NodeKeyFileFlag.Name, err)
		}
	case hex != "":
		if key, err = crypto.HexToECDSA(hex); err != nil {
			Fatalf("Option %q: %v", NodeKeyHexFlag.Name, err)
		}
	}
	return key
}

func MakeEthConfig(clientID, version string, ctx *cli.Context) *eth.Config {
	return &eth.Config{
		Name:            common.MakeName(clientID, version),
		DataDir:         ctx.GlobalString(DataDirFlag.Name),
		ProtocolVersion: ctx.GlobalInt(ProtocolVersionFlag.Name),
		NetworkId:       ctx.GlobalInt(NetworkIdFlag.Name),
		LogFile:         ctx.GlobalString(LogFileFlag.Name),
		LogLevel:        ctx.GlobalInt(LogLevelFlag.Name),
		LogJSON:         ctx.GlobalString(LogJSONFlag.Name),
		Etherbase:       ctx.GlobalString(EtherbaseFlag.Name),
		MinerThreads:    ctx.GlobalInt(MinerThreadsFlag.Name),
		AccountManager:  GetAccountManager(ctx),
		VmDebug:         ctx.GlobalBool(VMDebugFlag.Name),
		MaxPeers:        ctx.GlobalInt(MaxPeersFlag.Name),
		Port:            ctx.GlobalString(ListenPortFlag.Name),
		NAT:             GetNAT(ctx),
		NodeKey:         GetNodeKey(ctx),
		Shh:             true,
		Dial:            true,
		BootNodes:       ctx.GlobalString(BootnodesFlag.Name),
	}
}

func GetChain(ctx *cli.Context) (*core.ChainManager, common.Database, common.Database) {
	dataDir := ctx.GlobalString(DataDirFlag.Name)
	blockDb, err := ethdb.NewLDBDatabase(path.Join(dataDir, "blockchain"))
	if err != nil {
		Fatalf("Could not open database: %v", err)
	}

	stateDb, err := ethdb.NewLDBDatabase(path.Join(dataDir, "state"))
	if err != nil {
		Fatalf("Could not open database: %v", err)
	}
	return core.NewChainManager(blockDb, stateDb, new(event.TypeMux)), blockDb, stateDb
}

func GetAccountManager(ctx *cli.Context) *accounts.Manager {
	dataDir := ctx.GlobalString(DataDirFlag.Name)
	ks := crypto.NewKeyStorePassphrase(path.Join(dataDir, "keys"))
	return accounts.NewManager(ks)
}

func StartRPC(eth *eth.Ethereum, ctx *cli.Context) {
	config := rpc.RpcConfig{
		ListenAddress: ctx.GlobalString(RPCListenAddrFlag.Name),
		ListenPort:    uint(ctx.GlobalInt(RPCPortFlag.Name)),
		CorsDomain:    ctx.GlobalString(RPCCORSDomainFlag.Name),
	}

	xeth := xeth.New(eth, nil)
	_ = rpc.Start(xeth, config)
}
