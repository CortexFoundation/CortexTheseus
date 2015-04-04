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
	"bufio"
	"fmt"
	"io/ioutil"
	"os"
	"runtime"
	"strconv"
	"time"

	"github.com/codegangsta/cli"
	"github.com/ethereum/ethash"
	"github.com/ethereum/go-ethereum/accounts"
	"github.com/ethereum/go-ethereum/cmd/utils"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/eth"
	"github.com/ethereum/go-ethereum/logger"
	"github.com/peterh/liner"
)

const (
	ClientIdentifier = "Geth"
	Version          = "0.9.5"
)

var (
	clilogger = logger.NewLogger("CLI")
	app       = utils.NewApp(Version, "the go-ethereum command line interface")
)

func init() {
	app.Action = run
	app.HideVersion = true // we have a command to print the version
	app.Commands = []cli.Command{
		blocktestCmd,
		{
			Action: makedag,
			Name:   "makedag",
			Usage:  "generate ethash dag (for testing)",
			Description: `
The makedag command generates an ethash DAG in /tmp/dag.

This command exists to support the system testing project.
Regular users do not need to execute it.
`,
		},
		{
			Action: version,
			Name:   "version",
			Usage:  "print ethereum version numbers",
			Description: `
The output of this command is supposed to be machine-readable.
`,
		},

		{
			Name:  "wallet",
			Usage: "ethereum presale wallet",
			Subcommands: []cli.Command{
				{
					Action: importWallet,
					Name:   "import",
					Usage:  "import ethereum presale wallet",
				},
			},
		},
		{
			Action: accountList,
			Name:   "account",
			Usage:  "manage accounts",
			Description: `

Manage accounts lets you create new accounts, list all existing accounts,
import a private key into a new account.

It supports interactive mode, when you are prompted for password as well as
non-interactive mode where passwords are supplied via a given password file.
Non-interactive mode is only meant for scripted use on test networks or known
safe environments.

Make sure you remember the password you gave when creating a new account (with
either new or import). Without it you are not able to unlock your account.

Note that exporting your key in unencrypted format is NOT supported.

Keys are stored under <DATADIR>/keys.
It is safe to transfer the entire directory or the individual keys therein
between ethereum nodes.
Make sure you backup your keys regularly.

And finally. DO NOT FORGET YOUR PASSWORD.
`,
			Subcommands: []cli.Command{
				{
					Action: accountList,
					Name:   "list",
					Usage:  "print account addresses",
				},
				{
					Action: accountCreate,
					Name:   "new",
					Usage:  "create a new account",
					Description: `

    ethereum account new

Creates a new account. Prints the address.

The account is saved in encrypted format, you are prompted for a passphrase.

You must remember this passphrase to unlock your account in the future.

For non-interactive use the passphrase can be specified with the --password flag:

    ethereum --password <passwordfile> account new

Note, this is meant to be used for testing only, it is a bad idea to save your
password to file or expose in any other way.
					`,
				},
				{
					Action: accountImport,
					Name:   "import",
					Usage:  "import a private key into a new account",
					Description: `

    ethereum account import <keyfile>

Imports an unencrypted private key from <keyfile> and creates a new account.
Prints the address.

The keyfile is assumed to contain an unencrypted private key in canonical EC
raw bytes format.

The account is saved in encrypted format, you are prompted for a passphrase.

You must remember this passphrase to unlock your account in the future.

For non-interactive use the passphrase can be specified with the -password flag:

    ethereum --password <passwordfile> account import <keyfile>

Note:
As you can directly copy your encrypted accounts to another ethereum instance,
this import mechanism is not needed when you transfer an account between
nodes.
					`,
				},
			},
		},
		{
			Action: dump,
			Name:   "dump",
			Usage:  `dump a specific block from storage`,
			Description: `
The arguments are interpreted as block numbers or hashes.
Use "ethereum dump 0" to dump the genesis block.
`,
		},
		{
			Action: console,
			Name:   "console",
			Usage:  `Geth Console: interactive JavaScript environment`,
			Description: `
The Geth console is an interactive shell for the JavaScript runtime environment
which exposes a node admin interface as well as the DAPP JavaScript API.
See https://github.com/ethereum/go-ethereum/wiki/Frontier-Console
`,
		},
		{
			Action: execJSFiles,
			Name:   "js",
			Usage:  `executes the given JavaScript files in the Geth JavaScript VM`,
			Description: `
The JavaScript VM exposes a node admin interface as well as the DAPP
JavaScript API. See https://github.com/ethereum/go-ethereum/wiki/Javascipt-Console
`,
		},
		{
			Action: importchain,
			Name:   "import",
			Usage:  `import a blockchain file`,
		},
		{
			Action: exportchain,
			Name:   "export",
			Usage:  `export blockchain into file`,
		},
	}
	app.Flags = []cli.Flag{
		utils.UnlockedAccountFlag,
		utils.PasswordFileFlag,
		utils.BootnodesFlag,
		utils.DataDirFlag,
		utils.JSpathFlag,
		utils.ListenPortFlag,
		utils.LogFileFlag,
		utils.LogJSONFlag,
		utils.LogLevelFlag,
		utils.MaxPeersFlag,
		utils.EtherbaseFlag,
		utils.MinerThreadsFlag,
		utils.MiningEnabledFlag,
		utils.NATFlag,
		utils.NodeKeyFileFlag,
		utils.NodeKeyHexFlag,
		utils.RPCEnabledFlag,
		utils.RPCListenAddrFlag,
		utils.RPCPortFlag,
		utils.VMDebugFlag,
		utils.ProtocolVersionFlag,
		utils.NetworkIdFlag,
		utils.RPCCORSDomainFlag,
	}

	// missing:
	// flag.StringVar(&ConfigFile, "conf", defaultConfigFile, "config file")
	// flag.BoolVar(&DiffTool, "difftool", false, "creates output for diff'ing. Sets LogLevel=0")
	// flag.StringVar(&DiffType, "diff", "all", "sets the level of diff output [vm, all]. Has no effect if difftool=false")

	// potential subcommands:
	// flag.StringVar(&SecretFile, "import", "", "imports the file given (hex or mnemonic formats)")
	// flag.StringVar(&ExportDir, "export", "", "exports the session keyring to files in the directory given")
	// flag.BoolVar(&GenAddr, "genaddr", false, "create a new priv/pub key")
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	defer logger.Flush()
	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(ctx *cli.Context) {
	fmt.Printf("Welcome to the FRONTIER\n")
	utils.HandleInterrupt()
	cfg := utils.MakeEthConfig(ClientIdentifier, Version, ctx)
	ethereum, err := eth.New(cfg)
	if err != nil {
		utils.Fatalf("%v", err)
	}

	startEth(ctx, ethereum)
	// this blocks the thread
	ethereum.WaitForShutdown()
}

func console(ctx *cli.Context) {
	cfg := utils.MakeEthConfig(ClientIdentifier, Version, ctx)
	ethereum, err := eth.New(cfg)
	if err != nil {
		utils.Fatalf("%v", err)
	}

	startEth(ctx, ethereum)
	repl := newJSRE(ethereum, ctx.String(utils.JSpathFlag.Name), true)
	repl.interactive()

	ethereum.Stop()
	ethereum.WaitForShutdown()
}

func execJSFiles(ctx *cli.Context) {
	cfg := utils.MakeEthConfig(ClientIdentifier, Version, ctx)
	ethereum, err := eth.New(cfg)
	if err != nil {
		utils.Fatalf("%v", err)
	}

	startEth(ctx, ethereum)
	repl := newJSRE(ethereum, ctx.String(utils.JSpathFlag.Name), false)
	for _, file := range ctx.Args() {
		repl.exec(file)
	}

	ethereum.Stop()
	ethereum.WaitForShutdown()
}

func unlockAccount(ctx *cli.Context, am *accounts.Manager, account string) (passphrase string) {
	var err error
	// Load startup keys. XXX we are going to need a different format
	// Attempt to unlock the account
	passphrase = getPassPhrase(ctx, "", false)
	accbytes := common.FromHex(account)
	if len(accbytes) == 0 {
		utils.Fatalf("Invalid account address '%s'", account)
	}
	err = am.Unlock(accbytes, passphrase)
	if err != nil {
		utils.Fatalf("Unlock account failed '%v'", err)
	}
	return
}

func startEth(ctx *cli.Context, eth *eth.Ethereum) {
	utils.StartEthereum(eth)
	am := eth.AccountManager()

	account := ctx.GlobalString(utils.UnlockedAccountFlag.Name)
	if len(account) > 0 {
		if account == "primary" {
			accbytes, err := am.Primary()
			if err != nil {
				utils.Fatalf("no primary account: %v", err)
			}
			account = common.ToHex(accbytes)
		}
		unlockAccount(ctx, am, account)
	}
	// Start auxiliary services if enabled.
	if ctx.GlobalBool(utils.RPCEnabledFlag.Name) {
		utils.StartRPC(eth, ctx)
	}
	if ctx.GlobalBool(utils.MiningEnabledFlag.Name) {
		eth.StartMining()
	}
}

func accountList(ctx *cli.Context) {
	am := utils.GetAccountManager(ctx)
	accts, err := am.Accounts()
	if err != nil {
		utils.Fatalf("Could not list accounts: %v", err)
	}
	for _, acct := range accts {
		fmt.Printf("Address: %x\n", acct)
	}
}

func getPassPhrase(ctx *cli.Context, desc string, confirmation bool) (passphrase string) {
	passfile := ctx.GlobalString(utils.PasswordFileFlag.Name)
	if len(passfile) == 0 {
		fmt.Println(desc)
		auth, err := readPassword("Passphrase: ", true)
		if err != nil {
			utils.Fatalf("%v", err)
		}
		if confirmation {
			confirm, err := readPassword("Repeat Passphrase: ", false)
			if err != nil {
				utils.Fatalf("%v", err)
			}
			if auth != confirm {
				utils.Fatalf("Passphrases did not match.")
			}
		}
		passphrase = auth

	} else {
		passbytes, err := ioutil.ReadFile(passfile)
		if err != nil {
			utils.Fatalf("Unable to read password file '%s': %v", passfile, err)
		}
		passphrase = string(passbytes)
	}
	return
}

func accountCreate(ctx *cli.Context) {
	am := utils.GetAccountManager(ctx)
	passphrase := getPassPhrase(ctx, "Your new account is locked with a password. Please give a password. Do not forget this password.", true)
	acct, err := am.NewAccount(passphrase)
	if err != nil {
		utils.Fatalf("Could not create the account: %v", err)
	}
	fmt.Printf("Address: %x\n", acct)
}

func importWallet(ctx *cli.Context) {
	keyfile := ctx.Args().First()
	if len(keyfile) == 0 {
		utils.Fatalf("keyfile must be given as argument")
	}
	keyJson, err := ioutil.ReadFile(keyfile)
	if err != nil {
		utils.Fatalf("Could not read wallet file: %v", err)
	}

	am := utils.GetAccountManager(ctx)
	passphrase := getPassPhrase(ctx, "", false)

	acct, err := am.ImportPreSaleKey(keyJson, passphrase)
	if err != nil {
		utils.Fatalf("Could not create the account: %v", err)
	}
	fmt.Printf("Address: %x\n", acct)
}

func accountImport(ctx *cli.Context) {
	keyfile := ctx.Args().First()
	if len(keyfile) == 0 {
		utils.Fatalf("keyfile must be given as argument")
	}
	am := utils.GetAccountManager(ctx)
	passphrase := getPassPhrase(ctx, "Your new account is locked with a password. Please give a password. Do not forget this password.", true)
	acct, err := am.Import(keyfile, passphrase)
	if err != nil {
		utils.Fatalf("Could not create the account: %v", err)
	}
	fmt.Printf("Address: %x\n", acct)
}

func importchain(ctx *cli.Context) {
	if len(ctx.Args()) != 1 {
		utils.Fatalf("This command requires an argument.")
	}
	chainmgr, _, _ := utils.GetChain(ctx)
	start := time.Now()
	err := utils.ImportChain(chainmgr, ctx.Args().First())
	if err != nil {
		utils.Fatalf("Import error: %v\n", err)
	}
	fmt.Printf("Import done in %v", time.Since(start))
	return
}

func exportchain(ctx *cli.Context) {
	if len(ctx.Args()) != 1 {
		utils.Fatalf("This command requires an argument.")
	}
	chainmgr, _, _ := utils.GetChain(ctx)
	start := time.Now()
	err := utils.ExportChain(chainmgr, ctx.Args().First())
	if err != nil {
		utils.Fatalf("Export error: %v\n", err)
	}
	fmt.Printf("Export done in %v", time.Since(start))
	return
}

func dump(ctx *cli.Context) {
	chainmgr, _, stateDb := utils.GetChain(ctx)
	for _, arg := range ctx.Args() {
		var block *types.Block
		if hashish(arg) {
			block = chainmgr.GetBlock(common.HexToHash(arg))
		} else {
			num, _ := strconv.Atoi(arg)
			block = chainmgr.GetBlockByNumber(uint64(num))
		}
		if block == nil {
			fmt.Println("{}")
			utils.Fatalf("block not found")
		} else {
			statedb := state.New(block.Root(), stateDb)
			fmt.Printf("%s\n", statedb.Dump())
		}
	}
}

func makedag(ctx *cli.Context) {
	chain, _, _ := utils.GetChain(ctx)
	pow := ethash.New(chain)
	fmt.Println("making cache")
	pow.UpdateCache(true)
	fmt.Println("making DAG")
	pow.UpdateDAG()
}

func version(c *cli.Context) {
	fmt.Printf(`%v
Version: %v
Protocol Version: %d
Network Id: %d
GO: %s
OS: %s
GOPATH=%s
GOROOT=%s
`, ClientIdentifier, Version, c.GlobalInt(utils.ProtocolVersionFlag.Name), c.GlobalInt(utils.NetworkIdFlag.Name), runtime.Version(), runtime.GOOS, os.Getenv("GOPATH"), runtime.GOROOT())
}

// hashish returns true for strings that look like hashes.
func hashish(x string) bool {
	_, err := strconv.Atoi(x)
	return err != nil
}

func readPassword(prompt string, warnTerm bool) (string, error) {
	if liner.TerminalSupported() {
		lr := liner.NewLiner()
		defer lr.Close()
		return lr.PasswordPrompt(prompt)
	}
	if warnTerm {
		fmt.Println("!! Unsupported terminal, password will be echoed.")
	}
	fmt.Print(prompt)
	input, err := bufio.NewReader(os.Stdin).ReadString('\n')
	fmt.Println()
	return input, err
}
