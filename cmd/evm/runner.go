// Copyright 2017 The go-cortex Authors
// This file is part of go-cortex.
//
// go-cortex is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// go-cortex is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with go-cortex. If not, see <http://www.gnu.org/licenses/>.

package main

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/big"
	"os"
	goruntime "runtime"
	"runtime/pprof"
	"time"

	"github.com/CortexFoundation/CortexTheseus/cmd/evm/internal/compiler"
	"github.com/CortexFoundation/CortexTheseus/cmd/utils"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/core"
	"github.com/CortexFoundation/CortexTheseus/core/state"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/core/vm"
	"github.com/CortexFoundation/CortexTheseus/core/vm/runtime"
	"github.com/CortexFoundation/CortexTheseus/crypto"
	"github.com/CortexFoundation/CortexTheseus/db"
	infer "github.com/CortexFoundation/CortexTheseus/inference/synapse"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/CortexTheseus/rlp"
	cli "gopkg.in/urfave/cli.v1"
)

var runCommand = cli.Command{
	Action:      runCmd,
	Name:        "run",
	Usage:       "run arbitrary evm binary",
	ArgsUsage:   "<code>",
	Description: `The run command runs arbitrary EVM code.`,
}

// readGenesis will read the given JSON format genesis file and return
// the initialized Genesis structure
func readGenesis(genesisPath string) *core.Genesis {
	// Make sure we have a valid genesis JSON
	//genesisPath := ctx.Args().First()
	if len(genesisPath) == 0 {
		utils.Fatalf("Must supply path to genesis JSON file")
	}
	file, err := os.Open(genesisPath)
	if err != nil {
		utils.Fatalf("Failed to read genesis file: %v", err)
	}
	defer file.Close()

	genesis := new(core.Genesis)
	if err := json.NewDecoder(file).Decode(genesis); err != nil {
		utils.Fatalf("invalid genesis file: %v", err)
	}
	return genesis
}

func runCmd(ctx *cli.Context) error {
	glogger := log.NewGlogHandler(log.StreamHandler(os.Stderr, log.TerminalFormat(false)))
	glogger.Verbosity(log.Lvl(ctx.GlobalInt(VerbosityFlag.Name)))
	log.Root().SetHandler(glogger)
	logconfig := &vm.LogConfig{
		DisableMemory: ctx.GlobalBool(DisableMemoryFlag.Name),
		DisableStack:  ctx.GlobalBool(DisableStackFlag.Name),
		Debug:         ctx.GlobalBool(DebugFlag.Name),
	}

	var (
		tracer      vm.Tracer
		debugLogger *vm.StructLogger
		statedb     *state.StateDB
		chainConfig *params.ChainConfig
		sender      = common.BytesToAddress([]byte("sender"))
		receiver    = common.BytesToAddress([]byte("receiver"))
		blockNumber uint64
	)
	if ctx.GlobalBool(MachineFlag.Name) {
		tracer = NewJSONLogger(logconfig, os.Stdout)
	} else if ctx.GlobalBool(DebugFlag.Name) {
		debugLogger = vm.NewStructLogger(logconfig)
		tracer = debugLogger
	} else {
		debugLogger = vm.NewStructLogger(logconfig)
	}
	if ctx.GlobalString(GenesisFlag.Name) != "" {
		gen := readGenesis(ctx.GlobalString(GenesisFlag.Name))
		db := ctxcdb.NewMemDatabase()
		genesis := gen.ToBlock(db)
		statedb, _ = state.New(genesis.Root(), state.NewDatabase(db))
		chainConfig = gen.Config
		blockNumber = gen.Number
	} else {
		statedb, _ = state.New(common.Hash{}, state.NewDatabase(ctxcdb.NewMemDatabase()))
	}
	if ctx.GlobalString(SenderFlag.Name) != "" {
		sender = common.HexToAddress(ctx.GlobalString(SenderFlag.Name))
	}
	statedb.CreateAccount(sender)
	mh1, _ := hex.DecodeString("ca3d0286d5758697cdef653c1375960a868ac08a")
	testModelMeta1, _ := rlp.EncodeToBytes(
		&types.ModelMeta{
			Hash:          common.BytesToAddress(mh1),
			RawSize:       10000,
			InputShape:    []uint64{1, 28, 28},
			OutputShape:   []uint64{1},
			Gas:           1000,
			BlockNum:      *big.NewInt(10),
			AuthorAddress: common.BytesToAddress(crypto.Keccak256([]byte{0x2, 0x2})),
		})

	mh2, _ := hex.DecodeString("4d8bc8272b882f315c6a96449ad4568fac0e6038")
	testModelMeta2, _ := rlp.EncodeToBytes(
		&types.ModelMeta{
			Hash:          common.BytesToAddress(mh2),
			RawSize:       10000,
			InputShape:    []uint64{3, 224, 224},
			OutputShape:   []uint64{1},
			Gas:           1000,
			BlockNum:      *big.NewInt(10),
			AuthorAddress: common.BytesToAddress(crypto.Keccak256([]byte{0x2, 0x2})),
		})
	// new a modelmeta at 0x1001 and new a datameta at 0x2001

	ih, _ := hex.DecodeString("c35dde5292458e91c6533d671a9cfcf55fc46026")
	testInputMeta, _ := rlp.EncodeToBytes(
		&types.InputMeta{
			Hash:          common.BytesToAddress(ih),
			RawSize:       10000,
			Shape:         []uint64{1, 28, 28},
			AuthorAddress: common.BytesToAddress(crypto.Keccak256([]byte{0x3})),
		})
	fmt.Println("testModelMeta1", testModelMeta1)
	fmt.Println("testModelMeta2", testModelMeta2)
	fmt.Println("testInputMeta", testInputMeta)
	statedb.SetCode(common.HexToAddress("0xFCE5a78Bfb16e599E3d2628fA4b21aCFE25a190E"), append([]byte{0x0, 0x1}, []byte(testModelMeta1)...))
	statedb.SetCode(common.HexToAddress("0x049d8385c81200339fca354f2696fd57ea96255e"), append([]byte{0x0, 0x2}, []byte(testInputMeta)...))
	statedb.SetCode(common.HexToAddress("0x2001"), append([]byte{0x0, 0x2}, []byte(testInputMeta)...))
	// simple address for the sake of debuging
	statedb.SetCode(common.HexToAddress("0x1001"), append([]byte{0x0, 0x1}, []byte(testModelMeta1)...))
	statedb.SetCode(common.HexToAddress("0x1002"), append([]byte{0x0, 0x1}, []byte(testModelMeta2)...))
	statedb.SetNum(common.HexToAddress("0x1001"), big.NewInt(1))
	statedb.SetNum(common.HexToAddress("0x1002"), big.NewInt(1))
	if ctx.GlobalString(ReceiverFlag.Name) != "" {
		receiver = common.HexToAddress(ctx.GlobalString(ReceiverFlag.Name))
	}

	var (
		code []byte
		ret  []byte
		err  error
	)
	// The '--code' or '--codefile' flag overrides code in state
	if ctx.GlobalString(CodeFileFlag.Name) != "" {
		var hexcode []byte
		var err error
		// If - is specified, it means that code comes from stdin
		if ctx.GlobalString(CodeFileFlag.Name) == "-" {
			//Try reading from stdin
			if hexcode, err = ioutil.ReadAll(os.Stdin); err != nil {
				fmt.Printf("Could not load code from stdin: %v\n", err)
				os.Exit(1)
			}
		} else {
			// Codefile with hex assembly
			if hexcode, err = ioutil.ReadFile(ctx.GlobalString(CodeFileFlag.Name)); err != nil {
				fmt.Printf("Could not load code from file: %v\n", err)
				os.Exit(1)
			}
		}
		code = common.Hex2Bytes(string(bytes.TrimRight(hexcode, "\n")))

	} else if ctx.GlobalString(CodeFlag.Name) != "" {
		code = common.Hex2Bytes(ctx.GlobalString(CodeFlag.Name))
	} else if fn := ctx.Args().First(); len(fn) > 0 {
		// EASM-file to compile
		src, err := ioutil.ReadFile(fn)
		if err != nil {
			return err
		}
		bin, err := compiler.Compile(fn, src, false)
		if err != nil {
			return err
		}
		code = common.Hex2Bytes(bin)
	}

	storageDir := ""
	if ctx.GlobalString(StorageDir.Name) != "" {
		storageDir = ctx.GlobalString(StorageDir.Name)
	}

	initialGas := ctx.GlobalUint64(GasFlag.Name)
	runtimeConfig := runtime.Config{
		Origin:      sender,
		State:       statedb,
		GasLimit:    initialGas,
		GasPrice:    utils.GlobalBig(ctx, PriceFlag.Name),
		Value:       utils.GlobalBig(ctx, ValueFlag.Name),
		BlockNumber: new(big.Int).SetUint64(blockNumber),
		EVMConfig: vm.Config{
			Tracer:   tracer,
			Debug:    ctx.GlobalBool(DebugFlag.Name) || ctx.GlobalBool(MachineFlag.Name),
			StorageDir:  storageDir,
		},
	}

	if cpuProfilePath := ctx.GlobalString(CPUProfileFlag.Name); cpuProfilePath != "" {
		f, err := os.Create(cpuProfilePath)
		if err != nil {
			fmt.Println("could not create CPU profile: ", err)
			os.Exit(1)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			fmt.Println("could not start CPU profile: ", err)
			os.Exit(1)
		}
		defer pprof.StopCPUProfile()
	}

	if chainConfig != nil {
		runtimeConfig.ChainConfig = chainConfig
	}
	tstart := time.Now()
	var leftOverGas uint64
	fmt.Println("evm storageDir", storageDir)
	inferServer := infer.New(&infer.Config{
		StorageDir: storageDir,
		IsNotCache: false,
			IsRemoteInfer: false,
			DeviceType: "cpu",
	})

	if ctx.GlobalBool(CreateFlag.Name) {
		input := append(code, common.Hex2Bytes(ctx.GlobalString(InputFlag.Name))...)
		ret, _, leftOverGas, err = runtime.Create(input, &runtimeConfig)
	} else {
		if len(code) > 0 {
			statedb.SetCode(receiver, code)
		}
		ret, leftOverGas, err = runtime.Call(receiver, common.Hex2Bytes(ctx.GlobalString(InputFlag.Name)), &runtimeConfig)
	}
	execTime := time.Since(tstart)

	if ctx.GlobalBool(DumpFlag.Name) {
		statedb.IntermediateRoot(true)
		fmt.Println(string(statedb.Dump()))
	}

	if memProfilePath := ctx.GlobalString(MemProfileFlag.Name); memProfilePath != "" {
		f, err := os.Create(memProfilePath)
		if err != nil {
			fmt.Println("could not create memory profile: ", err)
			os.Exit(1)
		}
		if err := pprof.WriteHeapProfile(f); err != nil {
			fmt.Println("could not write memory profile: ", err)
			os.Exit(1)
		}
		f.Close()
	}

	if ctx.GlobalBool(DebugFlag.Name) {
		if debugLogger != nil {
			fmt.Fprintln(os.Stderr, "#### TRACE ####")
			vm.WriteTrace(os.Stderr, debugLogger.StructLogs())
		}
		fmt.Fprintln(os.Stderr, "#### LOGS ####")
		vm.WriteLogs(os.Stderr, statedb.Logs())
	}

	if ctx.GlobalBool(StatDumpFlag.Name) {
		var mem goruntime.MemStats
		goruntime.ReadMemStats(&mem)
		fmt.Fprintf(os.Stderr, `evm execution time: %v
heap objects:       %d
allocations:        %d
total allocations:  %d
GC calls:           %d
Gas used:           %d

`, execTime, mem.HeapObjects, mem.Alloc, mem.TotalAlloc, mem.NumGC, initialGas-leftOverGas)
	}
	if tracer == nil {
		fmt.Printf("0x%x\n", ret)
		if err != nil {
			fmt.Printf(" error: %v\n", err)
		}
	}
	inferServer.Close()
	return nil
}
