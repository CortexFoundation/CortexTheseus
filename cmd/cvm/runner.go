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
	"github.com/CortexFoundation/torrentfs"
	cli "gopkg.in/urfave/cli.v1"
)

var runCommand = cli.Command{
	Action:      runCmd,
	Name:        "run",
	Usage:       "run arbitrary cvm binary",
	ArgsUsage:   "<code>",
	Description: `The run command runs arbitrary CVM code.`,
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
		sender             = common.BytesToAddress([]byte("sender"))
		receiver           = common.BytesToAddress([]byte("receiver"))
		blockNumber uint64 = 0
	)
	if ctx.GlobalBool(MachineFlag.Name) {
		tracer = NewJSONLogger(logconfig, os.Stdout)
	} else if ctx.GlobalBool(DebugFlag.Name) {
		debugLogger = vm.NewStructLogger(logconfig)
		tracer = debugLogger
	} else {
		debugLogger = vm.NewStructLogger(logconfig)
	}

	if ctx.GlobalInt(BlockNumberFlag.Name) > 0 {
		blockNumber = uint64(ctx.GlobalInt(BlockNumberFlag.Name))
	}
	fmt.Println("Current blockNumber: ", blockNumber)

	if ctx.GlobalString(GenesisFlag.Name) != "" {
		gen := readGenesis(ctx.GlobalString(GenesisFlag.Name))
		db := ctxcdb.NewMemDatabase()
		genesis := gen.ToBlock(db)
		statedb, _ = state.New(genesis.Root(), state.NewDatabase(db), nil)
		chainConfig = gen.Config
		if blockNumber == 0 {
			blockNumber = gen.Number
		}
	} else {
		statedb, _ = state.New(common.Hash{}, state.NewDatabase(ctxcdb.NewMemDatabase()), nil)
	}
	if ctx.GlobalString(SenderFlag.Name) != "" {
		sender = common.HexToAddress(ctx.GlobalString(SenderFlag.Name))
	}
	statedb.CreateAccount(sender)
	mh1, _ := hex.DecodeString("32ce759a802ea862de7e1529e738e010449d6a69")
	testModelMeta1, _ := rlp.EncodeToBytes(
		&types.ModelMeta{
			Hash:          common.BytesToAddress(mh1),
			RawSize:       26047799,
			InputShape:    []uint64{3, 224, 224},
			OutputShape:   []uint64{1},
			Gas:           1000,
			AuthorAddress: common.BytesToAddress(crypto.Keccak256([]byte{0x2, 0x2})),
		})

	mh2, _ := hex.DecodeString("3145ad19228c1cd2d051314e72f26c1ce77b7f02")
	testModelMeta2, _ := rlp.EncodeToBytes(
		&types.ModelMeta{
			Hash:          common.BytesToAddress(mh2),
			RawSize:       1000000,
			InputShape:    []uint64{3, 32, 32},
			OutputShape:   []uint64{1},
			Gas:           1000,
			AuthorAddress: common.BytesToAddress(crypto.Keccak256([]byte{0x2, 0x2})),
		})
	mh3, _ := hex.DecodeString("d31d1b0f588069aa6f36de5a7025a8d73a9a49f6")
	testModelMeta3, _ := rlp.EncodeToBytes(
		&types.ModelMeta{
			Hash:          common.BytesToAddress(mh3),
			RawSize:       100000000,
			InputShape:    []uint64{3, 416, 416},
			OutputShape:   []uint64{1},
			Gas:           1000,
			AuthorAddress: common.BytesToAddress(crypto.Keccak256([]byte{0x2, 0x2})),
		})
	mh4, _ := hex.DecodeString("2d343a00ca1c533eeea6bd2ed5cd2182e62c9f0c")
	testModelMeta4, _ := rlp.EncodeToBytes(
		&types.ModelMeta{
			Hash:          common.BytesToAddress(mh4),
			RawSize:       6219271,
			InputShape:    []uint64{1, 38, 1},
			OutputShape:   []uint64{1},
			Gas:           1000,
			AuthorAddress: common.BytesToAddress(crypto.Keccak256([]byte{0x2, 0x2})),
		})
	mh5, _ := hex.DecodeString("821a22bac01e47b22bc8a917421b163006385bd9")
	testModelMeta5, _ := rlp.EncodeToBytes(
		&types.ModelMeta{
			Hash:          common.BytesToAddress(mh5),
			RawSize:       1000000,
			InputShape:    []uint64{3, 32, 32},
			OutputShape:   []uint64{1},
			Gas:           1000,
			AuthorAddress: common.BytesToAddress(crypto.Keccak256([]byte{0x2, 0x2})),
		})
	// new a modelmeta at 0x1001 and new a datameta at 0x2001

	ih1, _ := hex.DecodeString("4c5e20b86f46943422e0ac09749aed9882b4bf35")
	testInputMeta1, _ := rlp.EncodeToBytes(
		&types.InputMeta{
			Hash:    common.BytesToAddress(ih1),
			RawSize: 10000,
			Shape:   []uint64{3, 224, 224},
		})
	ih2, _ := hex.DecodeString("aea5584d0cd3865e90c80eace3bfcb062473d966")
	testInputMeta2, _ := rlp.EncodeToBytes(
		&types.InputMeta{
			Hash:    common.BytesToAddress(ih2),
			RawSize: 3200,
			Shape:   []uint64{3, 32, 32},
		})
	ih3, _ := hex.DecodeString("8e14bbd1c395b7fdcc36fbd3e5f3b6cb7931cc67")
	testInputMeta3, _ := rlp.EncodeToBytes(
		&types.InputMeta{
			Hash:    common.BytesToAddress(ih3),
			RawSize: 519296,
			Shape:   []uint64{3, 416, 416},
		})
	ih4, _ := hex.DecodeString("0fa499fb0966faf927d0c7a4c5f561a37ef8c3e3")
	testInputMeta4, _ := rlp.EncodeToBytes(
		&types.InputMeta{
			Hash:    common.BytesToAddress(ih4),
			RawSize: 10000,
			Shape:   []uint64{1, 38, 1},
		})
	ih5, _ := hex.DecodeString("91122004e230af0addc1f084fe0c7bbc6cf6c7fb")
	testInputMeta5, _ := rlp.EncodeToBytes(
		&types.InputMeta{
			Hash:    common.BytesToAddress(ih5),
			RawSize: 519296,
			Shape:   []uint64{3, 416, 416},
		})
	ih6, _ := hex.DecodeString("f302746f4e07c8dc4c9b4e09fac1cebfc336b585")
	testInputMeta6, _ := rlp.EncodeToBytes(
		&types.InputMeta{
			Hash:    common.BytesToAddress(ih6),
			RawSize: 3200,
			Shape:   []uint64{3, 32, 32},
		})
	if false {
		// statedb.SetCode(common.HexToAddress("0xFCE5a78Bfb16e599E3d2628fA4b21aCFE25a190E"),
		// append([]byte{0x0, 0x1}, []byte(testModelMeta1)...))
		// statedb.SetCode(common.HexToAddress("0x049d8385c81200339fca354f2696fd57ea96255e"),
		// append([]byte{0x0, 0x2}, []byte(testInputMeta1)...))
	}
	statedb.SetCode(common.HexToAddress("0x2001"), append([]byte{0x0, 0x2}, []byte(testInputMeta1)...))
	statedb.SetCode(common.HexToAddress("0x2002"), append([]byte{0x0, 0x2}, []byte(testInputMeta2)...))
	statedb.SetCode(common.HexToAddress("0x2003"), append([]byte{0x0, 0x2}, []byte(testInputMeta3)...))
	statedb.SetCode(common.HexToAddress("0x2004"), append([]byte{0x0, 0x2}, []byte(testInputMeta4)...))
	statedb.SetCode(common.HexToAddress("0x2005"), append([]byte{0x0, 0x2}, []byte(testInputMeta5)...))
	statedb.SetCode(common.HexToAddress("0x2006"), append([]byte{0x0, 0x2}, []byte(testInputMeta6)...))
	// simple address for the sake of debuging
	statedb.SetCode(common.HexToAddress("0x1001"), append([]byte{0x0, 0x1}, []byte(testModelMeta1)...))
	statedb.SetCode(common.HexToAddress("0x1002"), append([]byte{0x0, 0x1}, []byte(testModelMeta2)...))
	statedb.SetCode(common.HexToAddress("0x1003"), append([]byte{0x0, 0x1}, []byte(testModelMeta3)...))
	statedb.SetCode(common.HexToAddress("0x1004"), append([]byte{0x0, 0x1}, []byte(testModelMeta4)...))
	statedb.SetCode(common.HexToAddress("0x1005"), append([]byte{0x0, 0x1}, []byte(testModelMeta5)...))

	statedb.SetNum(common.HexToAddress("0x1001"), big.NewInt(1))
	statedb.SetNum(common.HexToAddress("0x1002"), big.NewInt(1))
	statedb.SetNum(common.HexToAddress("0x1003"), big.NewInt(1))
	statedb.SetNum(common.HexToAddress("0x1004"), big.NewInt(1))
	statedb.SetNum(common.HexToAddress("0x1005"), big.NewInt(1))
	statedb.SetNum(common.HexToAddress("0x1006"), big.NewInt(1))

	statedb.SetNum(common.HexToAddress("0x2001"), big.NewInt(1))
	statedb.SetNum(common.HexToAddress("0x2002"), big.NewInt(1))
	statedb.SetNum(common.HexToAddress("0x2003"), big.NewInt(1))
	statedb.SetNum(common.HexToAddress("0x2004"), big.NewInt(1))
	statedb.SetNum(common.HexToAddress("0x2005"), big.NewInt(1))
	statedb.SetNum(common.HexToAddress("0x2006"), big.NewInt(1))

	fmt.Println("model meta")
	fmt.Println(common.ToHex(statedb.GetCode(common.HexToAddress("0x1001"))))
	fmt.Println(common.ToHex(statedb.GetCode(common.HexToAddress("0x1002"))))
	fmt.Println(common.ToHex(statedb.GetCode(common.HexToAddress("0x1003"))))
	fmt.Println(common.ToHex(statedb.GetCode(common.HexToAddress("0x1004"))))
	fmt.Println(common.ToHex(statedb.GetCode(common.HexToAddress("0x1005"))))
	fmt.Println("input meta")
	fmt.Println(common.ToHex(statedb.GetCode(common.HexToAddress("0x2001"))))
	fmt.Println(common.ToHex(statedb.GetCode(common.HexToAddress("0x2002"))))
	fmt.Println(common.ToHex(statedb.GetCode(common.HexToAddress("0x2003"))))
	fmt.Println(common.ToHex(statedb.GetCode(common.HexToAddress("0x2004"))))
	fmt.Println(common.ToHex(statedb.GetCode(common.HexToAddress("0x2005"))))
	fmt.Println(common.ToHex(statedb.GetCode(common.HexToAddress("0x2006"))))
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
	}

	storageDir := ""
	if ctx.GlobalString(StorageDir.Name) != "" {
		storageDir = ctx.GlobalString(StorageDir.Name)
	}

	storagefs := torrentfs.CreateStorage("simple", torrentfs.Config{
		DataDir: storageDir,
	})

	initialGas := ctx.GlobalUint64(GasFlag.Name)
	runtimeConfig := runtime.Config{
		Origin:      sender,
		State:       statedb,
		GasLimit:    initialGas,
		GasPrice:    utils.GlobalBig(ctx, PriceFlag.Name),
		Value:       utils.GlobalBig(ctx, ValueFlag.Name),
		BlockNumber: new(big.Int).SetUint64(blockNumber),
		CVMConfig: vm.Config{
			Tracer:       tracer,
			Debug:        ctx.GlobalBool(DebugFlag.Name) || ctx.GlobalBool(MachineFlag.Name),
			StorageDir:   storageDir,
			DebugInferVM: true,
			Storagefs:    storagefs,
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
	fmt.Println("cvm storageDir", storageDir)
	inferServer := infer.New(&infer.Config{
		// StorageDir: storageDir,
		IsNotCache:    false,
		IsRemoteInfer: false,
		DeviceType:    "cpu",
		DeviceId:      0,
		Debug:         true,
		Storagefs:     storagefs,
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
		statedb.Commit(true)
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
		fmt.Fprintf(os.Stderr, `cvm execution time: %v
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
