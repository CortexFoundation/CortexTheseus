// Copyright 2021 The CortexTheseus Authors
// This file is part of CortexTheseus.
//
// CortexTheseus is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// CortexTheseus is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with CortexTheseus. If not, see <http://www.gnu.org/licenses/>.

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/big"
	"os"
	goruntime "runtime"
	"runtime/pprof"
	"time"

	"github.com/CortexFoundation/CortexTheseus/cmd/cvm/compiler"
	"github.com/CortexFoundation/CortexTheseus/cmd/utils"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/core"
	"github.com/CortexFoundation/CortexTheseus/core/rawdb"
	"github.com/CortexFoundation/CortexTheseus/core/state"
	"github.com/CortexFoundation/CortexTheseus/core/vm"
	"github.com/CortexFoundation/CortexTheseus/core/vm/runtime"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/inference/synapse"
	torrentfs "github.com/CortexFoundation/torrentfs"
	torrentfsType "github.com/CortexFoundation/torrentfs/types"
	"gopkg.in/urfave/cli.v1"
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

type execStats struct {
	time           time.Duration // The execution time.
	allocs         int64         // The number of heap allocations during execution.
	bytesAllocated int64         // The cumulative number of bytes allocated during execution.
}

func timedExec(execFunc func() ([]byte, uint64, error)) (output []byte, gasLeft uint64, stats execStats, err error) {
	var memStatsBefore, memStatsAfter goruntime.MemStats
	goruntime.ReadMemStats(&memStatsBefore)
	startTime := time.Now()
	output, gasLeft, err = execFunc()
	stats.time = time.Since(startTime)
	goruntime.ReadMemStats(&memStatsAfter)
	stats.allocs = int64(memStatsAfter.Mallocs - memStatsBefore.Mallocs)
	stats.bytesAllocated = int64(memStatsAfter.TotalAlloc - memStatsBefore.TotalAlloc)

	return output, gasLeft, stats, err
}

// set address of AI input
func setInputMeta(statedb *state.StateDB) (err error) {
	inputAddress := [20]byte{}
	inputAddress[18] = 0x20
	inputAddress[19] = 0x13
	inputAddress = common.Address(inputAddress)

	inputMeta := torrentfsType.InputMeta{
		Hash:     inputAddress,
		RawSize:  912,
		Shape:    []uint64{10},
		BlockNum: *big.NewInt(1),
	}

	inputCode, err := inputMeta.ToBytes()
	if err != nil {
		fmt.Println("could not encode input: ", err)
		os.Exit(1)
	}
	inputCodeAddPrefix := make([]byte, 0)
	inputCodeAddPrefix = append(inputCodeAddPrefix, 0x00, 0x02)
	inputCodeAddPrefix = append(inputCodeAddPrefix, inputCode...)

	statedb.SetCode(inputAddress, inputCodeAddPrefix)
	statedb.SetNum(inputAddress, &inputMeta.BlockNum)
	inputMetaRaw := statedb.GetCode(inputAddress)
	err = inputMeta.DecodeRLP(inputMetaRaw)

	return
}

// set address of AI model
func setModelMeta(statedb *state.StateDB) (err error) {
	modelAddress := [20]byte{}
	modelAddress[18] = 0x10
	modelAddress[19] = 0x13
	modelAddress = common.Address(modelAddress)

	modelMeta := torrentfsType.ModelMeta{
		Hash:        modelAddress,
		RawSize:     446905,
		InputShape:  []uint64{10},
		OutputShape: []uint64{10},
		Gas:         222,
		BlockNum:    *big.NewInt(1),
	}

	modelCode, err := modelMeta.ToBytes()
	if err != nil {
		fmt.Println("could not encode model: ", err)
		os.Exit(1)
	}
	modelCodeAddPrefix := make([]byte, 0)
	modelCodeAddPrefix = append(modelCodeAddPrefix, 0x00, 0x01)
	modelCodeAddPrefix = append(modelCodeAddPrefix, modelCode...)
	statedb.SetCode(modelAddress, modelCodeAddPrefix)
	statedb.SetNum(modelAddress, &modelMeta.BlockNum)
	modelMetaRaw := statedb.GetCode(modelAddress)
	err = modelMeta.DecodeRLP(modelMetaRaw)

	return
}

// prepare cvm-runtime engine for running Infer!
func startSynapse() (err error) {
	fsCfg := torrentfs.DefaultConfig
	fsCfg.DataDir = "tf_data"
	storagefs, fsErr := torrentfs.New(&fsCfg, true, false, true)
	if fsErr != nil {
		return fsErr
	}

	synapse.New(&synapse.Config{
		IsNotCache:     false,
		DeviceType:     "cpu",
		DeviceId:       0,
		MaxMemoryUsage: synapse.DefaultConfig.MaxMemoryUsage,
		IsRemoteInfer:  false,
		InferURI:       "",
		Storagefs:      storagefs,
	})

	return nil
}

func runCmd(ctx *cli.Context) error {
	fmt.Println("CVM Individual Runner Started!")
	logConfig := &vm.LogConfig{
		Debug: true,
	}

	var (
		tracer        vm.Tracer
		debugLogger   *vm.StructLogger
		statedb       *state.StateDB
		chainConfig   *params.ChainConfig
		sender        = common.BytesToAddress([]byte("sender"))
		receiver      = common.BytesToAddress([]byte("receiver"))
		genesisConfig *core.Genesis
	)

	debugLogger = vm.NewStructLogger(logConfig)
	tracer = debugLogger

	if ctx.GlobalString(GenesisFlag.Name) != "" {
		gen := readGenesis(ctx.GlobalString(GenesisFlag.Name))
		genesisConfig = gen
		db := rawdb.NewMemoryDatabase()
		genesis := gen.ToBlock(db)
		statedb, _ = state.New(genesis.Root(), state.NewDatabase(db), nil)
		chainConfig = gen.Config
	} else {
		statedb, _ = state.New(common.Hash{}, state.NewDatabase(rawdb.NewMemoryDatabase()), nil)
		genesisConfig = new(core.Genesis)
	}
	if ctx.GlobalString(SenderFlag.Name) != "" {
		sender = common.HexToAddress(ctx.GlobalString(SenderFlag.Name))
	}
	statedb.CreateAccount(sender)

	if ctx.GlobalString(ReceiverFlag.Name) != "" {
		receiver = common.HexToAddress(ctx.GlobalString(ReceiverFlag.Name))
	}

	var code []byte
	codeFileFlag := ctx.GlobalString(CodeFileFlag.Name)
	codeFlag := ctx.GlobalString(CodeFlag.Name)

	// The '--code' or '--codefile' flag overrides code in state
	if codeFileFlag != "" || codeFlag != "" {
		var hexcode []byte
		if codeFileFlag != "" {
			var err error
			// If - is specified, it means that code comes from stdin
			if codeFileFlag == "-" {
				//Try reading from stdin
				if hexcode, err = ioutil.ReadAll(os.Stdin); err != nil {
					fmt.Printf("Could not load code from stdin: %v\n", err)
					os.Exit(1)
				}
			} else {
				// Codefile with hex assembly
				if hexcode, err = ioutil.ReadFile(codeFileFlag); err != nil {
					fmt.Printf("Could not load code from file: %v\n", err)
					os.Exit(1)
				}
			}
		} else {
			hexcode = []byte(codeFlag)
		}
		hexcode = bytes.TrimSpace(hexcode)
		if len(hexcode)%2 != 0 {
			fmt.Printf("Invalid input length for hex data (%d)\n", len(hexcode))
			os.Exit(1)
		}
		code = common.FromHex(string(hexcode))
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
	initialGas := ctx.GlobalUint64(GasFlag.Name)
	if genesisConfig.GasLimit != 0 {
		initialGas = genesisConfig.GasLimit
	}
	runtimeConfig := runtime.Config{
		Origin:      sender,
		State:       statedb,
		GasLimit:    initialGas,
		GasPrice:    utils.GlobalBig(ctx, PriceFlag.Name),
		Value:       utils.GlobalBig(ctx, ValueFlag.Name),
		Difficulty:  genesisConfig.Difficulty,
		Time:        new(big.Int).SetUint64(genesisConfig.Timestamp),
		Coinbase:    genesisConfig.Coinbase,
		BlockNumber: new(big.Int).SetUint64(3230001), //new(big.Int).SetUint64(genesisConfig.Number),
		CVMConfig: vm.Config{
			Tracer:         tracer,
			Debug:          ctx.GlobalBool(DebugFlag.Name) || ctx.GlobalBool(MachineFlag.Name),
			CVMInterpreter: ctx.GlobalString(CVMInterpreterFlag.Name),
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
	} else {
		runtimeConfig.ChainConfig = params.MainnetChainConfig
	}

	var hexInput []byte
	if inputFileFlag := ctx.GlobalString(InputFileFlag.Name); inputFileFlag != "" {
		var err error
		if hexInput, err = ioutil.ReadFile(inputFileFlag); err != nil {
			fmt.Printf("could not load input from file: %v\n", err)
			os.Exit(1)
		}
	} else {
		hexInput = []byte(ctx.GlobalString(InputFlag.Name))
	}
	input := common.FromHex(string(bytes.TrimSpace(hexInput)))

	var execFunc func() ([]byte, uint64, error)
	if ctx.GlobalBool(CreateFlag.Name) {
		input = append(code, input...)
		execFunc = func() ([]byte, uint64, error) {
			output, _, gasLeft, err := runtime.Create(input, &runtimeConfig)
			return output, gasLeft, err
		}
	} else {
		if len(code) > 0 {
			statedb.SetCode(receiver, code)
		}
		execFunc = func() ([]byte, uint64, error) {
			return runtime.Call(receiver, input, &runtimeConfig)
		}
	}

	// set address of AI model
	err := setModelMeta(statedb)
	if err != nil {
		log.Error(fmt.Sprintf("could not decode model: %v", err))
		os.Exit(1)
	}

	// set address of AI input
	err = setInputMeta(statedb)
	if err != nil {
		log.Error(fmt.Sprintf("could not decode input: %v", err))
		os.Exit(1)
	}

	// prepare cvm-runtime engine for running Infer!
	err = startSynapse()
	if err != nil {
		log.Error(fmt.Sprintf("New torrentfs err: %v", err))
		os.Exit(1)
	}
	log.Info("New torrentfs and synapse Success!")

	// config done, start execution
	output, leftOverGas, stats, err := timedExec(execFunc)

	fmt.Printf("CVM Runner Execute Done! err? [%v]\n", err)

	if ctx.GlobalBool(DumpFlag.Name) {
		statedb.Commit(true)
		statedb.IntermediateRoot(true)
		fmt.Println(string(statedb.Dump(false, false, true)))
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
		fmt.Fprintf(os.Stderr, `CVM gas used:    %d
execution time:  %v
allocations:     %d
allocated bytes: %d
`, initialGas-leftOverGas, stats.time, stats.allocs, stats.bytesAllocated)
	}
	if tracer == nil {
		fmt.Printf("0x%x\n", output)
		if err != nil {
			fmt.Printf(" error: %v\n", err)
		}
	}

	return nil
}
