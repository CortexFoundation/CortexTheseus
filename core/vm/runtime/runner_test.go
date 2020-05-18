package runtime

import (
	"encoding/hex"
	"fmt"
	"math/big"
	"os"
	"testing"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/core/state"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/core/vm"
	"github.com/CortexFoundation/CortexTheseus/crypto"
	"github.com/CortexFoundation/CortexTheseus/ctxcdb"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/rlp"
)

func TestRunCmd(t *testing.T) {
	glogger := log.NewGlogHandler(log.StreamHandler(os.Stderr, log.TerminalFormat(false)))
	log.Root().SetHandler(glogger)

	var (
		tracer      vm.Tracer
		debugLogger *vm.StructLogger
		statedb     *state.StateDB
		sender      = common.BytesToAddress([]byte("sender"))
		receiver    = common.BytesToAddress([]byte("receiver"))
		blockNumber uint64
	)
	logconfig := &vm.LogConfig{
		Debug: true,
	}
	debugLogger = vm.NewStructLogger(logconfig)
	tracer = debugLogger
	{
		statedb, _ = state.New(common.Hash{}, state.NewDatabase(ctxcdb.NewMemDatabase()))
	}
	statedb.CreateAccount(sender)
	mh, _ := hex.DecodeString("5c4d1f84063be8e25e83da6452b1821926548b3c2a2a903a0724e14d5c917b00")
	ih, _ := hex.DecodeString("c0a1f3c82e11e314822679e4834e3bc575bd017d12d888acda4a851a62d261dc")
	testModelMeta, _ := rlp.EncodeToBytes(
		&types.ModelMeta{
			Hash:          common.BytesToHash(mh),
			RawSize:       10000,
			InputShape:    []uint64{10, 1},
			OutputShape:   []uint64{1},
			Gas:           100000,
			AuthorAddress: common.BytesToAddress(crypto.Keccak256([]byte{0x2, 0x2})),
		})
	// new a modelmeta at 0x1001 and new a datameta at 0x2001

	testInputMeta, _ := rlp.EncodeToBytes(
		&types.InputMeta{
			Hash:          common.BytesToHash(ih),
			RawSize:       10000,
			Shape:         []uint64{1},
			AuthorAddress: common.BytesToAddress(crypto.Keccak256([]byte{0x3})),
		})
	statedb.SetCode(common.HexToAddress("0x1001"), append([]byte{0x0, 0x1}, []byte(testModelMeta)...))
	statedb.SetCode(common.HexToAddress("0x2001"), append([]byte{0x0, 0x2}, []byte(testInputMeta)...))

	var (
		code []byte
		ret  []byte
		err  error
	)

	code = common.Hex2Bytes("60086000612001611001c000")
	input_flag := ""

	runtimeConfig := &Config{
		Origin:      sender,
		State:       statedb,
		GasLimit:    uint64(10000000),
		GasPrice:    new(big.Int),
		Value:       new(big.Int),
		BlockNumber: new(big.Int).SetUint64(blockNumber),
		CVMConfig: vm.Config{
			Tracer:   tracer,
			Debug:    true,
			InferURI: "http://127.0.0.1:5000/infer",
		},
	}

	if false {
		input := append(code, input_flag...)
		setDefaults(runtimeConfig)
		var (
			vmenv  = NewEnv(runtimeConfig)
			sender = vm.AccountRef(runtimeConfig.Origin)
		)
		code, address, leftOverGas, _, err := vmenv.Create(
			sender,
			input,
			runtimeConfig.GasLimit,
			runtimeConfig.Value,
		)
		fmt.Println(code, address, leftOverGas, err)
	} else {
		if len(code) > 0 {
			statedb.SetCode(receiver, code)
		}
		ret, _, err = Call(receiver, common.Hex2Bytes(input_flag), runtimeConfig)
	}

	if true {
		if debugLogger != nil {
			fmt.Fprintln(os.Stderr, "#### TRACE ####")
			vm.WriteTrace(os.Stderr, debugLogger.StructLogs())
		}
		fmt.Fprintln(os.Stderr, "#### LOGS ####")
		vm.WriteLogs(os.Stderr, statedb.Logs())
	}

	if tracer == nil {
		fmt.Printf("0x%x\n", ret)
		if err != nil {
			fmt.Printf(" error: %v\n", err)
		}
	}

}
