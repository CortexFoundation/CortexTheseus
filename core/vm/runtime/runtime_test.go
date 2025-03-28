// Copyright 2019 The go-ethereum Authors
// This file is part of the CortexFoundation library.
//
// The CortexFoundation library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexFoundation library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the CortexFoundation library. If not, see <http://www.gnu.org/licenses/>.

package runtime

import (
	"encoding/binary"
	"math/big"
	"strconv"
	"strings"
	"testing"

	"github.com/CortexFoundation/CortexTheseus/accounts/abi"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/consensus"
	"github.com/CortexFoundation/CortexTheseus/core"
	"github.com/CortexFoundation/CortexTheseus/core/rawdb"
	"github.com/CortexFoundation/CortexTheseus/core/state"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/core/vm"
	"github.com/CortexFoundation/CortexTheseus/params"
)

func TestDefaults(t *testing.T) {
	cfg := new(Config)
	setDefaults(cfg)

	if cfg.Difficulty == nil {
		t.Error("expected difficulty to be non nil")
	}

	if cfg.GasLimit == 0 {
		t.Error("didn't expect gaslimit to be zero")
	}
	if cfg.GasPrice == nil {
		t.Error("expected time to be non nil")
	}
	if cfg.Value == nil {
		t.Error("expected time to be non nil")
	}
	if cfg.GetHashFn == nil {
		t.Error("expected time to be non nil")
	}
	if cfg.BlockNumber == nil {
		t.Error("expected block number to be non nil")
	}
}

func TestCVM(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("crashed with: %v", r)
		}
	}()

	Execute([]byte{
		byte(vm.DIFFICULTY),
		byte(vm.TIMESTAMP),
		byte(vm.GASLIMIT),
		byte(vm.PUSH1),
		byte(vm.ORIGIN),
		byte(vm.BLOCKHASH),
		byte(vm.COINBASE),
	}, nil, nil)
}

func TestExecute(t *testing.T) {
	ret, _, err := Execute([]byte{
		byte(vm.PUSH1), 10,
		byte(vm.PUSH1), 0,
		byte(vm.MSTORE),
		byte(vm.PUSH1), 32,
		byte(vm.PUSH1), 0,
		byte(vm.RETURN),
	}, nil, nil)
	if err != nil {
		t.Fatal("didn't expect error", err)
	}

	num := new(big.Int).SetBytes(ret)
	if num.Cmp(big.NewInt(10)) != 0 {
		t.Error("Expected 10, got", num)
	}
}

func TestCall(t *testing.T) {
	state, _ := state.New(common.Hash{}, state.NewDatabase(rawdb.NewMemoryDatabase(), nil))
	address := common.HexToAddress("0x0a")
	state.SetCode(address, []byte{
		byte(vm.PUSH1), 10,
		byte(vm.PUSH1), 0,
		byte(vm.MSTORE),
		byte(vm.PUSH1), 32,
		byte(vm.PUSH1), 0,
		byte(vm.RETURN),
	})

	ret, _, err := Call(address, nil, &Config{State: state})
	if err != nil {
		t.Fatal("didn't expect error", err)
	}

	num := new(big.Int).SetBytes(ret)
	if num.Cmp(big.NewInt(10)) != 0 {
		t.Error("Expected 10, got", num)
	}
}

func BenchmarkCall(b *testing.B) {
	var definition = `[{"constant":true,"inputs":[],"name":"seller","outputs":[{"name":"","type":"address"}],"type":"function"},{"constant":false,"inputs":[],"name":"abort","outputs":[],"type":"function"},{"constant":true,"inputs":[],"name":"value","outputs":[{"name":"","type":"uint256"}],"type":"function"},{"constant":false,"inputs":[],"name":"refund","outputs":[],"type":"function"},{"constant":true,"inputs":[],"name":"buyer","outputs":[{"name":"","type":"address"}],"type":"function"},{"constant":false,"inputs":[],"name":"confirmReceived","outputs":[],"type":"function"},{"constant":true,"inputs":[],"name":"state","outputs":[{"name":"","type":"uint8"}],"type":"function"},{"constant":false,"inputs":[],"name":"confirmPurchase","outputs":[],"type":"function"},{"inputs":[],"type":"constructor"},{"anonymous":false,"inputs":[],"name":"Aborted","type":"event"},{"anonymous":false,"inputs":[],"name":"PurchaseConfirmed","type":"event"},{"anonymous":false,"inputs":[],"name":"ItemReceived","type":"event"},{"anonymous":false,"inputs":[],"name":"Refunded","type":"event"}]`

	var code = common.Hex2Bytes("6060604052361561006c5760e060020a600035046308551a53811461007457806335a063b4146100865780633fa4f245146100a6578063590e1ae3146100af5780637150d8ae146100cf57806373fac6f0146100e1578063c19d93fb146100fe578063d696069714610112575b610131610002565b610133600154600160a060020a031681565b610131600154600160a060020a0390811633919091161461015057610002565b61014660005481565b610131600154600160a060020a039081163391909116146102d557610002565b610133600254600160a060020a031681565b610131600254600160a060020a0333811691161461023757610002565b61014660025460ff60a060020a9091041681565b61013160025460009060ff60a060020a9091041681146101cc57610002565b005b600160a060020a03166060908152602090f35b6060908152602090f35b60025460009060a060020a900460ff16811461016b57610002565b600154600160a060020a03908116908290301631606082818181858883f150506002805460a060020a60ff02191660a160020a179055506040517f72c874aeff0b183a56e2b79c71b46e1aed4dee5e09862134b8821ba2fddbf8bf9250a150565b80546002023414806101dd57610002565b6002805460a060020a60ff021973ffffffffffffffffffffffffffffffffffffffff1990911633171660a060020a1790557fd5d55c8a68912e9a110618df8d5e2e83b8d83211c57a8ddd1203df92885dc881826060a15050565b60025460019060a060020a900460ff16811461025257610002565b60025460008054600160a060020a0390921691606082818181858883f150508354604051600160a060020a0391821694503090911631915082818181858883f150506002805460a060020a60ff02191660a160020a179055506040517fe89152acd703c9d8c7d28829d443260b411454d45394e7995815140c8cbcbcf79250a150565b60025460019060a060020a900460ff1681146102f057610002565b6002805460008054600160a060020a0390921692909102606082818181858883f150508354604051600160a060020a0391821694503090911631915082818181858883f150506002805460a060020a60ff02191660a160020a179055506040517f8616bbbbad963e4e65b1366f1d75dfb63f9e9704bbbf91fb01bec70849906cf79250a15056")

	abi, err := abi.JSON(strings.NewReader(definition))
	if err != nil {
		b.Fatal(err)
	}

	cpurchase, err := abi.Pack("confirmPurchase")
	if err != nil {
		b.Fatal(err)
	}
	creceived, err := abi.Pack("confirmReceived")
	if err != nil {
		b.Fatal(err)
	}
	refund, err := abi.Pack("refund")
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 400; j++ {
			Execute(code, cpurchase, nil)
			Execute(code, creceived, nil)
			Execute(code, refund, nil)
		}
	}
}
func benchmarkCVM_Create(bench *testing.B, code string) {
	var (
		statedb, _ = state.New(common.Hash{}, state.NewDatabase(rawdb.NewMemoryDatabase(), nil))
		sender     = common.BytesToAddress([]byte("sender"))
		receiver   = common.BytesToAddress([]byte("receiver"))
	)

	statedb.CreateAccount(sender)
	statedb.SetCode(receiver, common.FromHex(code))
	runtimeConfig := Config{
		Origin:      sender,
		State:       statedb,
		GasLimit:    10000000,
		Difficulty:  big.NewInt(0x200000),
		Time:        0,
		Coinbase:    common.Address{},
		BlockNumber: new(big.Int).SetUint64(1),
		ChainConfig: &params.ChainConfig{
			ChainID:             big.NewInt(1),
			HomesteadBlock:      new(big.Int),
			ByzantiumBlock:      new(big.Int),
			ConstantinopleBlock: new(big.Int),
			DAOForkBlock:        new(big.Int),
			DAOForkSupport:      false,
			EIP150Block:         new(big.Int),
			EIP155Block:         new(big.Int),
			EIP158Block:         new(big.Int),
		},
		CVMConfig: vm.Config{},
	}
	// Warm up the intpools and stuff
	bench.ResetTimer()
	for i := 0; i < bench.N; i++ {
		Call(receiver, []byte{}, &runtimeConfig)
	}
	bench.StopTimer()
}

func BenchmarkCVM_CREATE_500(bench *testing.B) {
	// initcode size 500K, repeatedly calls CREATE and then modifies the mem contents
	benchmarkCVM_Create(bench, "5b6207a120600080f0600152600056")
}
func BenchmarkCVM_CREATE2_500(bench *testing.B) {
	// initcode size 500K, repeatedly calls CREATE2 and then modifies the mem contents
	benchmarkCVM_Create(bench, "5b586207a120600080f5600152600056")
}
func BenchmarkCVM_CREATE_1200(bench *testing.B) {
	// initcode size 1200K, repeatedly calls CREATE and then modifies the mem contents
	benchmarkCVM_Create(bench, "5b62124f80600080f0600152600056")
}
func BenchmarkCVM_CREATE2_1200(bench *testing.B) {
	// initcode size 1200K, repeatedly calls CREATE2 and then modifies the mem contents
	benchmarkCVM_Create(bench, "5b5862124f80600080f5600152600056")
}

func BenchmarkEVM_RETURN(b *testing.B) {
	// returns a contract that returns a zero-byte slice of len size
	returnContract := func(size uint64) []byte {
		contract := []byte{
			byte(vm.PUSH8), 0, 0, 0, 0, 0, 0, 0, 0, // PUSH8 0xXXXXXXXXXXXXXXXX
			byte(vm.PUSH0),  // PUSH0
			byte(vm.RETURN), // RETURN
		}
		binary.BigEndian.PutUint64(contract[1:], size)
		return contract
	}

	state, _ := state.New(types.EmptyRootHash, state.NewDatabase(rawdb.NewMemoryDatabase(), nil))
	contractAddr := common.BytesToAddress([]byte("contract"))

	for _, n := range []uint64{1_000, 10_000, 100_000, 1_000_000} {
		b.Run(strconv.FormatUint(n, 10), func(b *testing.B) {
			b.ReportAllocs()

			contractCode := returnContract(n)
			state.SetCode(contractAddr, contractCode)

			for i := 0; i < b.N; i++ {
				ret, _, err := Call(contractAddr, []byte{}, &Config{State: state})
				if err != nil {
					b.Fatal(err)
				}
				if uint64(len(ret)) != n {
					b.Fatalf("expected return size %d, got %d", n, len(ret))
				}
			}
		})
	}
}

func fakeHeader(n uint64, parentHash common.Hash) *types.Header {
	header := types.Header{
		Coinbase:   common.HexToAddress("0x00000000000000000000000000000000deadbeef"),
		Number:     big.NewInt(int64(n)),
		ParentHash: parentHash,
		Time:       1000,
		Nonce:      types.BlockNonce{0x1},
		Extra:      []byte{},
		Difficulty: big.NewInt(0),
		GasLimit:   100000,
	}
	return &header
}

type dummyChain struct {
	counter int
}

// Engine retrieves the chain's consensus engine.
func (d *dummyChain) Engine() consensus.Engine {
	return nil
}

// GetHeader returns the hash corresponding to their hash.
func (d *dummyChain) GetHeader(h common.Hash, n uint64) *types.Header {
	d.counter++
	parentHash := common.Hash{}
	s := common.LeftPadBytes(big.NewInt(int64(n-1)).Bytes(), 32)
	copy(parentHash[:], s)

	//parentHash := common.Hash{byte(n - 1)}
	//fmt.Printf("GetHeader(%x, %d) => header with parent %x\n", h, n, parentHash)
	return fakeHeader(n, parentHash)
}

// TestBlockhash tests the blockhash operation. It's a bit special, since it internally
// requires access to a chain reader.
func TestBlockhash(t *testing.T) {
	// Current head
	n := uint64(1000)
	parentHash := common.Hash{}
	s := common.LeftPadBytes(big.NewInt(int64(n-1)).Bytes(), 32)
	copy(parentHash[:], s)
	header := fakeHeader(n, parentHash)

	// This is the contract we're using. It requests the blockhash for current num (should be all zeroes),
	// then iteratively fetches all blockhashes back to n-260.
	// It returns
	// 1. the first (should be zero)
	// 2. the second (should be the parent hash)
	// 3. the last non-zero hash
	// By making the chain reader return hashes which correlate to the number, we can
	// verify that it obtained the right hashes where it should

	/*

		pragma solidity ^0.5.3;
		contract Hasher{

			function test() public view returns (bytes32, bytes32, bytes32){
				uint256 x = block.number;
				bytes32 first;
				bytes32 last;
				bytes32 zero;
				zero = blockhash(x); // Should be zeroes
				first = blockhash(x-1);
				for(uint256 i = 2 ; i < 260; i++){
					bytes32 hash = blockhash(x - i);
					if (uint256(hash) != 0){
						last = hash;
					}
				}
				return (zero, first, last);
			}
		}

	*/
	// The contract above
	data := common.Hex2Bytes("6080604052348015600f57600080fd5b50600436106045576000357c010000000000000000000000000000000000000000000000000000000090048063f8a8fd6d14604a575b600080fd5b60506074565b60405180848152602001838152602001828152602001935050505060405180910390f35b600080600080439050600080600083409050600184034092506000600290505b61010481101560c35760008186034090506000816001900414151560b6578093505b5080806001019150506094565b508083839650965096505050505090919256fea165627a7a72305820462d71b510c1725ff35946c20b415b0d50b468ea157c8c77dff9466c9cb85f560029")
	// The method call to 'test()'
	input := common.Hex2Bytes("f8a8fd6d")
	chain := &dummyChain{}
	ret, _, err := Execute(data, input, &Config{
		GetHashFn:   core.GetHashFn(header, chain),
		BlockNumber: new(big.Int).Set(header.Number),
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(ret) != 96 {
		t.Fatalf("expected returndata to be 96 bytes, got %d", len(ret))
	}

	zero := new(big.Int).SetBytes(ret[0:32])
	first := new(big.Int).SetBytes(ret[32:64])
	last := new(big.Int).SetBytes(ret[64:96])
	if zero.BitLen() != 0 {
		t.Fatalf("expected zeroes, got %x", ret[0:32])
	}
	if first.Uint64() != 999 {
		t.Fatalf("second block should be 999, got %d (%x)", first, ret[32:64])
	}
	if last.Uint64() != 744 {
		t.Fatalf("last block should be 744, got %d (%x)", last, ret[64:96])
	}
	if exp, got := 255, chain.counter; exp != got {
		t.Errorf("suboptimal; too much chain iteration, expected %d, got %d", exp, got)
	}
}

// benchmarkNonModifyingCode benchmarks code, but if the code modifies the
// state, this should not be used, since it does not reset the state between runs.
func benchmarkNonModifyingCode(gas uint64, code []byte, name string, b *testing.B) {
	cfg := new(Config)
	setDefaults(cfg)
	cfg.State, _ = state.New(common.Hash{}, state.NewDatabase(rawdb.NewMemoryDatabase(), nil))
	cfg.GasLimit = gas
	var (
		destination = common.BytesToAddress([]byte("contract"))
		vmenv       = NewEnv(cfg)
	)
	cfg.State.CreateAccount(destination)
	eoa := common.HexToAddress("E0")
	{
		cfg.State.CreateAccount(eoa)
		cfg.State.SetNonce(eoa, 100)
	}
	reverting := common.HexToAddress("EE")
	{
		cfg.State.CreateAccount(reverting)
		cfg.State.SetCode(reverting, []byte{
			byte(vm.PUSH1), 0x00,
			byte(vm.PUSH1), 0x00,
			byte(vm.REVERT),
		})
	}

	//cfg.State.CreateAccount(cfg.Origin)
	// set the receiver's (the executing contract) code for execution.
	cfg.State.SetCode(destination, code)
	vmenv.Call(cfg.Origin, destination, nil, gas, cfg.Value)

	b.Run(name, func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			vmenv.Call(cfg.Origin, destination, nil, gas, cfg.Value)
		}
	})
}

// BenchmarkSimpleLoop test a pretty simple loop which loops until OOG
// 55 ms
func BenchmarkSimpleLoop(b *testing.B) {

	staticCallIdentity := []byte{
		byte(vm.JUMPDEST), //  [ count ]
		// push args for the call
		byte(vm.PUSH1), 0, // out size
		byte(vm.DUP1),       // out offset
		byte(vm.DUP1),       // out insize
		byte(vm.DUP1),       // in offset
		byte(vm.PUSH1), 0x4, // address of identity
		byte(vm.GAS), // gas
		byte(vm.STATICCALL),
		byte(vm.POP),      // pop return value
		byte(vm.PUSH1), 0, // jumpdestination
		byte(vm.JUMP),
	}

	callIdentity := []byte{
		byte(vm.JUMPDEST), //  [ count ]
		// push args for the call
		byte(vm.PUSH1), 0, // out size
		byte(vm.DUP1),       // out offset
		byte(vm.DUP1),       // out insize
		byte(vm.DUP1),       // in offset
		byte(vm.DUP1),       // value
		byte(vm.PUSH1), 0x4, // address of identity
		byte(vm.GAS), // gas
		byte(vm.CALL),
		byte(vm.POP),      // pop return value
		byte(vm.PUSH1), 0, // jumpdestination
		byte(vm.JUMP),
	}

	callInexistant := []byte{
		byte(vm.JUMPDEST), //  [ count ]
		// push args for the call
		byte(vm.PUSH1), 0, // out size
		byte(vm.DUP1),        // out offset
		byte(vm.DUP1),        // out insize
		byte(vm.DUP1),        // in offset
		byte(vm.DUP1),        // value
		byte(vm.PUSH1), 0xff, // address of existing contract
		byte(vm.GAS), // gas
		byte(vm.CALL),
		byte(vm.POP),      // pop return value
		byte(vm.PUSH1), 0, // jumpdestination
		byte(vm.JUMP),
	}

	callEOA := []byte{
		byte(vm.JUMPDEST), //  [ count ]
		// push args for the call
		byte(vm.PUSH1), 0, // out size
		byte(vm.DUP1),        // out offset
		byte(vm.DUP1),        // out insize
		byte(vm.DUP1),        // in offset
		byte(vm.DUP1),        // value
		byte(vm.PUSH1), 0xE0, // address of EOA
		byte(vm.GAS), // gas
		byte(vm.CALL),
		byte(vm.POP),      // pop return value
		byte(vm.PUSH1), 0, // jumpdestination
		byte(vm.JUMP),
	}

	loopingCode := []byte{
		byte(vm.JUMPDEST), //  [ count ]
		// push args for the call
		byte(vm.PUSH1), 0, // out size
		byte(vm.DUP1),       // out offset
		byte(vm.DUP1),       // out insize
		byte(vm.DUP1),       // in offset
		byte(vm.PUSH1), 0x4, // address of identity
		byte(vm.GAS), // gas

		byte(vm.POP), byte(vm.POP), byte(vm.POP), byte(vm.POP), byte(vm.POP), byte(vm.POP),
		byte(vm.PUSH1), 0, // jumpdestination
		byte(vm.JUMP),
	}

	loopingCode2 := []byte{
		byte(vm.JUMPDEST), //  [ count ]
		// push args for the call
		byte(vm.PUSH4), 1, 2, 3, 4,
		byte(vm.PUSH5), 1, 2, 3, 4, 5,

		byte(vm.POP), byte(vm.POP),
		byte(vm.PUSH6), 0, 0, 0, 0, 0, 0, // jumpdestination
		byte(vm.JUMP),
	}

	calllRevertingContractWithInput := []byte{
		byte(vm.JUMPDEST), //
		// push args for the call
		byte(vm.PUSH1), 0, // out size
		byte(vm.DUP1),        // out offset
		byte(vm.PUSH1), 0x20, // in size
		byte(vm.PUSH1), 0x00, // in offset
		byte(vm.PUSH1), 0x00, // value
		byte(vm.PUSH1), 0xEE, // address of reverting contract
		byte(vm.GAS), // gas
		byte(vm.CALL),
		byte(vm.POP),      // pop return value
		byte(vm.PUSH1), 0, // jumpdestination
		byte(vm.JUMP),
	}

	//tracer := vm.NewJSONLogger(nil, os.Stdout)
	//Execute(loopingCode, nil, &Config{
	//	CVMConfig: vm.Config{
	//		Debug:  true,
	//		Tracer: tracer,
	//	}})
	// 100M gas
	benchmarkNonModifyingCode(100000000, staticCallIdentity, "staticcall-identity-100M", b)
	benchmarkNonModifyingCode(100000000, callIdentity, "call-identity-100M", b)
	benchmarkNonModifyingCode(100000000, loopingCode, "loop-100M", b)
	benchmarkNonModifyingCode(100000000, loopingCode2, "loop2-100M", b)
	benchmarkNonModifyingCode(100000000, callInexistant, "call-nonexist-100M", b)
	benchmarkNonModifyingCode(100000000, callEOA, "call-EOA-100M", b)
	benchmarkNonModifyingCode(100000000, calllRevertingContractWithInput, "call-reverting-100M", b)

	//benchmarkNonModifyingCode(10000000, staticCallIdentity, "staticcall-identity-10M", b)
	//benchmarkNonModifyingCode(10000000, loopingCode, "loop-10M", b)
}

func BenchmarkVM_SWAP1(b *testing.B) {
	// returns a contract that does n swaps (SWAP1)
	swapContract := func(n uint64) []byte {
		contract := []byte{
			byte(vm.PUSH0), // PUSH0
			byte(vm.PUSH0), // PUSH0
		}
		for i := uint64(0); i < n; i++ {
			contract = append(contract, byte(vm.SWAP1))
		}
		return contract
	}

	state, _ := state.New(types.EmptyRootHash, state.NewDatabase(rawdb.NewMemoryDatabase(), nil))
	contractAddr := common.BytesToAddress([]byte("contract"))

	b.Run("10k", func(b *testing.B) {
		contractCode := swapContract(10_000)
		state.SetCode(contractAddr, contractCode)

		for i := 0; i < b.N; i++ {
			_, _, err := Call(contractAddr, []byte{}, &Config{State: state})
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}
