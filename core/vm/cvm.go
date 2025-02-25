// Copyright 2018 The go-ethereum Authors
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

package vm

import (
	_ "encoding/hex"
	"math/big"
	"sync/atomic"

	"github.com/holiman/uint256"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/crypto"
	"github.com/CortexFoundation/CortexTheseus/params"
)

type (
	// CanTransferFunc is the signature of a transfer guard function
	CanTransferFunc func(StateDB, common.Address, *big.Int) bool
	// TransferFunc is the signature of a transfer function
	TransferFunc func(StateDB, common.Address, common.Address, *big.Int)
	// GetHashFunc returns the nth block hash in the blockchain
	// and is used by the BLOCKHASH CVM op code.
	GetHashFunc func(uint64) common.Hash
)

// run runs the given contract and takes care of running precompiles with a fallback to the byte code interpreter.
func (cvm *CVM) precompile(addr common.Address) (PrecompiledContract, bool) {
	var precompiles map[common.Address]PrecompiledContract
	switch {
	case cvm.chainRules.IsNeo:
		precompiles = PrecompiledContractsNeo
	case cvm.chainRules.IsIstanbul:
		precompiles = PrecompiledContractsIstanbul
	case cvm.chainRules.IsByzantium:
		precompiles = PrecompiledContractsByzantium
	default:
		precompiles = PrecompiledContractsHomestead
	}
	p, ok := precompiles[addr]
	return p, ok
}

// Context provides the CVM with auxiliary information. Once provided
// it shouldn't be modified.
type BlockContext struct {
	// CanTransfer returns whether the account contains
	// sufficient ctxcer to transfer the value
	CanTransfer CanTransferFunc
	// Transfer transfers ctxcer from one account to the other
	Transfer TransferFunc
	// GetHash returns the hash corresponding to n
	GetHash GetHashFunc

	// Block information
	Coinbase    common.Address // Provides information for COINBASE
	GasLimit    uint64         // Provides information for GASLIMIT
	Quota       uint64
	BlockNumber *big.Int     // Provides information for NUMBER
	Time        uint64       // Provides information for TIME
	Difficulty  *big.Int     // Provides information for DIFFICULTY
	Random      *common.Hash // Provides information for RANDOM
}

// TxContext provides the CVM with information about a transaction.
// All fields can change between transactions.
type TxContext struct {
	// Message information
	Origin   common.Address // Provides information for ORIGIN
	GasPrice *big.Int       // Provides information for GASPRICE
}

// CVM is the Cortex Virtual Machine base object and provides
// the necessary tools to run a contract on the given state with
// the provided context. It should be noted that any error
// generated through any of the calls should be considered a
// revert-state-and-consume-all-gas operation, no checks on
// specific errors should ever be performed. The interpreter makes
// sure that any errors generated are to be considered faulty code.
//
// The CVM should never be reused and is not thread safe.
type CVM struct {
	// Context provides auxiliary blockchain related information
	Context BlockContext
	TxContext
	// StateDB gives access to the underlying state
	StateDB StateDB
	// Depth is the current call stack
	depth int

	// chainConfig contains information about the current chain
	chainConfig *params.ChainConfig
	// chain rules contains the chain rules for the current epoch
	chainRules params.Rules

	category Category
	// virtual machine configuration options used to initialise the
	// cvm.
	vmConfig Config
	// global (to this context) cortex virtual machine
	// used throughout the execution of the tx.
	interpreter *CVMInterpreter
	// abort is used to abort the CVM calling operations
	// NOTE: must be set atomically
	abort atomic.Bool
	// callGasTemp holds the gas available for the current call. This is needed because the
	// available gas is calculated in gasCall* according to the 63/64 rule and later
	// applied in opCall*.
	callGasTemp uint64
	//Fs          *torrentfs.FileStorage

	// precompiles holds the precompiled contracts for the current epoch
	precompiles map[common.Address]PrecompiledContract

	// jumpDests is the aggregated result of JUMPDEST analysis made through
	// the life cycle of EVM.
	jumpDests map[common.Hash]bitvec
}

// NewCVM returns a new CVM. The returned CVM is not thread safe and should
// only ever be used *once*.
func NewCVM(blockCtx BlockContext, statedb StateDB, chainConfig *params.ChainConfig, vmConfig Config) *CVM {

	cvm := &CVM{
		Context:     blockCtx,
		StateDB:     statedb,
		vmConfig:    vmConfig,
		chainConfig: chainConfig,
		category:    Category{},
		chainRules:  chainConfig.Rules(blockCtx.BlockNumber, blockCtx.Random != nil, blockCtx.Time),
		jumpDests:   make(map[common.Hash]bitvec),
	}

	cvm.precompiles = activePrecompiledContracts(cvm.chainRules)
	cvm.interpreter = NewCVMInterpreter(cvm)

	return cvm
}

// SetPrecompiles sets the precompiled contracts for the EVM.
// This method is only used through RPC calls.
// It is not thread-safe.
func (cvm *CVM) SetPrecompiles(precompiles PrecompiledContracts) {
	cvm.precompiles = precompiles
}

// Reset resets the CVM with a new transaction context.Reset
// This is not threadsafe and should only be done very cautiously.
func (cvm *CVM) Reset(txCtx TxContext, statedb StateDB) {
	cvm.TxContext = txCtx
	cvm.StateDB = statedb
}

func (cvm *CVM) SetTxContext(txCtx TxContext) {
	cvm.TxContext = txCtx
}

// Cancel cancels any running CVM operation. This may be called concurrently and
// it's safe to be called multiple times.
func (cvm *CVM) Cancel() {
	cvm.abort.Store(true)
}

func (cvm *CVM) Cancelled() bool {
	return cvm.abort.Load()
}

// Interpreter returns the current interpreter
func (cvm *CVM) Interpreter() *CVMInterpreter {
	return cvm.interpreter
}

func isSystemCall(caller common.Address) bool {
	return caller == params.SystemAddress
}

func (cvm *CVM) Config() Config {
	return cvm.vmConfig
}

func (cvm *CVM) SetExtraEips(extraEips []int) {
	cvm.vmConfig.ExtraEips = extraEips
}

// SetBlockContext updates the block context of the CVM.
func (cvm *CVM) SetBlockContext(blockCtx BlockContext) {
	cvm.Context = blockCtx
	num := blockCtx.BlockNumber
	timestamp := blockCtx.Time
	cvm.chainRules = cvm.chainConfig.Rules(num, blockCtx.Random != nil, timestamp)
}

// Call executes the contract associated with the addr with the given input as
// parameters. It also handles any necessary value transfer required and takes
// the necessary steps to create accounts and reverses the state in case of an
// execution error or failed value transfer.
func (cvm *CVM) Call(caller common.Address, addr common.Address, input []byte, gas uint64, value *big.Int) (ret []byte, leftOverGas uint64, modelGas map[common.Address]uint64, err error) {
	//if cvm.vmConfig.NoRecursion && cvm.depth > 0 {
	//	return nil, gas, nil, nil
	//}

	// Fail if we're trying to execute above the call depth limit
	if cvm.depth > int(params.CallCreateDepth) {
		return nil, gas, nil, ErrDepth
	}
	// Fail if we're trying to transfer more than the available balance
	if value.Sign() != 0 && !cvm.Context.CanTransfer(cvm.StateDB, caller, value) {
		return nil, gas, nil, ErrInsufficientBalance
	}
	snapshot := cvm.StateDB.Snapshot()
	p, isPrecompile := cvm.precompile(addr)
	debug := cvm.Config().Tracer != nil

	if !cvm.StateDB.Exist(addr) {
		if !isPrecompile && cvm.chainRules.IsEIP158 && value.Sign() == 0 {
			// Calling a non existing account, don't do anything, but ping the tracer
			if debug {
				if cvm.depth == 0 {
					cvm.vmConfig.Tracer.CaptureStart(cvm, caller, addr, false, input, gas, value)
					cvm.vmConfig.Tracer.CaptureEnd(ret, 0, nil)
				} else {
					cvm.Config().Tracer.CaptureEnter(CALL, caller, addr, input, gas, value)
					cvm.Config().Tracer.CaptureExit(ret, 0, nil)
				}
			}
			return nil, gas, nil, nil
		}
		cvm.StateDB.CreateAccount(addr)
	}
	cvm.Context.Transfer(cvm.StateDB, caller, addr, value)

	// Capture the tracer start/end events in debug mode
	if debug {
		if cvm.depth == 0 {
			cvm.vmConfig.Tracer.CaptureStart(cvm, caller, addr, false, input, gas, value)
			defer func(startGas uint64) { // Lazy evaluation of the parameters
				cvm.vmConfig.Tracer.CaptureEnd(ret, startGas-gas, err)
			}(gas)
		} else {
			// Handle tracer events for entering and exiting a call frame
			cvm.Config().Tracer.CaptureEnter(CALL, caller, addr, input, gas, value)
			defer func(startGas uint64) {
				cvm.Config().Tracer.CaptureExit(ret, startGas-gas, err)
			}(gas)
		}
	}

	if isPrecompile {
		ret, gas, err = RunPrecompiledContract(p, input, gas)
	} else {
		// Initialise a new contract and set the code that is to be used by the CVM.
		// The contract is a scoped environment for this execution context only.
		code := cvm.StateDB.GetCode(addr)
		if len(code) == 0 {
			ret, err = nil, nil // gas is unchanged
		} else {
			contract := NewContract(caller, addr, value, gas, cvm.jumpDests)
			contract.IsSystemCall = isSystemCall(caller)
			contract.SetCallCode(cvm.resolveCodeHash(addr), code)
			ret, err = cvm.interpreter.Run(contract, input, false)
			gas = contract.Gas
			modelGas = contract.ModelGas
			if cvm.vmConfig.RPC_GetInternalTransaction {
				ret = append(ret, []byte(caller.String()+"-"+addr.String()+"-"+value.String()+",")...)
			}
		}
	}
	// When an error was returned by the CVM or when setting the creation code
	// above we revert to the snapshot and consume any gas remaining. Additionally
	// when we're in homestead this also counts for code storage gas errors.
	if err != nil {
		cvm.StateDB.RevertToSnapshot(snapshot)
		if err != ErrExecutionReverted {
			gas = 0
		}
		// TODO: consider clearing up unused snapshots:
		//} else {
		//	cvm.StateDB.DiscardSnapshot(snapshot)
	}
	return ret, gas, modelGas, err
}

// CallCode executes the contract associated with the addr with the given input
// as parameters. It also handles any necessary value transfer required and takes
// the necessary steps to create accounts and reverses the state in case of an
// execution error or failed value transfer.
//
// CallCode differs from Call in the sense that it executes the given address'
// code with the caller as context.
func (cvm *CVM) CallCode(caller common.Address, addr common.Address, input []byte, gas uint64, value *big.Int) (ret []byte, leftOverGas uint64, modelGas map[common.Address]uint64, err error) {
	//if cvm.vmConfig.NoRecursion && cvm.depth > 0 {
	//	return nil, gas, nil, nil
	//}

	// Fail if we're trying to execute above the call depth limit
	if cvm.depth > int(params.CallCreateDepth) {
		return nil, gas, nil, ErrDepth
	}
	// Fail if we're trying to transfer more than the available balance
	if !cvm.Context.CanTransfer(cvm.StateDB, caller, value) {
		return nil, gas, nil, ErrInsufficientBalance
	}
	var snapshot = cvm.StateDB.Snapshot()

	// It is allowed to call precompiles, even via delegatecall
	if p, isPrecompile := cvm.precompile(addr); isPrecompile {
		ret, gas, err = RunPrecompiledContract(p, input, gas)
	} else {
		contract := NewContract(caller, caller, value, gas, cvm.jumpDests)
		contract.SetCallCode(cvm.resolveCodeHash(addr), cvm.resolveCode(addr))
		ret, err = cvm.interpreter.Run(contract, input, false)
		gas = contract.Gas
		modelGas = contract.ModelGas
	}
	if err != nil {
		cvm.StateDB.RevertToSnapshot(snapshot)
		if err != ErrExecutionReverted {
			gas = 0
		}
	}
	return ret, gas, modelGas, err
}

// DelegateCall executes the contract associated with the addr with the given input
// as parameters. It reverses the state in case of an execution error.
//
// DelegateCall differs from CallCode in the sense that it executes the given address'
// code with the caller as context and the caller is set to the caller of the caller.
func (cvm *CVM) DelegateCall(originCaller common.Address, caller common.Address, addr common.Address, input []byte, gas uint64, value *big.Int) (ret []byte, leftOverGas uint64, modelGas map[common.Address]uint64, err error) {
	//if cvm.vmConfig.NoRecursion && cvm.depth > 0 {
	//	return nil, gas, nil, nil
	//}
	// Fail if we're trying to execute above the call depth limit
	if cvm.depth > int(params.CallCreateDepth) {
		return nil, gas, nil, ErrDepth
	}
	var snapshot = cvm.StateDB.Snapshot()

	// It is allowed to call precompiles, even via delegatecall
	if p, isPrecompile := cvm.precompile(addr); isPrecompile {
		ret, gas, err = RunPrecompiledContract(p, input, gas)
	} else {
		// Initialise a new contract and make initialise the delegate values
		//
		// Note: The value refers to the original value from the parent call.
		contract := NewContract(originCaller, caller, value, gas, cvm.jumpDests)
		contract.SetCallCode(cvm.resolveCodeHash(addr), cvm.resolveCode(addr))
		ret, err = cvm.interpreter.Run(contract, input, false)
		gas = contract.Gas
		modelGas = contract.ModelGas
	}
	if err != nil {
		cvm.StateDB.RevertToSnapshot(snapshot)
		if err != ErrExecutionReverted {
			gas = 0
		}
	}
	return ret, gas, modelGas, err
}

// StaticCall executes the contract associated with the addr with the given input
// as parameters while disallowing any modifications to the state during the call.
// Opcodes that attempt to perform such modifications will result in exceptions
// instead of performing the modifications.
func (cvm *CVM) StaticCall(caller common.Address, addr common.Address, input []byte, gas uint64) (ret []byte, leftOverGas uint64, modelGas map[common.Address]uint64, err error) {
	//if cvm.vmConfig.NoRecursion && cvm.depth > 0 {
	//	return nil, gas, nil, nil
	//}
	// Fail if we're trying to execute above the call depth limit
	if cvm.depth > int(params.CallCreateDepth) {
		return nil, gas, nil, ErrDepth
	}
	// We take a snapshot here. This is a bit counter-intuitive, and could probably be skipped.
	// However, even a staticcall is considered a 'touch'. On mainnet, static calls were introduced
	// after all empty accounts were deleted, so this is not required. However, if we omit this,
	// then certain tests start failing; stRevertTest/RevertPrecompiledTouchExactOOG.json.
	// We could change this, but for now it's left for legacy reasons
	var snapshot = cvm.StateDB.Snapshot()

	// We do an AddBalance of zero here, just in order to trigger a touch.
	// This doesn't matter on Mainnet, where all empties are gone at the time of Byzantium,
	// but is the correct thing to do and matters on other networks, in tests, and potential
	// future scenarios
	cvm.StateDB.AddBalance(addr, big0)

	if p, isPrecompile := cvm.precompile(addr); isPrecompile {
		ret, gas, err = RunPrecompiledContract(p, input, gas)
	} else {
		// At this point, we use a copy of address. If we don't, the go compiler will
		// leak the 'contract' to the outer scope, and make allocation for 'contract'
		// even if the actual execution ends on RunPrecompiled above.
		contract := NewContract(caller, addr, new(big.Int), gas, cvm.jumpDests)
		contract.SetCallCode(cvm.resolveCodeHash(addr), cvm.resolveCode(addr))
		// When an error was returned by the CVM or when setting the creation code
		// above we revert to the snapshot and consume any gas remaining. Additionally
		// when we're in Homestead this also counts for code storage gas errors.
		ret, err = cvm.interpreter.Run(contract, input, true)
		gas = contract.Gas
		modelGas = contract.ModelGas
	}
	if err != nil {
		cvm.StateDB.RevertToSnapshot(snapshot)
		if err != ErrExecutionReverted {
			gas = 0
		}
	}
	return ret, gas, modelGas, err
}

// create creates a new contract using code as deployment code.
func (cvm *CVM) create(caller common.Address, code []byte, gas uint64, value *big.Int, address common.Address, typ OpCode) ([]byte, common.Address, uint64, map[common.Address]uint64, error) {
	// Depth check execution. Fail if we're trying to execute above the
	// limit.
	if cvm.depth > int(params.CallCreateDepth) {
		return nil, common.Address{}, gas, nil, ErrDepth
	}
	if !cvm.Context.CanTransfer(cvm.StateDB, caller, value) {
		return nil, common.Address{}, gas, nil, ErrInsufficientBalance
	}
	nonce := cvm.StateDB.GetNonce(caller)
	if nonce+1 < nonce {
		return nil, common.Address{}, gas, nil, ErrNonceUintOverflow
	}
	cvm.StateDB.SetNonce(caller, nonce+1)

	// Ensure there's no existing contract already at the designated address
	contractHash := cvm.StateDB.GetCodeHash(address)
	storageRoot := cvm.StateDB.GetStorageRoot(address)
	if cvm.StateDB.GetNonce(address) != 0 ||
		(contractHash != (common.Hash{}) && contractHash != types.EmptyCodeHash) || // non-empty code
		(storageRoot != (common.Hash{}) && storageRoot != types.EmptyRootHash) { // non-empty storage
		return nil, common.Address{}, 0, nil, ErrContractAddressCollision
	}
	// Create a new account on the state
	snapshot := cvm.StateDB.Snapshot()
	if !cvm.StateDB.Exist(address) {
		cvm.StateDB.CreateAccount(address)
	}
	// CreateContract means that regardless of whether the account previously existed
	// in the state trie or not, it _now_ becomes created as a _contract_ account.
	// This is performed _prior_ to executing the initcode,  since the initcode
	// acts inside that account.
	cvm.StateDB.CreateContract(address)
	if cvm.chainRules.IsEIP158 {
		cvm.StateDB.SetNonce(address, 1)
	}
	cvm.Context.Transfer(cvm.StateDB, caller, address, value)

	// initialise a new contract and set the code that is to be used by the
	// CVM. The contract is a scoped environment for this execution contex only.
	contract := NewContract(caller, address, value, gas, cvm.jumpDests)

	// Explicitly set the code to a null hash to prevent caching of jump analysis
	// for the initialization code.
	contract.SetCallCode(common.Hash{}, code)

	if cvm.Config().Tracer != nil {
		if cvm.depth == 0 {
			cvm.vmConfig.Tracer.CaptureStart(cvm, caller, address, true, code, gas, value)
		} else {
			cvm.Config().Tracer.CaptureEnter(typ, caller, address, code, gas, value)
		}
	}

	ret, err := cvm.initNewContract(contract, address)

	// When an error was returned by the CVM or when setting the creation code
	// above we revert to the snapshot and consume any gas remaining. Additionally
	// when we're in homestead this also counts for code storage gas errors.
	if err != nil && (cvm.chainRules.IsHomestead || err != ErrCodeStoreOutOfGas) {
		cvm.StateDB.RevertToSnapshot(snapshot)
		if err != ErrExecutionReverted {
			contract.UseGas(contract.Gas)
		}
	}

	if cvm.vmConfig.RPC_GetInternalTransaction {
		ret = append(ret, []byte(caller.String()+"-"+address.String()+"-"+value.String()+",")...)
	}

	if cvm.Config().Tracer != nil {
		if cvm.depth == 0 {
			cvm.vmConfig.Tracer.CaptureEnd(ret, gas-contract.Gas, err)
		} else {
			cvm.Config().Tracer.CaptureExit(ret, gas-contract.Gas, err)
		}
	}

	return ret, address, contract.Gas, contract.ModelGas, err
}

// initNewContract runs a new contract's creation code, performs checks on the
// resulting code that is to be deployed, and consumes necessary gas.
func (cvm *CVM) initNewContract(contract *Contract, address common.Address) ([]byte, error) {
	ret, err := cvm.interpreter.Run(contract, nil, false)
	if err != nil {
		return ret, err
	}

	// check whether the max code size has been exceeded
	if err == nil && cvm.chainRules.IsEIP158 && len(ret) > params.MaxCodeSize {
		return ret, ErrMaxCodeSizeExceeded
	}
	// if the contract creation ran successfully and no errors were returned
	// calculate the gas required to store the code. If the code could not
	// be stored due to not enough gas set an error and let it be handled
	// by the error checking condition below.
	if err == nil {
		createDataGas := uint64(len(ret)) * params.CreateDataGas
		if contract.UseGas(createDataGas) {
			cvm.StateDB.SetCode(address, ret)
		} else {
			return ret, ErrCodeStoreOutOfGas
		}
	}

	return ret, err
}

// Create creates a new contract using code as deployment code.
func (cvm *CVM) Create(caller common.Address, code []byte, gas uint64, value *big.Int) (ret []byte, contractAddr common.Address, leftOverGas uint64, modelGas map[common.Address]uint64, err error) {
	//contractAddr = crypto.CreateAddress(caller.Address(), cvm.StateDB.GetNonce(caller.Address()))
	contractAddr = crypto.CreateAddress(caller, cvm.StateDB.GetNonce(caller))
	//return cvm.create(caller, code, gas, value, contractAddr)
	return cvm.create(caller, code, gas, value, contractAddr, CREATE)
}

// Create2 creates a new contract using code as deployment code.
//
// The different between Create2 with Create is Create2 uses sha3(0xff ++ msg.sender ++ salt ++ sha3(init_code))[12:]
// instead of the usual sender-and-nonce-hash as the address where the contract is initialized at.
func (cvm *CVM) Create2(caller common.Address, code []byte, gas uint64, endowment *big.Int, salt *uint256.Int) (ret []byte, contractAddr common.Address, leftOverGas uint64, modelGas map[common.Address]uint64, err error) {
	//contractAddr = crypto.CreateAddress2(caller.Address(), common.BigToHash(salt), code)
	//	return cvm.create(caller, code, gas, endowment, contractAddr)
	contractAddr = crypto.CreateAddress2(caller, salt.Bytes32(), crypto.Keccak256(code))
	return cvm.create(caller, code, gas, endowment, contractAddr, CREATE2)
}

// resolveCode returns the code associated with the provided account. After
// Prague, it can also resolve code pointed to by a delegation designator.
func (cvm *CVM) resolveCode(addr common.Address) []byte {
	code := cvm.StateDB.GetCode(addr)
	return code
}

// resolveCodeHash returns the code hash associated with the provided address.
// After Prague, it can also resolve code hash of the account pointed to by a
// delegation designator. Although this is not accessible in the EVM it is used
// internally to associate jumpdest analysis to code.
func (cvm *CVM) resolveCodeHash(addr common.Address) common.Hash {
	return cvm.StateDB.GetCodeHash(addr)
}

// ChainConfig returns the environment's chain configuration
func (cvm *CVM) ChainConfig() *params.ChainConfig { return cvm.chainConfig }

/*const interv = 5

func (cvm *CVM) DataSync(meta common.Address, dir string, errCh chan error) {
	street := big.NewInt(0).Sub(cvm.PeekNumber, cvm.BlockNumber)
	point := big.NewInt(time.Now().Add(confirmTime).Unix())
	if point.Cmp(cvm.Context.Time) > 0 || street.Cmp(big.NewInt(params.CONFIRM_BLOCKS)) > 0 {
		cost := big.NewInt(0)
		duration := big.NewInt(0).Sub(big.NewInt(time.Now().Unix()), cvm.Context.Time)
		for i := 0; i < 3600 && duration.Cmp(cost) > 0; i++ {
			if !torrentfs.ExistTorrent(meta.String()) {
				log.Warn("Inference synchronizing ... ...", "point", point, "tvm", cvm.Context.Time, "ago", common.PrettyDuration(time.Duration(duration.Uint64()*1000000000)), "level", i, "number", cvm.BlockNumber, "street", street)
				cost.Add(cost, big.NewInt(interv))
				time.Sleep(time.Second * interv)
				continue
			} else {
				errCh <- nil
				return
			}
		}
		log.Error("Torrent synchronized timeout", "address", meta.Hex(), "number", cvm.BlockNumber, "meta", meta, "storage", dir, "street", street, "duration", duration, "cost", cost)
	} else {
		if !torrentfs.Exist(meta.String()) {
			log.Warn("Data not exist", "address", meta.Hex(), "number", cvm.BlockNumber, "current", cvm.BlockNumber, "meta", meta, "storage", dir)
			errCh <- synapse.ErrModelFileNotExist
			return
		} else {
			errCh <- nil
			return
		}
	}

	if !torrentfs.Exist(meta.String()) {
		log.Warn("Data not exist", "address", meta.Hex(), "number", cvm.BlockNumber, "current", cvm.BlockNumber, "meta", meta, "storage", dir)
		errCh <- synapse.ErrModelFileNotExist
		return
	} else {
		errCh <- nil
		return
	}

	//log.Error("Torrent synchronized timeout", "address", meta.Hex(), "number", cvm.BlockNumber, "meta", meta, "storage", dir, "street", street)
	//errCh <- synapse.ErrModelFileNotExist
	//return
}*/

// infer function that returns an int64 as output, can be used a categorical output
/*func (cvm *CVM) Infer(modelInfoHash, inputInfoHash string, modelRawSize, inputRawSize uint64) ([]byte, error) {
	var (
		inferRes []byte
		errRes   error
	)

	start := mclock.Now()

	cvmVersion := synapse.CVMVersion(cvm.chainConfig, cvm.Context.BlockNumber)
	model := common.StorageEntry{
		Hash: modelInfoHash,
		Size: modelRawSize,
	}
	input := common.StorageEntry{
		Hash: inputInfoHash,
		Size: inputRawSize,
	}
	inferRes, errRes = synapse.Engine().InferByInfoHashWithSize(model, input, cvmVersion, cvm.chainConfig.ChainID.Int64())

	elapsed := time.Duration(mclock.Now()) - time.Duration(start)

	if errRes == nil {
		log.Debug("[hash ] succeed", "label", inferRes, "model", modelInfoHash, "input", inputInfoHash, "number", cvm.Context.BlockNumber, "elapsed", common.PrettyDuration(elapsed))
	}
	// ret := synapse.ArgMax(inferRes)
	if cvm.vmConfig.DebugInferVM {
		fmt.Println("infer result: ", inferRes, errRes)
	}
	return inferRes, errRes
}

// infer function that returns an int64 as output, can be used a categorical output
func (cvm *CVM) InferArray(modelInfoHash string, inputArray []byte, modelRawSize uint64) ([]byte, error) {
	log.Trace("Detail", "Input Content", hexutil.Encode(inputArray))

	if cvm.vmConfig.DebugInferVM {
		fmt.Println("Model Hash", modelInfoHash, "number", cvm.Context.BlockNumber, "Input Content", hexutil.Encode(inputArray))
	}

	var (
		inferRes []byte
		errRes   error
	)

	start := mclock.Now()

	cvmVersion := synapse.CVMVersion(cvm.chainConfig, cvm.Context.BlockNumber)
	model := common.StorageEntry{
		Hash: modelInfoHash,
		Size: modelRawSize,
	}
	inferRes, errRes = synapse.Engine().InferByInputContentWithSize(model, inputArray, cvmVersion, cvm.chainConfig.ChainID.Int64())
	elapsed := time.Duration(mclock.Now()) - time.Duration(start)

	if errRes == nil {
		log.Debug("[array] succeed", "label", inferRes, "model", modelInfoHash, "array", inputArray, "number", cvm.Context.BlockNumber, "elapsed", common.PrettyDuration(elapsed))
	}
	return inferRes, errRes
}

// infer function that returns an int64 as output, can be used a categorical output
func (cvm *CVM) OpsInfer(addr common.Address) (opsRes uint64, errRes error) {
	modelMeta, err := cvm.GetModelMeta(addr)
	if err != nil {
		return 0, err
	}

	start := mclock.Now()
	model := common.StorageEntry{
		Hash: modelMeta.Hash.Hex(),
		Size: modelMeta.RawSize,
	}
	opsRes, errRes = synapse.Engine().GetGasByInfoHashWithSize(model, cvm.chainConfig.ChainID.Int64())

	elapsed := time.Duration(mclock.Now()) - time.Duration(start)

	if errRes == nil {
		log.Debug("[ops  ] succeed", "ops", opsRes, "addr", addr, "elapsed", common.PrettyDuration(elapsed))
	}

	return opsRes, errRes
}

func (cvm *CVM) GetModelMeta(addr common.Address) (meta *torrentfs.ModelMeta, err error) {
	modelMetaRaw := cvm.StateDB.GetCode(addr)
	var modelMeta torrentfs.ModelMeta
	if err := modelMeta.DecodeRLP(modelMetaRaw); err != nil {
		return nil, err
	} else {
		return &modelMeta, nil
	}
}

func (cvm *CVM) GetInputMeta(addr common.Address) (meta *torrentfs.InputMeta, err error) {
	inputMetaRaw := cvm.StateDB.GetCode(addr)
	var inputMeta torrentfs.InputMeta
	if err := inputMeta.DecodeRLP(inputMetaRaw); err != nil {
		return nil, err
	} else {
		return &inputMeta, nil
	}
}*/
