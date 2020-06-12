// Copyright 2018 The CortexTheseus Authors
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
	"fmt"
	"hash"
	"math/big"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/math"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
	torrentfs "github.com/CortexFoundation/torrentfs/types"
	"sync/atomic"
)

var (
//MIN_UPLOAD_BYTES     uint64 = 0
//MAX_UPLOAD_BYTES     uint64 = 1024 * 1024 * 1024 * 1024
//DEFAULT_UPLOAD_BYTES uint64 = 10 * 512 * 1024
//MODEL_GAS_LIMIT      uint64 = 65536
)

// Config are the configuration options for the Interpreter
type Config struct {
	// Debug enabled debugging Interpreter options
	Debug bool
	// Tracer is the op code logger
	Tracer Tracer
	// NoRecursion disabled Interpreter call, callcode,
	// delegate call and create.
	NoRecursion bool
	// Enable recording of SHA3/keccak preimages
	EnablePreimageRecording bool
	// JumpTable contains the CVM instruction table. This
	// may be left uninitialised and will be set to the default
	// table.
	JumpTable [256]operation
	// uri for remote infer service
	// InferURI string
	// rpc getInternalTransaction flag
	RPC_GetInternalTransaction bool

	// opCall flag
	CallFakeVM   bool
	DebugInferVM bool
	StorageDir   string
	// Storagefs    torrentfs.CVMStorage

	CWASMInterpreter string // External CWASM interpreter options
	CVMInterpreter   string // External CVM interpreter options

	ExtraEips []int // Additional EIPS that are to be enabled
}

// only for the sake of debug info of NewPublicBlockChainAPI
type ConfigAux struct {
	InferURI string
}

// Interpreter is used to run Cortex based contracts and will utilise the
// passed environment to query external sources for state information.
// The Interpreter will run the byte code VM based on the passed
// configuration.
type Interpreter interface {
	// Run loops and evaluates the contract's code with the given input data and returns
	// the return byte-slice and an error if one occurred.
	Run(contract *Contract, input []byte, static bool) ([]byte, error)
	// CanRun tells if the contract, passed as an argument, can be
	// run by the current interpreter. This is meant so that the
	// caller can do something like:
	//
	// ```golang
	// for _, interpreter := range interpreters {
	//   if interpreter.CanRun(contract.code) {
	//     interpreter.Run(contract.code, input)
	//   }
	// }
	// ```
	CanRun([]byte) bool
	//IsModelMeta([]byte) bool
	//IsInputMeta([]byte) bool
}

// callCtx contains the things that are per-call, such as stack and memory,
// but not transients like pc and gas
type callCtx struct {
	memory   *Memory
	stack    *Stack
	rstack   *ReturnStack
	contract *Contract
}

// keccakState wraps sha3.state. In addition to the usual hash methods, it also supports
// Read to get a variable amount of data from the hash state. Read is faster than Sum
// because it doesn't copy the internal state, but also modifies the internal state.
type keccakState interface {
	hash.Hash
	Read([]byte) (int, error)
}

// CVMInterpreter represents an CVM interpreter
type CVMInterpreter struct {
	cvm      *CVM
	cfg      Config
	gasTable params.GasTable

	hasher    keccakState // Keccak256 hasher instance shared across opcodes
	hasherBuf common.Hash // Keccak256 hasher result array shared aross opcodes

	readOnly   bool   // Whether to throw on stateful modifications
	returnData []byte // Last CALL's return data for subsequent reuse

	//Code bool
	//ModelMeta bool
	//InputMeta bool
}

// NewCVMInterpreter returns a new instance of the Interpreter.
func NewCVMInterpreter(cvm *CVM, cfg Config) *CVMInterpreter {
	// We use the STOP instruction whether to see
	// the jump table was initialised. If it was not
	// we'll set the default jump table.
	// log.Debug("NewCVMInterpreter", "cvm.ChainConfig().IsByzantium(cvm.BlockNumber)", cvm.ChainConfig().IsByzantium(cvm.BlockNumber), "cvm.ChainConfig().IsConstantinople(cvm.BlockNumber)", cvm.ChainConfig().IsConstantinople(cvm.BlockNumber))
	if !cfg.JumpTable[STOP].valid {
		var jt JumpTable
		switch {
		case cvm.chainRules.IsIstanbul:
			jt = istanbulInstructionSet
		case cvm.chainRules.IsConstantinople:
			jt = constantinopleInstructionSet
		case cvm.chainRules.IsByzantium:
			jt = byzantiumInstructionSet
		case cvm.chainRules.IsEIP158:
			jt = spuriousDragonInstructionSet
		case cvm.chainRules.IsEIP150:
			jt = tangerineWhistleInstructionSet
		case cvm.chainRules.IsHomestead:
			jt = homesteadInstructionSet
		default:
			jt = frontierInstructionSet
		}
		for i, eip := range cfg.ExtraEips {
			if err := EnableEIP(eip, &jt); err != nil {
				// Disable it, so caller can check if it's activated or not
				cfg.ExtraEips = append(cfg.ExtraEips[:i], cfg.ExtraEips[i+1:]...)
				log.Error("EIP activation failed", "eip", eip, "error", err)
			}
		}
		cfg.JumpTable = jt
	}

	return &CVMInterpreter{
		cvm:      cvm,
		cfg:      cfg,
		gasTable: cvm.ChainConfig().GasTable(cvm.BlockNumber),
	}
}

func (in *CVMInterpreter) enforceRestrictions(op OpCode, operation operation, stack *Stack) error {
	if in.cvm.chainRules.IsByzantium {
		if in.readOnly {
			// If the interpreter is operating in readonly mode, make sure no
			// state-modifying operation is performed. The 3rd stack item
			// for a call operation is the value. Transferring value from one
			// account to the others means the state is modified and should also
			// return with an error.
			//if operation.writes || (op == CALL && stack.Back(2).BitLen() > 0) {
			if operation.writes || (op == CALL && stack.Back(2).Sign() != 0) {
				return errWriteProtection
			}
		}
	}
	return nil
}

/*func (in *CVMInterpreter) IsCode(code []byte) bool {
	if len(code) >= 2 && code[0] == 0 && code[1] == 0 {
		return true
	}
	return false
}
func (in *CVMInterpreter) IsModelMeta(code []byte) bool {
	if len(code) >= 2 && code[0] == 0 && code[1] == 1 {
		return true
	}
	return false
}

func (in *CVMInterpreter) IsInputMeta(code []byte) bool {
	if len(code) >= 2 && code[0] == 0 && code[1] == 2 {
		return true
	}
	return false
}*/

// Run loops and evaluates the contract's code with the given input data and returns
// the return byte-slice and an error if one occurred.
//
// It's important to note that any errors returned by the interpreter should be
// considered a revert-and-consume-all-gas operation except for
// errExecutionReverted which means revert-and-keep-gas-left.
func (in *CVMInterpreter) Run(contract *Contract, input []byte, readOnly bool) (ret []byte, err error) {

	// Increment the call depth which is restricted to 1024
	in.cvm.depth++
	defer func() { in.cvm.depth-- }()

	// Make sure the readOnly is only set if we aren't in readOnly yet.
	// This makes also sure that the readOnly flag isn't removed for child calls.
	if readOnly && !in.readOnly {
		in.readOnly = true
		defer func() { in.readOnly = false }()
	}

	// Reset the previous call's return data. It's unimportant to preserve the old buffer
	// as every returning call will return new data anyway.
	in.returnData = nil

	// Don't bother with the execution if there's no code.
	if contract == nil || len(contract.Code) == 0 {
		return nil, nil
	}

	//if in.IsModelMeta(contract.Code) {
	if in.cvm.category.IsModel {
		if in.cvm.vmConfig.RPC_GetInternalTransaction {
			return nil, nil
		}

		if input != nil {
			log.Debug("Readonly for model meta")
			return nil, nil
		}

		var modelMeta torrentfs.ModelMeta
		if err := modelMeta.DecodeRLP(contract.Code); err != nil {
			log.Error("Failed decode model meta", "code", contract.Code, "err", err)
			return nil, err
		} else {
			log.Debug("Model meta",
				"meta", modelMeta,
				"modelMeta.RawSize", modelMeta.RawSize,
				"Upload", in.cvm.StateDB.Upload(contract.Address()),
				"params.MODEL_MIN_UPLOAD_BYTES", params.MODEL_MIN_UPLOAD_BYTES)
			if modelMeta.BlockNum.Sign() == 0 {
				if modelMeta.RawSize > params.MODEL_MIN_UPLOAD_BYTES && modelMeta.RawSize <= params.MODEL_MAX_UPLOAD_BYTES { // 1Byte ~ 1TB

					//must in rawbytes if it is too small
					//if modelMeta.RawSize <= params.MaxRawSize {
					//if modelMeta.RawSize != uint64(len(modelMeta.RawBytes)) {
					//return nil, ErrInvalidMetaRawSize
					//}
					//} else {
					//deal with the big model
					//}

					if modelMeta.RawSize > params.DEFAULT_UPLOAD_BYTES {
						in.cvm.StateDB.SetUpload(contract.Address(), new(big.Int).SetUint64(modelMeta.RawSize-params.DEFAULT_UPLOAD_BYTES))
					}
				} else {
					return nil, ErrInvalidMetaRawSize
				}

				if !common.IsHexAddress(modelMeta.AuthorAddress.String()) {
					return nil, ErrInvalidMetaAuthor
				}

				if modelMeta.Gas == uint64(0) {
					//modelMeta.SetGas(params.MODEL_GAS_LIMIT)
					modelMeta.SetGas(0)
				} else if modelMeta.Gas > params.MODEL_GAS_UP_LIMIT {
					modelMeta.SetGas(params.MODEL_GAS_LIMIT)
				} else if int64(modelMeta.Gas) < 0 {
					modelMeta.SetGas(0)
				}

				in.cvm.StateDB.SetNum(contract.Address(), in.cvm.BlockNumber)
				modelMeta.SetBlockNum(*in.cvm.BlockNumber)
				if tmpCode, err := modelMeta.ToBytes(); err != nil {
					return nil, err
				} else {
					contract.Code = append([]byte{0, 1}, tmpCode...)
				}
				log.Debug("Model created", "size", modelMeta.RawSize, "hash", modelMeta.Hash.Hex(), "author", modelMeta.AuthorAddress.Hex(), "gas", modelMeta.Gas, "birth", modelMeta.BlockNum.Uint64())
			} else {
				log.Debug("Invalid model meta", "size", modelMeta.RawSize, "hash", modelMeta.Hash.Hex(), "author", modelMeta.AuthorAddress.Hex(), "gas", modelMeta.Gas, "birth", modelMeta.BlockNum.Uint64())
			}
			return contract.Code, nil
		}
	}

	//if in.IsInputMeta(contract.Code) {
	if in.cvm.category.IsInput {
		if in.cvm.vmConfig.RPC_GetInternalTransaction {
			return nil, nil
		}

		if input != nil {
			log.Debug("Readonly for input meta")
			return nil, nil
		}

		var inputMeta torrentfs.InputMeta
		if err := inputMeta.DecodeRLP(contract.Code); err != nil {
			log.Error("Failed decode input meta", "code", contract.Code, "err", err)
			return nil, err
		} else {
			if inputMeta.BlockNum.Sign() == 0 {
				if inputMeta.RawSize > 0 {
					if inputMeta.RawSize > params.DEFAULT_UPLOAD_BYTES {
						in.cvm.StateDB.SetUpload(contract.Address(), new(big.Int).SetUint64(inputMeta.RawSize-params.DEFAULT_UPLOAD_BYTES))
					}
				} else {
					return nil, ErrInvalidMetaRawSize
				}

				inputMeta.SetBlockNum(*in.cvm.BlockNumber)
				in.cvm.StateDB.SetNum(contract.Address(), in.cvm.BlockNumber)
				if tmpCode, err := inputMeta.ToBytes(); err != nil {
					return nil, err
				} else {
					contract.Code = append([]byte{0, 2}, tmpCode...)
				}
				//log.Info("Input meta created", "size", inputMeta.RawSize, "author", inputMeta.AuthorAddress)
			} else {
				log.Warn("Invalid input meta", "size", inputMeta.RawSize, "hash", inputMeta.Hash.Hex(), "birth", inputMeta.BlockNum.Uint64())
			}
			return contract.Code, nil
		}
	}

	var (
		op          OpCode             // current opcode
		mem         = NewMemory()      // bound memory
		stack       = newstack()       // local stack
		returns     = newReturnStack() // local returns stack
		callContext = &callCtx{
			memory:   mem,
			stack:    stack,
			rstack:   returns,
			contract: contract,
		}
		// For optimisation reason we're using uint64 as the program counter.
		// It's theoretically possible to go above 2^64. The YP defines the PC
		// to be uint256. Practically much less so feasible.
		pc   = uint64(0) // program counter
		cost uint64
		// copies used by tracer
		pcCopy  uint64 // needed for the deferred Tracer
		gasCopy uint64 // for Tracer to log gas remaining before execution
		logged  bool   // deferred Tracer should ignore already logged steps
	)
	contract.Input = input

	// Reclaim the stack as an int pool when the execution stops
	if in.cfg.Debug {
		defer func() {
			if err != nil {
				if !logged {
					in.cfg.Tracer.CaptureState(in.cvm, pcCopy, op, gasCopy, cost, mem, stack, returns, contract, in.cvm.depth, err)
				} else {
					in.cfg.Tracer.CaptureFault(in.cvm, pcCopy, op, gasCopy, cost, mem, stack, returns, contract, in.cvm.depth, err)
				}
			}
		}()
	}
	// The Interpreter main run loop (contextual). This loop runs until either an
	// explicit STOP, RETURN or SELFDESTRUCT is executed, an error occurred during
	// the execution of one of the operations or until the done flag is set by the
	// parent context.
	//if in.IsCode(contract.Code) {
	if in.cvm.category.IsCode {
		contract.Code = contract.Code[2:]
	}
	cgas := uint64(0)
	res := make([]byte, 10)
	for atomic.LoadInt32(&in.cvm.abort) == 0 {
		if in.cfg.Debug {
			// Capture pre-execution values for tracing.
			logged, pcCopy, gasCopy = false, pc, contract.Gas
		}

		// Get the operation from the jump table and validate the stack to ensure there are
		// enough stack items available to perform the operation.
		op = contract.GetOp(pc)
		operation := in.cfg.JumpTable[op]
		if !operation.valid {
			return nil, fmt.Errorf("invalid opcode 0x%x", int(op))
		}
		if err := operation.validateStack(stack); err != nil {
			return nil, err
		}
		// If the operation is valid, enforce and write restrictions
		if err := in.enforceRestrictions(op, operation, stack); err != nil {
			return nil, err
		}

		var memorySize uint64
		// calculate the new memory size and expand the memory to fit
		// the operation
		if operation.memorySize != nil {
			memSize, overflow := operation.memorySize(stack)
			if overflow {
				return nil, ErrGasUintOverflow
			}
			// memory is expanded in words of 32 bytes. Gas
			// is also calculated in words.
			if memorySize, overflow = math.SafeMul(toWordSize(memSize), 32); overflow {
				return nil, ErrGasUintOverflow
			}
		}

		cost, err = operation.gasCost(in.gasTable, in.cvm, contract, stack, mem, memorySize)
		cgas += cost

		if in.cvm.vmConfig.DebugInferVM {
			fmt.Println("gasCost: ", cost, "err: ", err, " op: ", op, "cgas: ", cgas)
		}

		// gasCost will check model's metainfo before checking available gas
		if err == ErrRuntime {
			return nil, err
		}

		if op.IsInfer() {
			modelMeta, err := in.cvm.GetModelMeta(common.Address(stack.Back(0).Bytes20()))
			if err != nil {
				return nil, err
			}
			//todo model validation
			if modelMeta.AuthorAddress != common.EmptyAddress {
				contract.ModelGas[modelMeta.AuthorAddress] += modelMeta.Gas
				log.Debug("Model gas earn", "author", modelMeta.AuthorAddress.Hex(), "gas", modelMeta.Gas)
			}
			var overflow bool
			if cost, overflow = math.SafeAdd(cost, modelMeta.Gas); overflow {
				log.Warn("overflow", "cost", cost, "gas", modelMeta.Gas)
				return nil, ErrGasUintOverflow
			}
		}

		if err != nil || !contract.UseGas(cost) {
			log.Debug("Interpreter", "cost", cost, "err", err, "cgas", cgas)
			return nil, ErrOutOfGas
		}

		if memorySize > 0 {
			mem.Resize(memorySize)
		}

		if in.cfg.Debug {
			in.cfg.Tracer.CaptureState(in.cvm, pc, op, gasCopy, cost, mem, stack, returns, contract, in.cvm.depth, err)
			logged = true
		}

		// execute the operation
		ret, err = operation.execute(&pc, in, callContext)
		if in.cvm.vmConfig.RPC_GetInternalTransaction {
			if op == CALL {
				res = append(res, ret...)
			}
		} else {
			res = ret
		}

		// if the operation clears the return data (e.g. it has returning data)
		// set the last return to the result of the operation.
		if operation.returns {
			in.returnData = res
		}
		switch {
		case err != nil:
			return nil, err
		case operation.reverts:
			return res, ErrExecutionReverted
		case operation.halts:
			return res, nil
		case !operation.jumps:
			pc++
		}
	}
	return nil, nil
}

// CanRun tells if the contract, passed as an argument, can be
// run by the current interpreter.
func (in *CVMInterpreter) CanRun(code []byte) bool {
	//in.Code = in.IsCode(code)
	//in.ModelMeta = in.IsModelMeta(code)
	//in.InputMeta = in.IsInputMeta(code)
	return true
}
