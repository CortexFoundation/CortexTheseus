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
	"fmt"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/math"
	"github.com/CortexFoundation/CortexTheseus/crypto"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
)

// Config are the configuration options for the Interpreter
type Config struct {
	// Tracer is the op code logger
	Tracer CVMLogger
	// NoRecursion disabled Interpreter call, callcode,
	// delegate call and create.
	//NoRecursion bool
	// Enable recording of SHA3/keccak preimages
	EnablePreimageRecording bool
	// uri for remote infer service
	// InferURI string
	// rpc getInternalTransaction flag
	RPC_GetInternalTransaction bool

	// opCall flag
	CallFakeVM   bool
	DebugInferVM bool
	StorageDir   string
	// Storagefs    torrentfs.CVMStorage

	ExtraEips               []int // Additional EIPS that are to be enabled
	EnableWitnessCollection bool  // true if witness collection is enabled
}

// only for the sake of debug info of NewPublicBlockChainAPI
type ConfigAux struct {
	InferURI string
}

// ScopeContext contains the things that are per-call, such as stack and memory,
// but not transients like pc and gas
type ScopeContext struct {
	Memory   *Memory
	Stack    *Stack
	Contract *Contract
}

// CVMInterpreter represents an CVM interpreter
type CVMInterpreter struct {
	cvm      *CVM
	table    *JumpTable
	gasTable params.GasTable

	hasher    crypto.KeccakState // Keccak256 hasher instance shared across opcodes
	hasherBuf common.Hash        // Keccak256 hasher result array shared aross opcodes

	readOnly   bool   // Whether to throw on stateful modifications
	returnData []byte // Last CALL's return data for subsequent reuse

	//Code bool
	//ModelMeta bool
	//InputMeta bool
}

// NewCVMInterpreter returns a new instance of the Interpreter.
func NewCVMInterpreter(cvm *CVM) *CVMInterpreter {
	// If jump table was not initialised we set the default one.
	var table *JumpTable
	switch {
	case cvm.chainRules.IsMerge:
		table = &mergeInstructionSet
	case cvm.chainRules.IsNeo:
		table = &neoInstructionSet
	case cvm.chainRules.IsIstanbul:
		table = &istanbulInstructionSet
	case cvm.chainRules.IsConstantinople:
		table = &constantinopleInstructionSet
	case cvm.chainRules.IsByzantium:
		table = &byzantiumInstructionSet
	case cvm.chainRules.IsEIP158:
		table = &spuriousDragonInstructionSet
	case cvm.chainRules.IsEIP150:
		table = &tangerineWhistleInstructionSet
	case cvm.chainRules.IsHomestead:
		table = &homesteadInstructionSet
	default:
		table = &frontierInstructionSet
	}
	var extraEips []int
	if len(cvm.Config().ExtraEips) > 0 {
		// Deep-copy jumptable to prevent modification of opcodes in other tables
		table = copyJumpTable(table)
	}
	for _, eip := range cvm.Config().ExtraEips {
		if err := EnableEIP(eip, table); err != nil {
			// Disable it, so caller can check if it's activated or not
			log.Error("EIP activation failed", "eip", eip, "error", err)
		} else {
			extraEips = append(extraEips, eip)
		}
	}
	//cvm.Config().ExtraEips = extraEips
	cvm.SetExtraEips(extraEips)

	return &CVMInterpreter{
		cvm:      cvm,
		table:    table,
		gasTable: cvm.ChainConfig().GasTable(cvm.Context.BlockNumber),
		hasher:   crypto.NewKeccakState(),
	}
}

func (in *CVMInterpreter) enforceRestrictions(op OpCode, operation *operation, stack *Stack) error {
	/*if in.cvm.chainRules.IsByzantium {
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
	}*/
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
	// Cortex code category solved
	in.cvm.category.IsCode, in.cvm.category.IsModel, in.cvm.category.IsInput = in.cvm.IsCode(contract.Code), in.cvm.IsModel(contract.Code), in.cvm.IsInput(contract.Code)

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

	// Inference data prepare
	if in.cvm.category.IsModel || in.cvm.category.IsInput {
		return in.prepareData(contract, input)
	}

	var (
		op          OpCode     // current opcode
		jumpTable   *JumpTable = in.table
		mem                    = NewMemory() // bound memory
		stack                  = newstack()  // local stack
		callContext            = &ScopeContext{
			Memory:   mem,
			Stack:    stack,
			Contract: contract,
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
		res     []byte
		debug   = in.cvm.Config().Tracer != nil
	)
	// Don't move this deferrred function, it's placed before the capturestate-deferred method,
	// so that it get's executed _after_: the capturestate needs the stacks before
	// they are returned to the pools
	defer func() {
		returnStack(stack)
		mem.Free()
	}()
	contract.Input = input

	// Reclaim the stack as an int pool when the execution stops
	if debug {
		defer func() {
			if err != nil {
				if !logged {
					in.cvm.Config().Tracer.CaptureState(pcCopy, op, gasCopy, cost, callContext, in.returnData, in.cvm.depth, err)
				} else {
					in.cvm.Config().Tracer.CaptureFault(pcCopy, op, gasCopy, cost, callContext, in.cvm.depth, err)
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
	//for atomic.LoadInt32(&in.cvm.abort) == 0 {
	_ = jumpTable[0] // nil-check the jumpTable out of the loop
	for {
		if debug {
			// Capture pre-execution values for tracing.
			logged, pcCopy, gasCopy = false, pc, contract.Gas
		}

		// Get the operation from the jump table and validate the stack to ensure there are
		// enough stack items available to perform the operation.
		op = contract.GetOp(pc)
		operation := jumpTable[op]
		//if operation == nil {
		//	return nil, fmt.Errorf("invalid opcode 0x%x", int(op))
		//}
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

		if in.cvm.Config().DebugInferVM {
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

		if err != nil {
			return nil, ErrOutOfGas
		}

		// for tracing: this gas consumption event is emitted below in the debug section.
		if contract.Gas < cost {
			return nil, ErrOutOfGas
		} else {
			contract.Gas -= cost
		}

		if debug {
			in.cvm.Config().Tracer.CaptureState(pc, op, gasCopy, cost, callContext, in.returnData, in.cvm.depth, err)
			logged = true
		}

		if memorySize > 0 {
			mem.Resize(memorySize)
		}

		// execute the operation
		ret, err = operation.execute(&pc, in, callContext)
		if in.cvm.Config().RPC_GetInternalTransaction {
			if op == CALL {
				res = append(res, ret...)
			}
		} else {
			res = ret
		}

		if err != nil {
			break
		}
		pc++
	}
	if err == errStopToken {
		err = nil // clear stop token error
	}
	return res, err
}
