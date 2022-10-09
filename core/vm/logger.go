// Copyright 2015 The CortexTheseus Authors
// This file is part of the CortexTheseus library.
//
// The CortexTheseus library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexTheseus library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the CortexTheseus library. If not, see <http://www.gnu.org/licenses/>.

package vm

import (
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math/big"
	"strings"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/common/math"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/holiman/uint256"
)

var errTraceLimitReached = errors.New("the number of logs reached the specified limit")

// Storage represents a contract's storage.
type Storage map[common.Hash]common.Hash

// Copy duplicates the current storage.
func (s Storage) Copy() Storage {
	cpy := make(Storage)
	for key, value := range s {
		cpy[key] = value
	}
	return cpy
}

// LogConfig are the configuration options for structured logger the CVM
type LogConfig struct {
	DisableMemory     bool // disable memory capture
	DisableStack      bool // disable stack capture
	DisableStorage    bool // disable storage capture
	DisableReturnData bool // disable return data capture
	Debug             bool // print output during capture end
	Limit             int  // maximum length of output, but zero means unlimited
}

//go:generate go run github.com/fjl/gencodec@latest -type StructLog -field-override structLogMarshaling -out gen_structlog.go

// StructLog is emitted to the CVM each cycle and lists information about the current internal state
// prior to the execution of the statement.
type StructLog struct {
	Pc            uint64                      `json:"pc"`
	Op            OpCode                      `json:"op"`
	Gas           uint64                      `json:"gas"`
	GasCost       uint64                      `json:"gasCost"`
	Memory        []byte                      `json:"memory"`
	MemorySize    int                         `json:"memSize"`
	Stack         []uint256.Int               `json:"stack"`
	ReturnData    []byte                      `json:"returnData"`
	Storage       map[common.Hash]common.Hash `json:"-"`
	Depth         int                         `json:"depth"`
	RefundCounter uint64                      `json:"refund"`
	Err           error                       `json:"-"`
}

// overrides for gencodec
type structLogMarshaling struct {
	Gas         math.HexOrDecimal64
	GasCost     math.HexOrDecimal64
	Memory      hexutil.Bytes
	ReturnData  hexutil.Bytes
	OpName      string `json:"opName"` // adds call to OpName() in MarshalJSON
	ErrorString string `json:"error"`  // adds call to ErrorString() in MarshalJSON
}

// OpName formats the operand name in a human-readable format.
func (s *StructLog) OpName() string {
	return s.Op.String()
}

// ErrorString formats the log's error as a string.
func (s *StructLog) ErrorString() string {
	if s.Err != nil {
		return s.Err.Error()
	}
	return ""
}

// Tracer is used to collect execution traces from an CVM transaction
// execution. CaptureState is called for each step of the VM with the
// current VM state.
// Note that reference types are actual VM data structures; make copies
// if you need to retain them beyond the current call.
type CVMLogger interface {
	CaptureStart(env *CVM, from common.Address, to common.Address, create bool, input []byte, gas uint64, value *big.Int)
	//CaptureState(env *CVM, pc uint64, op OpCode, gas, cost uint64, memory *Memory, stack *Stack, rData []byte, contract *Contract, depth int, err error) error
	CaptureState(pc uint64, op OpCode, gas, cost uint64, scope *ScopeContext, rData []byte, depth int, err error)
	//CaptureFault(env *CVM, pc uint64, op OpCode, gas, cost uint64, memory *Memory, stack *Stack, contract *Contract, depth int, err error) error
	CaptureFault(pc uint64, op OpCode, gas, cost uint64, scope *ScopeContext, depth int, err error)
	CaptureEnd(output []byte, gasUsed uint64, t time.Duration, err error)
}

// StructLogger is an CVM state logger and implements Tracer.
//
// StructLogger can capture state based on the given Log configuration and also keeps
// a track record of modified storage which is used in reporting snapshots of the
// contract their storage.
type StructLogger struct {
	cfg LogConfig
	env *CVM

	storage map[common.Address]Storage
	logs    []StructLog
	output  []byte
	err     error

	gasLimit uint64
	usedGas  uint64

	interrupt uint32 // Atomic flag to signal execution interruption
	reason    error  // Textual reason for the interruption
}

// NewStructLogger returns a new logger
func NewStructLogger(cfg *LogConfig) *StructLogger {
	logger := &StructLogger{
		storage: make(map[common.Address]Storage),
	}
	if cfg != nil {
		logger.cfg = *cfg
	}
	return logger
}

// Reset clears the data held by the logger.
func (l *StructLogger) Reset() {
	l.storage = make(map[common.Address]Storage)
	l.output = make([]byte, 0)
	l.logs = l.logs[:0]
	l.err = nil
}

// CaptureStart implements the Tracer interface to initialize the tracing operation.
func (l *StructLogger) CaptureStart(env *CVM, from common.Address, to common.Address, create bool, input []byte, gas uint64, value *big.Int) {
	l.env = env
}

// CaptureState logs a new structured log message and pushes it out to the environment
//
// CaptureState also tracks SLOAD/SSTORE ops to track storage change.
// func (l *StructLogger) CaptureState(env *CVM, pc uint64, op OpCode, gas, cost uint64, memory *Memory, stack *Stack, rData []byte, contract *Contract, depth int, err error){
func (l *StructLogger) CaptureState(pc uint64, op OpCode, gas, cost uint64, scope *ScopeContext, rData []byte, depth int, err error) {
	// check if already accumulated the specified number of logs
	if l.cfg.Limit != 0 && l.cfg.Limit <= len(l.logs) {
		err = errTraceLimitReached
	}
	// Copy a snapshot of the current memory state to a new buffer
	var mem []byte
	if !l.cfg.DisableMemory {
		mem = make([]byte, len(scope.Memory.Data()))
		copy(mem, scope.Memory.Data())
	}
	// Copy a snapshot of the current stack state to a new buffer
	var stck []uint256.Int
	if !l.cfg.DisableStack {
		stck = make([]uint256.Int, len(scope.Stack.Data()))
		for i, item := range scope.Stack.Data() {
			stck[i] = item
		}
	}
	// Copy a snapshot of the current storage to a new container
	var storage Storage
	if !l.cfg.DisableStorage && (op == SLOAD || op == SSTORE) {
		// initialise new changed values storage container for this contract
		// if not present.
		if l.storage[scope.Contract.Address()] == nil {
			l.storage[scope.Contract.Address()] = make(Storage)
		}
		// capture SLOAD opcodes and record the read entry in the local storage
		if op == SLOAD && scope.Stack.len() >= 1 {
			var (
				address = common.Hash(scope.Stack.data[scope.Stack.len()-1].Bytes32())
				value   = l.env.StateDB.GetState(scope.Contract.Address(), address)
			)
			l.storage[scope.Contract.Address()][address] = value
			storage = l.storage[scope.Contract.Address()].Copy()
		} else if op == SSTORE && scope.Stack.len() >= 2 {
			// capture SSTORE opcodes and record the written entry in the local storage.
			var (
				value   = common.Hash(scope.Stack.data[scope.Stack.len()-2].Bytes32())
				address = common.Hash(scope.Stack.data[scope.Stack.len()-1].Bytes32())
			)
			l.storage[scope.Contract.Address()][address] = value
		}
		storage = l.storage[scope.Contract.Address()].Copy()
	}
	var rdata []byte
	if !l.cfg.DisableReturnData {
		rdata = make([]byte, len(rData))
		copy(rdata, rData)
	}
	// create a new snapshot of the CVM.
	log := StructLog{pc, op, gas, cost, mem, scope.Memory.Len(), stck, rdata, storage, depth, l.env.StateDB.GetRefund(), err}
	l.logs = append(l.logs, log)
}

// CaptureFault implements the Tracer interface to trace an execution fault
// while running an opcode.
func (l *StructLogger) CaptureFault(env *CVM, pc uint64, op OpCode, gas, cost uint64, memory *Memory, stack *Stack, contract *Contract, depth int, err error) {
}

// CaptureEnd is called after the call finishes to finalize the tracing.
func (l *StructLogger) CaptureEnd(output []byte, gasUsed uint64, t time.Duration, err error) {
	l.output = output
	l.err = err
	if l.cfg.Debug {
		fmt.Printf("0x%x\n", output)
		if err != nil {
			fmt.Printf(" error: %v\n", err)
		}
	}
}

// StructLogs returns the captured log entries.
func (l *StructLogger) StructLogs() []StructLog { return l.logs }

// Error returns the VM error captured by the trace.
func (l *StructLogger) Error() error { return l.err }

// Output returns the VM return value captured by the trace.
func (l *StructLogger) Output() []byte { return l.output }

// WriteTrace writes a formatted trace to the given writer
func WriteTrace(writer io.Writer, logs []StructLog) {
	for _, log := range logs {
		fmt.Fprintf(writer, "%-16spc=%08d gas=%v cost=%v", log.Op, log.Pc, log.Gas, log.GasCost)
		if log.Err != nil {
			fmt.Fprintf(writer, " ERROR: %v", log.Err)
		}
		fmt.Fprintln(writer)

		if len(log.Stack) > 0 {
			fmt.Fprintln(writer, "Stack:")
			for i := len(log.Stack) - 1; i >= 0; i-- {
				fmt.Fprintf(writer, "%08d %s\n", len(log.Stack)-i-1, log.Stack[i].Hex())
			}
		}
		if len(log.Memory) > 0 {
			fmt.Fprintln(writer, "Memory:")
			fmt.Fprint(writer, hex.Dump(log.Memory))
		}
		if len(log.Storage) > 0 {
			fmt.Fprintln(writer, "Storage:")
			for h, item := range log.Storage {
				fmt.Fprintf(writer, "%x: %x\n", h, item)
			}
		}
		if len(log.ReturnData) > 0 {
			fmt.Fprintln(writer, "ReturnData:")
			fmt.Fprint(writer, hex.Dump(log.ReturnData))
		}
		fmt.Fprintln(writer)
	}
}

// WriteLogs writes vm logs in a readable format to the given writer
func WriteLogs(writer io.Writer, logs []*types.Log) {
	for _, log := range logs {
		fmt.Fprintf(writer, "LOG%d: %x bn=%d txi=%x\n", len(log.Topics), log.Address, log.BlockNumber, log.TxIndex)

		for i, topic := range log.Topics {
			fmt.Fprintf(writer, "%08d  %x\n", i, topic)
		}

		fmt.Fprint(writer, hex.Dump(log.Data))
		fmt.Fprintln(writer)
	}
}

type mdLogger struct {
	out io.Writer
	cfg *LogConfig
}

// NewMarkdownLogger creates a logger which outputs information in a format adapted
// for human readability, and is also a valid markdown table
func NewMarkdownLogger(cfg *LogConfig, writer io.Writer) *mdLogger {
	l := &mdLogger{writer, cfg}
	if l.cfg == nil {
		l.cfg = &LogConfig{}
	}
	return l
}

func (t *mdLogger) CaptureStart(from common.Address, to common.Address, create bool, input []byte, gas uint64, value *big.Int) {
	if !create {
		fmt.Fprintf(t.out, "From: `%v`\nTo: `%v`\nData: `0x%x`\nGas: `%d`\nValue `%v` wei\n",
			from.String(), to.String(),
			input, gas, value)
	} else {
		fmt.Fprintf(t.out, "From: `%v`\nCreate at: `%v`\nData: `0x%x`\nGas: `%d`\nValue `%v` wei\n",
			from.String(), to.String(),
			input, gas, value)
	}

	fmt.Fprintf(t.out, `
|  Pc   |      Op     | Cost |   Stack   |   RStack  |
|-------|-------------|------|-----------|-----------|
`)
}

func (t *mdLogger) CaptureState(pc uint64, op OpCode, gas, cost uint64, scope *ScopeContext, rData []byte, depth int, err error) {
	fmt.Fprintf(t.out, "| %4d  | %10v  |  %3d |", pc, op, cost)

	if !t.cfg.DisableStack {
		// format stack
		var a []string
		for _, elem := range scope.Stack.data {
			a = append(a, elem.Hex())
		}
		b := fmt.Sprintf("[%v]", strings.Join(a, ","))
		fmt.Fprintf(t.out, "%10v |", b)

		// format return stack
	}
	fmt.Fprintln(t.out, "")
	if err != nil {
		fmt.Fprintf(t.out, "Error: %v\n", err)
	}
}

func (t *mdLogger) CaptureFault(env *CVM, pc uint64, op OpCode, gas, cost uint64, memory *Memory, stack *Stack, contract *Contract, depth int, err error) {

	fmt.Fprintf(t.out, "\nError: at pc=%d, op=%v: %v\n", pc, op, err)
}

func (t *mdLogger) CaptureEnd(output []byte, gasUsed uint64, tm time.Duration, err error) {
	fmt.Fprintf(t.out, `
Output: 0x%x
Consumed gas: %d
Error: %v
`,
		output, gasUsed, err)
}
