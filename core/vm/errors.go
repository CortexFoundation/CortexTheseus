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
	"errors"
	"fmt"

	"github.com/CortexFoundation/inference/synapse"
)

// List execution errors
var (
	// ErrInvalidSubroutineEntry means that a BEGINSUB was reached via iteration,
	// as opposed to from a JUMPSUB instruction
	ErrInvalidSubroutineEntry   = errors.New("invalid subroutine entry")
	ErrOutOfGas                 = errors.New("out of gas")
	ErrCodeStoreOutOfGas        = errors.New("contract creation code storage out of gas")
	ErrDepth                    = errors.New("max call depth exceeded")
	ErrTraceLimitReached        = errors.New("the number of logs reached the specified limit")
	ErrInsufficientBalance      = errors.New("insufficient balance for transfer")
	ErrContractAddressCollision = errors.New("contract address collision")

	ErrInvalidMetaRawSize      = errors.New("invalid meta raw size")
	ErrNoCompatibleInterpreter = errors.New("no compatible interpreter")
	ErrInvalidMetaAuthor       = errors.New("invalid meta author")

	ErrGasUintOverflow     = errors.New("gas uint64 overflow")
	ErrInvalidJump         = errors.New("invalid jump destination")
	ErrWriteProtection     = errors.New("write protection")
	ErrInvalidRetsub       = errors.New("invalid retsub")
	ErrReturnStackExceeded = errors.New("return stack limit reached")

	ErrDownloading    = errors.New("downloading")
	ErrFileNotExist   = errors.New("file not exist")
	ErrInvalidTorrent = errors.New("invalid torrent")
	ErrInfer          = errors.New("infer error")

	ErrRuntime               = synapse.KERNEL_RUNTIME_ERROR
	ErrLogic                 = synapse.KERNEL_LOGIC_ERROR
	ErrReturnDataOutOfBounds = errors.New("return data out of bounds")
	ErrExecutionReverted     = errors.New("execution reverted")

	ErrNonceUintOverflow = errors.New("nonce uint64 overflow")

	errStopToken = errors.New("stop token")
)

// ErrStackUnderflow wraps an evm error when the items on the stack less
// than the minimal requirement.
type ErrStackUnderflow struct {
	stackLen int
	required int
}

func (e *ErrStackUnderflow) Error() string {
	return fmt.Sprintf("stack underflow (%d <=> %d)", e.stackLen, e.required)
}

// ErrStackOverflow wraps an evm error when the items on the stack exceeds
// the maximum allowance.
type ErrStackOverflow struct {
	stackLen int
	limit    int
}

func (e *ErrStackOverflow) Error() string {
	return fmt.Sprintf("stack limit reached %d (%d)", e.stackLen, e.limit)
}

// ErrInvalidOpCode wraps an evm error when an invalid opcode is encountered.
type ErrInvalidOpCode struct {
	opcode OpCode
}

func (e *ErrInvalidOpCode) Error() string { return fmt.Sprintf("invalid opcode: %s", e.opcode) }
