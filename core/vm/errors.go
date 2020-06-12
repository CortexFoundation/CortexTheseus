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
	"errors"
	"github.com/CortexFoundation/CortexTheseus/inference/synapse"
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
)
