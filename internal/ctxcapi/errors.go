// Copyright 2024 The go-ethereum Authors
// This file is part of the go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package ctxcapi

import (
	"github.com/CortexFoundation/CortexTheseus/common"
)

type txSyncTimeoutError struct {
	msg  string
	hash common.Hash
}

const (
	errCodeNonceTooHigh            = -38011
	errCodeNonceTooLow             = -38010
	errCodeIntrinsicGas            = -38013
	errCodeInsufficientFunds       = -38014
	errCodeBlockGasLimitReached    = -38015
	errCodeBlockNumberInvalid      = -38020
	errCodeBlockTimestampInvalid   = -38021
	errCodeSenderIsNotEOA          = -38024
	errCodeMaxInitCodeSizeExceeded = -38025
	errCodeClientLimitExceeded     = -38026
	errCodeInternalError           = -32603
	errCodeInvalidParams           = -32602
	errCodeReverted                = -32000
	errCodeVMError                 = -32015
	errCodeTxSyncTimeout           = 4
)

func (e *txSyncTimeoutError) Error() string          { return e.msg }
func (e *txSyncTimeoutError) ErrorCode() int         { return errCodeTxSyncTimeout }
func (e *txSyncTimeoutError) ErrorData() interface{} { return e.hash.Hex() }
