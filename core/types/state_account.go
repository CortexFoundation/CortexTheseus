// Copyright 2021 The CortexTheseus Authors
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

package types

import (
	"math/big"

	"github.com/CortexFoundation/CortexTheseus/common"
)

//go:generate go run ../../rlp/rlpgen -type StateAccount -out gen_account_rlp.go

// StateAccount is the Cortex consensus representation of accounts.
// These objects are stored in the main account trie.
type StateAccount struct {
	Nonce    uint64
	Balance  *big.Int
	Root     common.Hash // merkle root of the storage trie
	CodeHash []byte
	Upload   *big.Int //bytes
	Num      *big.Int
}

// NewEmptyStateAccount constructs an empty state account.
func NewEmptyStateAccount() *StateAccount {
	return &StateAccount{
		Balance:  new(big.Int),
		Root:     EmptyRootHash,
		CodeHash: EmptyCodeHash.Bytes(),
		Upload:   new(big.Int),
		Num:      new(big.Int),
	}
}

// Copy returns a deep-copied state account object.
func (acct *StateAccount) Copy() *StateAccount {
	var (
		balance *big.Int
		upload  *big.Int
		num     *big.Int
	)

	if acct.Balance != nil {
		balance = new(big.Int).Set(acct.Balance)
	}
	if acct.Upload != nil {
		upload = new(big.Int).Set(acct.Upload)
	}
	if acct.Num != nil {
		num = new(big.Int).Set(acct.Num)
	}
	return &StateAccount{
		Nonce:    acct.Nonce,
		Balance:  balance,
		Root:     acct.Root,
		CodeHash: common.CopyBytes(acct.CodeHash),
		Upload:   upload,
		Num:      num,
	}
}
