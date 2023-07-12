// Copyright 2019 The CortexTheseus Authors
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

package snapshot

import (
	"bytes"
	"math/big"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/rlp"
)

// Account is a slim version of a state.Account, where the root and code hash
// are replaced with a nil byte slice for empty accounts.
type Account struct {
	Nonce    uint64
	Balance  *big.Int
	Root     []byte
	CodeHash []byte
	Upload   *big.Int //bytes
	Num      *big.Int
}

// AccountRLP converts a state.Account content into a slim snapshot version RLP
// encoded.
func SlimAccount(nonce uint64, balance *big.Int, root common.Hash, codehash []byte, upload *big.Int, num *big.Int) Account {
	slim := Account{
		Nonce:   nonce,
		Balance: balance,
		Upload:  upload,
		Num:     num,
	}
	if root != types.EmptyRootHash {
		slim.Root = root[:]
	}
	if !bytes.Equal(codehash, types.EmptyCodeHash[:]) {
		slim.CodeHash = codehash
	}
	return slim
}

// SlimAccountRLP converts a state.Account content into a slim snapshot
// version RLP encoded.
func SlimAccountRLP(nonce uint64, balance *big.Int, root common.Hash, codehash []byte, upload *big.Int, num *big.Int) []byte {
	data, err := rlp.EncodeToBytes(SlimAccount(nonce, balance, root, codehash, upload, num))
	if err != nil {
		panic(err)
	}
	return data
}

// FullAccount decodes the data on the 'slim RLP' format and return
// the consensus format account.
func FullAccount(data []byte) (Account, error) {
	var account Account
	if err := rlp.DecodeBytes(data, &account); err != nil {
		return Account{}, err
	}
	if len(account.Root) == 0 {
		account.Root = types.EmptyRootHash[:]
	}
	if len(account.CodeHash) == 0 {
		account.CodeHash = types.EmptyCodeHash[:]
	}
	return account, nil
}

// FullAccountRLP converts data on the 'slim RLP' format into the full RLP-format.
func FullAccountRLP(data []byte) ([]byte, error) {
	account, err := FullAccount(data)
	if err != nil {
		return nil, err
	}
	return rlp.EncodeToBytes(account)
}
