// Copyright 2017 The go-ethereum Authors
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

package cuckoo

import (
	"encoding/binary"

	"github.com/PoolMiner/common"
	"github.com/PoolMiner/crypto"
)

func Sha3Solution(sol *common.BlockSolution) []byte {
	// maximum cycle length 42
	buf := make([]byte, 42*4)
	for i := 0; i < len(sol); i++ {
		binary.BigEndian.PutUint32(buf[i*4:], sol[i])
	}
	ret := crypto.Keccak256(buf)
	// fmt.Println("Sha3Solution: ", ret, "buf: ", buf, "sol: ", sol)
	return ret
}

func CuckooVerifyHeader(hash []byte, nonce uint64, sol *common.BlockSolution) (ok bool, sha3hash common.Hash) {
	r := CuckooVerifyProof(hash, uint64(nonce), &sol[0], common.CuckooCycleLength, common.CuckooEdgeBits)
	if r != 1 {
		return false, common.Hash{}
	}
	sha3 := common.BytesToHash(Sha3Solution(sol))
	return true, sha3
}
