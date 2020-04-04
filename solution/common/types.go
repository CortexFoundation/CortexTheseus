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

package common

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"math/big"
)

// Lengths of hashes and addresses in bytes.
const (
	// HashLength is the expected length of the hash
	HashLength = 32
	// AddressLength is the expected length of the address
	AddressLength = 20
	Prefix        = "0x"
	Prefix_caps   = "0X"
)

const (
	CuckooCycleLength uint8 = 12
	CuckooEdgeBits    uint8 = 28
)

type BlockSolution [42]uint32

// Hash represents the 32 byte Keccak256 hash of arbitrary data.
type Hash [HashLength]byte

func Uint64ToHexString(value uint64) string {
	buf := make([]byte, 8)
	binary.BigEndian.PutUint64(buf, value)
	s := hex.EncodeToString(buf)
	return "0x" + s
}

func Uint32ArrayToHexString(value []uint32) string {
	//buf := make([]byte, len(value)*4)
	//for i := 0; i < len(value); i++ {
	//	binary.BigEndian.PutUint32(buf[i*4:], value[i])
	//}
	buf := new(bytes.Buffer)
	for _, v := range value {
		binary.Write(buf, binary.BigEndian, v)
	}
	return "0x" + hex.EncodeToString(buf.Bytes())
}

func Uint32ToHexString(value uint32) string {
	buf := make([]byte, 4)
	binary.BigEndian.PutUint32(buf, value)
	s := hex.EncodeToString(buf)
	for len(s) < 8 {
		s = s + "0"
	}
	return "0x" + s
}

// Hex2Bytes returns the bytes represented by the hexadecimal string str.
func Hex2Bytes(str string) []byte {
	h, _ := hex.DecodeString(str)
	return h
}

// FromHex returns the bytes represented by the hexadecimal string s.
// s may be prefixed with "0x".
func FromHex(s string) []byte {
	if len(s) > 1 {
		if s[0:2] == Prefix || s[0:2] == Prefix_caps {
			s = s[2:]
		}
	}
	if len(s)%2 == 1 {
		s = "0" + s
	}
	return Hex2Bytes(s)
}

// BytesToHash sets b to hash.
// If b is larger than len(h), b will be cropped from the left.
func BytesToHash(b []byte) Hash {
	var h Hash
	h.SetBytes(b)
	return h
}

// BigToHash sets byte representation of b to hash.
// If b is larger than len(h), b will be cropped from the left.
func BigToHash(b *big.Int) Hash { return BytesToHash(b.Bytes()) }

// HexToHash sets byte representation of s to hash.
// If b is larger than len(h), b will be cropped from the left.
func HexToHash(s string) Hash { return BytesToHash(FromHex(s)) }

// SetBytes sets the hash to the value of b.
// If b is larger than len(h), b will be cropped from the left.
func (h *Hash) SetBytes(b []byte) {
	if len(b) > len(h) {
		b = b[len(b)-HashLength:]
	}

	copy(h[HashLength-len(b):], b)
}

// Bytes gets the byte representation of the underlying hash.
func (h Hash) Bytes() []byte { return h[:] }

// Big converts a hash to a big integer.
func (h Hash) Big() *big.Int { return new(big.Int).SetBytes(h[:]) }
