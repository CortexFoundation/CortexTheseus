// Copyright 2023 The CortexTheseus Authors
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

package state

import (
	"bytes"
	"encoding/binary"
	"math/big"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/crypto"
)

func (s *StateDB) GetUpload(addr common.Address) *big.Int {
	stateObject := s.getStateObject(addr)
	if stateObject != nil {
		return stateObject.Upload()
	}
	return common.Big0
}

func (s *StateDB) GetNum(addr common.Address) *big.Int {
	stateObject := s.getStateObject(addr)
	if stateObject != nil {
		return stateObject.Num()
	}
	return common.Big0
}

// GetState returns a value in account storage.
func (s *StateDB) GetSolidityBytes(addr common.Address, slot common.Hash) ([]byte, error) {
	length := s.GetState(addr, slot).Big().Uint64()
	if length == uint64(0) {
		return nil, nil
	}
	hashBig := new(big.Int).SetBytes(crypto.Keccak256(slot.Bytes()))
	//log.Warn(fmt.Sprintf("Pos %v, %v => %v, %v", addr, slot, length, hash))
	//log.Trace("solid", "addr", addr, "slot", slot, "length", length, "hash", hash, "x", s.GetState(addr, slot), "y", common.Hash{})
	// fmt.Println(fmt.Sprintf("Pos %v, %v => %v, %v", addr, slot, length, hash))

	//buffSize := length * 32

	buff := new(bytes.Buffer) //make([]byte, length * 32)
	for i := int64(0); i < int64(length); i++ {
		slotAddr := common.BigToHash(big.NewInt(0).Add(hashBig, big.NewInt(i)))
		payload := s.GetState(addr, slotAddr).Bytes()
		//copy(buff[idx*32:], payload[:])
		binary.Write(buff, binary.LittleEndian, payload[:])
		//binary.LittleEndian.Put(buff[idx*32:], payload[:])
		// fmt.Println(fmt.Sprintf("load[%v]: %x, %x => %x, %x", idx, addr, slotAddr, payload, hash))
	}
	// fmt.Println(fmt.Sprintf("data: %v", buff))
	return buff.Bytes(), nil
}

func (s *StateDB) Upload(addr common.Address) *big.Int {
	stateObject := s.getStateObject(addr)
	if stateObject != nil {
		return stateObject.Upload()
	}
	return nil
}

func (s *StateDB) SetUpload(addr common.Address, amount *big.Int) {
	stateObject := s.getOrNewStateObject(addr)
	if stateObject != nil {
		stateObject.SetUpload(amount)
	}
}

func (s *StateDB) SetNum(addr common.Address, num *big.Int) {
	stateObject := s.getOrNewStateObject(addr)
	if stateObject != nil {
		stateObject.SetNum(num)
	}
}

func (s *StateDB) Uploading(addr common.Address) bool {
	stateObject := s.getStateObject(addr)
	if stateObject != nil {
		return stateObject.Upload().Sign() > 0
	}
	return false
}

func (s *StateDB) SubUpload(addr common.Address, amount *big.Int) *big.Int {
	stateObject := s.getOrNewStateObject(addr)
	if stateObject != nil {
		return stateObject.SubUpload(amount)
	}
	return big0
}
