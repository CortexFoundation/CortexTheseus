// Copyright 2017 The CortexFoundation Authors
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

package asm

import (
	"encoding/hex"
	"fmt"
	"testing"
)

// Tests disassembling the instructions for valid cvm code
func TestInstructionIteratorValid(t *testing.T) {
	cnt := 0
	script, _ := hex.DecodeString("61000000")

	it := NewInstructionIterator(script)
	for it.Next() {
		cnt++
	}

	if err := it.Error(); err != nil {
		t.Errorf("Expected 2, but encountered error %v instead.", err)
	}
	if cnt != 2 {
		t.Errorf("Expected 2, but got %v instead.", cnt)
	}
}

// Tests disassembling the instructions for invalid cvm code
func TestInstructionIteratorInvalid(t *testing.T) {
	cnt := 0
	script, _ := hex.DecodeString("6100")

	it := NewInstructionIterator(script)
	for it.Next() {
		cnt++
	}

	if it.Error() == nil {
		t.Errorf("Expected an error, but got %v instead.", cnt)
	}
}

// Tests disassembling the instructions for empty cvm code
func TestInstructionIteratorEmpty(t *testing.T) {
	cnt := 0
	script, _ := hex.DecodeString("")

	it := NewInstructionIterator(script)
	for it.Next() {
		cnt++
	}

	if err := it.Error(); err != nil {
		t.Errorf("Expected 0, but encountered error %v instead.", err)
	}
	if cnt != 0 {
		t.Errorf("Expected 0, but got %v instead.", cnt)
	}
}

func TestHasInferOp(t *testing.T) {
	//cnt := 0
	fmt.Println("test ..")
	//script, _ := hex.DecodeString("608060c05260043610603f576000357c0100000000000000000000000000000000000000000000000000000000900463ffffffff1680630dc7f960146044575b600080fd5b348015604f57600080fd5b50606f600480360381019080803560ff1690602001909291905050506071565b005b806000806101000a81548160ff021916908360ff160217905550505600a165627a")
	//script, _ := hex.DecodeString("60086000612001611001c0")
	//script, _ := hex.DecodeString("60086000612001611001010060086000612001611001c0c0")
	script, _ := hex.DecodeString("60c0c0c0c0c0c")
	HasInferOp(script)
}
