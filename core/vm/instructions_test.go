// Copyright 2017 The go-ethereum Authors
// This file is part of The go-ethereum library.
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
// along with The go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package vm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/crypto"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/holiman/uint256"
)

type TwoOperandTestcase struct {
	X        string
	Y        string
	Expected string
}

type twoOperandParams struct {
	x string
	y string
}

var commonParams []*twoOperandParams
var twoOpMethods map[string]executionFunc

func init() {

	// Params is a list of common edgecases that should be used for some common tests
	params := []string{
		"0000000000000000000000000000000000000000000000000000000000000000", // 0
		"0000000000000000000000000000000000000000000000000000000000000001", // +1
		"0000000000000000000000000000000000000000000000000000000000000005", // +5
		"7ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe", // + max -1
		"7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", // + max
		"8000000000000000000000000000000000000000000000000000000000000000", // - max
		"8000000000000000000000000000000000000000000000000000000000000001", // - max+1
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffb", // - 5
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", // - 1
	}
	// Params are combined so each param is used on each 'side'
	commonParams = make([]*twoOperandParams, len(params)*len(params))
	for i, x := range params {
		for j, y := range params {
			commonParams[i*len(params)+j] = &twoOperandParams{x, y}
		}
	}
	twoOpMethods = map[string]executionFunc{
		"add":     opAdd,
		"sub":     opSub,
		"mul":     opMul,
		"div":     opDiv,
		"sdiv":    opSdiv,
		"mod":     opMod,
		"smod":    opSmod,
		"exp":     opExp,
		"signext": opSignExtend,
		"lt":      opLt,
		"gt":      opGt,
		"slt":     opSlt,
		"sgt":     opSgt,
		"eq":      opEq,
		"and":     opAnd,
		"or":      opOr,
		"xor":     opXor,
		"byte":    opByte,
		"shl":     opSHL,
		"shr":     opSHR,
		"sar":     opSAR,
	}
}

func testTwoOperandOp(t *testing.T, tests []TwoOperandTestcase, opFn executionFunc, name string) {

	var (
		env            = NewCVM(BlockContext{}, nil, params.TestChainConfig, Config{})
		stack          = newstack()
		pc             = uint64(0)
		cvmInterpreter = env.interpreter
	)

	for i, test := range tests {
		x := new(uint256.Int).SetBytes(common.Hex2Bytes(test.X))
		y := new(uint256.Int).SetBytes(common.Hex2Bytes(test.Y))
		expected := new(uint256.Int).SetBytes(common.Hex2Bytes(test.Expected))
		stack.push(x)
		stack.push(y)
		opFn(&pc, cvmInterpreter, &ScopeContext{nil, stack, nil})
		if len(stack.data) != 1 {
			t.Errorf("Expected one item on stack after %v, got %d: ", name, len(stack.data))
		}
		actual := stack.pop()

		if actual.Cmp(expected) != 0 {
			t.Errorf("Testcase %v %d, %v(%x, %x): expected  %x, got %x", name, i, name, x, y, expected, actual)
		}
	}
}

func TestByteOp(t *testing.T) {
	tests := []TwoOperandTestcase{
		{"ABCDEF0908070605040302010000000000000000000000000000000000000000", "00", "AB"},
		{"ABCDEF0908070605040302010000000000000000000000000000000000000000", "01", "CD"},
		{"00CDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff", "00", "00"},
		{"00CDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff", "01", "CD"},
		{"0000000000000000000000000000000000000000000000000000000000102030", "1F", "30"},
		{"0000000000000000000000000000000000000000000000000000000000102030", "1E", "20"},
		{"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "20", "00"},
		{"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "FFFFFFFFFFFFFFFF", "00"},
	}
	testTwoOperandOp(t, tests, opByte, "byte")
}

func TestSHL(t *testing.T) {
	// Testcases from https://github.com/cortex/EIPs/blob/master/EIPS/eip-145.md#shl-shift-left
	tests := []TwoOperandTestcase{
		{"0000000000000000000000000000000000000000000000000000000000000001", "01", "0000000000000000000000000000000000000000000000000000000000000002"},
		{"0000000000000000000000000000000000000000000000000000000000000001", "ff", "8000000000000000000000000000000000000000000000000000000000000000"},
		{"0000000000000000000000000000000000000000000000000000000000000001", "0100", "0000000000000000000000000000000000000000000000000000000000000000"},
		{"0000000000000000000000000000000000000000000000000000000000000001", "0101", "0000000000000000000000000000000000000000000000000000000000000000"},
		{"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "00", "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"},
		{"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "01", "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe"},
		{"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "ff", "8000000000000000000000000000000000000000000000000000000000000000"},
		{"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "0100", "0000000000000000000000000000000000000000000000000000000000000000"},
		{"0000000000000000000000000000000000000000000000000000000000000000", "01", "0000000000000000000000000000000000000000000000000000000000000000"},
		{"7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "01", "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe"},
	}
	testTwoOperandOp(t, tests, opSHL, "shl")
}

func TestSHR(t *testing.T) {
	// Testcases from https://github.com/cortex/EIPs/blob/master/EIPS/eip-145.md#shr-logical-shift-right
	tests := []TwoOperandTestcase{
		{"0000000000000000000000000000000000000000000000000000000000000001", "00", "0000000000000000000000000000000000000000000000000000000000000001"},
		{"0000000000000000000000000000000000000000000000000000000000000001", "01", "0000000000000000000000000000000000000000000000000000000000000000"},
		{"8000000000000000000000000000000000000000000000000000000000000000", "01", "4000000000000000000000000000000000000000000000000000000000000000"},
		{"8000000000000000000000000000000000000000000000000000000000000000", "ff", "0000000000000000000000000000000000000000000000000000000000000001"},
		{"8000000000000000000000000000000000000000000000000000000000000000", "0100", "0000000000000000000000000000000000000000000000000000000000000000"},
		{"8000000000000000000000000000000000000000000000000000000000000000", "0101", "0000000000000000000000000000000000000000000000000000000000000000"},
		{"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "00", "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"},
		{"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "01", "7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"},
		{"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "ff", "0000000000000000000000000000000000000000000000000000000000000001"},
		{"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "0100", "0000000000000000000000000000000000000000000000000000000000000000"},
		{"0000000000000000000000000000000000000000000000000000000000000000", "01", "0000000000000000000000000000000000000000000000000000000000000000"},
	}
	testTwoOperandOp(t, tests, opSHR, "shr")
}

func TestSAR(t *testing.T) {
	// Testcases from https://github.com/cortex/EIPs/blob/master/EIPS/eip-145.md#sar-arithmetic-shift-right
	tests := []TwoOperandTestcase{
		{"0000000000000000000000000000000000000000000000000000000000000001", "00", "0000000000000000000000000000000000000000000000000000000000000001"},
		{"0000000000000000000000000000000000000000000000000000000000000001", "01", "0000000000000000000000000000000000000000000000000000000000000000"},
		{"8000000000000000000000000000000000000000000000000000000000000000", "01", "c000000000000000000000000000000000000000000000000000000000000000"},
		{"8000000000000000000000000000000000000000000000000000000000000000", "ff", "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"},
		{"8000000000000000000000000000000000000000000000000000000000000000", "0100", "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"},
		{"8000000000000000000000000000000000000000000000000000000000000000", "0101", "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"},
		{"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "00", "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"},
		{"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "01", "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"},
		{"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "ff", "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"},
		{"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "0100", "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"},
		{"0000000000000000000000000000000000000000000000000000000000000000", "01", "0000000000000000000000000000000000000000000000000000000000000000"},
		{"4000000000000000000000000000000000000000000000000000000000000000", "fe", "0000000000000000000000000000000000000000000000000000000000000001"},
		{"7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "f8", "000000000000000000000000000000000000000000000000000000000000007f"},
		{"7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "fe", "0000000000000000000000000000000000000000000000000000000000000001"},
		{"7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "ff", "0000000000000000000000000000000000000000000000000000000000000000"},
		{"7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "0100", "0000000000000000000000000000000000000000000000000000000000000000"},
	}

	testTwoOperandOp(t, tests, opSAR, "sar")
}

func TestAddMod(t *testing.T) {
	var (
		env            = NewCVM(BlockContext{}, nil, params.TestChainConfig, Config{})
		stack          = newstack()
		cvmInterpreter = NewCVMInterpreter(env)
		pc             = uint64(0)
	)
	tests := []struct {
		x        string
		y        string
		z        string
		expected string
	}{
		{"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
			"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe",
			"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
			"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe",
		},
	}
	// x + y = 0x1fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffd
	// in 256 bit repr, fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffd

	for i, test := range tests {
		x := new(uint256.Int).SetBytes(common.Hex2Bytes(test.x))
		y := new(uint256.Int).SetBytes(common.Hex2Bytes(test.y))
		z := new(uint256.Int).SetBytes(common.Hex2Bytes(test.z))
		expected := new(uint256.Int).SetBytes(common.Hex2Bytes(test.expected))
		stack.push(z)
		stack.push(y)
		stack.push(x)
		opAddmod(&pc, cvmInterpreter, &ScopeContext{nil, stack, nil})
		actual := stack.pop()
		if actual.Cmp(expected) != 0 {
			t.Errorf("Testcase %d, expected  %x, got %x", i, expected, actual)
		}
	}
}

// getResult is a convenience function to generate the expected values
func getResult(args []*twoOperandParams, opFn executionFunc) []TwoOperandTestcase {
	var (
		env         = NewCVM(BlockContext{}, nil, params.TestChainConfig, Config{})
		stack       = newstack()
		pc          = uint64(0)
		interpreter = env.interpreter
	)
	result := make([]TwoOperandTestcase, len(args))
	for i, param := range args {
		x := new(uint256.Int).SetBytes(common.Hex2Bytes(param.x))
		y := new(uint256.Int).SetBytes(common.Hex2Bytes(param.y))
		stack.push(x)
		stack.push(y)
		opFn(&pc, interpreter, &ScopeContext{nil, stack, nil})
		actual := stack.pop()
		result[i] = TwoOperandTestcase{param.x, param.y, fmt.Sprintf("%064x", actual)}
	}
	return result
}

// utility function to fill the json-file with testcases
// Enable this test to generate the 'testcases_xx.json' files
func TestWriteExpectedValues(t *testing.T) {
	t.Skip("Enable this test to create json test cases.")

	for name, method := range twoOpMethods {
		data, err := json.Marshal(getResult(commonParams, method))
		if err != nil {
			t.Fatal(err)
		}
		_ = os.WriteFile(fmt.Sprintf("testdata/testcases_%v.json", name), data, 0644)
		if err != nil {
			t.Fatal(err)
		}
	}
}

// TestJsonTestcases runs through all the testcases defined as json-files
func TestJsonTestcases(t *testing.T) {
	for name := range twoOpMethods {
		data, err := os.ReadFile(fmt.Sprintf("testdata/testcases_%v.json", name))
		if err != nil {
			t.Fatal("Failed to read file", err)
		}
		var testcases []TwoOperandTestcase
		json.Unmarshal(data, &testcases)
		testTwoOperandOp(t, testcases, twoOpMethods[name], name)
	}
}

func opBenchmark(bench *testing.B, op executionFunc, args ...string) {
	var (
		env            = NewCVM(BlockContext{}, nil, params.TestChainConfig, Config{})
		stack          = newstack()
		cvmInterpreter = NewCVMInterpreter(env)
	)

	env.interpreter = cvmInterpreter
	// convert args
	byteArgs := make([][]byte, len(args))
	for i, arg := range args {
		byteArgs[i] = common.Hex2Bytes(arg)
	}
	pc := uint64(0)
	bench.ResetTimer()
	for i := 0; i < bench.N; i++ {
		for _, arg := range byteArgs {
			a := new(uint256.Int)
			a.SetBytes(arg)
			stack.push(a)
		}
		op(&pc, cvmInterpreter, &ScopeContext{nil, stack, nil})
		stack.pop()
	}
}

func BenchmarkOpAdd64(b *testing.B) {
	x := "ffffffff"
	y := "fd37f3e2bba2c4f"

	opBenchmark(b, opAdd, x, y)
}

func BenchmarkOpAdd128(b *testing.B) {
	x := "ffffffffffffffff"
	y := "f5470b43c6549b016288e9a65629687"

	opBenchmark(b, opAdd, x, y)
}

func BenchmarkOpAdd256(b *testing.B) {
	x := "0802431afcbce1fc194c9eaa417b2fb67dc75a95db0bc7ec6b1c8af11df6a1da9"
	y := "a1f5aac137876480252e5dcac62c354ec0d42b76b0642b6181ed099849ea1d57"

	opBenchmark(b, opAdd, x, y)
}

func BenchmarkOpSub64(b *testing.B) {
	x := "51022b6317003a9d"
	y := "a20456c62e00753a"

	opBenchmark(b, opSub, x, y)
}

func BenchmarkOpSub128(b *testing.B) {
	x := "4dde30faaacdc14d00327aac314e915d"
	y := "9bbc61f5559b829a0064f558629d22ba"

	opBenchmark(b, opSub, x, y)
}

func BenchmarkOpSub256(b *testing.B) {
	x := "4bfcd8bb2ac462735b48a17580690283980aa2d679f091c64364594df113ea37"
	y := "97f9b1765588c4e6b69142eb00d20507301545acf3e1238c86c8b29be227d46e"

	opBenchmark(b, opSub, x, y)
}

func BenchmarkOpMul(b *testing.B) {
	x := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"
	y := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"

	opBenchmark(b, opMul, x, y)
}

func BenchmarkOpDiv256(b *testing.B) {
	x := "ff3f9014f20db29ae04af2c2d265de17"
	y := "fe7fb0d1f59dfe9492ffbf73683fd1e870eec79504c60144cc7f5fc2bad1e611"
	opBenchmark(b, opDiv, x, y)
}

func BenchmarkOpDiv128(b *testing.B) {
	x := "fdedc7f10142ff97"
	y := "fbdfda0e2ce356173d1993d5f70a2b11"
	opBenchmark(b, opDiv, x, y)
}

func BenchmarkOpDiv64(b *testing.B) {
	x := "fcb34eb3"
	y := "f97180878e839129"
	opBenchmark(b, opDiv, x, y)
}

func BenchmarkOpSdiv(b *testing.B) {
	x := "ff3f9014f20db29ae04af2c2d265de17"
	y := "fe7fb0d1f59dfe9492ffbf73683fd1e870eec79504c60144cc7f5fc2bad1e611"

	opBenchmark(b, opSdiv, x, y)
}

func BenchmarkOpMod(b *testing.B) {
	x := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"
	y := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"

	opBenchmark(b, opMod, x, y)
}

func BenchmarkOpSmod(b *testing.B) {
	x := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"
	y := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"

	opBenchmark(b, opSmod, x, y)
}

func BenchmarkOpExp(b *testing.B) {
	x := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"
	y := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"

	opBenchmark(b, opExp, x, y)
}

func BenchmarkOpSignExtend(b *testing.B) {
	x := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"
	y := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"

	opBenchmark(b, opSignExtend, x, y)
}

func BenchmarkOpLt(b *testing.B) {
	x := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"
	y := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"

	opBenchmark(b, opLt, x, y)
}

func BenchmarkOpGt(b *testing.B) {
	x := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"
	y := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"

	opBenchmark(b, opGt, x, y)
}

func BenchmarkOpSlt(b *testing.B) {
	x := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"
	y := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"

	opBenchmark(b, opSlt, x, y)
}

func BenchmarkOpSgt(b *testing.B) {
	x := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"
	y := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"

	opBenchmark(b, opSgt, x, y)
}

func BenchmarkOpEq(b *testing.B) {
	x := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"
	y := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"

	opBenchmark(b, opEq, x, y)
}
func BenchmarkOpEq2(b *testing.B) {
	x := "FBCDEF090807060504030201ffffffffFBCDEF090807060504030201ffffffff"
	y := "FBCDEF090807060504030201ffffffffFBCDEF090807060504030201fffffffe"
	opBenchmark(b, opEq, x, y)
}
func BenchmarkOpAnd(b *testing.B) {
	x := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"
	y := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"

	opBenchmark(b, opAnd, x, y)
}

func BenchmarkOpOr(b *testing.B) {
	x := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"
	y := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"

	opBenchmark(b, opOr, x, y)
}

func BenchmarkOpXor(b *testing.B) {
	x := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"
	y := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"

	opBenchmark(b, opXor, x, y)
}

func BenchmarkOpByte(b *testing.B) {
	x := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"
	y := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"

	opBenchmark(b, opByte, x, y)
}

func BenchmarkOpAddmod(b *testing.B) {
	x := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"
	y := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"
	z := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"

	opBenchmark(b, opAddmod, x, y, z)
}

func BenchmarkOpMulmod(b *testing.B) {
	x := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"
	y := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"
	z := "ABCDEF090807060504030201ffffffffffffffffffffffffffffffffffffffff"

	opBenchmark(b, opMulmod, x, y, z)
}

func BenchmarkOpSHL(b *testing.B) {
	x := "FBCDEF090807060504030201ffffffffFBCDEF090807060504030201ffffffff"
	y := "ff"

	opBenchmark(b, opSHL, x, y)
}
func BenchmarkOpSHR(b *testing.B) {
	x := "FBCDEF090807060504030201ffffffffFBCDEF090807060504030201ffffffff"
	y := "ff"

	opBenchmark(b, opSHR, x, y)
}
func BenchmarkOpSAR(b *testing.B) {
	x := "FBCDEF090807060504030201ffffffffFBCDEF090807060504030201ffffffff"
	y := "ff"

	opBenchmark(b, opSAR, x, y)
}
func BenchmarkOpIsZero(b *testing.B) {
	x := "FBCDEF090807060504030201ffffffffFBCDEF090807060504030201ffffffff"
	opBenchmark(b, opIszero, x)
}

func TestOpMstore(t *testing.T) {
	var (
		env            = NewCVM(BlockContext{}, nil, params.TestChainConfig, Config{})
		stack          = newstack()
		mem            = NewMemory()
		cvmInterpreter = NewCVMInterpreter(env)
	)

	env.interpreter = cvmInterpreter
	mem.Resize(64)
	pc := uint64(0)
	v := "abcdef00000000000000abba000000000deaf000000c0de00100000000133700"
	stack.pushN(*new(uint256.Int).SetBytes(common.Hex2Bytes(v)), *new(uint256.Int))
	opMstore(&pc, cvmInterpreter, &ScopeContext{mem, stack, nil})
	if got := common.Bytes2Hex(mem.GetCopy(0, 32)); got != v {
		t.Fatalf("Mstore fail, got %v, expected %v", got, v)
	}
	stack.pushN(*new(uint256.Int).SetUint64(0x1), *new(uint256.Int))
	opMstore(&pc, cvmInterpreter, &ScopeContext{mem, stack, nil})
	if common.Bytes2Hex(mem.GetCopy(0, 32)) != "0000000000000000000000000000000000000000000000000000000000000001" {
		t.Fatalf("Mstore failed to overwrite previous value")
	}
}

func BenchmarkOpMstore(bench *testing.B) {
	var (
		env            = NewCVM(BlockContext{}, nil, params.TestChainConfig, Config{})
		stack          = newstack()
		mem            = NewMemory()
		cvmInterpreter = NewCVMInterpreter(env)
	)

	env.interpreter = cvmInterpreter
	mem.Resize(64)
	pc := uint64(0)
	memStart := new(uint256.Int)
	value := new(uint256.Int).SetUint64(0x1337)

	bench.ResetTimer()
	for i := 0; i < bench.N; i++ {
		stack.pushN(*value, *memStart)
		opMstore(&pc, cvmInterpreter, &ScopeContext{mem, stack, nil})
	}
}

func BenchmarkOpSHA3(bench *testing.B) {
	var (
		env            = NewCVM(BlockContext{}, nil, params.TestChainConfig, Config{})
		stack          = newstack()
		mem            = NewMemory()
		cvmInterpreter = NewCVMInterpreter(env)
	)
	env.interpreter = cvmInterpreter
	mem.Resize(32)
	pc := uint64(0)
	start := new(uint256.Int)

	bench.ResetTimer()
	for i := 0; i < bench.N; i++ {
		stack.pushN(*uint256.NewInt(32), *start)
		opKeccak256(&pc, cvmInterpreter, &ScopeContext{mem, stack, nil})
	}
}

func TestCreate2Addreses(t *testing.T) {
	type testcase struct {
		origin   string
		salt     string
		code     string
		expected string
	}

	for i, tt := range []testcase{
		{
			origin:   "0x0000000000000000000000000000000000000000",
			salt:     "0x0000000000000000000000000000000000000000",
			code:     "0x00",
			expected: "0x4d1a2e2bb4f88f0250f26ffff098b0b30b26bf38",
		},
		{
			origin:   "0xdeadbeef00000000000000000000000000000000",
			salt:     "0x0000000000000000000000000000000000000000",
			code:     "0x00",
			expected: "0xB928f69Bb1D91Cd65274e3c79d8986362984fDA3",
		},
		{
			origin:   "0xdeadbeef00000000000000000000000000000000",
			salt:     "0xfeed000000000000000000000000000000000000",
			code:     "0x00",
			expected: "0xD04116cDd17beBE565EB2422F2497E06cC1C9833",
		},
		{
			origin:   "0x0000000000000000000000000000000000000000",
			salt:     "0x0000000000000000000000000000000000000000",
			code:     "0xdeadbeef",
			expected: "0x70f2b2914A2a4b783FaEFb75f459A580616Fcb5e",
		},
		{
			origin:   "0x00000000000000000000000000000000deadbeef",
			salt:     "0xcafebabe",
			code:     "0xdeadbeef",
			expected: "0x60f3f640a8508fC6a86d45DF051962668E1e8AC7",
		},
		{
			origin:   "0x00000000000000000000000000000000deadbeef",
			salt:     "0xcafebabe",
			code:     "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
			expected: "0x1d8bfDC5D46DC4f61D6b6115972536eBE6A8854C",
		},
		{
			origin:   "0x0000000000000000000000000000000000000000",
			salt:     "0x0000000000000000000000000000000000000000",
			code:     "0x",
			expected: "0xE33C0C7F7df4809055C3ebA6c09CFe4BaF1BD9e0",
		},
	} {

		origin := common.BytesToAddress(common.FromHex(tt.origin))
		salt := common.BytesToHash(common.FromHex(tt.salt))
		code := common.FromHex(tt.code)
		codeHash := crypto.Keccak256(code)
		address := crypto.CreateAddress2(origin, salt, codeHash)
		/*
			stack          := newstack()
			// salt, but we don't need that for this test
			stack.push(big.NewInt(int64(len(code)))) //size
			stack.push(big.NewInt(0)) // memstart
			stack.push(big.NewInt(0)) // value
			gas, _ := gasCreate2(params.GasTable{}, nil, nil, stack, nil, 0)
			fmt.Printf("Example %d\n* address `0x%x`\n* salt `0x%x`\n* init_code `0x%x`\n* gas (assuming no mem expansion): `%v`\n* result: `%s`\n\n", i,origin, salt, code, gas, address.String())
		*/
		expected := common.BytesToAddress(common.FromHex(tt.expected))
		if !bytes.Equal(expected.Bytes(), address.Bytes()) {
			t.Errorf("test %d: expected %s, got %s", i, expected.String(), address.String())
		}
	}
}

// TestPush sanity-checks how code with immediates are handled when the code size is
// smaller than the size of the immediate.
func TestPush(t *testing.T) {
	code := common.FromHex("0011223344556677889900aabbccddeeff0102030405060708090a0b0c0d0e0ff1e1d1c1b1a19181716151413121")

	push32 := makePush(32, 32)

	scope := &ScopeContext{
		Memory: nil,
		Stack:  newstack(),
		Contract: &Contract{
			Code: code,
		},
	}
	for i, want := range []string{
		"0x11223344556677889900aabbccddeeff0102030405060708090a0b0c0d0e0ff1",
		"0x223344556677889900aabbccddeeff0102030405060708090a0b0c0d0e0ff1e1",
		"0x3344556677889900aabbccddeeff0102030405060708090a0b0c0d0e0ff1e1d1",
		"0x44556677889900aabbccddeeff0102030405060708090a0b0c0d0e0ff1e1d1c1",
		"0x556677889900aabbccddeeff0102030405060708090a0b0c0d0e0ff1e1d1c1b1",
		"0x6677889900aabbccddeeff0102030405060708090a0b0c0d0e0ff1e1d1c1b1a1",
		"0x77889900aabbccddeeff0102030405060708090a0b0c0d0e0ff1e1d1c1b1a191",
		"0x889900aabbccddeeff0102030405060708090a0b0c0d0e0ff1e1d1c1b1a19181",
		"0x9900aabbccddeeff0102030405060708090a0b0c0d0e0ff1e1d1c1b1a1918171",
		"0xaabbccddeeff0102030405060708090a0b0c0d0e0ff1e1d1c1b1a191817161",
		"0xaabbccddeeff0102030405060708090a0b0c0d0e0ff1e1d1c1b1a19181716151",
		"0xbbccddeeff0102030405060708090a0b0c0d0e0ff1e1d1c1b1a1918171615141",
		"0xccddeeff0102030405060708090a0b0c0d0e0ff1e1d1c1b1a191817161514131",
		"0xddeeff0102030405060708090a0b0c0d0e0ff1e1d1c1b1a19181716151413121",
		"0xeeff0102030405060708090a0b0c0d0e0ff1e1d1c1b1a1918171615141312100",
		"0xff0102030405060708090a0b0c0d0e0ff1e1d1c1b1a191817161514131210000",
		"0x102030405060708090a0b0c0d0e0ff1e1d1c1b1a19181716151413121000000",
		"0x2030405060708090a0b0c0d0e0ff1e1d1c1b1a1918171615141312100000000",
		"0x30405060708090a0b0c0d0e0ff1e1d1c1b1a191817161514131210000000000",
		"0x405060708090a0b0c0d0e0ff1e1d1c1b1a19181716151413121000000000000",
		"0x5060708090a0b0c0d0e0ff1e1d1c1b1a1918171615141312100000000000000",
		"0x60708090a0b0c0d0e0ff1e1d1c1b1a191817161514131210000000000000000",
		"0x708090a0b0c0d0e0ff1e1d1c1b1a19181716151413121000000000000000000",
		"0x8090a0b0c0d0e0ff1e1d1c1b1a1918171615141312100000000000000000000",
		"0x90a0b0c0d0e0ff1e1d1c1b1a191817161514131210000000000000000000000",
		"0xa0b0c0d0e0ff1e1d1c1b1a19181716151413121000000000000000000000000",
		"0xb0c0d0e0ff1e1d1c1b1a1918171615141312100000000000000000000000000",
		"0xc0d0e0ff1e1d1c1b1a191817161514131210000000000000000000000000000",
		"0xd0e0ff1e1d1c1b1a19181716151413121000000000000000000000000000000",
		"0xe0ff1e1d1c1b1a1918171615141312100000000000000000000000000000000",
		"0xff1e1d1c1b1a191817161514131210000000000000000000000000000000000",
		"0xf1e1d1c1b1a19181716151413121000000000000000000000000000000000000",
		"0xe1d1c1b1a1918171615141312100000000000000000000000000000000000000",
		"0xd1c1b1a191817161514131210000000000000000000000000000000000000000",
		"0xc1b1a19181716151413121000000000000000000000000000000000000000000",
		"0xb1a1918171615141312100000000000000000000000000000000000000000000",
		"0xa191817161514131210000000000000000000000000000000000000000000000",
		"0x9181716151413121000000000000000000000000000000000000000000000000",
		"0x8171615141312100000000000000000000000000000000000000000000000000",
		"0x7161514131210000000000000000000000000000000000000000000000000000",
		"0x6151413121000000000000000000000000000000000000000000000000000000",
		"0x5141312100000000000000000000000000000000000000000000000000000000",
		"0x4131210000000000000000000000000000000000000000000000000000000000",
		"0x3121000000000000000000000000000000000000000000000000000000000000",
		"0x2100000000000000000000000000000000000000000000000000000000000000",
		"0x0",
	} {
		pc := new(uint64)
		*pc = uint64(i)
		push32(pc, nil, scope)
		res := scope.Stack.pop()
		if have := res.Hex(); have != want {
			t.Fatalf("case %d, have %v want %v", i, have, want)
		}
	}
}

func TestOpCLZ(t *testing.T) {
	evm := NewCVM(BlockContext{}, nil, params.TestChainConfig, Config{})

	tests := []struct {
		inputHex string
		want     uint64 // expected CLZ result
	}{
		{"0x0", 256},
		{"0x1", 255},
		{"0x6ff", 245},        // 0x6ff = 0b11011111111 (11 bits), so 256-11 = 245
		{"0xffffffffff", 216}, // 40 bits, so 256-40 = 216
		{"0x4000000000000000000000000000000000000000000000000000000000000000", 1},
		{"0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", 1},
		{"0x8000000000000000000000000000000000000000000000000000000000000000", 0},
		{"0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", 0},
	}
	for _, tc := range tests {
		// prepare a fresh stack and PC
		stack := newstack()
		pc := uint64(0)

		// parse input
		val := new(uint256.Int)
		if err := val.SetFromHex(tc.inputHex); err != nil {
			t.Fatal("invalid hex uint256:", tc.inputHex)
		}

		stack.push(val)
		opCLZ(&pc, evm.interpreter, &ScopeContext{Stack: stack})

		if gotLen := stack.len(); gotLen != 1 {
			t.Fatalf("stack length = %d; want 1", gotLen)
		}
		result := stack.pop()
		if got := result.Uint64(); got != tc.want {
			t.Fatalf("clz(%q) = %d; want %d", tc.inputHex, got, tc.want)
		}
	}
}
