package vm

import (
	"math/big"

	"github.com/ethereum/go-ethereum/ethutil"
)

func analyseJumpDests(code []byte) (dests map[uint64]*big.Int) {
	dests = make(map[uint64]*big.Int)

	lp := false
	var lpv *big.Int
	for pc := uint64(0); pc < uint64(len(code)); pc++ {
		var op OpCode = OpCode(code[pc])
		switch op {
		case PUSH1, PUSH2, PUSH3, PUSH4, PUSH5, PUSH6, PUSH7, PUSH8, PUSH9, PUSH10, PUSH11, PUSH12, PUSH13, PUSH14, PUSH15, PUSH16, PUSH17, PUSH18, PUSH19, PUSH20, PUSH21, PUSH22, PUSH23, PUSH24, PUSH25, PUSH26, PUSH27, PUSH28, PUSH29, PUSH30, PUSH31, PUSH32:
			a := uint64(op) - uint64(PUSH1) + 1
			if uint64(len(code)) > pc+1+a {
				lpv = ethutil.BigD(code[pc+1 : pc+1+a])
			}

			pc += a
			lp = true
		case JUMP, JUMPI:
			if lp {
				dests[pc] = lpv
			}

		default:
			lp = false
		}
	}
	return
}
