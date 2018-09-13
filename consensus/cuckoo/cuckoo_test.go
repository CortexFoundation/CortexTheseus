package cuckoo

import (
	"fmt"
	"math/big"
	"testing"

	"github.com/ethereum/go-ethereum/core/types"
)

func TestTestMode(t *testing.T) {

	cuckoo := NewTester()
	for i := 0; i < 1; i++ {
		head := &types.Header{
			Number:     big.NewInt(int64(i)),
			Difficulty: big.NewInt(2),
			Time:       big.NewInt(int64(i) * int64(i)),
		}
		target := new(big.Int).Div(maxUint256, head.Difficulty).Bytes()
		fmt.Println(target)
		// t.Fatal()
		// target := big.NewInt(0)
		// target.Exp(big.NewInt(2), big.NewInt(256), nil)
		// target.Sub(target, head.Difficulty)
		// target.Sub(target, big.NewInt(1))
		// head.Difficulty = target
		nb := types.NewBlockWithHeader(head)
		block, err := cuckoo.Seal(nil, nb, nil)
		if err != nil {
			t.Fatalf("failed to seal block: %v", err)
		}
		// t.Fatalf("1")
		// t.Fatal(block.Solution())
		// t.Fatal(block.Header().SolutionHash)
		// head.Nonce = types.EncodeNonce(block.Nonce())
		// head.Solution = block.Solution()
		// // t.Log(head.Solution)
		// // head.Solution[0] = head.Solution[1]
		// head.SolutionHash = block.Header().SolutionHash
		t.Log(block.Header())
		t.Fatal()
		if err := cuckoo.VerifySeal(nil, block.Header()); err != nil {
			t.Fatalf("unexpected verification error: %v", err)
		}
	}

}
