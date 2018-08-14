package cuckoo

import (
	"math/big"
	"testing"

	"github.com/ethereum/go-ethereum/core/types"
)

func TestTestMode(t *testing.T) {

	for i := 0; i < 5; i++ {
		head := &types.Header{
			Number:     big.NewInt(int64(i)),
			Difficulty: big.NewInt(int64(i)*123 + 1),
			Time:       big.NewInt(int64(i) * int64(i)),
		}
		// target := big.NewInt(0)
		// target.Exp(big.NewInt(2), big.NewInt(256), nil)
		// target.Sub(target, head.Difficulty)
		// target.Sub(target, big.NewInt(1))
		// head.Difficulty = target
		cuckoo := NewTester()
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

		if err := cuckoo.VerifySeal(nil, block.Header()); err != nil {
			t.Fatalf("unexpected verification error: %v", err)
		}

	}

}
