package cuckoo

import (
	"math/big"
	"testing"

	"github.com/ethereum/go-ethereum/core/types"
)

func TestTestMode(t *testing.T) {
	head := &types.Header{
		Number:     big.NewInt(123),
		Difficulty: big.NewInt(1),
		Time:       big.NewInt(0),
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
	head.Nonce = types.EncodeNonce(block.Nonce())
	head.Solution = block.Solution()
	// t.Log(head.Solution)
	// head.Solution[0] = head.Solution[1]
	head.SolutionHash = block.Header().SolutionHash

	if err := cuckoo.VerifySeal(nil, head); err != nil {
		t.Fatalf("unexpected verification error: %v", err)
	}

}
