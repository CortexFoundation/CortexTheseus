package cuckoo

import (
	"fmt"
	"math/big"
	"testing"

	"github.com/ethereum/go-ethereum/core/types"
)

func TestTestMode(t *testing.T) {
	head := &types.Header{Number: big.NewInt(1), Difficulty: big.NewInt(100)}

	cuckoo := NewTester()
	t.Log("HEAD INFO: ", head)
	block, err := cuckoo.Seal(nil, types.NewBlockWithHeader(head), nil)
	if err != nil {
		t.Fatalf("failed to seal block: %v", err)
	}

	fmt.Println("HEAD INFO: ", head)
	head.Nonce = types.EncodeNonce(block.Nonce())
	head.Solution = block.Solution()

	if err := cuckoo.VerifySeal(nil, head); err != nil {
		t.Fatalf("unexpected verification error: %v", err)
	}

	t.Log("HEAD INFO: ", head)
	t.Logf("HEAD INFO: %v", head)
}
