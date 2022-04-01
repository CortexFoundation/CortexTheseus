package cuckoo

// go test ./consensus/cuckoo -v -cover

import (
	"fmt"
	"math/big"
	"math/rand"
	"testing"
	"time"

	"github.com/CortexFoundation/CortexTheseus/core/types"
)

func TestSha3Sol(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	cuckoo := New(Config{})
	// len is 42
	var result types.BlockSolution
	header := &types.Header{Difficulty: big.NewInt(66666)}
	hash := cuckoo.SealHash(header).Bytes()
	targetDiff := new(big.Int).Div(maxUint256, header.Difficulty)

	nonce := uint64(0)
	for {
		fmt.Println("nonce is:", nonce)
		r, sols := cuckoo.GenSha3Solution(hash, nonce)
		if r > 0 && len(sols) > 0 {
			copy(result[:], sols[0][0:len(sols[0])])
			res := cuckoo.CuckooVerifyHeader_SHA3(hash, nonce, &result, targetDiff)
			if res {
				fmt.Printf("nonce[%v] verify result(%v)\n", nonce, res)
				break
			}

		}
		nonce += 1
		//nonce = rand.Uint64()
	}
	//t.Fatal("Sha3Sol")
}
