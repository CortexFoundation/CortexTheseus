package cuckoo

import (
	"encoding/binary"
	"math/big"
	"math/rand"
	"testing"

	"github.com/CortexFoundation/CortexTheseus/core/types"
)

func TestReward(t *testing.T) {
	for i := int64(1); i < 10; i++ {
		reward := calculateRewardByNumber(big.NewInt(8409600*i-1), uint64(21))
		t.Log("Period:", i, "reward:", reward)
	}
}

func TestDifficultyCalculators(t *testing.T) {
	rand.Seed(2)
	diff := big.NewInt(10)
	for i := 0; i < 30; i++ {
		var timeDelta = uint64(18)
		if i > 5 {
			timeDelta = uint64(8)
		}

		if i > 10 {
			timeDelta = uint64(17)
		}

		if i > 15 {
			timeDelta = uint64(13)
		}

		if i > 20 {
			timeDelta = uint64(9)
		}

		if i > 25 {
			timeDelta = uint64(18)
		}

		header := &types.Header{
			Difficulty: diff,
			Number:     new(big.Int).SetUint64(rand.Uint64() % 50_000_000),
			Time:       rand.Uint64(),
		}
		header.UncleHash = types.EmptyUncleHash

		diff = calcDifficultyNeo(header.Time+timeDelta, header, true)
		t.Log("interval:", timeDelta, "diff:", diff)
	}
}

func randSlice(min, max uint32) []byte {
	var b = make([]byte, 4)
	rand.Read(b)
	a := binary.LittleEndian.Uint32(b)
	size := min + a%(max-min)
	out := make([]byte, size)
	rand.Read(out)
	return out
}
