package cuckoo

import (
	"encoding/binary"
	"math/big"
	"math/rand"
	"testing"

	"github.com/CortexFoundation/CortexTheseus/core/types"
)

func TestReward(t *testing.T) {
	reward := calculateRewardByNumber(big.NewInt(8409600-1), uint64(21))
	t.Log(reward)
	reward = calculateRewardByNumber(big.NewInt(8409600), uint64(21))
	t.Log(reward)
	reward = calculateRewardByNumber(big.NewInt(8409600*2), uint64(21))
	t.Log(reward)
	reward = calculateRewardByNumber(big.NewInt(8409600*3), uint64(21))
	t.Log(reward)
	reward = calculateRewardByNumber(big.NewInt(8409600*4), uint64(21))
	t.Log(reward)
	reward = calculateRewardByNumber(big.NewInt(8409600*5), uint64(21))
	t.Log(reward)
	reward = calculateRewardByNumber(big.NewInt(8409600*6), uint64(21))
	t.Log(reward)
	reward = calculateRewardByNumber(big.NewInt(8409600*7), uint64(21))
	t.Log(reward)
}

func TestDifficultyCalculators(t *testing.T) {
	rand.Seed(2)
	diff := big.NewInt(10)
	for i := 0; i < 30; i++ {

		// 1 to 300 seconds diff
		var timeDelta = uint64(18)
		//rand.Read(difficulty)
		header := &types.Header{
			Difficulty: diff,
			Number:     new(big.Int).SetUint64(rand.Uint64() % 50_000_000),
			Time:       rand.Uint64(),
		}
		//if rand.Uint32()&1 == 0 {
		header.UncleHash = types.EmptyUncleHash
		//}

		diff = calcDifficultyNeo(header.Time+timeDelta, header, true)
		//t.Log(header.Difficulty)
		t.Log(diff)
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
