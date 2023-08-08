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
		b := 8409600*i - 1
		reward := calculateRewardByNumber(big.NewInt(b), uint64(21))
		t.Log("block:", b, "period:", i, "reward:", reward)
		b += 1
		reward_1 := calculateRewardByNumber(big.NewInt(b), uint64(21))
		t.Log("block:", b, "period:", i+1, "reward:", reward_1)
		if reward_1.Cmp(new(big.Int).Div(reward, big.NewInt(2))) != 0 {
			t.Fatalf("failed")
		}
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

func TestGasLimitCheck(t *testing.T) {
	if validate := checkGasLimit(0, 7992189, 8000000, nil, nil); !validate {
		t.Fatalf("failed")
	}

	if validate := checkGasLimit(7992189, 7992189, 8000000, nil, nil); !validate {
		t.Fatalf("failed")
	}

	if validate := checkGasLimit(0, 8000000, 7992189, nil, nil); validate {
		t.Fatalf("failed")
	}

	if validate := checkGasLimit(7980000, 8000000, 8003878, nil, nil); !validate {
		t.Fatalf("failed")
	}

	if validate := checkGasLimit(7980000, 8000000, 8003879, nil, nil); validate {
		t.Fatalf("failed")
	}
}
