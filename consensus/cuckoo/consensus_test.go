package cuckoo

import (
	"math/big"
	"testing"
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
}
