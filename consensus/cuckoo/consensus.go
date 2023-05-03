// Copyright 2018 The CortexTheseus Authors
// This file is part of the CortexFoundation library.
//
// The CortexFoundation library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexFoundation library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the CortexFoundation library. If not, see <http://www.gnu.org/licenses/>.

package cuckoo

import (
	"encoding/binary"
	// "encoding/hex"
	// "bytes"
	"errors"
	"fmt"
	"math/big"
	"runtime"
	"time"

	// "strconv"
	// "strings"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/math"
	"github.com/CortexFoundation/CortexTheseus/consensus"
	"github.com/CortexFoundation/CortexTheseus/consensus/cuckoo/plugins"
	"github.com/CortexFoundation/CortexTheseus/consensus/misc"
	"github.com/CortexFoundation/CortexTheseus/core/state"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/crypto"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/CortexTheseus/rlp"
	"github.com/CortexFoundation/CortexTheseus/trie"
	mapset "github.com/deckarep/golang-set/v2"
	"golang.org/x/crypto/sha3"
	//	"github.com/CortexFoundation/CortexTheseus/solution/miner/libcuckoo"
)

// Cuckoo proof-of-work protocol constants.
var (
	FrontierBlockReward           *big.Int = big.NewInt(7e+18) // Block reward in wei for successfully mining a block
	ByzantiumBlockReward          *big.Int = big.NewInt(7e+18) // Block reward in wei for successfully mining a block upward from Byzantium
	ConstantinopleBlockReward              = big.NewInt(7e+18)
	maxUncles                              = 2 // Maximum number of uncles allowed in a single block
	allowedFutureBlockTimeSeconds          = int64(15)
	FixHashes                              = map[common.Hash]bool{
		common.HexToHash("0x367e111f0f274d54f357ed3dc2d16107b39772c3a767138b857f5c02b5c30607"): true,
		common.HexToHash("0xbde83a87b6d526ada5a02e394c5f21327acb080568f7cc6f8fff423620f0eec3"): true,
	}
	// calcDifficultyConstantinople is the difficulty adjustment algorithm for Constantinople.
	// It returns the difficulty that a new block should have when created at time given the
	// parent block's time and difficulty. The calculation uses the Byzantium rules, but with
	// bomb offset 5M.
	// Specification EIP-1234: https://eips.cortex.org/EIPS/eip-1234
	//calcDifficultyConstantinople = makeDifficultyCalculator(big.NewInt(5000000))

	// calcDifficultyByzantium is the difficulty adjustment algorithm. It returns
	// the difficulty that a new block should have when created at time given the
	// parent block's time and difficulty. The calculation uses the Byzantium rules.
	// Specification EIP-649: https://eips.cortex.org/EIPS/eip-649
	//calcDifficultyByzantium = makeDifficultyCalculator(big.NewInt(3000000))
)

// Various error messages to mark blocks invalid. These should be private to
// prevent engine specific errors from being referenced in the remainder of the
// codebase, inherently breaking if the engine is swapped out. Please put common
// error types into the consensus package.
var (
	errOlderBlockTime    = errors.New("timestamp older than parent")
	errTooManyUncles     = errors.New("too many uncles")
	errDuplicateUncle    = errors.New("duplicate uncle")
	errUncleIsAncestor   = errors.New("uncle is ancestor")
	errDanglingUncle     = errors.New("uncle's parent is not ancestor")
	errInvalidDifficulty = errors.New("non-positive difficulty")
	errInvalidMixDigest  = errors.New("invalid mix digest")
	errInvalidPoW        = errors.New("invalid proof-of-work")
)

// Author implements consensus.Engine, returning the header's coinbase as the
// proof-of-work verified author of the block.
func (cuckoo *Cuckoo) Author(header *types.Header) (common.Address, error) {
	return header.Coinbase, nil
}

// VerifyHeader checks whether a header conforms to the consensus rules of the
// stock Cortex cuckoo engine.
func (cuckoo *Cuckoo) VerifyHeader(chain consensus.ChainHeaderReader, header *types.Header, seal bool) error {
	// If we're running a full engine faking, accept any input as valid
	if cuckoo.config.PowMode == ModeFullFake {
		return nil
	}
	// Short circuit if the header is known, or it's parent not
	number := header.Number.Uint64()
	if chain.GetHeader(header.Hash(), number) != nil {
		return nil
	}
	parent := chain.GetHeader(header.ParentHash, number-1)
	if parent == nil {
		return consensus.ErrUnknownAncestor
	}
	// Sanity checks passed, do a proper verification
	return cuckoo.verifyHeader(chain, header, parent, false, seal, time.Now().Unix())
}

// VerifyHeaders is similar to VerifyHeader, but verifies a batch of headers
// concurrently. The method returns a quit channel to abort the operations and
// a results channel to retrieve the async verifications.
func (cuckoo *Cuckoo) VerifyHeaders(chain consensus.ChainHeaderReader, headers []*types.Header, seals []bool) (chan<- struct{}, <-chan error) {
	// If we're running a full engine faking, accept any input as valid
	if cuckoo.config.PowMode == ModeFullFake || len(headers) == 0 {
		abort, results := make(chan struct{}), make(chan error, len(headers))
		for i := 0; i < len(headers); i++ {
			results <- nil
		}
		return abort, results
	}

	// Spawn as many workers as allowed threads
	workers := runtime.GOMAXPROCS(0)
	if len(headers) < workers {
		workers = len(headers)
	}

	// Create a task channel and spawn the verifiers
	var (
		inputs = make(chan int)
		done   = make(chan int, workers)
		errors = make([]error, len(headers))
		abort  = make(chan struct{})
		utcNow = time.Now().Unix()
	)
	for i := 0; i < workers; i++ {
		go func() {
			for index := range inputs {
				errors[index] = cuckoo.verifyHeaderWorker(chain, headers, seals, index, utcNow)
				done <- index
			}
		}()
	}

	errorsOut := make(chan error, len(headers))
	go func() {
		defer close(inputs)
		var (
			in, out = 0, 0
			checked = make([]bool, len(headers))
			inputs  = inputs
		)
		for {
			select {
			case inputs <- in:
				if in++; in == len(headers) {
					// Reached end of headers. Stop sending to workers.
					inputs = nil
				}
			case index := <-done:
				for checked[index] = true; checked[out]; out++ {
					errorsOut <- errors[out]
					if out == len(headers)-1 {
						return
					}
				}
			case <-abort:
				return
			}
		}
	}()
	return abort, errorsOut
}

func (cuckoo *Cuckoo) verifyHeaderWorker(chain consensus.ChainHeaderReader, headers []*types.Header, seals []bool, index int, utcNow int64) error {
	var parent *types.Header
	if index == 0 {
		parent = chain.GetHeader(headers[0].ParentHash, headers[0].Number.Uint64()-1)
	} else if headers[index-1].Hash() == headers[index].ParentHash {
		parent = headers[index-1]
	}
	if parent == nil {
		return consensus.ErrUnknownAncestor
	}
	return cuckoo.verifyHeader(chain, headers[index], parent, false, seals[index], utcNow)
}

// VerifyUncles verifies that the given block's uncles conform to the consensus
// rules of the stock Cortex cuckoo engine.
func (cuckoo *Cuckoo) VerifyUncles(chain consensus.ChainReader, block *types.Block) error {
	// If we're running a full engine faking, accept any input as valid
	if cuckoo.config.PowMode == ModeFullFake {
		return nil
	}
	// Verify that there are at most 2 uncles included in this block
	if len(block.Uncles()) > maxUncles {
		return errTooManyUncles
	}

	if len(block.Uncles()) == 0 {
		return nil
	}
	// Gather the set of past uncles and ancestors
	uncles, ancestors := mapset.NewSet[common.Hash](), make(map[common.Hash]*types.Header)

	number, parent := block.NumberU64()-1, block.ParentHash()
	for i := 0; i < 7; i++ {
		ancestor := chain.GetBlock(parent, number)
		if ancestor == nil {
			break
		}
		ancestors[ancestor.Hash()] = ancestor.Header()
		for _, uncle := range ancestor.Uncles() {
			uncles.Add(uncle.Hash())
		}
		parent, number = ancestor.ParentHash(), number-1
	}
	ancestors[block.Hash()] = block.Header()
	uncles.Add(block.Hash())

	// Verify each of the uncles that it's recent, but not an ancestor
	for _, uncle := range block.Uncles() {
		// Make sure every uncle is rewarded only once
		hash := uncle.Hash()
		if uncles.Contains(hash) {
			return errDuplicateUncle
		}
		uncles.Add(hash)

		// Make sure the uncle has a valid ancestry
		if ancestors[hash] != nil {
			return errUncleIsAncestor
		}
		if ancestors[uncle.ParentHash] == nil || uncle.ParentHash == block.ParentHash() {
			return errDanglingUncle
		}
		if err := cuckoo.verifyHeader(chain, uncle, ancestors[uncle.ParentHash], true, true, time.Now().Unix()); err != nil {
			return err
		}
	}
	return nil
}

// verifyHeader checks whether a header conforms to the consensus rules of the
// stock Cortex cuckoo engine.
// See YP section 4.3.4. "Block Header Validity"
func (cuckoo *Cuckoo) verifyHeader(chain consensus.ChainHeaderReader, header, parent *types.Header, uncle, seal bool, utcNow int64) error {
	// Ensure that the header's extra-data section is of a reasonable size
	if uint64(len(header.Extra)) > params.MaximumExtraDataSize {
		return fmt.Errorf("extra-data too long: %d > %d", len(header.Extra), params.MaximumExtraDataSize)
	}
	// Verify the header's timestamp
	if !uncle {
		if header.Time > uint64(utcNow+allowedFutureBlockTimeSeconds) {
			return consensus.ErrFutureBlock
		}
	}
	if header.Time <= parent.Time {
		return errOlderBlockTime
	}
	// Verify the block's difficulty based in it's timestamp and parent's difficulty
	expected := cuckoo.CalcDifficulty(chain, header.Time, parent)

	if expected.Cmp(header.Difficulty) != 0 {
		return fmt.Errorf("invalid difficulty: have %v, want %v", header.Difficulty, expected)
	}
	// Verify that the gas limit is <= 2^63-1
	if header.GasLimit > params.MaxGasLimit {
		return fmt.Errorf("invalid gasLimit: have %v, max %v", header.GasLimit, params.MaxGasLimit)
	}
	// Verify that the gasUsed is <= gasLimit
	if header.GasUsed > header.GasLimit {
		return fmt.Errorf("invalid gasUsed: have %d, gasLimit %d", header.GasUsed, header.GasLimit)
	}

	validate := checkGasLimit(parent.GasUsed, parent.GasLimit, header.GasLimit)
	if !validate {
		return fmt.Errorf("invalid gas limit trend: have %d, want %d used %d", header.GasLimit, parent.GasLimit, parent.GasUsed)
	}

	// Verify that the gas limit remains within allowed bounds
	diff := int64(parent.GasLimit) - int64(header.GasLimit)
	if diff < 0 {
		diff *= -1
	}

	limit := parent.GasLimit / params.GasLimitBoundDivisor

	if uint64(diff) >= limit || header.GasLimit < params.MinGasLimit {
		return fmt.Errorf("invalid gas limit: have %d, want %d += %d", header.GasLimit, parent.GasLimit, limit)
	}

	// Verify that the block number is parent's +1
	if diff := new(big.Int).Sub(header.Number, parent.Number); diff.Cmp(big.NewInt(1)) != 0 {
		return consensus.ErrInvalidNumber
	}

	if header.Quota < parent.Quota || header.Quota != parent.Quota+chain.Config().GetBlockQuota(header.Number) {
		return fmt.Errorf("invalid quota %v, %v, %v", header.Quota, parent.Quota, chain.Config().GetBlockQuota(header.Number))
	}

	bigInitReward := calculateRewardByNumber(header.Number, chain.Config().ChainID.Uint64())

	uncleMaxReward := big.NewInt(0).Div(big.NewInt(0).Mul(bigInitReward, big7), big8)
	nephewReward := big.NewInt(0).Div(bigInitReward, big32)

	//final with uncle
	bigMaxReward := big.NewInt(0).Add(big.NewInt(0).Mul(big2, big.NewInt(0).Add(uncleMaxReward, nephewReward)), bigInitReward)

	if header.UncleHash == types.EmptyUncleHash {
		if _, ok := FixHashes[header.Hash()]; ok {
		} else {
			if header.Supply.Cmp(new(big.Int).Add(parent.Supply, bigInitReward)) > 0 {
				return fmt.Errorf("invalid supply without uncle %v, %v, %v, %v, %v", header.Supply, parent.Supply, header.Hash().Hex(), header.Number, bigInitReward)
			}
		}
	} else {
		if header.Supply.Cmp(new(big.Int).Add(parent.Supply, bigMaxReward)) > 0 {
			return fmt.Errorf("invalid supply with uncle of max reward %v, %v, %v", header.Supply, parent.Supply, bigMaxReward)
		}
	}
	// Verify the engine specific seal securing the block
	if seal {
		if err := cuckoo.VerifySeal(chain, header); err != nil {
			return err
		}
	}
	// If all checks passed, validate any special fields for hard forks
	//if err := misc.VerifyDAOHeaderExtraData(chain.Config(), header); err != nil {
	//	return err
	//}
	if err := misc.VerifyForkHashes(chain.Config(), header, uncle); err != nil {
		return err
	}
	return nil
}

// CalcDifficulty is the difficulty adjustment algorithm. It returns
// the difficulty that a new block should have when created at time
// given the parent block's time and difficulty.
func (cuckoo *Cuckoo) CalcDifficulty(chain consensus.ChainHeaderReader, time uint64, parent *types.Header) *big.Int {
	return CalcDifficulty(chain.Config(), time, parent)
}

// CalcDifficulty is the difficulty adjustment algorithm. It returns
// the difficulty that a new block should have when created at time
// given the parent block's time and difficulty.
func CalcDifficulty(config *params.ChainConfig, time uint64, parent *types.Header) *big.Int {
	next := new(big.Int).Add(parent.Number, big1)
	switch {
	case config.IsNeo(next):
		return calcDifficultyNeo(time, parent, true)
	case config.IsIstanbul(next):
		return calcDifficultyIstanbul(time, parent, false)
	case config.IsConstantinople(next):
		return calcDifficultyConstantinople(time, parent, false)
	case config.IsByzantium(next):
		return calcDifficultyByzantium(time, parent, false)
	case config.IsHomestead(next):
		return calcDifficultyHomestead(time, parent)
	default:
		return calcDifficultyFrontier(time, parent)
	}
}

// important add gas limit to consensus
func checkGasLimit(gasUsed, gasLimit, currentGasLimit uint64) bool {
	contrib := (gasUsed + gasUsed/2) / params.GasLimitBoundDivisor
	decay := gasLimit/params.GasLimitBoundDivisor - 1
	limit := gasLimit - decay + contrib

	if limit < params.MinGasLimit {
		limit = params.MinGasLimit
	}

	if limit < params.MinerGasFloor {
		limit = gasLimit + decay
		if limit > params.MinerGasFloor {
			limit = params.MinerGasFloor
		}
	} else if limit > params.MinerGasCeil {
		limit = gasLimit - decay
		if limit < params.MinerGasCeil {
			limit = params.MinerGasCeil
		}
	}

	return limit == currentGasLimit
}

// Some weird constants to avoid constant memory allocs for them.
var (
	expDiffPeriod = big.NewInt(100000)
	big1          = big.NewInt(1)
	big2          = big.NewInt(2)
	big3          = big.NewInt(3)
	big5          = big.NewInt(5)
	big9          = big.NewInt(9)
	big10         = big.NewInt(10)
	big15         = big.NewInt(15)
	bigMinus1     = big.NewInt(-1)
	bigMinus9     = big.NewInt(-9)
	bigMinus99    = big.NewInt(-99)
)

func calcDifficultyIstanbul(time uint64, parent *types.Header, neo bool) *big.Int {
	return calcDifficultyConstantinople(time, parent, neo)
}

func calcDifficultyConstantinople(time uint64, parent *types.Header, neo bool) *big.Int {
	return calcDifficultyByzantium(time, parent, neo)
}

func calcDifficultyNeo(time uint64, parent *types.Header, neo bool) *big.Int {
	return calcDifficultyIstanbul(time, parent, neo)
}

// calcDifficultyByzantium is the difficulty adjustment algorithm. It returns
// the difficulty that a new block should have when created at time given the
// parent block's time and difficulty. The calculation uses the Byzantium rules.
func calcDifficultyByzantium(time uint64, parent *types.Header, neo bool) *big.Int {
	// https://github.com/cortex/EIPs/issues/100.
	// algorithm:
	// diff = (parent_diff +
	//         (parent_diff / 2048 * max((2 if len(parent.uncles) else 1) - ((timestamp - parent.timestamp) // 9), -99))
	//        ) + 2^(periodCount - 2)

	bigTime := new(big.Int).SetUint64(time)
	bigParentTime := new(big.Int).SetUint64(parent.Time)

	// holds intermediate values to make the algo easier to read & audit
	x := new(big.Int)
	y := new(big.Int)

	// (2 if len(parent_uncles) else 1) - (block_timestamp - parent_timestamp) // 9
	x.Sub(bigTime, bigParentTime)
	x.Div(x, big9)
	if parent.UncleHash == types.EmptyUncleHash {
		x.Sub(big1, x)
	} else {
		x.Sub(big2, x)
	}
	// max((2 if len(parent_uncles) else 1) - (block_timestamp - parent_timestamp) // 9, -99)
	if bigParentTime.Cmp(big0) > 0 {
		if x.Cmp(bigMinus99) < 0 {
			x.Set(bigMinus99)
		}
	} else {
		x.Set(big0)
	}

	if parent.Difficulty.Cmp(params.MeanDifficultyBoundDivisor) >= 0 && parent.Difficulty.Cmp(params.HighDifficultyBoundDivisor) < 0 {
		y.Div(parent.Difficulty, params.MeanDifficultyBoundDivisor)
	} else if parent.Difficulty.Cmp(params.HighDifficultyBoundDivisor) >= 0 {
		y.Div(parent.Difficulty, params.HighDifficultyBoundDivisor)
	} else {
		if neo {
			y = params.MinimumDifficulty // delta 2
		} else {
			y.Div(parent.Difficulty, params.DifficultyBoundDivisor_2)
		}

		if x.Cmp(big0) > 0 {
			x.Set(big1)
		}

		if x.Cmp(big0) < 0 {
			x.Set(bigMinus1)
		}
	}

	//log.Info("cal diff", "x", x, "parent.Difficulty", parent.Difficulty, "y", y)

	// parent_diff + (parent_diff / 2048 * max((2 if len(parent.uncles) else 1) - ((timestamp - parent.timestamp) // 9), -99))
	//y.Div(parent.Difficulty, params.DifficultyBoundDivisor)
	x.Mul(y, x)
	x.Add(parent.Difficulty, x)

	//log.Info("cal diff", "x", x, "parent.Difficulty", parent.Difficulty, "y", y)

	// minimum difficulty can ever be (before exponential factor)
	if x.Cmp(params.MinimumDifficulty) < 0 {
		x.Set(params.MinimumDifficulty)
	}
	// calculate a fake block number for the ice-age delay:
	// https://github.com/cortex/EIPs/pull/669
	// fake_block_number = max(0, block.number - 3_000_000)
	//fakeBlockNumber := new(big.Int)
	//if parent.Number.Cmp(big2999999) >= 0 {
	//	fakeBlockNumber = fakeBlockNumber.Sub(parent.Number, big2999999) // Note, parent is 1 less than the actual block number
	//}
	// for the exponential factor
	//periodCount := fakeBlockNumber
	//periodCount.Div(periodCount, expDiffPeriod)

	// the exponential factor, commonly referred to as "the bomb"
	// diff = diff + 2^(periodCount - 2)
	//if periodCount.Cmp(big1) > 0 {
	//	y.Sub(periodCount, big2)
	//	y.Exp(big2, y, nil)
	//	x.Add(x, y)
	//}
	return x
}

// makeDifficultyCalculator creates a difficultyCalculator with the given bomb-delay.
// the difficulty is calculated with Byzantium rules, which differs from Homestead in
// how uncles affect the calculation
func makeDifficultyCalculator(bombDelay *big.Int) func(time uint64, parent *types.Header) *big.Int {
	// Note, the calculations below looks at the parent number, which is 1 below
	// the block number. Thus we remove one from the delay given
	bombDelayFromParent := new(big.Int).Sub(bombDelay, big1)
	return func(time uint64, parent *types.Header) *big.Int {
		// https://github.com/cortex/EIPs/issues/100.
		// algorithm:
		// diff = (parent_diff +
		//         (parent_diff / 2048 * max((2 if len(parent.uncles) else 1) - ((timestamp - parent.timestamp) // 9), -99))
		//        ) + 2^(periodCount - 2)

		bigTime := new(big.Int).SetUint64(time)
		bigParentTime := new(big.Int).SetUint64(parent.Time)

		// holds intermediate values to make the algo easier to read & audit
		x := new(big.Int)
		y := new(big.Int)

		// (2 if len(parent_uncles) else 1) - (block_timestamp - parent_timestamp) // 9
		x.Sub(bigTime, bigParentTime)
		x.Div(x, big9)
		if parent.UncleHash == types.EmptyUncleHash {
			x.Sub(big1, x)
		} else {
			x.Sub(big2, x)
		}
		// max((2 if len(parent_uncles) else 1) - (block_timestamp - parent_timestamp) // 9, -99)
		if x.Cmp(bigMinus99) < 0 {
			x.Set(bigMinus99)
		}
		// parent_diff + (parent_diff / 2048 * max((2 if len(parent.uncles) else 1) - ((timestamp - parent.timestamp) // 9), -99))
		y.Div(parent.Difficulty, params.DifficultyBoundDivisor_2)
		x.Mul(y, x)
		x.Add(parent.Difficulty, x)

		// minimum difficulty can ever be (before exponential factor)
		if x.Cmp(params.MinimumDifficulty) < 0 {
			x.Set(params.MinimumDifficulty)
		}
		// calculate a fake block number for the ice-age delay
		// Specification: https://eips.cortex.org/EIPS/eip-1234
		fakeBlockNumber := new(big.Int)
		if parent.Number.Cmp(bombDelayFromParent) >= 0 {
			fakeBlockNumber = fakeBlockNumber.Sub(parent.Number, bombDelayFromParent)
		}
		// for the exponential factor
		periodCount := fakeBlockNumber
		periodCount.Div(periodCount, expDiffPeriod)

		// the exponential factor, commonly referred to as "the bomb"
		// diff = diff + 2^(periodCount - 2)
		if periodCount.Cmp(big1) > 0 {
			y.Sub(periodCount, big2)
			y.Exp(big2, y, nil)
			x.Add(x, y)
		}
		return x
	}
}

// calcDifficultyHomestead is the difficulty adjustment algorithm. It returns
// the difficulty that a new block should have when created at time given the
// parent block's time and difficulty. The calculation uses the Homestead rules.
func calcDifficultyHomestead(time uint64, parent *types.Header) *big.Int {
	// https://github.com/cortex/EIPs/blob/master/EIPS/eip-2.md
	// algorithm:
	// diff = (parent_diff +
	//         (parent_diff / 2048 * max(1 - (block_timestamp - parent_timestamp) // 10, -99))
	//        ) + 2^(periodCount - 2)

	bigTime := new(big.Int).SetUint64(time)
	bigParentTime := new(big.Int).SetUint64(parent.Time)

	// holds intermediate values to make the algo easier to read & audit
	x := new(big.Int)
	y := new(big.Int)

	// 1 - (block_timestamp - parent_timestamp) // 10
	x.Sub(bigTime, bigParentTime)
	x.Div(x, big10)
	x.Sub(big1, x)

	// max(1 - (block_timestamp - parent_timestamp) // 10, -99)
	if x.Cmp(bigMinus99) < 0 {
		x.Set(bigMinus99)
	}
	// (parent_diff + parent_diff // 2048 * max(1 - (block_timestamp - parent_timestamp) // 10, -99))
	if parent.Difficulty.Cmp(params.MeanDifficultyBoundDivisor) >= 0 && parent.Difficulty.Cmp(params.HighDifficultyBoundDivisor) < 0 {
		y.Div(parent.Difficulty, params.MeanDifficultyBoundDivisor)
	} else if parent.Difficulty.Cmp(params.HighDifficultyBoundDivisor) >= 0 {
		y.Div(parent.Difficulty, params.HighDifficultyBoundDivisor)
	} else {
		y.Div(parent.Difficulty, params.DifficultyBoundDivisor_2)
	}
	x.Mul(y, x)
	x.Add(parent.Difficulty, x)

	// minimum difficulty can ever be (before exponential factor)
	if x.Cmp(params.MinimumDifficulty) < 0 {
		x.Set(params.MinimumDifficulty)
	}
	// for the exponential factor
	//periodCount := new(big.Int).Add(parent.Number, big1)
	//periodCount.Div(periodCount, expDiffPeriod)

	// the exponential factor, commonly referred to as "the bomb"
	// diff = diff + 2^(periodCount - 2)
	//if periodCount.Cmp(big1) > 0 {
	//	y.Sub(periodCount, big2)
	//	y.Exp(big2, y, nil)
	//	x.Add(x, y)
	//}
	return x
}

// calcDifficultyFrontier is the difficulty adjustment algorithm. It returns the
// difficulty that a new block should have when created at time given the parent
// block's time and difficulty. The calculation uses the Frontier rules.
func calcDifficultyFrontier(time uint64, parent *types.Header) *big.Int {
	diff := new(big.Int)
	adjust := new(big.Int).Div(parent.Difficulty, params.DifficultyBoundDivisor_2)
	bigTime := new(big.Int)
	bigParentTime := new(big.Int)

	bigTime.SetUint64(time)
	bigParentTime.SetUint64(parent.Time)

	if bigTime.Sub(bigTime, bigParentTime).Cmp(params.DurationLimit) < 0 {
		diff.Add(parent.Difficulty, adjust)
	} else {
		diff.Sub(parent.Difficulty, adjust)
	}
	if diff.Cmp(params.MinimumDifficulty) < 0 {
		diff.Set(params.MinimumDifficulty)
	}

	periodCount := new(big.Int).Add(parent.Number, big1)
	periodCount.Div(periodCount, expDiffPeriod)
	if periodCount.Cmp(big1) > 0 {
		// diff = diff + 2^(periodCount - 2)
		expDiff := periodCount.Sub(periodCount, big2)
		expDiff.Exp(big2, expDiff, nil)
		diff.Add(diff, expDiff)
		diff = math.BigMax(diff, params.MinimumDifficulty)
	}
	return diff
}

// VerifySeal implements consensus.Engine, checking whether the given block satisfies
// the PoW difficulty requirements.

func (cuckoo *Cuckoo) VerifySeal(chain consensus.ChainHeaderReader, header *types.Header) error {
	// If we're running a fake PoW, accept any seal as valid
	if cuckoo.config.PowMode == ModeFake || cuckoo.config.PowMode == ModeFullFake {
		time.Sleep(cuckoo.fakeDelay)
		if cuckoo.fakeFail == header.Number.Uint64() {
			return errInvalidPoW
		}
		return nil
	}
	if header.Difficulty.Sign() <= 0 {
		return errInvalidDifficulty
	}

	var (
		result        = header.Solution
		nonce  uint64 = uint64(header.Nonce.Uint64())
		hash          = cuckoo.SealHash(header).Bytes()
	)

	targetDiff := new(big.Int).Div(maxUint256, header.Difficulty)
	// fmt.Println("uint8_t a[80] = {" + strings.Trim(strings.Join(strings.Fields(fmt.Sprint(hash)), ","), "[]") + "};")
	// fmt.Println("uint32_t nonce =  ", nonce, ";")
	// fmt.Println("uint32_t result[42] =  {" + strings.Trim(strings.Join(strings.Fields(fmt.Sprint(result)), ","), "[]") + "};")
	// fmt.Println("uint8_t t[32] = {" + strings.Trim(strings.Join(strings.Fields(fmt.Sprint(diff)), ","), "[]") + "};")
	// fmt.Println("uint8_t h[32] = {" + strings.Trim(strings.Join(strings.Fields(fmt.Sprint(result_hash)), ","), "[]") + "};")
	// r := CuckooVerify(&hash[0], len(hash), uint32(nonce), &result[0], &diff[0], &result_hash[0])
	//fmt.Println("VerifySeal: ", result, nonce, uint32((nonce)), hash)
	//r := cuckoo.CuckooVerifyHeader(hash, nonce, &result, header.Number.Uint64(), targetDiff)
	r := cuckoo.CuckooVerifyHeader(hash, nonce, &result, targetDiff)
	if !r {
		log.Trace(fmt.Sprintf("VerifySeal: %v, %v", r, targetDiff))
		return errInvalidPoW
	}

	return nil
}

// Prepare implements consensus.Engine, initializing the difficulty field of a
// header to conform to the cuckoo protocol. The changes are done inline.
func (cuckoo *Cuckoo) Prepare(chain consensus.ChainHeaderReader, header *types.Header) error {
	parent := chain.GetHeader(header.ParentHash, header.Number.Uint64()-1)
	if parent == nil {
		return consensus.ErrUnknownAncestor
	}
	header.Difficulty = cuckoo.CalcDifficulty(chain, header.Time, parent)
	header.Supply = new(big.Int).Set(parent.Supply)
	header.Quota = parent.Quota + chain.Config().GetBlockQuota(header.Number)
	if header.Quota < parent.Quota {
		panic("quota reaches the upper limit of uint64")
	}
	header.QuotaUsed = parent.QuotaUsed
	return nil
}

// Finalize implements consensus.Engine, accumulating the block and uncle rewards,
// setting the final state on the header
func (cuckoo *Cuckoo) Finalize(chain consensus.ChainHeaderReader, header *types.Header, state *state.StateDB, txs []*types.Transaction, uncles []*types.Header) error {
	// Always need parent to caculate the reward of current block
	parent := chain.GetHeader(header.ParentHash, header.Number.Uint64()-1)
	if parent == nil {
		return consensus.ErrUnknownAncestor
	}
	// Accumulate any block and uncle rewards and commit the final state root
	accumulateRewards(chain.Config(), state, header, parent, uncles)
	header.Root = state.IntermediateRoot(chain.Config().IsEIP158(header.Number))
	return nil
}

// Finalize implements consensus.Engine, accumulating the block and uncle rewards,
// setting the final state and assembling the block.
func (cuckoo *Cuckoo) FinalizeAndAssemble(chain consensus.ChainHeaderReader, header *types.Header, state *state.StateDB, txs []*types.Transaction, uncles []*types.Header, receipts []*types.Receipt) (*types.Block, error) {
	//log.Info(fmt.Sprintf("parent: %v, current: %v, number: %v, total: %v, epoch: %v", parent.Number, header.Hash(), header.Number, params.CTXC_TOP, params.CortexBlockRewardPeriod))
	// Accumulate any block and uncle rewards and commit the final state root
	err := cuckoo.Finalize(chain, header, state, txs, uncles)
	if err != nil {
		return nil, err
	}

	// Header seems complete, assemble into a block and return
	return types.NewBlock(header, txs, uncles, receipts, trie.NewStackTrie(nil)), nil
}

// FinalizeAndAssemble implements consensus.Engine, accumulating the block and
// uncle rewards, setting the final state and assembling the block.
func (cuckoo *Cuckoo) FinalizeWithoutParent(chain consensus.ChainHeaderReader, header *types.Header, state *state.StateDB, txs []*types.Transaction, uncles []*types.Header, receipts []*types.Receipt) (*types.Block, error) {
	// Accumulate any block and uncle rewards and commit the final state root
	accumulateRewardsWithoutParent(chain.Config(), state, header, uncles)
	header.Root = state.IntermediateRoot(chain.Config().IsEIP158(header.Number))

	// Header seems complete, assemble into a block and return
	return types.NewBlock(header, txs, uncles, receipts, trie.NewStackTrie(nil)), nil
}

// SealHash returns the hash of a block prior to it being sealed.
func (cuckoo *Cuckoo) SealHash(header *types.Header) (hash common.Hash) {
	hasher := sha3.NewLegacyKeccak256()

	rlp.Encode(hasher, []interface{}{
		header.ParentHash,
		header.UncleHash,
		header.Coinbase,
		header.Root,
		header.TxHash,
		header.ReceiptHash,
		header.Bloom,
		header.Difficulty,
		header.Number,
		header.GasLimit,
		header.GasUsed,
		header.Time,
		header.Extra,
		//header.Quota,
		//header.QuotaUsed,
		//header.Supply,
	})
	hasher.Sum(hash[:0])
	return hash
}

// Some weird constants to avoid constant memory allocs for them.
var (
	big0    = big.NewInt(0)
	big4    = big.NewInt(4)
	big7    = big.NewInt(7)
	big8    = big.NewInt(8)
	big32   = big.NewInt(32)
	big64   = big.NewInt(64)
	big128  = big.NewInt(128)
	big4096 = big.NewInt(4096)
	//bigInitReward = big.NewInt(7000000000000000000)
	bigFix = big.NewInt(6343750000000000000)
	//bigMidReward  = big.NewInt(0).Mul(big.NewInt(13343750000), big.NewInt(1000000000))
	//bigMaxReward  = big.NewInt(0).Mul(big.NewInt(19687500000), big.NewInt(1000000000))
)

func calculateRewardByNumber(num *big.Int, chainId uint64) (blockReward *big.Int) {
	blockReward = big.NewInt(0).Set(FrontierBlockReward)

	if chainId == 21 {
		if num.Cmp(params.CortexBlockRewardPeriod) >= 0 {
			d := new(big.Int).Div(num, params.CortexBlockRewardPeriod)
			e := new(big.Int).Exp(big2, d, nil)
			blockReward = new(big.Int).Div(blockReward, e)
		}
	} else if chainId == 42 {
		if num.Cmp(params.BernardBlockRewardPeriod) >= 0 {
			d := new(big.Int).Div(num, params.BernardBlockRewardPeriod)
			e := new(big.Int).Exp(big2, d, nil)
			blockReward = new(big.Int).Div(blockReward, e)
		}
	} else if chainId == 43 {
		if num.Cmp(params.DoloresBlockRewardPeriod) >= 0 {
			d := new(big.Int).Div(num, params.DoloresBlockRewardPeriod)
			e := new(big.Int).Exp(big2, d, nil)
			blockReward = new(big.Int).Div(blockReward, e)
		}
	} else {
		if num.Cmp(params.CortexBlockRewardPeriod) >= 0 {
			d := new(big.Int).Div(num, params.CortexBlockRewardPeriod)
			e := new(big.Int).Exp(big2, d, nil)
			blockReward = new(big.Int).Div(blockReward, e)
		}
	}

	return
}

// AccumulateRewards credits the coinbase of the given block with the mining
// reward. The total reward consists of the static block reward and rewards for
// included uncles. The coinbase of each uncle block is also rewarded.
func accumulateRewards(config *params.ChainConfig, state *state.StateDB, header, parent *types.Header, uncles []*types.Header) {

	if parent == nil {
		return
	}

	headerInitialHash := header.Hash()

	blockReward := calculateRewardByNumber(header.Number, config.ChainID.Uint64())

	log.Trace("Parent status", "number", parent.Number, "hash", parent.Hash(), "supply", toCoin(parent.Supply))
	if header.Supply == nil {
		header.Supply = new(big.Int)
	}
	header.Supply.Set(parent.Supply)

	if header.Supply.Cmp(params.CTXC_INIT) < 0 && config.ChainID.Uint64() != 42 {
		header.Supply.Set(params.CTXC_INIT)
	}

	if header.Supply.Cmp(params.CTXC_TOP) >= 0 {
		blockReward.Set(big0)
		header.Supply.Set(params.CTXC_TOP)
	}

	if blockReward.Cmp(big0) > 0 {
		remain := new(big.Int).Sub(params.CTXC_TOP, header.Supply)
		header.Supply.Add(header.Supply, blockReward)
		if header.Supply.Cmp(params.CTXC_TOP) >= 0 {
			blockReward.Set(remain)
			header.Supply.Set(params.CTXC_TOP)
			log.Warn("Congratulations!!! We have mined all cortex", "number", header.Number, "last reward", toCoin(remain))
		}

		if blockReward.Cmp(big0) <= 0 {
			//should never happend
			return
		}

		log.Trace("Block mining reward", "parent", toCoin(parent.Supply), "current", toCoin(header.Supply), "number", header.Number, "reward", toCoin(blockReward))
		// Accumulate the rewards for the miner and any included uncles
		reward := new(big.Int).Set(blockReward)
		r := new(big.Int)

		//for hash := range FixHashes {
		//	if hash == headerInitialHash {
		//		header.Supply.Add(header.Supply, bigFix)
		//	}
		//}

		if len(uncles) > 0 {
			for _, uncle := range uncles {
				r.Add(uncle.Number, big8)
				r.Sub(r, header.Number)
				r.Mul(r, blockReward)
				r.Div(r, big8)

				header.Supply.Add(header.Supply, r)
				if header.Supply.Cmp(params.CTXC_TOP) > 0 {
					header.Supply.Sub(header.Supply, r)
					r.Set(big0)
					break
				}
				state.AddBalance(uncle.Coinbase, r)
				log.Trace("Uncle mining reward", "miner", uncle.Coinbase, "reward", toCoin(r), "total", toCoin(header.Supply))

				r.Div(blockReward, big32)
				header.Supply.Add(header.Supply, r)
				if header.Supply.Cmp(params.CTXC_TOP) > 0 {
					header.Supply.Sub(header.Supply, r)
					r.Set(big0)
					break
				}

				log.Trace("Nephew mining reward", "reward", toCoin(r), "total", toCoin(header.Supply))
				reward.Add(reward, r)
			}
		} else {

			if _, ok := FixHashes[headerInitialHash]; ok {
				header.Supply.Add(header.Supply, bigFix)
			}
		}

		state.AddBalance(header.Coinbase, reward)

		if config.ChainID.Uint64() == 21 && config.IstanbulBlock != nil && header.Number.Cmp(config.IstanbulBlock) == 0 {
			state.AddBalance(common.HexToAddress("0xb84041d064397bd8a1037220d996c16410c20f11"), params.CTXC_F1)
			state.AddBalance(common.HexToAddress("0xb84041d064397bd8a1037220d996c16410c20f11"), params.CTXC_F2)
		}
	}
}

// AccumulateRewards credits the coinbase of the given block with the mining
// reward. The total reward consists of the static block reward and rewards for
// included uncles. The coinbase of each uncle block is also rewarded.
func accumulateRewardsWithoutParent(config *params.ChainConfig, state *state.StateDB, header *types.Header, uncles []*types.Header) {
	headerInitialHash := header.Hash()

	blockReward := calculateRewardByNumber(header.Number, config.ChainID.Uint64())

	if header.Supply == nil {
		header.Supply = new(big.Int)
	}
	header.Supply.Set(params.CTXC_INIT)

	if header.Supply.Cmp(params.CTXC_INIT) < 0 && config.ChainID.Uint64() != 42 {
		header.Supply.Set(params.CTXC_INIT)
	}

	if header.Supply.Cmp(params.CTXC_TOP) >= 0 {
		blockReward.Set(big0)
		header.Supply.Set(params.CTXC_TOP)
	}

	if blockReward.Cmp(big0) > 0 {
		remain := new(big.Int).Sub(params.CTXC_TOP, header.Supply)
		header.Supply.Add(header.Supply, blockReward)
		if header.Supply.Cmp(params.CTXC_TOP) >= 0 {
			blockReward.Set(remain)
			header.Supply.Set(params.CTXC_TOP)
			log.Warn("Congratulations!!! We have mined all cortex", "number", header.Number, "last reward", toCoin(remain))
		}

		if blockReward.Cmp(big0) <= 0 {
			//should never happend
			return
		}

		// Accumulate the rewards for the miner and any included uncles
		reward := new(big.Int).Set(blockReward)
		r := new(big.Int)

		//for hash := range FixHashes {
		//	if hash == headerInitialHash {
		//		header.Supply.Add(header.Supply, bigFix)
		//	}
		//}

		if len(uncles) > 0 {

			for _, uncle := range uncles {
				r.Add(uncle.Number, big8)
				r.Sub(r, header.Number)
				r.Mul(r, blockReward)
				r.Div(r, big8)

				header.Supply.Add(header.Supply, r)
				if header.Supply.Cmp(params.CTXC_TOP) > 0 {
					header.Supply.Sub(header.Supply, r)
					r.Set(big0)
					break
				}
				state.AddBalance(uncle.Coinbase, r)
				log.Trace("Uncle mining reward", "miner", uncle.Coinbase, "reward", toCoin(r), "total", toCoin(header.Supply))

				r.Div(blockReward, big32)
				header.Supply.Add(header.Supply, r)
				if header.Supply.Cmp(params.CTXC_TOP) > 0 {
					header.Supply.Sub(header.Supply, r)
					r.Set(big0)
					break
				}

				log.Trace("Nephew mining reward", "reward", toCoin(r), "total", toCoin(header.Supply))
				reward.Add(reward, r)
			}
		} else {

			if _, ok := FixHashes[headerInitialHash]; ok {
				header.Supply.Add(header.Supply, bigFix)
			}
		}

		state.AddBalance(header.Coinbase, reward)
	}
}

func toCoin(wei *big.Int) *big.Float {
	return new(big.Float).Quo(new(big.Float).SetInt(wei), new(big.Float).SetInt(big.NewInt(params.Cortex)))
}

func (cuckoo *Cuckoo) Sha3Solution(sol *types.BlockSolution) []byte {
	buf := make([]byte, 42*4)
	for i, s := range sol {
		binary.BigEndian.PutUint32(buf[i*4:], s)
	}
	ret := crypto.Keccak256(buf)
	return ret
}

func (cuckoo *Cuckoo) CuckooVerifyHeader(hash []byte, nonce uint64, sol *types.BlockSolution, targetDiff *big.Int) bool {
	return plugins.CuckooVerify_cuckaroo(&hash[0], nonce, *sol, cuckoo.Sha3Solution(sol), targetDiff)
}
