package cuckoo

import (
	crand "crypto/rand"
	"errors"
	"math"
	"math/big"
	"math/rand"
	"sync"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"

	//"runtime"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/consensus"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/log"
)

var (
	errNoMiningWork      = errors.New("no mining work available yet")
	errInvalidSealResult = errors.New("invalid or stale proof-of-work solution")
)

const (
	// staleThreshold is the maximum depth of the acceptable stale but valid solution.
	staleThreshold = 7
)

func (cuckoo *Cuckoo) Seal(chain consensus.ChainHeaderReader, block *types.Block, results chan<- *types.Block, stop <-chan struct{}) error {
	// If we're running a fake PoW, simply return a 0 nonce immediately
	if cuckoo.config.PowMode == ModeFake || cuckoo.config.PowMode == ModeFullFake {
		header := block.Header()
		header.Nonce, header.MixDigest = types.BlockNonce{}, common.Hash{}
		select {
		case results <- block.WithSeal(header):
		default:
			//log.Warn("Sealing result is not read by miner", "mode", "fake", "sealhash", cuckoo.SealHash(block.Header()))
		}
		return nil
	}
	// If we're running a shared PoW, delegate sealing to it
	if cuckoo.shared != nil {
		return cuckoo.shared.Seal(chain, block, results, stop)
	}
	// Create a runner and the multiple search threads it directs
	abort := make(chan struct{})
	cuckoo.lock.Lock()
	if cuckoo.rand == nil {
		seed, err := crand.Int(crand.Reader, big.NewInt(math.MaxInt64))
		if err != nil {
			cuckoo.lock.Unlock()
			return err
		}
		cuckoo.rand = rand.New(rand.NewSource(seed.Int64()))
	}
	cuckoo.lock.Unlock()

	// Push new work to remote sealer
	if cuckoo.workCh != nil {
		cuckoo.workCh <- block
	}

	//if !cuckoo.config.UseCuda || cuckoo.threads <= 0 {
	//	return nil
	//}

	//err := cuckoo.InitOnce()
	//if err != nil {
	//	log.Error("cuckoo init error", "error", err)
	//	return err
	//}

	var pend sync.WaitGroup
	for i := 0; i < cuckoo.threads; i++ {
		pend.Add(1)
		var err error
		go func(id int, nonce uint64, err *error) {
			defer pend.Done()
			cuckoo.lock.Lock()
			*err = cuckoo.Mine(block, id, nonce, abort, cuckoo.resultCh)
			cuckoo.lock.Unlock()
		}(i, uint64(cuckoo.rand.Int63()), &err)
		if err != nil {
			return err
		}
	}
	// Wait until sealing is terminated or a nonce is found
	go func() {
		var result *types.Block
		select {
		case <-stop:
			// Outside abort, stop all miner threads
			close(abort)
		case result = <-cuckoo.resultCh:
			// One of the threads found a block, abort all others
			select {
			case results <- result:
			default:
				log.Warn("Sealing result is not read by miner", "mode", "local", "sealhash", cuckoo.SealHash(block.Header()))
			}
			close(abort)
		case <-cuckoo.update:
			// Thread count was changed on user request, restart
			close(abort)
			// pend.Wait()
			if err := cuckoo.Seal(chain, block, results, stop); err != nil {
				log.Error("Failed to restart sealing after update", "err", err)
			}
			// return cuckoo.Seal(chain, block, stop)
		}
		// Wait for all miners to terminate and return the block
		pend.Wait()
	}()
	return nil
	// return result, nil
}

func (cuckoo *Cuckoo) Verify(block Block, hashNoNonce common.Hash, shareDiff *big.Int, solution *types.BlockSolution) (bool, bool, int64) {

	blockDiff := block.Difficulty()
	if blockDiff.Sign() == 0 {
		log.Info("invalid block difficulty")
		return false, false, 0
	}

	if shareDiff.Sign() == 0 {
		log.Info("invalid share difficulty")
		return false, false, 0
	}
	//fmt.Println(hashNoNonce, solution, block.NumberU64(), blockDiff, shareDiff, block.Nonce())
	targetDiff := new(big.Int).Div(maxUint256, shareDiff)
	ok := cuckoo.CuckooVerifyHeader(hashNoNonce.Bytes(), block.Nonce(), solution, targetDiff)
	if !ok {
		//fmt.Println("invalid solution")
		return false, false, 0
	}
	sha3Hash := common.BytesToHash(cuckoo.Sha3Solution(solution))
	blockTarget := new(big.Int).Div(maxUint256, blockDiff)
	shareTarget := new(big.Int).Div(maxUint256, shareDiff)
	actualDiff := new(big.Int).Div(maxUint256, sha3Hash.Big())
	// fmt.Printf("%v\n%v\n%v\n",
	// 	common.BytesToHash(blockTarget.Bytes()).Hex(),
	// 	common.BytesToHash(shareTarget.Bytes()).Hex(),
	// 	sha3Hash.Hex())
	return sha3Hash.Big().Cmp(shareTarget) <= 0, sha3Hash.Big().Cmp(blockTarget) <= 0, actualDiff.Int64()
}

func (cuckoo *Cuckoo) VerifySolution(hash []byte, nonce uint32, solution *types.BlockSolution, target big.Int) (bool, common.Hash) {
	return false, common.Hash{}
}

// remote starts a standalone goroutine to handle remote mining related stuff.
func (cuckoo *Cuckoo) remote() {
	var (
		works       = make(map[common.Hash]*types.Block)
		rates       = make(map[common.Hash]hashrate)
		currentWork *types.Block
	)

	// getWork returns a work package for external miner.
	//
	// The work package consists of 3 strings:
	//   result[0], 32 bytes hex encoded current block header pow-hash
	//   result[1], 32 bytes hex encoded seed hash used for DAG
	//   result[2], 32 bytes hex encoded boundary condition ("target"), 2^256/difficulty
	getWork := func() ([4]string, error) {
		var res [4]string
		if currentWork == nil {
			return res, errNoMiningWork
		}
		sealhash := cuckoo.SealHash(currentWork.Header())
		res[0] = sealhash.Hex()
		//res[1] = common.BytesToHash(SeedHash(currentWork.NumberU64())).Hex()
		res[1] = "0x0000000000000000000000000000000000000000000000000000000000000000"

		// Calculate the "target" to be returned to the external sealer.
		//n := big.NewInt(1)
		//n.Lsh(n, 255)
		//n.Div(n, currentWork.Difficulty())
		//n.Lsh(n, 1)
		//res[2] = common.BytesToHash(n.Bytes()).Hex()

		res[2] = common.BytesToHash(new(big.Int).Div(two256, currentWork.Difficulty()).Bytes()).Hex()
		res[3] = hexutil.EncodeBig(currentWork.Number())
		// Trace the seal work fetched by remote sealer.
		works[sealhash] = currentWork
		return res, nil
	}

	// submitWork verifies the submitted pow solution, returning
	// whether the solution was accepted or not (not can be both a bad pow as well as
	// any other error, like no pending work or stale mining result).
	// submitWork := func(nonce types.BlockNonce, mixDigest common.Hash, hash common.Hash, sol types.BlockSolution) bool {
	submitWork := func(nonce types.BlockNonce, hash common.Hash, sol types.BlockSolution) bool {
		if currentWork == nil {
			//log.Error("Pending work without block", "sealhash", sealhash)
			return false
		}
		// Make sure the work submitted is present
		block := works[hash]
		if block == nil {
			log.Info("Work submitted but none pending", "hash", hash)
			return false
		}

		// Verify the correctness of submitted result.
		header := block.Header()
		header.Nonce = nonce
		//header.MixDigest = mixDigest
		header.Solution = sol
		if err := cuckoo.VerifySeal(nil, header); err != nil {
			log.Warn("Invalid proof-of-work submitted", "hash", hash, "err", err)
			return false
		}

		// Make sure the result channel is created.
		if cuckoo.resultCh == nil {
			log.Warn("Cuckoo result channel is empty, submitted mining result is rejected")
			return false
		}

		// Solutions seems to be valid, return to the miner and notify acceptance.
		solution := block.WithSeal(header)
		if solution.NumberU64()+staleThreshold > currentWork.NumberU64() {
			// Solutions seems to be valid, return to the miner and notify acceptance.
			select {
			case cuckoo.resultCh <- solution:
				//delete(works, hash)
				return true
			default:
				log.Warn("Sealing result is not read by miner", "mode", "remote", "sealhash", hash)
				return false
			}
		}
		//log.Warn("Work submitted is too old", "number", solution.NumberU64(), "sealhash", sealhash, "hash", solution.Hash())
		return false
	}

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-cuckoo.exitCh:
			// Exit remote loop if cuckoo is closed and return relevant error.
			log.Warn("Cuckoo cycle remote sealer is exiting")
			return
		case block := <-cuckoo.workCh:
			//if currentWork != nil && block.ParentHash() != currentWork.ParentHash() {
			// Start new round mining, throw out all previous work.
			//	works = make(map[common.Hash]*types.Block)
			//}
			// Update current work with new received block.
			// Note same work can be past twice, happens when changing CPU threads.
			currentWork = block

		case work := <-cuckoo.fetchWorkCh:
			// Return current mining work to remote miner.
			miningWork, err := getWork()
			if err != nil {
				work.errc <- err
			} else {
				work.res <- miningWork
			}

		case result := <-cuckoo.submitWorkCh:
			// Verify submitted PoW solution based on maintained mining blocks.
			//if submitWork(result.nonce, result.mixDigest, result.hash, result.solution) {
			if submitWork(result.nonce, result.hash, result.solution) {
				result.errc <- nil
			} else {
				result.errc <- errInvalidSealResult
			}

		case result := <-cuckoo.submitRateCh:
			// Trace remote sealer's hash rate by submitted value.
			rates[result.id] = hashrate{rate: result.rate, ping: time.Now()}
			close(result.done)

		case req := <-cuckoo.fetchRateCh:
			// Gather all hash rate submitted by remote sealer.
			var total uint64
			for _, rate := range rates {
				// this could overflow
				total += rate.rate
			}
			req <- total

		case <-ticker.C:
			// Clear stale submitted hash rate.
			for id, rate := range rates {
				if time.Since(rate.ping) > 10*time.Second {
					delete(rates, id)
				}
			}

			if currentWork != nil {
				for hash, block := range works {
					if block.NumberU64()+staleThreshold <= currentWork.NumberU64() {
						delete(works, hash)
						log.Debug("Stale work removed", "hash", hash, "number", block.NumberU64(), "current", currentWork.NumberU64())
					}
				}
			}
		}
	}
}
