package cuckoo

import (
	crand "crypto/rand"
	"errors"
	"math"
	"math/big"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/core/types"
	//"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/ethereum/go-ethereum/log"
)

var (
	errNoMiningWork      = errors.New("no mining work available yet")
	errInvalidSealResult = errors.New("invalid or stale proof-of-work solution")
)

func (cuckoo *Cuckoo) Seal(chain consensus.ChainReader, block *types.Block, results chan<- *types.Block, stop <-chan struct{}) error {
	// If we're running a fake PoW, simply return a 0 nonce immediately
	if cuckoo.config.PowMode == ModeFake || cuckoo.config.PowMode == ModeFullFake {
		header := block.Header()
		header.Nonce, header.MixDigest = types.BlockNonce{}, common.Hash{}
		select {
		case results <- block.WithSeal(header):
		default:
			log.Warn("Sealing result is not read by miner", "mode", "fake", "sealhash", cuckoo.SealHash(block.Header()))
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
	threads := cuckoo.threads
	if cuckoo.rand == nil {
		seed, err := crand.Int(crand.Reader, big.NewInt(math.MaxInt64))
		if err != nil {
			cuckoo.lock.Unlock()
			return err
		}
		cuckoo.rand = rand.New(rand.NewSource(seed.Int64()))
	}
	cuckoo.lock.Unlock()
	if threads == 0 {
		threads = runtime.NumCPU()
	}
	if threads < 0 {
		threads = 0 // Allows disabling local mining without extra logic around local/remote
	}

	// Push new work to remote sealer
	if cuckoo.workCh != nil {
		cuckoo.workCh <- block
	}
	var pend sync.WaitGroup
	for i := 0; i < threads; i++ {
		pend.Add(1)
		go func(id int, nonce uint32) {
			defer pend.Done()
			cuckoo.mine(block, id, nonce, abort, cuckoo.resultCh)
		}(i, uint32(cuckoo.rand.Int31()))
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

func (cuckoo *Cuckoo) VerifyShare(block Block, shareDiff *big.Int, solution types.BlockSolution) (bool, bool, int64) {
	// For return arguments
	zeroHash := common.Hash{}

	blockDiff := block.Difficulty()
	if blockDiff.Cmp(common.Big0) == 0 {
		log.Debug("invalid block difficulty")
		return false, false, 0
	}

	if shareDiff.Cmp(common.Big0) == 0 {
		log.Debug("invalid share difficulty")
		return false, false, 0
	}

	// avoid mixdigest malleability as it's not included in a block's "hashNononce"
        if blkMix := block.MixDigest(); blkMix != zeroHash {
                return false, false, 0
        }

	//ok, mixDigest, result := cache.compute(uint64(dagSize), block.HashNoNonce(), block.Nonce())
	ok, result := cuckoo.VerifySolution(block.MixDigest().Bytes(), uint32(block.Nonce()), solution, *shareDiff)
	if !ok {
		return false, false, 0
	}

	// The actual check.
	blockTarget := new(big.Int).Div(maxUint256, blockDiff)
	shareTarget := new(big.Int).Div(maxUint256, shareDiff)
	actualDiff := new(big.Int).Div(maxUint256, result.Big())
	return result.Big().Cmp(shareTarget) <= 0, result.Big().Cmp(blockTarget) <= 0, actualDiff.Int64()
}

func (cuckoo *Cuckoo) VerifySolution(hash []byte, nonce uint32, solution types.BlockSolution, target big.Int) (bool, common.Hash) {
	var (
		result_hash [32]byte
		//result_len uint32
	)
	diff := target.Bytes()
	r := CuckooVerify(&hash[0], len(hash), uint32(nonce), &solution[0], &diff[0], &result_hash[0])
	/* r := C.CuckooVerify(
	(*C.char)(unsafe.Pointer(&hash[0])),
	C.uint(len(hash)),
	C.uint(uint32(nonce)),
	(*C.uint)(unsafe.Pointer(&solution[0])),
	//                        (*C.uint)(unsafe.Pointer(&result_len)),
	(*C.uchar)(unsafe.Pointer(&diff[0])),
	(*C.uchar)(unsafe.Pointer(&result_hash[0]))) */
	if r != 0 {
		return true, common.BytesToHash(result_hash[:])
	}
	return false, common.BytesToHash(result_hash[:])
}

func (cuckoo *Cuckoo) mine(block *types.Block, id int, seed uint32, abort chan struct{}, found chan *types.Block) {
	cuckoo.InitOnce()

	var (
		header = block.Header()
		hash   = cuckoo.SealHash(header).Bytes()
		target = new(big.Int).Div(maxUint256, header.Difficulty)

		result     types.BlockSolution
		result_len uint32
	)
	var (
		attempts = int32(0)
		nonce    = seed
	)

	logger := log.New("miner", id)
	logger.Trace("Started cuckoo search for new solution", "seed", seed)

search:
	for {
		select {
		case errc := <-cuckoo.exitCh:
			errc <- nil
			break search
		case <-abort:
			//Mining terminated, update stats and abort
			logger.Trace("Cuckoo solution search aborted", "attempts", nonce-seed)
			cuckoo.hashrate.Mark(int64(attempts))
			break search
		default:
			attempts++
			if attempts%(1<<15) == 0 {
				cuckoo.hashrate.Mark(int64(attempts))
				attempts = 0
			}

			var result_hash [32]byte
			diff := target.Bytes()
			// cuckoo.cMutex.Lock()
			r := CuckooSolve(&hash[0], len(hash), uint32(nonce), &result[0], &result_len, &diff[0], &result_hash[0])
			/* r := C.CuckooSolve(
			(*C.char)(unsafe.Pointer(&hash[0])),
			C.uint(len(hash)),
			C.uint(uint32(nonce)),
			(*C.uint)(unsafe.Pointer(&result[0])),
			(*C.uint)(unsafe.Pointer(&result_len)),
			(*C.uchar)(unsafe.Pointer(&diff[0])),
			(*C.uchar)(unsafe.Pointer(&result_hash[0]))) */
			if r == 0 {
				// cuckoo.cMutex.Unlock()
				nonce++
				continue
			}
			r = CuckooVerify(&hash[0], len(hash), uint32(nonce), &result[0], &diff[0], &result_hash[0])
			/* r = C.CuckooVerify(
			(*C.char)(unsafe.Pointer(&hash[0])),
			C.uint(len(hash)),
			C.uint(uint32(nonce)),
			(*C.uint)(unsafe.Pointer(&result[0])),
			(*C.uchar)(unsafe.Pointer(&diff[0])),
			(*C.uchar)(unsafe.Pointer(&result_hash[0]))) */
			// cuckoo.cMutex.Unlock()

			if r != 0 {
				// Correct solution found, create a new header with it
				header = types.CopyHeader(header)
				header.Nonce = types.EncodeNonce(uint64(nonce))
				header.Solution = result
				header.SolutionHash = result_hash

				select {
				case found <- block.WithSeal(header):
					logger.Trace("Cuckoo solution found and reported", "attempts", nonce-seed, "nonce", nonce)
				case <-abort:
					logger.Trace("Cuckoo solution found but discarded", "attempts", nonce-seed, "nonce", nonce)
				}
				break search
			}
			nonce++
		}
	}
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
	getWork := func() ([3]string, error) {
		var res [3]string
		if currentWork == nil {
			return res, errNoMiningWork
		}
		res[0] = cuckoo.SealHash(currentWork.Header()).Hex()
		res[1] = common.BytesToHash(SeedHash(currentWork.NumberU64())).Hex()

		// Calculate the "target" to be returned to the external sealer.
		n := big.NewInt(1)
		n.Lsh(n, 255)
		n.Div(n, currentWork.Difficulty())
		n.Lsh(n, 1)
		res[2] = common.BytesToHash(n.Bytes()).Hex()

		// Trace the seal work fetched by remote sealer.
		works[cuckoo.SealHash(currentWork.Header())] = currentWork
		return res, nil
	}

	// submitWork verifies the submitted pow solution, returning
	// whether the solution was accepted or not (not can be both a bad pow as well as
	// any other error, like no pending work or stale mining result).
	submitWork := func(nonce types.BlockNonce, mixDigest common.Hash, hash common.Hash) bool {
		// Make sure the work submitted is present
		block := works[hash]
		if block == nil {
			log.Info("Work submitted but none pending", "hash", hash)
			return false
		}

		// Verify the correctness of submitted result.
		header := block.Header()
		header.Nonce = nonce
		header.MixDigest = mixDigest
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
		select {
		case cuckoo.resultCh <- block.WithSeal(header):
			delete(works, hash)
			return true
		default:
			log.Info("Work submitted is stale", "hash", hash)
			return false
		}
	}

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case block := <-cuckoo.workCh:
			if currentWork != nil && block.ParentHash() != currentWork.ParentHash() {
				// Start new round mining, throw out all previous work.
				works = make(map[common.Hash]*types.Block)
			}
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
			if submitWork(result.nonce, result.mixDigest, result.hash) {
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

		case errc := <-cuckoo.exitCh:
			// Exit remote loop if cuckoo is closed and return relevant error.
			errc <- nil
			log.Trace("Cuckoo remote sealer is exiting")
			return
		}
	}
}
