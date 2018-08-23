package cuckoo

/*
#cgo LDFLAGS:  -lstdc++ -lgominer
#include "gominer.h"
*/
import "C"
import (
	crand "crypto/rand"
	"errors"
	"math"
	"math/big"
	"math/rand"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
)

var (
	errNoMiningWork      = errors.New("no mining work available yet")
	errInvalidSealResult = errors.New("invalid or stale proof-of-work solution")
)

func (cuckoo *Cuckoo) Seal(chain consensus.ChainReader, block *types.Block, stop <-chan struct{}) (*types.Block, error) {
	// If we're running a fake PoW, simply return a 0 nonce immediately
	if cuckoo.config.PowMode == ModeFake || cuckoo.config.PowMode == ModeFullFake {
		header := block.Header()
		header.Nonce, header.MixDigest = types.BlockNonce{}, common.Hash{}
		return block.WithSeal(header), nil
	}
	// If we're running a shared PoW, delegate sealing to it
	if cuckoo.shared != nil {
		return cuckoo.shared.Seal(chain, block, stop)
	}
	// Create a runner and the multiple search threads it directs
	abort := make(chan struct{})
	cuckoo.lock.Lock()
	threads := cuckoo.threads
	if cuckoo.rand == nil {
		seed, err := crand.Int(crand.Reader, big.NewInt(math.MaxInt64))
		if err != nil {
			cuckoo.lock.Unlock()
			return nil, err
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
	for i := 0; i < 1; i++ {
		pend.Add(1)
		go func(id int, nonce uint32) {
			defer pend.Done()
			cuckoo.mine(block, id, nonce, abort, cuckoo.resultCh)
		}(i, uint32(cuckoo.rand.Int31()))
	}
	// Wait until sealing is terminated or a nonce is found
	var result *types.Block
	select {
	case <-stop:
		// Outside abort, stop all miner threads
		close(abort)
	case result = <-cuckoo.resultCh:
		// One of the threads found a block, abort all others
		close(abort)
	case <-cuckoo.update:
		// Thread count was changed on user request, restart
		close(abort)
		pend.Wait()
		return cuckoo.Seal(chain, block, stop)
	}
	// Wait for all miners to terminate and return the block
	pend.Wait()
	return result, nil
}

func (cuckoo *Cuckoo) mine(block *types.Block, id int, seed uint32, abort chan struct{}, found chan *types.Block) {
	var (
		header = block.Header()
		hash   = header.HashNoNonce().Bytes()
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
			// fmt.Println("hash", hash)
			var result_hash [32]byte
			diff := target.Bytes()
			// fmt.Println("diff", header.Difficulty)
			// fmt.Println("target", diff)
			cuckoo.cMutex.Lock()
			r := C.CuckooSolve(
				(*C.char)(unsafe.Pointer(&hash[0])),
				C.uint(len(hash)),
				C.uint(uint32(nonce)),
				(*C.uint)(unsafe.Pointer(&result[0])),
				(*C.uint)(unsafe.Pointer(&result_len)),
				(*C.uchar)(unsafe.Pointer(&diff[0])),
				(*C.uchar)(unsafe.Pointer(&result_hash[0])))
			// fmt.Println("target", diff)
			if byte(r) == 0 {
				cuckoo.cMutex.Unlock()
				nonce++
				continue
			}
			// fmt.Println("result", result)
			r = C.CuckooVerify(
				(*C.char)(unsafe.Pointer(&hash[0])),
				C.uint(len(hash)),
				C.uint(uint32(nonce)),
				(*C.uint)(unsafe.Pointer(&result[0])),
				(*C.uchar)(unsafe.Pointer(&diff[0])),
				(*C.uchar)(unsafe.Pointer(&result_hash[0])))
			// fmt.Println(result)
			cuckoo.cMutex.Unlock()

			if byte(r) != 0 {
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
		res[0] = currentWork.HashNoNonce().Hex()
		res[1] = common.BytesToHash(SeedHash(currentWork.NumberU64())).Hex()

		// Calculate the "target" to be returned to the external sealer.
		n := big.NewInt(1)
		n.Lsh(n, 255)
		n.Div(n, currentWork.Difficulty())
		n.Lsh(n, 1)
		res[2] = common.BytesToHash(n.Bytes()).Hex()

		// Trace the seal work fetched by remote sealer.
		works[currentWork.HashNoNonce()] = currentWork
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
