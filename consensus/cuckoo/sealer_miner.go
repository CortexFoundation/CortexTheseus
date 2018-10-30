// +build !disable_miner

package cuckoo

import (
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
	"math/big"
)

func (cuckoo *Cuckoo) Mine(block *types.Block, id int, seed uint64, abort chan struct{}, found chan *types.Block) {
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
			r := CuckooSolve(&hash[0], len(hash), (nonce), &result[0], &result_len, &diff[0], &result_hash[0])
			if r == 0 {
				nonce++
				continue
			}
			r = CuckooVerify(&hash[0], len(hash), (nonce), &result[0], &diff[0], &result_hash[0])

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

func (cuckoo *Cuckoo) SetThreads(threads int) {
	cuckoo.lock.Lock()
	defer cuckoo.lock.Unlock()

	// If we're running a shared PoW, set the thread count on that instead
	if cuckoo.shared != nil {
		cuckoo.shared.SetThreads(threads)
		return
	}
	// Update the threads and ping any running seal to pull in any changes
	cuckoo.threads = threads
	select {
	case cuckoo.update <- struct{}{}:
	default:
	}
}
