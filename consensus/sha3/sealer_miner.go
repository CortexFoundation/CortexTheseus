//go:build !disable_miner
// +build !disable_miner

package sha3

import (
	"math/big"

	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/log"
)

func (sha3 *SHAThree) Mine(block *types.Block, id int, seed uint64, abort chan struct{}, found chan *types.Block) (err error) {
	var (
		header = block.Header()
		hash   = sha3.SealHash(header).Bytes()
		target = new(big.Int).Div(maxUint256, header.Difficulty)

		result types.BlockSolution
	)
	var (
		attempts = int32(0x0)
		nonce    = seed
	)

	logger := log.New("miner", id)
	logger.Trace("Started sha3 search for new solution", "seed", seed)

search:
	for {
		select {
		case <-sha3.exitCh:
			return
		case <-abort:
			//Mining terminated, update stats and abort
			logger.Trace("SHAThree solution search aborted", "attempts", nonce-seed)
			sha3.hashrate.Mark(int64(attempts))
			break search
		default:
			attempts++
			if attempts%(1<<15) == 0 {
				sha3.hashrate.Mark(int64(attempts))
				attempts = 0
			}

			r, res := sha3.GenSha3Solution(hash, nonce)

			if r == 0 {
				nonce++
				continue
			}
			copy(result[:], res[0][0:len(res[0])])

			// Check solution
			if sha3.SHAThreeVerifyHeader(hash, nonce, &result, target) {
				// Correct solution found, create a new header with it
				header = types.CopyHeader(header)
				header.Nonce = types.EncodeNonce(uint64(nonce))
				header.Solution = result

				select {
				case found <- block.WithSeal(header):
					logger.Trace("SHAThree solution found and reported", "attempts", nonce-seed, "nonce", nonce)
				case <-abort:
					logger.Trace("SHAThree solution found but discarded", "attempts", nonce-seed, "nonce", nonce)
				}
				break search
			}

			//next loop
			nonce++
		}
	}
	return nil
}

func (sha3 *SHAThree) SetThreads(threads int) {
	// Enable CPU Mining
	threads = 1
	log.Info("#### SHAThree SetThreads", "threads set_to", threads)
	sha3.lock.Lock()
	defer sha3.lock.Unlock()

	// If we're running a shared PoW, set the thread count on that instead
	if sha3.shared != nil {
		sha3.shared.SetThreads(threads)
		return
	}
	// Update the threads and ping any running seal to pull in any changes
	sha3.threads = threads
	select {
	case sha3.update <- struct{}{}:
	default:
	}
}
