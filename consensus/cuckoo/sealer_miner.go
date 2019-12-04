// +build !disable_miner

package cuckoo

import (
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/log"
	"math/big"
)

func (cuckoo *Cuckoo) Mine(block *types.Block, id int, seed uint64, abort chan struct{}, found chan *types.Block) (err error) {
	err = cuckoo.InitPlugin()
	if err != nil {
		log.Error("cuckoo init error", "error", err)
		return err
	}

	var (
		header = block.Header()
		hash   = cuckoo.SealHash(header).Bytes()
		target = new(big.Int).Div(maxUint256, header.Difficulty)

		result types.BlockSolution
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

			ret := false
			if cuckoo.config.Algorithm == CUCKAROO {
				m, err := cuckoo.minerPlugin.Lookup("CuckooFindSolutions")
				if err != nil {
					return err
				}
				r, res := m.(func([]byte, uint64) (uint32, [][]uint32))(hash, nonce)
				if r == 0 {
					nonce++
					continue
				}
				copy(result[:], res[0][0:len(res[0])])

				m, err = cuckoo.minerPlugin.Lookup("CuckooVerify_cuckaroo")
				if err != nil {
					return err
				}
				ret = m.(func(*byte, uint64, types.BlockSolution, []byte, *big.Int) bool)(&hash[0], nonce, result, cuckoo.Sha3Solution(&result), target)
			} else {
				m, err := cuckoo.xcortexPlugin.Lookup("Verify")
				if err != nil {
					log.Error("cuckoo", "lookup xcortex verify error", err)
					return err
				}
				ret = m.(func(*byte, uint64, string) bool)(&hash[0], nonce, string(common.ToHex(target.Bytes())))
			}
			if ret {
				// Correct solution found, create a new header with it
				header = types.CopyHeader(header)
				header.Nonce = types.EncodeNonce(uint64(nonce))
				header.Solution = result

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
	return nil
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
