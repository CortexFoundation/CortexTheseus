package cuckoo

/*
#cgo LDFLAGS:  -lstdc++ -lgominer
#include "gominer.h"
*/
import "C"
import (
	crand "crypto/rand"
	"fmt"
	"math"
	"math/big"
	"math/rand"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
)

func (cuckoo *Cuckoo) Seal(chain consensus.ChainReader, block *types.Block, stop <-chan struct{}) (*types.Block, error) {
	if cuckoo.config.PowMode == ModeFake || cuckoo.config.PowMode == ModeFullFake {
		header := block.Header()
		header.Nonce = types.BlockNonce{}
		return block.WithSeal(header), nil
	}

	abort := make(chan struct{})
	found := make(chan *types.Block)

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

	var pend sync.WaitGroup
	for i := 0; i < threads; i++ {
		pend.Add(1)
		go func(id int, nonce uint32) {
			defer pend.Done()
			cuckoo.mine(block, id, nonce, abort, found)
		}(i, uint32(cuckoo.rand.Int31()))
	}

	var result *types.Block
	select {
	case <-stop:
		close(abort)
	case result = <-found:
		close(abort)
	case <-cuckoo.update:
		close(abort)
		pend.Wait()
		return cuckoo.Seal(chain, block, stop)
	}

	pend.Wait()
	return result, nil
}

func (cuckoo *Cuckoo) mine(block *types.Block, id int, seed uint32, abort chan struct{}, found chan *types.Block) {
	var (
		header = block.Header()
		hash   = header.HashNoNonce().Bytes()

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

			fmt.Println("hash", hexutil.Bytes(hash[:]).String())
			fmt.Println("hash", hash)
			var result_hash [32]byte
			diff := difficultyTarget(block.Header().Difficulty).Bytes()
			fmt.Println("diff", block.Header().Difficulty)
			r := C.CuckooSolve(
				(*C.char)(unsafe.Pointer(&hash[0])),
				C.uint(len(hash)),
				C.uint(uint32(nonce)),
				(*C.uint)(unsafe.Pointer(&result[0])),
				(*C.uint)(unsafe.Pointer(&result_len)),
				(*C.uchar)(unsafe.Pointer(&diff[0])),
				(*C.uchar)(unsafe.Pointer(&result_hash[0])))
			fmt.Println("diff", diff)
			if byte(r) == 0 {
				nonce++
				continue
			}
			fmt.Println("result", result)
			r = C.CuckooVerify(
				(*C.char)(unsafe.Pointer(&hash[0])),
				C.uint(len(hash)),
				C.uint(uint32(nonce)),
				(*C.uint)(unsafe.Pointer(&result[0])),
				(*C.uchar)(unsafe.Pointer(&diff[0])),
				(*C.uchar)(unsafe.Pointer(&result_hash[0])))
			fmt.Println(result)

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
