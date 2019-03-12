// +build disable_miner

package cuckoo

import (
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
)

func (cuckoo *Cuckoo) Mine(block *types.Block, id int, seed uint64, abort chan struct{}, found chan *types.Block) {
	logger := log.New("miner", id)
	logger.Trace("not miner available for this version of testnet")

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
}
