package xeth

import (
	"github.com/ethereum/ethash"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/miner"
)

type RemoteAgent struct {
	work        *types.Block
	currentWork *types.Block

	quit     chan struct{}
	workCh   chan *types.Block
	returnCh chan<- miner.Work
}

func NewRemoteAgent() *RemoteAgent {
	agent := &RemoteAgent{}
	go agent.run()

	return agent
}

func (a *RemoteAgent) Work() chan<- *types.Block {
	return a.workCh
}

func (a *RemoteAgent) SetWorkCh(returnCh chan<- miner.Work) {
	a.returnCh = returnCh
}

func (a *RemoteAgent) Start() {
	a.quit = make(chan struct{})
	a.workCh = make(chan *types.Block, 1)
}

func (a *RemoteAgent) Stop() {
	close(a.quit)
	close(a.workCh)
}

func (a *RemoteAgent) GetHashRate() int64 { return 0 }

func (a *RemoteAgent) run() {
out:
	for {
		select {
		case <-a.quit:
			break out
		case work := <-a.workCh:
			a.work = work
		}
	}
}

func (a *RemoteAgent) GetWork() [3]string {
	var res [3]string

	// XXX Wait here until work != nil ?
	if a.work != nil {
		res[0] = a.work.HashNoNonce().Hex()
		seedHash, _ := ethash.GetSeedHash(a.currentWork.NumberU64())
		res[1] = common.Bytes2Hex(seedHash)
		res[2] = common.Bytes2Hex(a.work.Difficulty().Bytes())
	}

	return res
}

func (a *RemoteAgent) SubmitWork(nonce uint64, mixDigest, seedHash common.Hash) bool {
	// Return true or false, but does not indicate if the PoW was correct

	// Make sure the external miner was working on the right hash
	if a.currentWork != nil && a.work != nil && a.currentWork.Hash() == a.work.Hash() {
		a.returnCh <- miner.Work{a.currentWork.Number().Uint64(), nonce, mixDigest.Bytes(), seedHash.Bytes()}
		return true
	}

	return false
}
