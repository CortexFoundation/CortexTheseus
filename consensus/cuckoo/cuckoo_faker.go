package cuckoo

import (
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/consensus"
	"github.com/CortexFoundation/CortexTheseus/core/state"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"github.com/CortexFoundation/CortexTheseus/trie"
	"math/big"
	"time"
)

func NewFaker() *Cuckoo {
	return &Cuckoo{
		config: Config{
			PowMode: ModeFake,
		},
	}
}

func NewFakeFailer(number uint64) *Cuckoo {
	return &Cuckoo{
		config: Config{
			PowMode: ModeFake,
		},
		fakeFail: number,
	}
}

func NewFakeDelayer(seconds time.Duration) *Cuckoo {
	return &Cuckoo{
		config: Config{
			PowMode: ModeFake,
		},
	}
}

func NewFullFaker() *Cuckoo {
	return &Cuckoo{
		config: Config{
			PowMode: ModeFullFake,
		},
	}
}

type CuckooFake struct {
}

func (cuckoo *CuckooFake) APIs(chain consensus.ChainReader) []rpc.API {
	return []rpc.API{}
}

func (cuckoo *CuckooFake) Author(header *types.Header) (common.Address, error) {
	return header.Coinbase, nil
}

func (cuckoo *CuckooFake) CalcDifficulty(chain consensus.ChainReader, time uint64, parent *types.Header) *big.Int {
	return big.NewInt(0)
}

func (cuckoo *CuckooFake) Close() error {
	return nil
}

func (cuckoo *CuckooFake) Finalize(chain consensus.ChainReader, header *types.Header, state *state.StateDB, txs []*types.Transaction, uncles []*types.Header, receipts []*types.Receipt) (*types.Block, error) {
	return types.NewBlock(header, txs, uncles, receipts, new(trie.Trie)), nil
}

func (cuckoo *CuckooFake) FinalizeWithoutParent(chain consensus.ChainReader, header *types.Header, state *state.StateDB, txs []*types.Transaction, uncles []*types.Header, receipts []*types.Receipt) (*types.Block, error) {
	return types.NewBlock(header, txs, uncles, receipts, new(trie.Trie)), nil
}

func (cuckoo *CuckooFake) Prepare(chain consensus.ChainReader, header *types.Header) error {
	return nil
}

func (cuckoo *CuckooFake) Seal(chain consensus.ChainReader, block *types.Block, results chan<- *types.Block, stop <-chan struct{}) error {
	return nil
}

func (cuckoo *CuckooFake) SealHash(header *types.Header) (hash common.Hash) {
	return common.Hash{}
}

func (cuckoo *CuckooFake) VerifyHeader(chain consensus.ChainReader, header *types.Header, seal bool) error {
	return nil
}

func (cuckoo *CuckooFake) VerifyHeaders(chain consensus.ChainReader, headers []*types.Header, seals []bool) (chan<- struct{}, <-chan error) {
	abort := make(chan struct{})
	errorsOut := make(chan error, len(headers))
	go func() {
		for _, _ = range headers {
			errorsOut <- nil
		}
	}()
	return abort, errorsOut
}

func (cuckoo *CuckooFake) VerifySeal(chain consensus.ChainReader, header *types.Header) error {
	return nil
}

func (cuckoo *CuckooFake) VerifyUncles(chain consensus.ChainReader, block *types.Block) error {
	return nil
}
