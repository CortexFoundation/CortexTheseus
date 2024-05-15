// Copyright 2019 The go-ethereum Authors
// This file is part of the CortexFoundation library.
//
// The CortexFoundation library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexFoundation library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the CortexFoundation library. If not, see <http://www.gnu.org/licenses/>.

package core

import (
	"fmt"
	"math/big"
	"testing"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/consensus"
	"github.com/CortexFoundation/CortexTheseus/core/rawdb"
	"github.com/CortexFoundation/CortexTheseus/core/state"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/core/vm"
	"github.com/CortexFoundation/CortexTheseus/crypto"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"github.com/CortexFoundation/CortexTheseus/trie"
)

type CuckooFakeForTest struct {
}

func (cuckoo *CuckooFakeForTest) APIs(chain consensus.ChainHeaderReader) []rpc.API {
	return []rpc.API{}
}

func (cuckoo *CuckooFakeForTest) Author(header *types.Header) (common.Address, error) {
	return header.Coinbase, nil
}

func (cuckoo *CuckooFakeForTest) CalcDifficulty(chain consensus.ChainHeaderReader, time uint64, parent *types.Header) *big.Int {
	return big.NewInt(0)
}

func (cuckoo *CuckooFakeForTest) Close() error {
	return nil
}

// Finalize implements consensus.Engine, ensuring no uncles are set, nor block
// rewards given.
func (cuckoo *CuckooFakeForTest) Finalize(chain consensus.ChainHeaderReader, header *types.Header, state *state.StateDB, txs []*types.Transaction, uncles []*types.Header) error {
	// No block rewards in PoA, so the state remains as is and uncles are dropped
	header.Root = state.IntermediateRoot(chain.Config().IsEIP158(header.Number))
	header.UncleHash = types.CalcUncleHash(nil)
	return nil
}

func (cuckoo *CuckooFakeForTest) FinalizeAndAssemble(chain consensus.ChainHeaderReader, header *types.Header, state *state.StateDB, txs []*types.Transaction, uncles []*types.Header, receipts []*types.Receipt) (*types.Block, error) {
	// No block rewards in PoA, so the state remains as is and uncles are dropped
	header.Root = state.IntermediateRoot(chain.Config().IsEIP158(header.Number))
	header.UncleHash = types.CalcUncleHash(nil)
	return types.NewBlock(header, &types.Body{Transactions: txs, Uncles: uncles}, receipts, trie.NewStackTrie(nil)), nil
}

func (cuckoo *CuckooFakeForTest) FinalizeWithoutParent(chain consensus.ChainHeaderReader, header *types.Header, state *state.StateDB, txs []*types.Transaction, uncles []*types.Header, receipts []*types.Receipt) (*types.Block, error) {
	// No block rewards in PoA, so the state remains as is and uncles are dropped
	header.Root = state.IntermediateRoot(chain.Config().IsEIP158(header.Number))
	header.UncleHash = types.CalcUncleHash(nil)
	return types.NewBlock(header, &types.Body{Transactions: txs, Uncles: uncles}, receipts, trie.NewStackTrie(nil)), nil
}

func (cuckoo *CuckooFakeForTest) Prepare(chain consensus.ChainHeaderReader, header *types.Header) error {
	return nil
}

func (cuckoo *CuckooFakeForTest) Seal(chain consensus.ChainHeaderReader, block *types.Block, results chan<- *types.Block, stop <-chan struct{}) error {
	return nil
}

func (cuckoo *CuckooFakeForTest) SealHash(header *types.Header) (hash common.Hash) {
	return common.Hash{}
}

func (cuckoo *CuckooFakeForTest) VerifyHeader(chain consensus.ChainHeaderReader, header *types.Header, seal bool) error {
	return nil
}

func (cuckoo *CuckooFakeForTest) VerifyHeaders(chain consensus.ChainHeaderReader, headers []*types.Header, seals []bool) (chan<- struct{}, <-chan error) {
	abort := make(chan struct{})
	errorsOut := make(chan error, len(headers))
	go func() {
		for range headers {
			errorsOut <- nil
		}
	}()
	return abort, errorsOut
}

func (cuckoo *CuckooFakeForTest) VerifySeal(chain consensus.ChainHeaderReader, header *types.Header) error {
	return nil
}

func (cuckoo *CuckooFakeForTest) VerifyUncles(chain consensus.ChainReader, block *types.Block) error {
	return nil
}

func TestExampleGenerateChain(t *testing.T) {
	var (
		key1, _ = crypto.HexToECDSA("b71c71a67e1177ad4e901695e1b4b9ee17ae16c6668d313eac2f96dbcda3f291")
		key2, _ = crypto.HexToECDSA("8a1f9a8f95be41cd7ccb6168179afb4504aefe388d1e14474d32c45c72ce7b7a")
		key3, _ = crypto.HexToECDSA("49a7b37aa6f6645917e7b807e9d1c00d4fa71f18343b0d4122a4d2df64dd6fee")
		addr1   = crypto.PubkeyToAddress(key1.PublicKey)
		addr2   = crypto.PubkeyToAddress(key2.PublicKey)
		addr3   = crypto.PubkeyToAddress(key3.PublicKey)
		db      = rawdb.NewMemoryDatabase()
	)

	// Ensure that key1 has some funds in the genesis block.
	gspec := &Genesis{
		Config: &params.ChainConfig{ChainID: big0, HomesteadBlock: new(big.Int)},
		Alloc:  GenesisAlloc{addr1: {Balance: big.NewInt(1000000)}},
	}
	genesis := gspec.MustCommit(db)

	// This call generates a chain of 5 blocks. The function runs for
	// each block and adds different features to gen based on the
	// block index.
	signer := types.HomesteadSigner{}
	chain, _ := GenerateChain(gspec.Config, genesis, &CuckooFakeForTest{}, db, 5, func(i int, gen *BlockGen) {
		switch i {
		case 0:
			// In block 1, addr1 sends addr2 some ether.
			tx, _ := types.SignTx(types.NewTransaction(gen.TxNonce(addr1), addr2, big.NewInt(10000), params.TxGas, nil, nil), signer, key1)
			gen.AddTx(tx)
			//case 1:
			//	// In block 2, addr1 sends some more ether to addr2.
			//	// addr2 passes it on to addr3.
			//	tx1, _ := types.SignTx(types.NewTransaction(gen.TxNonce(addr1), addr2, big.NewInt(1000), params.TxGas, nil, nil), signer, key1)
			//	gen.AddTx(tx1)
			//	tx2, _ := types.SignTx(types.NewTransaction(gen.TxNonce(addr2), addr3, big.NewInt(1000), params.TxGas, nil, nil), signer, key2)
			//	gen.AddTx(tx2)
			//case 2:
			//	// Block 3 is empty but was mined by addr3.
			//	gen.SetCoinbase(addr3)
			//	gen.SetExtra([]byte("yeehaw"))
			//case 3:
			//	// Block 4 includes blocks 2 and 3 as uncle headers (with modified extra data).
			//	b2 := gen.PrevBlock(1).Header()
			//	b2.Extra = []byte("foo")
			//	gen.AddUncle(b2)
			//	b3 := gen.PrevBlock(2).Header()
			//	b3.Extra = []byte("foo")
			//	gen.AddUncle(b3)
		}
	})

	// Import the chain. This runs all block validation rules.
	blockchain, _ := NewBlockChain(db, nil, gspec.Config, &CuckooFakeForTest{}, vm.Config{}, nil, nil)
	defer blockchain.Stop()

	if i, err := blockchain.InsertChain(chain); err != nil {
		fmt.Printf("insert error (block %d): %v\n", chain[i].NumberU64(), err)
		return
	}

	state, _ := blockchain.State()
	fmt.Printf("last block: #%d\n", blockchain.CurrentBlock().Number())
	fmt.Println("balance of addr1:", state.GetBalance(addr1))
	fmt.Println("balance of addr2:", state.GetBalance(addr2))
	fmt.Println("balance of addr3:", state.GetBalance(addr3))
	// Output:
	// last block: #5
	// balance of addr1: 989000
	// balance of addr2: 10000
	// balance of addr3: 19687500000000001000
}
