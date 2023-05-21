// Copyright 2019 The CortexTheseus Authors
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

package ctxc

import (
	"context"
	"errors"
	"fmt"
	"math/big"

	"github.com/CortexFoundation/CortexTheseus"
	"github.com/CortexFoundation/CortexTheseus/accounts"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/consensus"
	"github.com/CortexFoundation/CortexTheseus/core"
	"github.com/CortexFoundation/CortexTheseus/core/bloombits"
	"github.com/CortexFoundation/CortexTheseus/core/rawdb"
	"github.com/CortexFoundation/CortexTheseus/core/state"
	"github.com/CortexFoundation/CortexTheseus/core/txpool"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/core/vm"
	"github.com/CortexFoundation/CortexTheseus/ctxc/gasprice"
	"github.com/CortexFoundation/CortexTheseus/ctxc/tracers"
	"github.com/CortexFoundation/CortexTheseus/ctxcdb"
	"github.com/CortexFoundation/CortexTheseus/event"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/CortexTheseus/rpc"
)

// CortexAPIBackend implements ctxcapi.Backend for full nodes
type CortexAPIBackend struct {
	allowUnprotectedTxs bool
	ctxc                *Cortex
	gpo                 *gasprice.Oracle
}

// ChainConfig returns the active chain configuration.
func (b *CortexAPIBackend) ChainConfig() *params.ChainConfig {
	return b.ctxc.chainConfig
}

func (b *CortexAPIBackend) CurrentBlock() *types.Block {
	return b.ctxc.blockchain.CurrentBlock()
}

func (b *CortexAPIBackend) SetHead(number uint64) {
	b.ctxc.protocolManager.downloader.Cancel()
	b.ctxc.blockchain.SetHead(number)
}

func (b *CortexAPIBackend) SeedingLocal(filePath string, isLinkMode bool) (string, error) {
	return b.ctxc.synapse.SeedingLocal(filePath, isLinkMode)
}

func (b *CortexAPIBackend) PauseLocalSeed(ih string) error {
	return b.ctxc.synapse.PauseLocalSeed(ih)
}

func (b *CortexAPIBackend) ResumeLocalSeed(ih string) error {
	return b.ctxc.synapse.ResumeLocalSeed(ih)
}

func (b *CortexAPIBackend) Download(ih string) error {
	return b.ctxc.synapse.DownloadSeed(ih)
}

func (b *CortexAPIBackend) ListAllTorrents() map[string]map[string]int {
	return b.ctxc.synapse.ListAllTorrents()
}

func (b *CortexAPIBackend) HeaderByNumber(ctx context.Context, number rpc.BlockNumber) (*types.Header, error) {
	// Pending block is only known by the miner
	if number == rpc.PendingBlockNumber {
		block := b.ctxc.miner.PendingBlock()
		return block.Header(), nil
	}
	// Otherwise resolve and return the block
	if number == rpc.LatestBlockNumber {
		return b.ctxc.blockchain.CurrentBlock().Header(), nil
	}
	if number == rpc.FinalizedBlockNumber {
		block := b.ctxc.blockchain.CurrentFinalizedBlock()
		if block != nil {
			return block.Header(), nil
		}
		return nil, errors.New("finalized block not found")
	}
	if number == rpc.SafeBlockNumber {
		block := b.ctxc.blockchain.CurrentSafeBlock()
		if block != nil {
			return block.Header(), nil
		}
		return nil, errors.New("safe block not found")
	}
	return b.ctxc.blockchain.GetHeaderByNumber(uint64(number)), nil
}

func (b *CortexAPIBackend) HeaderByHash(ctx context.Context, hash common.Hash) (*types.Header, error) {
	return b.ctxc.blockchain.GetHeaderByHash(hash), nil
}

func (b *CortexAPIBackend) BlockByNumber(ctx context.Context, blockNr rpc.BlockNumber) (*types.Block, error) {
	// Pending block is only known by the miner
	if blockNr == rpc.PendingBlockNumber {
		block := b.ctxc.miner.PendingBlock()
		return block, nil
	}
	// Otherwise resolve and return the block
	if blockNr == rpc.LatestBlockNumber {
		return b.ctxc.blockchain.CurrentBlock(), nil
	}
	return b.ctxc.blockchain.GetBlockByNumber(uint64(blockNr)), nil
}

func (b *CortexAPIBackend) BlockByHash(ctx context.Context, hash common.Hash) (*types.Block, error) {
	return b.ctxc.blockchain.GetBlockByHash(hash), nil
}

func (b *CortexAPIBackend) BlockByNumberOrHash(ctx context.Context, blockNrOrHash rpc.BlockNumberOrHash) (*types.Block, error) {
	if blockNr, ok := blockNrOrHash.Number(); ok {
		return b.BlockByNumber(ctx, blockNr)
	}
	if hash, ok := blockNrOrHash.Hash(); ok {
		header := b.ctxc.blockchain.GetHeaderByHash(hash)
		if header == nil {
			return nil, errors.New("header for hash not found")
		}
		if blockNrOrHash.RequireCanonical && b.ctxc.blockchain.GetCanonicalHash(header.Number.Uint64()) != hash {
			return nil, errors.New("hash is not currently canonical")
		}
		block := b.ctxc.blockchain.GetBlock(hash, header.Number.Uint64())
		if block == nil {
			return nil, errors.New("header found, but block body is missing")
		}
		return block, nil
	}
	return nil, errors.New("invalid arguments; neither block nor hash specified")
}

func (b *CortexAPIBackend) PendingBlockAndReceipts() (*types.Block, types.Receipts) {
	return b.ctxc.miner.PendingBlockAndReceipts()
}

func (b *CortexAPIBackend) StateAndHeaderByNumber(ctx context.Context, blockNr rpc.BlockNumber) (*state.StateDB, *types.Header, error) {
	// Pending state is only known by the miner
	if blockNr == rpc.PendingBlockNumber {
		block, state := b.ctxc.miner.Pending()
		return state, block.Header(), nil
	}
	// Otherwise resolve the block number and return its state
	header, err := b.HeaderByNumber(ctx, blockNr)
	if err != nil {
		return nil, nil, err
	}
	if header == nil {
		return nil, nil, errors.New("header not found")
	}
	stateDb, err := b.ctxc.BlockChain().StateAt(header.Root)
	if err != nil {
		fmt.Println("StateAndHeaderByNumber error: ", err)
	}
	return stateDb, header, err
}

func (b *CortexAPIBackend) GetBlock(ctx context.Context, hash common.Hash) (*types.Block, error) {
	return b.ctxc.blockchain.GetBlockByHash(hash), nil
}

func (b *CortexAPIBackend) StateAndHeaderByNumberOrHash(ctx context.Context, blockNrOrHash rpc.BlockNumberOrHash) (*state.StateDB, *types.Header, error) {
	if blockNr, ok := blockNrOrHash.Number(); ok {
		return b.StateAndHeaderByNumber(ctx, blockNr)
	}
	if hash, ok := blockNrOrHash.Hash(); ok {
		header, err := b.HeaderByHash(ctx, hash)
		if err != nil {
			return nil, nil, err
		}
		if header == nil {
			return nil, nil, errors.New("header for hash not found")
		}
		if blockNrOrHash.RequireCanonical && b.ctxc.blockchain.GetCanonicalHash(header.Number.Uint64()) != hash {
			return nil, nil, errors.New("hash is not currently canonical")
		}
		stateDb, err := b.ctxc.BlockChain().StateAt(header.Root)
		return stateDb, header, err
	}
	return nil, nil, errors.New("invalid arguments; neither block nor hash specified")
}

func (b *CortexAPIBackend) GetReceipts(ctx context.Context, hash common.Hash) (types.Receipts, error) {
	return b.ctxc.blockchain.GetReceiptsByHash(hash), nil
}

/*func (b *CortexAPIBackend) GetLogs(ctx context.Context, hash common.Hash) ([][]*types.Log, error) {
	db := b.ctxc.ChainDb()
	number := rawdb.ReadHeaderNumber(db, hash)
	if number == nil {
		return nil, fmt.Errorf("failed to get block number for hash %#x", hash)
	}
	logs := rawdb.ReadLogs(db, hash, *number, b.ctxc.blockchain.Config())
	if logs == nil {
		return nil, fmt.Errorf("failed to get logs for block #%d (0x%s)", *number, hash.TerminalString())
	}
	return logs, nil
}*/

func (b *CortexAPIBackend) GetLogs(ctx context.Context, hash common.Hash, number uint64) ([][]*types.Log, error) {
	return rawdb.ReadLogs(b.ctxc.chainDb, hash, number, b.ChainConfig()), nil
}

func (b *CortexAPIBackend) GetTd(ctx context.Context, blockHash common.Hash) *big.Int {
	return b.ctxc.blockchain.GetTdByHash(blockHash)
}

func (b *CortexAPIBackend) GetCVM(ctx context.Context, msg *core.Message, state *state.StateDB, header *types.Header, vmCfg vm.Config) (*vm.CVM, func() error) {

	txContext := core.NewCVMTxContext(msg)
	context := core.NewCVMBlockContext(header, b.ctxc.BlockChain(), nil)
	return vm.NewCVM(context, txContext, state, b.ctxc.chainConfig, vmCfg), state.Error
}

func (b *CortexAPIBackend) SubscribeRemovedLogsEvent(ch chan<- core.RemovedLogsEvent) event.Subscription {
	return b.ctxc.BlockChain().SubscribeRemovedLogsEvent(ch)
}

func (b *CortexAPIBackend) SubscribeChainEvent(ch chan<- core.ChainEvent) event.Subscription {
	return b.ctxc.BlockChain().SubscribeChainEvent(ch)
}

func (b *CortexAPIBackend) SubscribePendingLogsEvent(ch chan<- []*types.Log) event.Subscription {
	return b.ctxc.miner.SubscribePendingLogs(ch)
}

func (b *CortexAPIBackend) SubscribeChainHeadEvent(ch chan<- core.ChainHeadEvent) event.Subscription {
	return b.ctxc.BlockChain().SubscribeChainHeadEvent(ch)
}

func (b *CortexAPIBackend) SubscribeChainSideEvent(ch chan<- core.ChainSideEvent) event.Subscription {
	return b.ctxc.BlockChain().SubscribeChainSideEvent(ch)
}

func (b *CortexAPIBackend) SubscribeLogsEvent(ch chan<- []*types.Log) event.Subscription {
	return b.ctxc.BlockChain().SubscribeLogsEvent(ch)
}

func (b *CortexAPIBackend) SendTx(ctx context.Context, signedTx *types.Transaction) error {
	return b.ctxc.txPool.AddLocal(signedTx)
}

func (b *CortexAPIBackend) GetPoolTransactions() (types.Transactions, error) {
	pending := b.ctxc.txPool.Pending(false)
	var txs types.Transactions
	for _, batch := range pending {
		txs = append(txs, batch...)
	}
	return txs, nil
}

func (b *CortexAPIBackend) GetPoolTransaction(hash common.Hash) *types.Transaction {
	return b.ctxc.txPool.Get(hash)
}

func (b *CortexAPIBackend) GetTransaction(ctx context.Context, txHash common.Hash) (*types.Transaction, common.Hash, uint64, uint64, error) {
	tx, blockHash, blockNumber, index := rawdb.ReadTransaction(b.ctxc.ChainDb(), txHash)
	return tx, blockHash, blockNumber, index, nil
}

func (b *CortexAPIBackend) GetPoolNonce(ctx context.Context, addr common.Address) (uint64, error) {
	return b.ctxc.txPool.Nonce(addr), nil
}

func (b *CortexAPIBackend) Stats() (pending int, queued int) {
	return b.ctxc.txPool.Stats()
}

func (b *CortexAPIBackend) TxPoolContent() (map[common.Address]types.Transactions, map[common.Address]types.Transactions) {
	return b.ctxc.TxPool().Content()
}

func (b *CortexAPIBackend) TxPoolContentFrom(addr common.Address) (types.Transactions, types.Transactions) {
	return b.ctxc.TxPool().ContentFrom(addr)
}

func (b *CortexAPIBackend) TxPool() *txpool.TxPool {
	return b.ctxc.TxPool()
}

func (b *CortexAPIBackend) SubscribeNewTxsEvent(ch chan<- core.NewTxsEvent) event.Subscription {
	return b.ctxc.TxPool().SubscribeNewTxsEvent(ch)
}

func (b *CortexAPIBackend) SyncProgress() cortex.SyncProgress {
	return b.ctxc.Downloader().Progress()
}

func (b *CortexAPIBackend) ProtocolVersion() int {
	return b.ctxc.CortexVersion()
}

func (b *CortexAPIBackend) SuggestPrice(ctx context.Context) (*big.Int, error) {
	return b.gpo.SuggestPrice(ctx)
}

func (b *CortexAPIBackend) ChainDb() ctxcdb.Database {
	return b.ctxc.ChainDb()
}

func (b *CortexAPIBackend) EventMux() *event.TypeMux {
	return b.ctxc.EventMux()
}

func (b *CortexAPIBackend) AccountManager() *accounts.Manager {
	return b.ctxc.AccountManager()
}

func (b *CortexAPIBackend) UnprotectedAllowed() bool {
	return b.allowUnprotectedTxs
}

func (b *CortexAPIBackend) RPCGasCap() uint64 {
	return b.ctxc.config.RPCGasCap
}
func (b *CortexAPIBackend) RPCTxFeeCap() float64 {
	return b.ctxc.config.RPCTxFeeCap
}

func (b *CortexAPIBackend) BloomStatus() (uint64, uint64) {
	sections, _, _ := b.ctxc.bloomIndexer.Sections()
	return params.BloomBitsBlocks, sections
}

func (b *CortexAPIBackend) ServiceFilter(ctx context.Context, session *bloombits.MatcherSession) {
	for i := 0; i < bloomFilterThreads; i++ {
		go session.Multiplex(bloomRetrievalBatch, bloomRetrievalWait, b.ctxc.bloomRequests)
	}
}

func (b *CortexAPIBackend) Engine() consensus.Engine {
	return b.ctxc.engine
}

func (b *CortexAPIBackend) CurrentHeader() *types.Header {
	return b.ctxc.blockchain.CurrentHeader()
}

func (b *CortexAPIBackend) StateAtBlock(ctx context.Context, block *types.Block, reexec uint64, base *state.StateDB, readOnly bool, preferDisk bool) (*state.StateDB, tracers.StateReleaseFunc, error) {
	return b.ctxc.StateAtBlock(block, reexec, base, readOnly, preferDisk)
}

func (b *CortexAPIBackend) StateAtTransaction(ctx context.Context, block *types.Block, txIndex int, reexec uint64) (*core.Message, vm.BlockContext, *state.StateDB, tracers.StateReleaseFunc, error) {
	return b.ctxc.stateAtTransaction(block, txIndex, reexec)
}
