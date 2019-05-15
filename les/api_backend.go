// Copyright 2016 The go-cortex Authors
// This file is part of the go-cortex library.
//
// The go-cortex library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-cortex library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the go-cortex library. If not, see <http://www.gnu.org/licenses/>.

package les

import (
	"context"
	"math/big"

	"github.com/CortexFoundation/CortexTheseus/accounts"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/math"
	"github.com/CortexFoundation/CortexTheseus/core"
	"github.com/CortexFoundation/CortexTheseus/core/bloombits"
	"github.com/CortexFoundation/CortexTheseus/core/rawdb"
	"github.com/CortexFoundation/CortexTheseus/core/state"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/core/vm"
	"github.com/CortexFoundation/CortexTheseus/eth/downloader"
	"github.com/CortexFoundation/CortexTheseus/eth/gasprice"
	"github.com/CortexFoundation/CortexTheseus/ethdb"
	"github.com/CortexFoundation/CortexTheseus/event"
	"github.com/CortexFoundation/CortexTheseus/light"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/CortexTheseus/rpc"
)

type LesApiBackend struct {
	ctxc *LightCortex
	gpo *gasprice.Oracle
}

func (b *LesApiBackend) ChainConfig() *params.ChainConfig {
	return b.ctxc.chainConfig
}

func (b *LesApiBackend) CurrentBlock() *types.Block {
	return types.NewBlockWithHeader(b.ctxc.BlockChain().CurrentHeader())
}

func (b *LesApiBackend) SetHead(number uint64) {
	b.ctxc.protocolManager.downloader.Cancel()
	b.ctxc.blockchain.SetHead(number)
}

func (b *LesApiBackend) HeaderByNumber(ctx context.Context, blockNr rpc.BlockNumber) (*types.Header, error) {
	if blockNr == rpc.LatestBlockNumber || blockNr == rpc.PendingBlockNumber {
		return b.ctxc.blockchain.CurrentHeader(), nil
	}
	return b.ctxc.blockchain.GetHeaderByNumberOdr(ctx, uint64(blockNr))
}

func (b *LesApiBackend) HeaderByHash(ctx context.Context, hash common.Hash) (*types.Header, error) {
	return b.ctxc.blockchain.GetHeaderByHash(hash), nil
}

func (b *LesApiBackend) BlockByNumber(ctx context.Context, blockNr rpc.BlockNumber) (*types.Block, error) {
	header, err := b.HeaderByNumber(ctx, blockNr)
	if header == nil || err != nil {
		return nil, err
	}
	return b.GetBlock(ctx, header.Hash())
}

func (b *LesApiBackend) StateAndHeaderByNumber(ctx context.Context, blockNr rpc.BlockNumber) (*state.StateDB, *types.Header, error) {
	header, err := b.HeaderByNumber(ctx, blockNr)
	if header == nil || err != nil {
		return nil, nil, err
	}
	return light.NewState(ctx, header, b.ctxc.odr), header, nil
}

func (b *LesApiBackend) GetBlock(ctx context.Context, blockHash common.Hash) (*types.Block, error) {
	return b.ctxc.blockchain.GetBlockByHash(ctx, blockHash)
}

func (b *LesApiBackend) GetReceipts(ctx context.Context, hash common.Hash) (types.Receipts, error) {
	if number := rawdb.ReadHeaderNumber(b.ctxc.chainDb, hash); number != nil {
		return light.GetBlockReceipts(ctx, b.ctxc.odr, hash, *number)
	}
	return nil, nil
}

func (b *LesApiBackend) GetLogs(ctx context.Context, hash common.Hash) ([][]*types.Log, error) {
	if number := rawdb.ReadHeaderNumber(b.ctxc.chainDb, hash); number != nil {
		return light.GetBlockLogs(ctx, b.ctxc.odr, hash, *number)
	}
	return nil, nil
}

func (b *LesApiBackend) GetTd(hash common.Hash) *big.Int {
	return b.ctxc.blockchain.GetTdByHash(hash)
}

func (b *LesApiBackend) GetEVM(ctx context.Context, msg core.Message, state *state.StateDB, header *types.Header, vmCfg vm.Config) (*vm.EVM, func() error, error) {
	state.SetBalance(msg.From(), math.MaxBig256)
	context := core.NewEVMContext(msg, header, b.ctxc.blockchain, nil)
	return vm.NewEVM(context, state, b.ctxc.chainConfig, vmCfg), state.Error, nil
}

func (b *LesApiBackend) SendTx(ctx context.Context, signedTx *types.Transaction) error {
	return b.ctxc.txPool.Add(ctx, signedTx)
}

func (b *LesApiBackend) RemoveTx(txHash common.Hash) {
	b.ctxc.txPool.RemoveTx(txHash)
}

func (b *LesApiBackend) GetPoolTransactions() (types.Transactions, error) {
	return b.ctxc.txPool.GetTransactions()
}

func (b *LesApiBackend) GetPoolTransaction(txHash common.Hash) *types.Transaction {
	return b.ctxc.txPool.GetTransaction(txHash)
}

func (b *LesApiBackend) GetPoolNonce(ctx context.Context, addr common.Address) (uint64, error) {
	return b.ctxc.txPool.GetNonce(ctx, addr)
}

func (b *LesApiBackend) Stats() (pending int, queued int) {
	return b.ctxc.txPool.Stats(), 0
}

func (b *LesApiBackend) TxPoolContent() (map[common.Address]types.Transactions, map[common.Address]types.Transactions) {
	return b.ctxc.txPool.Content()
}

func (b *LesApiBackend) SubscribeNewTxsEvent(ch chan<- core.NewTxsEvent) event.Subscription {
	return b.ctxc.txPool.SubscribeNewTxsEvent(ch)
}

func (b *LesApiBackend) SubscribeChainEvent(ch chan<- core.ChainEvent) event.Subscription {
	return b.ctxc.blockchain.SubscribeChainEvent(ch)
}

func (b *LesApiBackend) SubscribeChainHeadEvent(ch chan<- core.ChainHeadEvent) event.Subscription {
	return b.ctxc.blockchain.SubscribeChainHeadEvent(ch)
}

func (b *LesApiBackend) SubscribeChainSideEvent(ch chan<- core.ChainSideEvent) event.Subscription {
	return b.ctxc.blockchain.SubscribeChainSideEvent(ch)
}

func (b *LesApiBackend) SubscribeLogsEvent(ch chan<- []*types.Log) event.Subscription {
	return b.ctxc.blockchain.SubscribeLogsEvent(ch)
}

func (b *LesApiBackend) SubscribeRemovedLogsEvent(ch chan<- core.RemovedLogsEvent) event.Subscription {
	return b.ctxc.blockchain.SubscribeRemovedLogsEvent(ch)
}

func (b *LesApiBackend) Downloader() *downloader.Downloader {
	return b.ctxc.Downloader()
}

func (b *LesApiBackend) ProtocolVersion() int {
	return b.ctxc.LesVersion() + 10000
}

func (b *LesApiBackend) SuggestPrice(ctx context.Context) (*big.Int, error) {
	return b.gpo.SuggestPrice(ctx)
}

func (b *LesApiBackend) ChainDb() ctxcdb.Database {
	return b.ctxc.chainDb
}

func (b *LesApiBackend) EventMux() *event.TypeMux {
	return b.ctxc.eventMux
}

func (b *LesApiBackend) AccountManager() *accounts.Manager {
	return b.ctxc.accountManager
}

func (b *LesApiBackend) BloomStatus() (uint64, uint64) {
	if b.ctxc.bloomIndexer == nil {
		return 0, 0
	}
	sections, _, _ := b.ctxc.bloomIndexer.Sections()
	return params.BloomBitsBlocksClient, sections
}

func (b *LesApiBackend) ServiceFilter(ctx context.Context, session *bloombits.MatcherSession) {
	for i := 0; i < bloomFilterThreads; i++ {
		go session.Multiplex(bloomRetrievalBatch, bloomRetrievalWait, b.ctxc.bloomRequests)
	}
}
