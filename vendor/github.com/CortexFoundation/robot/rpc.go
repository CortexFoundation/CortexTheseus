// Copyright 2023 The CortexTheseus Authors
// This file is part of the CortexTheseus library.
//
// The CortexTheseus library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexTheseus library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the CortexTheseus library. If not, see <http://www.gnu.org/licenses/>.

package robot

import (
	"context"
	"errors"
	"fmt"
	//"strconv"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/types"
)

// SetConnection method builds connection to remote or local communicator.
func (m *Monitor) buildConnection(ipcPath, rpcURI string) (*rpc.Client, error) {
	log.Debug("Building connection", "terminated", m.terminated.Load())

	if ipcPath == "" && rpcURI == "" {
		return nil, errors.New("both ipcPath and rpcURI are empty — cannot build connection")
	}

	const maxRetries = 30
	retryInterval := time.Second * params.QueryTimeInterval * 2

	if ipcPath != "" {
		for i := 0; i < maxRetries; i++ {
			if m.terminated.Load() {
				log.Info("Connection build terminated during IPC")
				return nil, errors.New("ipc connection terminated")
			}

			cl, err := rpc.Dial(ipcPath)
			if err == nil {
				m.local = true
				log.Info("Internal IPC connection established", "ipc", ipcPath, "rpc", rpcURI, "local", m.local)
				return cl, nil
			}

			log.Warn("Retrying IPC connection...", "attempt", i+1, "max", maxRetries, "ipc", ipcPath, "rpc", rpcURI, "error", err)
			time.Sleep(retryInterval)
		}
		log.Warn("IPC connection attempts exhausted, fallback to RPC", "ipc", ipcPath, "rpc", rpcURI)
	}

	if rpcURI != "" {
		cl, err := rpc.Dial(rpcURI)
		if err == nil {
			log.Info("Internal RPC connection established", "ipc", ipcPath, "rpc", rpcURI, "local", m.local)
			return cl, nil
		}

		log.Error("Failed to build RPC connection", "ipc", ipcPath, "rpc", rpcURI, "error", err)
		return nil, fmt.Errorf("failed to establish rpc connection: %w", err)
	}

	return nil, errors.New("no valid connection endpoint found (both IPC and RPC failed or missing)")
}

func (m *Monitor) rpcBlockByNumber(blockNumber uint64) (*types.Block, error) {
	block := &types.Block{}

	rpcBlockMeter.Mark(1)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	err := m.cl.CallContext(ctx, block, "ctxc_getBlockByNumber", hexutil.EncodeUint64(blockNumber), true)
	if err == nil {
		return block, nil
	}

	return nil, err //errors.New("[ Internal IPC Error ] try to get block out of times")
}

/*func (m *Monitor) rpcBatchBlockByNumber(from, to uint64) ([]*types.Block, error) {
	batch := to - from
	result := make([]*types.Block, batch)
	reqs := make([]rpc.BatchElem, batch)
	for i := range reqs {
		reqs[i] = rpc.BatchElem{
			Method: "ctxc_getBlockByNumber",
			Args:   []any{hexutil.EncodeUint64(from + uint64(i)), true},
			Result: &result[i],
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := m.cl.BatchCallContext(ctx, reqs); err != nil {
		return nil, err
	}

	for i, req := range reqs {
		if req.Error != nil {
			return nil, req.Error
		}
		if result[i] == nil {
			return nil, fmt.Errorf("got null block %d", i)
		}
	}

	return result, nil
}*/

func (m *Monitor) rpcBatchBlockByNumber(from, to uint64) ([]*types.Block, error) {
	if from >= to {
		return nil, nil
	}

	batch := to - from
	result := make([]*types.Block, batch)
	reqs := make([]rpc.BatchElem, batch)

	for i := uint64(0); i < batch; i++ {
		blockNum := from + i
		reqs[i] = rpc.BatchElem{
			Method: "ctxc_getBlockByNumber",
			Args:   []any{hexutil.EncodeUint64(blockNum), true},
			Result: &result[i],
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := m.cl.BatchCallContext(ctx, reqs); err != nil {
		return nil, fmt.Errorf("failed to make batch RPC call: %w", err)
	}

	for i, req := range reqs {
		if req.Error != nil {
			return nil, fmt.Errorf("batch RPC call for block %d failed: %w", from+uint64(i), req.Error)
		}
		if result[i] == nil {
			return nil, fmt.Errorf("batch RPC call returned nil for block %d", from+uint64(i))
		}
	}

	return result, nil
}

func (m *Monitor) rpcBatchBlockByNumberLegacy(from, to uint64) (result []*types.Block, err error) {
	batch := to - from
	result = make([]*types.Block, batch)
	var e error = nil
	for i := 0; i < int(batch); i++ {
		m.rpcWg.Add(1)
		go func(index int) {
			defer m.rpcWg.Done()
			//log.Info("bach rpc", "from", from, "to", to, "i", index)
			result[index], e = m.rpcBlockByNumber(from + uint64(index))
			if e != nil {
				err = e
			}
		}(i)
	}

	m.rpcWg.Wait()

	return
}

func (m *Monitor) getRemainingSize(address string) (uint64, error) {
	if size, suc := m.sizeCache.Get(address); suc && size == 0 {
		return size, nil
	}
	var remainingSize hexutil.Uint64
	rpcUploadMeter.Mark(1)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := m.cl.CallContext(ctx, &remainingSize, "ctxc_getUpload", address, "latest"); err != nil {
		return 0, err
	}
	remain := uint64(remainingSize)
	if remain == 0 {
		m.sizeCache.Add(address, remain)
	}
	return remain, nil
}

func (m *Monitor) getReceipt(tx string) (receipt types.Receipt, err error) {
	rpcReceiptMeter.Mark(1)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err = m.cl.CallContext(ctx, &receipt, "ctxc_getTransactionReceipt", tx); err != nil {
		log.Warn("R is nil", "R", tx, "err", err)
	}

	return
}

func (m *Monitor) getBlockReceipts(hash string) (receipt []types.Receipt, err error) {
	rpcReceiptMeter.Mark(1)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err = m.cl.CallContext(ctx, &receipt, "ctxc_getBlockReceipts", hash); err != nil {
		log.Warn("R array is nil", "R", hash, "err", err)
	}

	return
}

func (m *Monitor) currentBlock() (uint64, bool, error) {
	var (
		currentNumber hexutil.Uint64
		update        bool
	)

	rpcCurrentMeter.Mark(1)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := m.cl.CallContext(ctx, &currentNumber, "ctxc_blockNumber"); err != nil {
		log.Error("Call ipc method ctxc_blockNumber failed", "error", err)
		return m.currentNumber.Load(), false, err
	}
	if m.currentNumber.Load() != uint64(currentNumber) {
		m.currentNumber.Store(uint64(currentNumber))
		update = true
	}

	return uint64(currentNumber), update, nil
}
