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
	"errors"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/types"
	"strconv"
	"time"
)

// SetConnection method builds connection to remote or local communicator.
func (m *Monitor) buildConnection(ipcpath string, rpcuri string) (*rpc.Client, error) {
	log.Debug("Building connection", "terminated", m.terminated.Load())

	if len(ipcpath) > 0 {
		for i := 0; i < 30; i++ {
			time.Sleep(time.Second * params.QueryTimeInterval * 2)
			cl, err := rpc.Dial(ipcpath)
			if err != nil {
				log.Warn("Building internal ipc connection ... ", "ipc", ipcpath, "rpc", rpcuri, "error", err, "terminated", m.terminated.Load())
			} else {
				m.local = true
				log.Info("Internal ipc connection established", "ipc", ipcpath, "rpc", rpcuri, "local", m.local)
				return cl, nil
			}

			if m.terminated.Load() {
				log.Info("Connection builder break")
				return nil, errors.New("ipc connection terminated")
			}
		}
	} else {
		log.Warn("IPC is empty, try remote RPC instead")
	}

	cl, err := rpc.Dial(rpcuri)
	if err != nil {
		log.Warn("Building internal rpc connection ... ", "ipc", ipcpath, "rpc", rpcuri, "error", err, "terminated", m.terminated.Load())
	} else {
		log.Info("Internal rpc connection established", "ipc", ipcpath, "rpc", rpcuri, "local", m.local)
		return cl, nil
	}

	return nil, errors.New("building internal ipc connection failed")
}

func (m *Monitor) rpcBlockByNumber(blockNumber uint64) (*types.Block, error) {
	block := &types.Block{}

	rpcBlockMeter.Mark(1)
	err := m.cl.Call(block, "ctxc_getBlockByNumber", "0x"+strconv.FormatUint(blockNumber, 16), true)
	if err == nil {
		return block, nil
	}

	return nil, err //errors.New("[ Internal IPC Error ] try to get block out of times")
}

func (m *Monitor) rpcBatchBlockByNumber(from, to uint64) (result []*types.Block, err error) {
	batch := to - from
	result = make([]*types.Block, batch)
	var e error = nil
	for i := 0; i < int(batch); i++ {
		m.rpcWg.Add(1)
		go func(index int) {
			defer m.rpcWg.Done()
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
	if err := m.cl.Call(&remainingSize, "ctxc_getUpload", address, "latest"); err != nil {
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
	if err = m.cl.Call(&receipt, "ctxc_getTransactionReceipt", tx); err != nil {
		log.Warn("R is nil", "R", tx, "err", err)
	}

	return
}

func (m *Monitor) currentBlock() (uint64, bool, error) {
	var (
		currentNumber hexutil.Uint64
		update        bool
	)

	rpcCurrentMeter.Mark(1)
	if err := m.cl.Call(&currentNumber, "ctxc_blockNumber"); err != nil {
		log.Error("Call ipc method ctxc_blockNumber failed", "error", err)
		return m.currentNumber.Load(), false, err
	}
	if m.currentNumber.Load() != uint64(currentNumber) {
		m.currentNumber.Store(uint64(currentNumber))
		update = true
	}

	return uint64(currentNumber), update, nil
}
