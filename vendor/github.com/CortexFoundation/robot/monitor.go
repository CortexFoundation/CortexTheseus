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
	"encoding/json"
	"errors"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/metrics"
	params1 "github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"github.com/CortexFoundation/robot/backend"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/types"
	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/ucwong/golang-kv"
	"math"
	"math/big"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

const (
	batch   = 4096 * 2 //params.SyncBatch
	delay   = 12       //params.Delay
	timeout = 30 * time.Second
)

var (
	rpcBlockMeter   = metrics.NewRegisteredMeter("torrent/block/call", nil)
	rpcCurrentMeter = metrics.NewRegisteredMeter("torrent/current/call", nil)
	rpcUploadMeter  = metrics.NewRegisteredMeter("torrent/upload/call", nil)
	rpcReceiptMeter = metrics.NewRegisteredMeter("torrent/receipt/call", nil)
)

// Monitor observes the data changes on the blockchain and synchronizes.
// cl for ipc/rpc communication, dl for download manager, and fs for data storage.
type Monitor struct {
	config *params.Config
	cl     *rpc.Client
	fs     *backend.ChainDB
	engine kv.Bucket
	//dl     *backend.TorrentManager

	exitCh chan any
	srvCh  chan int
	//exitSyncCh    chan any
	terminated    atomic.Bool
	lastNumber    atomic.Uint64
	startNumber   atomic.Uint64
	currentNumber atomic.Uint64
	scope         uint64
	wg            sync.WaitGroup
	rpcWg         sync.WaitGroup

	taskCh chan *types.Block
	errCh  chan error
	//newTaskHook func(*types.Block)
	blockCache *lru.Cache[uint64, string]
	sizeCache  *lru.Cache[string, uint64]
	ckp        *params.TrustedCheckpoint
	start      mclock.AbsTime

	local  bool
	listen bool

	startOnce sync.Once
	closeOnce sync.Once
	mode      string

	lock sync.RWMutex

	callback chan any

	srv atomic.Int32
}

// NewMonitor creates a new instance of monitor.
// Once Ipcpath is settle, this method prefers to build socket connection in order to
// get higher communicating performance.
// IpcPath is unavailable on windows.
// func New(flag *params.Config, cache, compress, listen bool, fs *backend.ChainDB, tMana *backend.TorrentManager, callback chan any) (*Monitor, error) {
func New(flag *params.Config, cache, compress, listen bool, callback chan any) (m *Monitor, err error) {
	/*fs, fsErr := NewChainDB(flag)
	if fsErr != nil {
		log.Error("file storage failed", "err", fsErr)
		return nil, fsErr
	}
	log.Info("File storage initialized")

	tMana, err := NewTorrentManager(flag, fs.ID(), cache, compress)
	if err != nil || tMana == nil {
		log.Error("fs manager failed")
		return nil, errors.New("fs download manager initialise failed")
	}
	log.Info("Fs manager initialized")*/

	m = &Monitor{
		config: flag,
		cl:     nil,
		//fs:     fs,
		//dl:            tMana,
		exitCh: make(chan any),
		srvCh:  make(chan int),
		//exitSyncCh: make(chan any),
		scope: uint64(math.Max(float64(runtime.NumCPU()*2), float64(4))),
		//taskCh: make(chan *types.Block, batch),
		//taskCh:        make(chan *types.Block, 1),
		//start: mclock.Now(),
	}
	m.errCh = make(chan error, m.scope)
	m.taskCh = make(chan *types.Block, m.scope)
	// TODO https://github.com/ucwong/golang-kv
	if fs_, err := backend.NewChainDB(flag); err != nil {
		log.Error("file storage failed", "err", err)
		return nil, err
	} else {
		m.fs = fs_
	}
	m.lastNumber.Store(0)
	m.currentNumber.Store(0)
	m.startNumber.Store(0)

	m.terminated.Store(false)
	m.blockCache, _ = lru.New[uint64, string](delay)
	m.sizeCache, _ = lru.New[string, uint64](batch)
	m.listen = listen
	m.callback = callback

	//if err := m.dl.Start(); err != nil {
	//	log.Warn("Fs start error")
	//	return nil, err
	//}

	m.mode = flag.Mode

	m.srv.Store(SRV_MODEL)

	m.engine = kv.Pebble(filepath.Join(flag.DataDir, ".srv"))

	/*torrents, _ := fs.initTorrents()
	if m.mode != params.LAZY {
		for k, v := range torrents {
			if err := GetStorage().Download(context.Background(), k, v); err != nil {
				return nil, err
			}
		}
	}

	if len(torrents) == 0 {
		log.Warn("Data reloading", "mode", m.mode)
		m.indexInit()
	}*/

	return m, nil
}

func (m *Monitor) CurrentNumber() uint64 {
	return m.currentNumber.Load()
}

func (m *Monitor) ID() uint64 {
	return m.fs.ID()
}

func (m *Monitor) DB() *backend.ChainDB {
	return m.fs
}

func (m *Monitor) Callback() chan any {
	return m.callback
}

func (m *Monitor) loadHistory() error {
	torrents, _ := m.fs.InitTorrents()
	if m.mode != params.LAZY {
		for k, v := range torrents {
			ctx, cancel := context.WithTimeout(context.Background(), timeout)
			defer cancel()
			m.download(ctx, k, v)
		}
	}

	if len(torrents) == 0 {
		log.Warn("Data reloading", "mode", m.mode)
		if err := m.indexInit(); err != nil {
			return err
		}
	}

	return nil
}

func (m *Monitor) download(ctx context.Context, k string, v uint64) error {
	if m.mode != params.LAZY && m.callback != nil {
		select {
		case m.callback <- types.NewBitsFlow(k, v):
		case <-ctx.Done():
			return ctx.Err()
		case <-m.exitCh:
			return errors.New("terminated")
		}
	}
	return nil
}

func (m *Monitor) indexCheck() error {
	log.Info("Loading storage data ... ...", "latest", m.fs.LastListenBlockNumber(), "checkpoint", m.fs.CheckPoint(), "root", m.fs.Root(), "version", m.fs.Version(), "current", m.currentNumber.Load())
	genesis, err := m.rpcBlockByNumber(0)
	if err != nil {
		return err
	}

	if checkpoint, ok := params.TrustedCheckpoints[genesis.Hash]; ok {
		m.ckp = checkpoint

		version := m.fs.GetRoot(checkpoint.TfsCheckPoint)
		if common.BytesToHash(version) != checkpoint.TfsRoot {
			m.lastNumber.Store(0)
			m.startNumber.Store(0)
			if m.lastNumber.Load() > checkpoint.TfsCheckPoint {
				//m.fs.LastListenBlockNumber = 0
				m.fs.Anchor(0)
				//m.lastNumber = 0
				//if err := m.fs.Reset(); err != nil {
				//	return err
				//}
			}
			log.Warn("Fs storage is reloading ...", "name", m.ckp.Name, "number", checkpoint.TfsCheckPoint, "version", common.BytesToHash(version), "checkpoint", checkpoint.TfsRoot, "blocks", len(m.fs.Blocks()), "files", len(m.fs.Files()), "txs", m.fs.Txs(), "lastNumber", m.lastNumber.Load(), "last", m.fs.LastListenBlockNumber())
		} else {
			log.Info("Fs storage version check passed", "name", m.ckp.Name, "number", checkpoint.TfsCheckPoint, "version", common.BytesToHash(version), "blocks", len(m.fs.Blocks()), "files", len(m.fs.Files()), "txs", m.fs.Txs())
		}
	}

	return nil
}

func (m *Monitor) indexInit() error {
	fileMap := make(map[string]*types.FileInfo, len(m.fs.Files()))
	for _, file := range m.fs.Files() {
		if f, ok := fileMap[file.Meta.InfoHash]; ok {
			if f.LeftSize > file.LeftSize {
				fileMap[file.Meta.InfoHash] = file
			}
		} else {
			fileMap[file.Meta.InfoHash] = file
		}
	}

	var (
		capcity = uint64(0)
		seed    = 0
		pause   = 0
		pending = 0
	)

	for _, file := range fileMap {
		var bytesRequested uint64
		if file.Meta.RawSize > file.LeftSize {
			bytesRequested = file.Meta.RawSize - file.LeftSize
		}
		capcity += bytesRequested
		log.Debug("File storage info", "addr", file.ContractAddr, "ih", file.Meta.InfoHash, "remain", common.StorageSize(file.LeftSize), "raw", common.StorageSize(file.Meta.RawSize), "request", common.StorageSize(bytesRequested))
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		if err := m.download(ctx, file.Meta.InfoHash, bytesRequested); err != nil {
			return err
		}
		if file.LeftSize == 0 {
			seed++
		} else if file.Meta.RawSize == file.LeftSize && file.LeftSize > 0 {
			pending++
		} else if file.Meta.RawSize > file.LeftSize && file.LeftSize > 0 {
			pause++
		}
	}
	log.Info("Storage current state", "total", len(m.fs.Files()), "dis", len(fileMap), "seed", seed, "pause", pause, "pending", pending, "capcity", common.StorageSize(capcity), "blocks", len(m.fs.Blocks()), "txs", m.fs.Txs())
	return nil
}

func (m *Monitor) taskLoop() {
	log.Info("Task channel started")
	defer m.wg.Done()
	for {
		select {
		case task := <-m.taskCh:
			//if m.newTaskHook != nil {
			//	m.newTaskHook(task)
			//}

			/*if err := m.solve(task); err != nil {
				m.errCh <- err
				log.Warn("Block solved failed, try again", "err", err, "num", task.Number, "last", m.lastNumber.Load())
			} else {
				m.errCh <- nil
			}*/
			m.errCh <- m.solve(task)
		case <-m.exitCh:
			log.Info("Monitor task channel closed")
			return
		}
	}
}

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

func (m *Monitor) parseFileMeta(tx *types.Transaction, meta *types.FileMeta, b *types.Block) error {
	log.Debug("Monitor", "FileMeta", meta)

	receipt, err := m.getReceipt(tx.Hash.String())
	if err != nil {
		return err
	}

	if receipt.ContractAddr == nil {
		log.Warn("contract address is nil, waiting for indexing", "tx.Hash.String()", tx.Hash.String())
		return errors.New("contract address is nil")
	}

	log.Debug("Transaction Receipt", "address", receipt.ContractAddr.String(), "gas", receipt.GasUsed, "status", receipt.Status) //, "tx", receipt.TxHash.String())

	if receipt.Status != 1 {
		log.Warn("receipt.Status is wrong", "receipt.Status", receipt.Status)
		return nil
	}

	log.Debug("Meta data", "meta", meta)

	info := m.fs.NewFileInfo(meta)

	info.LeftSize = meta.RawSize
	info.ContractAddr = receipt.ContractAddr
	info.Relate = append(info.Relate, *info.ContractAddr)
	op, update, err := m.fs.AddFile(info)
	if err != nil {
		log.Warn("Create file failed", "err", err)
		return err
	}
	if update && op == 1 {
		log.Debug("Create new file", "ih", meta.InfoHash, "op", op)

		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		if m.mode == params.FULL {
			return m.download(ctx, meta.InfoHash, 512*1024)
		} else {
			return m.download(ctx, meta.InfoHash, 0)
		}
	}
	return nil
}

func (m *Monitor) parseBlockTorrentInfo(b *types.Block) (bool, error) {
	var (
		record bool
		start  = mclock.Now()
		final  []types.Transaction
	)
	for _, tx := range b.Txs {
		if meta := tx.Parse(); meta != nil {
			log.Debug("Data encounter", "ih", meta.InfoHash, "number", b.Number, "meta", meta)
			if err := m.parseFileMeta(&tx, meta, b); err != nil {
				log.Error("Parse file meta error", "err", err, "number", b.Number)
				return false, err
			}
			record = true
			final = append(final, tx)
		} else if tx.IsFlowControl() {
			if tx.Recipient == nil {
				continue
			}
			file := m.fs.GetFileByAddr(*tx.Recipient)
			if file == nil {
				continue
			}
			receipt, err := m.getReceipt(tx.Hash.String())
			if err != nil {
				return false, err
			}
			if receipt.Status != 1 || receipt.GasUsed != params.UploadGas {
				continue
			}
			remainingSize, err := m.getRemainingSize((*tx.Recipient).String())
			if err != nil {
				log.Error("Get remain failed", "err", err, "addr", (*tx.Recipient).String())
				return false, err
			}
			if file.LeftSize > remainingSize {
				file.LeftSize = remainingSize
				if _, progress, err := m.fs.AddFile(file); err != nil {
					return false, err
				} else if progress { // && progress {
					log.Debug("Update storage success", "ih", file.Meta.InfoHash, "left", file.LeftSize)
					var bytesRequested uint64
					if file.Meta.RawSize > file.LeftSize {
						bytesRequested = file.Meta.RawSize - file.LeftSize
					}
					if file.LeftSize == 0 {
						log.Debug("Data processing completed !!!", "ih", file.Meta.InfoHash, "addr", (*tx.Recipient).String(), "remain", common.StorageSize(remainingSize), "request", common.StorageSize(bytesRequested), "raw", common.StorageSize(file.Meta.RawSize), "number", b.Number)
					} else {
						log.Debug("Data processing ...", "ih", file.Meta.InfoHash, "addr", (*tx.Recipient).String(), "remain", common.StorageSize(remainingSize), "request", common.StorageSize(bytesRequested), "raw", common.StorageSize(file.Meta.RawSize), "number", b.Number)
					}
					ctx, cancel := context.WithTimeout(context.Background(), timeout)
					defer cancel()
					if err := m.download(ctx, file.Meta.InfoHash, bytesRequested); err != nil {
						return false, err
					}
				}
			}
			record = true
			final = append(final, tx)
		}
	}
	if len(final) > 0 && len(final) < len(b.Txs) {
		log.Debug("Final txs layout", "total", len(b.Txs), "final", len(final), "num", b.Number, "txs", m.fs.Txs())
		b.Txs = final
	}
	if record {
		m.fs.AddBlock(b)
	}
	if len(b.Txs) > 0 {
		elapsed := time.Duration(mclock.Now()) - time.Duration(start)
		log.Trace("Transactions scanning", "count", len(b.Txs), "number", b.Number, "elapsed", common.PrettyDuration(elapsed))
	}
	return record, nil
}

func (m *Monitor) exit() {
	m.closeOnce.Do(func() {
		if m.exitCh != nil {
			close(m.exitCh)
			m.wg.Wait()
			m.exitCh = nil
		} else {
			log.Warn("Listener has already been stopped")
		}
	})
}

func (m *Monitor) Stop() error {
	m.lock.Lock()
	defer m.lock.Unlock()
	if m.terminated.Swap(true) {
		return nil
	}

	m.exit()
	log.Info("Monitor is waiting to be closed")
	m.blockCache.Purge()
	m.sizeCache.Purge()

	//log.Info("Fs client listener synchronizing closing")
	//if err := m.dl.Close(); err != nil {
	//	log.Error("Monitor Fs Manager closed", "error", err)
	//}

	// TODO dirty statics deal with
	if m.engine != nil {
		log.Info("Golang-kv engine close", "engine", m.engine.Name())
		return m.engine.Close()
	}

	if err := m.fs.Close(); err != nil {
		log.Error("Monitor File Storage closed", "error", err)
		return err
	}
	log.Info("Fs listener synchronizing closed")
	return nil
}

// Start ... start ListenOn on the rpc port of a blockchain full node
func (m *Monitor) Start() error {
	//if !m.listen {
	//log.Info("Disable listener")
	//return nil
	//}

	m.startOnce.Do(func() {
		m.wg.Add(1)
		go func() {
			defer m.wg.Done()
			m.fs.Init()
			if err := m.run(); err != nil {
				log.Error("Fs monitor start failed", "err", err)
			}
		}()
	})
	return nil
}

func (m *Monitor) run() error {
	var ipcpath string
	if runtime.GOOS != "windows" && m.config.IpcPath != "" {
		ipcpath = m.config.IpcPath
		//} else {
		//	if m.config.RpcURI == "" {
		//		log.Warn("Fs rpc uri is empty")
		//		return errors.New("fs RpcURI is empty")
		//	}
		//	clientURI = m.config.RpcURI
	}

	rpcClient, rpcErr := m.buildConnection(ipcpath, m.config.RpcURI)
	if rpcErr != nil {
		log.Error("Fs rpc client is wrong", "uri", ipcpath, "error", rpcErr, "config", m.config)
		return rpcErr
	}
	m.cl = rpcClient

	if m.srv.Load() == SRV_MODEL {
		m.lastNumber.Store(m.fs.LastListenBlockNumber())
		if err := m.indexCheck(); err != nil {
			return err
		}
	}

	m.currentBlock()
	m.startNumber.Store(uint64(math.Min(float64(m.fs.LastListenBlockNumber()), float64(m.currentNumber.Load())))) // ? m.currentNumber:m.fs.LastListenBlockNumber

	//if err := m.loadHistory(); err != nil {
	//	return err
	//}
	m.wg.Add(1)
	go m.taskLoop()
	m.wg.Add(1)
	go m.listenLatestBlock()
	m.wg.Add(1)
	go m.syncLatestBlock()

	return nil
}

func (m *Monitor) listenLatestBlock() {
	defer m.wg.Done()
	timer := time.NewTimer(time.Second * params.QueryTimeInterval)
	defer timer.Stop()
	for {
		select {
		case <-timer.C:
			if cur, update, err := m.currentBlock(); err == nil && update {
				log.Debug("Blockchain update", "cur", cur)
			}
			if m.local {
				timer.Reset(time.Second * params.QueryTimeInterval)
			} else {
				timer.Reset(time.Second * params.QueryTimeInterval * 10)
			}
		case <-m.exitCh:
			log.Info("Block listener stopped")
			return
		}
	}
}

func (m *Monitor) syncLatestBlock() {
	defer m.wg.Done()
	timer := time.NewTimer(time.Second * params.QueryTimeInterval)
	defer timer.Stop()
	progress, counter, end := uint64(0), 0, false
	for {
		select {
		case sv := <-m.srvCh:
			if err := m.doSwitch(sv); err != nil {
				log.Error("Service switch failed", "srv", sv, "err", err)
			}
		case <-timer.C:
			progress = m.syncLastBlock()
			// Avoid sync in full mode, fresh interval may be less.
			if progress >= delay {
				end = false
				timer.Reset(time.Millisecond * 0)
			} else if progress > 1 {
				end = false
				timer.Reset(time.Millisecond * 2000)
			} else if progress == 1 {
				end = true
				timer.Reset(time.Millisecond * 6750)
			} else {
				if !m.listen {
					if (m.ckp != nil && m.currentNumber.Load() >= m.ckp.TfsCheckPoint) || (m.ckp == nil && m.currentNumber.Load() > 0) {
						if !end {
							end = true
							timer.Reset(time.Millisecond * 6750)
							continue
						}
						m.fs.Flush()
						//elapsed := time.Duration(mclock.Now()) - time.Duration(m.start)
						//log.Debug("Finish sync, listener will be paused", "current", m.currentNumber.Load(), "elapsed", common.PrettyDuration(elapsed), "progress", progress, "end", end, "last", m.lastNumber.Load())
						//return
						timer.Reset(time.Millisecond * 6750)
						end = false
						continue
					}
				}
				timer.Reset(time.Millisecond * 6750)
			}
			counter++
			if counter%100 == 0 {
				log.Info("Monitor status", "blocks", progress, "current", m.CurrentNumber(), "latest", m.lastNumber.Load(), "end", end, "txs", m.fs.Txs(), "ckp", m.fs.CheckPoint(), "last", m.fs.LastListenBlockNumber())
				counter = 0
			}
			m.fs.Flush()
		case <-m.exitCh:
			log.Info("Block syncer stopped")
			return
		}
	}
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

func (m *Monitor) skip(i uint64) bool {
	if m.srv.Load() != SRV_MODEL {
		return false
	}

	if len(m.ckp.Skips) == 0 || i > m.ckp.Skips[len(m.ckp.Skips)-1].To || i < m.ckp.Skips[0].From {
		return false
	}

	for _, skip := range m.ckp.Skips {
		if i > skip.From && i < skip.To {
			//m.lastNumber = i - 1
			return true
		}
	}
	return false
}

func (m *Monitor) syncLastBlock() uint64 {
	/*currentNumber, err := m.currentBlock()
	if err != nil {
		return 0
	}*/

	currentNumber := m.currentNumber.Load()

	if currentNumber < m.lastNumber.Load() {
		log.Warn("Fs sync rollback", "current", currentNumber, "last", m.lastNumber.Load(), "offset", m.lastNumber.Load()-currentNumber)
		if currentNumber > 65536 {
			m.lastNumber.Store(currentNumber - 65536)
		} else {
			m.lastNumber.Store(0)
		}
		m.startNumber.Store(m.lastNumber.Load())
	}

	minNumber := m.lastNumber.Load() + 1
	maxNumber := uint64(0)
	if currentNumber > delay {
		maxNumber = currentNumber - delay
		//maxNumber = currentNumber
	}

	if m.lastNumber.Load() > currentNumber {
		if m.lastNumber.Load() > batch {
			minNumber = m.lastNumber.Load() - batch
		}
	}

	if maxNumber > batch+minNumber {
		maxNumber = minNumber + batch
	}

	// replay
	if minNumber >= delay {
		//minNumber = minNumber - delay
	}

	if maxNumber < minNumber {
		return 0
	}

	//if m.start == 0 {
	m.start = mclock.Now()
	//}

	counter := 0
	for i := minNumber; i <= maxNumber; { // i++ {
		if m.terminated.Load() {
			log.Warn("Fs scan terminated", "number", i)
			m.lastNumber.Store(i - 1)
			return 0
		}
		//if maxNumber > minNumber && i%2048 == 0 {
		//	log.Info("Running", "min", minNumber, "max", maxNumber, "cur", currentNumber, "last", m.lastNumber.Load(), "batch", batch, "i", i, "srv", m.srv.Load(), "count", maxNumber-minNumber, "progress", float64(i)/float64(currentNumber))
		//}
		if m.ckp != nil && m.skip(i) {
			i++
			continue
		}

		if maxNumber-i >= m.scope {
			blocks, rpcErr := m.rpcBatchBlockByNumber(i, i+m.scope)
			if rpcErr != nil {
				log.Error("Sync old block failed", "number", i, "error", rpcErr)
				m.lastNumber.Store(i - 1)
				return 0
			}

			// batch blocks operation according service category
			for _, rpcBlock := range blocks {
				m.taskCh <- rpcBlock
			}

			for n := 0; n < len(blocks); n++ {
				select {
				case err := <-m.errCh:
					if err != nil {
						m.lastNumber.Store(i - 1)
						log.Error("solve err", "err", err, "last", m.lastNumber.Load(), "i", i, "scope", m.scope, "min", minNumber, "max", maxNumber, "cur", currentNumber)
						return 0
					}
				case <-m.exitCh:
					m.lastNumber.Store(i - 1)
					log.Info("Task checker quit")
					return 0
				}
			}
			i += uint64(len(blocks))
			counter += len(blocks)
		} else {
			rpcBlock, rpcErr := m.rpcBlockByNumber(i)
			if rpcErr != nil {
				log.Error("Sync old block failed", "number", i, "error", rpcErr)
				m.lastNumber.Store(i - 1)
				return 0
			}
			if err := m.solve(rpcBlock); err != nil {
				log.Error("solve err", "err", err)
				m.lastNumber.Store(i - 1)
				return 0
			}
			i++
			counter++
		}
	}
	//log.Debug("Last number changed", "min", minNumber, "max", maxNumber, "cur", currentNumber, "last", m.lastNumber.Load(), "batch", batch)
	m.lastNumber.Store(maxNumber)
	elapsedA := time.Duration(mclock.Now()) - time.Duration(m.start)
	log.Info("Chain segment frozen", "from", minNumber, "to", maxNumber, "range", uint64(maxNumber-minNumber+1), "counter", counter, "scope", m.scope, "current", m.CurrentNumber(), "prog", float64(maxNumber)/float64(m.CurrentNumber()), "last", m.lastNumber.Load(), "bps", float64(counter)*1000*1000*1000/float64(elapsedA), "elapsed", common.PrettyDuration(elapsedA))
	return uint64(maxNumber - minNumber)
}

// solve block from node
func (m *Monitor) solve(block *types.Block) error {
	switch m.srv.Load() {
	case SRV_MODEL:
		return m.forModelService(block)
	//case 1:
	//	return m.forExplorerService(block) // others service, explorer, exchange, zkp, nft, etc.
	//case 2:
	//	return m.forExchangeService(block)
	case SRV_RECORD:
		return m.forRecordService(block)
	default:
		return errors.New("no block operation service found")
	}
}

func (m *Monitor) SwitchService(srv int) error {
	log.Debug("Srv switch start", "srv", srv, "ch", cap(m.srvCh))
	select {
	case m.srvCh <- srv:
	case <-m.exitCh:
		return nil
	}
	log.Debug("Srv switch end", "srv", srv, "ch", cap(m.srvCh))
	return nil
}

func (m *Monitor) doSwitch(srv int) error {
	if m.srv.Load() != int32(srv) {
		switch m.srv.Load() {
		case SRV_MODEL:
			if m.lastNumber.Load() > 0 {
				m.fs.Anchor(m.lastNumber.Load())
				m.fs.Flush()
				log.Debug("Model srv flush", "last", m.lastNumber.Load())
			}
		case SRV_RECORD:
			if m.lastNumber.Load() > 0 {
				log.Debug("Record srv flush", "last", m.lastNumber.Load())
				m.engine.Set([]byte("srv_record_last"), []byte(strconv.FormatUint(m.lastNumber.Load(), 16)))
			}
		default:
			return errors.New("Invalid current service")
		}

		switch srv {
		case SRV_MODEL:
			m.fs.InitBlockNumber()
			m.lastNumber.Store(m.fs.LastListenBlockNumber())
			log.Debug("Model srv load", "last", m.lastNumber.Load())
		case SRV_RECORD:
			if v := m.engine.Get([]byte("srv_record_last")); v != nil {
				if number, err := strconv.ParseUint(string(v), 16, 64); err == nil {
					m.lastNumber.Store(number)
				} else {
					m.lastNumber.Store(0)
				}
			} else {
				m.lastNumber.Store(0)
			}
			log.Debug("Record srv load", "last", m.lastNumber.Load())
		default:
			return errors.New("Unknow service")
		}
		m.srv.Store(int32(srv))
		log.Info("Service switch", "srv", m.srv.Load(), "last", m.lastNumber.Load())
	}

	return nil
}

func (m *Monitor) storeLastNumber(last uint64) {
	log.Info("Last number changed", "last", last)
	m.lastNumber.Store(last)
}

// only for examples
func (m *Monitor) forExplorerService(block *types.Block) error {
	return errors.New("not support")
}

func (m *Monitor) forExchangeService(block *types.Block) error {
	return errors.New("not support")
}

func (m *Monitor) forRecordService(block *types.Block) error {
	if block.Number%4096 == 0 {
		log.Info("Block record", "num", block.Number, "hash", block.Hash, "txs", len(block.Txs), "last", m.lastNumber.Load())
	}
	if len(block.Txs) > 0 {
		for _, t := range block.Txs {
			x := new(big.Float).Quo(new(big.Float).SetInt(t.Amount), new(big.Float).SetInt(big.NewInt(params1.Cortex)))
			log.Debug("Tx record", "hash", t.Hash, "amount", x, "gas", t.GasLimit, "receipt", t.Recipient, "payload", t.Payload)

			if v, err := json.Marshal(t); err != nil {
				return err
			} else {
				m.engine.Set(t.Hash.Bytes(), v)
			}
		}
	}

	if v, err := json.Marshal(block); err != nil {
		return err
	} else {
		m.engine.Set(block.Hash.Bytes(), v)
	}

	m.engine.Set([]byte("srv_record_last"), []byte(strconv.FormatUint(block.Number, 16)))
	return nil
}

func (m *Monitor) forModelService(block *types.Block) error {
	i := block.Number
	if i%65536 == 0 {
		defer func() {
			//elapsedA := time.Duration(mclock.Now()) - time.Duration(m.start)
			//log.Info("Nas monitor", "start", m.startNumber.Load(), "max", uint64(m.CurrentNumber()), "last", m.lastNumber.Load(), "cur", i, "bps", math.Abs(float64(i)-float64(m.startNumber.Load()))*1000*1000*1000/float64(elapsedA), "elapsed", common.PrettyDuration(elapsedA), "scope", m.scope, "db", common.PrettyDuration(m.fs.Metrics()), "blocks", len(m.fs.Blocks()), "txs", m.fs.Txs(), "files", len(m.fs.Files()), "root", m.fs.Root())
			m.fs.SkipPrint()
		}()
	}

	if hash, suc := m.blockCache.Get(i); !suc || hash != block.Hash.Hex() {
		if record, parseErr := m.parseBlockTorrentInfo(block); parseErr != nil {
			log.Error("Parse new block", "number", block.Number, "block", block, "error", parseErr)
			return parseErr
		} else if record {
			elapsed := time.Duration(mclock.Now()) - time.Duration(m.start)

			if m.ckp != nil && m.ckp.TfsCheckPoint > 0 && i == m.ckp.TfsCheckPoint {
				if common.BytesToHash(m.fs.GetRoot(i)) == m.ckp.TfsRoot {
					log.Warn("FIRST MILESTONE PASS", "number", i, "root", m.fs.Root(), "blocks", len(m.fs.Blocks()), "txs", m.fs.Txs(), "files", len(m.fs.Files()), "elapsed", common.PrettyDuration(elapsed))
				} else {
					log.Error("Fs checkpoint failed", "number", i, "root", m.fs.Root(), "blocks", len(m.fs.Blocks()), "files", len(m.fs.Files()), "txs", m.fs.Txs(), "elapsed", common.PrettyDuration(elapsed), "exp", m.ckp.TfsRoot, "leaves", len(m.fs.Leaves()))
					panic("FIRST MILESTONE ERROR, run './cortex removedb' command to solve this problem")
				}
			}

			log.Debug("Seal fs record", "number", i, "record", record, "root", m.fs.Root().Hex(), "blocks", len(m.fs.Blocks()), "txs", m.fs.Txs(), "files", len(m.fs.Files()), "ckp", m.fs.CheckPoint())
		} else {
			if m.fs.LastListenBlockNumber() < i {
				m.fs.Anchor(i)
			}

			log.Trace("Confirm to seal the fs record", "number", i)
		}
		m.blockCache.Add(i, block.Hash.Hex())
	}

	return nil
}
