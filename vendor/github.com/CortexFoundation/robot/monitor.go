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
	//"context"
	"errors"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/metrics"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"github.com/CortexFoundation/robot/backend"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/types"
	lru "github.com/hashicorp/golang-lru"
	"math"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

const (
	batch = params.SyncBatch
	delay = params.Delay
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
	//dl     *backend.TorrentManager

	exitCh        chan any
	terminated    atomic.Bool
	lastNumber    uint64
	startNumber   uint64
	scope         uint64
	currentNumber atomic.Uint64
	wg            sync.WaitGroup
	rpcWg         sync.WaitGroup

	//taskCh      chan *types.Block
	//newTaskHook func(*types.Block)
	blockCache *lru.Cache
	sizeCache  *lru.Cache
	ckp        *params.TrustedCheckpoint
	start      mclock.AbsTime

	local  bool
	listen bool

	startOnce sync.Once
	closeOnce sync.Once
	mode      string

	lock sync.RWMutex

	callback chan any
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
		exitCh:     make(chan any),
		lastNumber: uint64(0),
		scope:      uint64(math.Min(float64(runtime.NumCPU()), float64(8))),
		//taskCh:        make(chan *types.Block, batch),
		//start: mclock.Now(),
	}
	fs_, err := backend.NewChainDB(flag)
	if err != nil {
		log.Error("file storage failed", "err", err)
		return nil, err
	}
	m.fs = fs_
	m.currentNumber.Store(0)
	m.terminated.Store(false)
	m.blockCache, _ = lru.New(delay)
	m.sizeCache, _ = lru.New(batch)
	m.listen = listen
	m.callback = callback

	//if err := m.dl.Start(); err != nil {
	//	log.Warn("Fs start error")
	//	return nil, err
	//}

	m.mode = flag.Mode

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

//func (m *Monitor) DB() *backend.ChainDB {
//	return m.fs
//}

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
			m.download(k, v)
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

func (m *Monitor) download(k string, v uint64) {
	if m.mode != params.LAZY && m.callback != nil {
		task := types.NewBitsFlow(k, v)
		m.callback <- task
	}
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
			m.lastNumber = 0
			m.startNumber = 0
			if m.lastNumber > checkpoint.TfsCheckPoint {
				//m.fs.LastListenBlockNumber = 0
				m.fs.Anchor(0)
				//m.lastNumber = 0
				//if err := m.fs.Reset(); err != nil {
				//	return err
				//}
			}
			log.Warn("Fs storage is reloading ...", "name", m.ckp.Name, "number", checkpoint.TfsCheckPoint, "version", common.BytesToHash(version), "checkpoint", checkpoint.TfsRoot, "blocks", len(m.fs.Blocks()), "files", len(m.fs.Files()), "txs", m.fs.Txs(), "lastNumber", m.lastNumber, "last in db", m.fs.LastListenBlockNumber())
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
	capcity := uint64(0)
	seed := 0
	pause := 0
	pending := 0

	for _, file := range fileMap {
		var bytesRequested uint64
		if file.Meta.RawSize > file.LeftSize {
			bytesRequested = file.Meta.RawSize - file.LeftSize
		}
		capcity += bytesRequested
		log.Debug("File storage info", "addr", file.ContractAddr, "ih", file.Meta.InfoHash, "remain", common.StorageSize(file.LeftSize), "raw", common.StorageSize(file.Meta.RawSize), "request", common.StorageSize(bytesRequested))
		m.download(file.Meta.InfoHash, bytesRequested)
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

/*func (m *Monitor) taskLoop() {
	defer m.wg.Done()
	for {
		select {
		case task := <-m.taskCh:
			if m.newTaskHook != nil {
				m.newTaskHook(task)
			}

			if err := m.solve(task); err != nil {
				log.Warn("Block solved failed, try again", "err", err, "num", task.Number)
			}
		case <-m.exitCh:
			if cap(m.taskCh) > 0 {
				continue
			}
			log.Info("Monitor task channel closed")
			return
		}
	}
}*/

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
		log.Warn("IPC is empty")
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
	if size, suc := m.sizeCache.Get(address); suc && size.(uint64) == 0 {
		return size.(uint64), nil
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

		if m.mode == params.FULL {
			m.download(meta.InfoHash, 512*1024)
		} else {
			m.download(meta.InfoHash, 0)
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
					m.download(file.Meta.InfoHash, bytesRequested)
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

func (m *Monitor) Stop() {
	m.lock.Lock()
	defer m.lock.Unlock()
	//m.closeOnce.Do(func() {
	if m.terminated.Load() {
		return
	}
	m.terminated.Store(true)

	m.exit()
	log.Info("Monitor is waiting to be closed")
	m.blockCache.Purge()
	m.sizeCache.Purge()

	//log.Info("Fs client listener synchronizing closing")
	//if err := m.dl.Close(); err != nil {
	//	log.Error("Monitor Fs Manager closed", "error", err)
	//}

	if err := m.fs.Close(); err != nil {
		log.Error("Monitor File Storage closed", "error", err)
	}
	log.Info("Fs listener synchronizing closed")
	//})
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

	m.lastNumber = m.fs.LastListenBlockNumber()
	m.currentBlock()
	m.startNumber = uint64(math.Min(float64(m.fs.LastListenBlockNumber()), float64(m.currentNumber.Load()))) // ? m.currentNumber:m.fs.LastListenBlockNumber

	if err := m.indexCheck(); err != nil {
		return err
	}

	//if err := m.loadHistory(); err != nil {
	//	return err
	//}
	//m.wg.Add(1)
	//go m.taskLoop()
	//m.wg.Add(1)
	//go m.listenLatestBlock()
	m.wg.Add(1)
	go m.syncLatestBlock()

	return nil
}

/*func (m *Monitor) listenLatestBlock() {
	defer m.wg.Done()
	timer := time.NewTimer(time.Second * queryTimeInterval)
	defer timer.Stop()
	for {
		select {
		case <-timer.C:
			m.currentBlock()
			if m.local {
				timer.Reset(time.Second * queryTimeInterval)
			} else {
				timer.Reset(time.Second * queryTimeInterval * 10)
			}
		case <-m.exitCh:
			log.Debug("Block listener stopped")
			return
		}
	}
}*/

func (m *Monitor) syncLatestBlock() {
	defer m.wg.Done()
	timer := time.NewTimer(time.Second * params.QueryTimeInterval)
	defer timer.Stop()
	progress, counter, end := uint64(0), 0, false
	for {
		select {
		case <-timer.C:
			//m.currentBlock()
			progress = m.syncLastBlock()
			// Avoid sync in full mode, fresh interval may be less.
			if progress >= delay {
				//timer.Reset(0)
				end = false
				timer.Reset(time.Millisecond * 2000)
			} else if progress >= 1 {
				end = false
				timer.Reset(time.Millisecond * 3000)
			} else {
				if !m.listen {
					if (m.ckp != nil && m.currentNumber.Load() >= m.ckp.TfsCheckPoint) || (m.ckp == nil && m.currentNumber.Load() > 0) {
						if !end {
							end = true
							timer.Reset(time.Millisecond * 15000)
							continue
						}
						m.fs.Flush()
						//go m.exit()
						elapsed := time.Duration(mclock.Now()) - time.Duration(m.start)
						log.Info("Finish sync, listener will be paused", "current", m.currentNumber.Load(), "elapsed", common.PrettyDuration(elapsed), "progress", progress, "end", end, "last", m.lastNumber)
						//return
						timer.Reset(time.Millisecond * 1000 * 180)
						end = false
						continue
					}
				}
				timer.Reset(time.Millisecond * 6000)
			}
			counter++
			if counter%10 == 0 {
				log.Info("Monitor status", "blocks", progress, "current", m.CurrentNumber(), "latest", m.lastNumber, "end", end, "txs", m.fs.Txs(), "ckp", m.fs.CheckPoint(), "last", m.fs.LastListenBlockNumber())
				counter = 0
			}
			m.fs.Flush()
		case <-m.exitCh:
			log.Info("Block syncer stopped")
			return
		}
	}
}

func (m *Monitor) currentBlock() (uint64, error) {
	var currentNumber hexutil.Uint64

	rpcCurrentMeter.Mark(1)
	if err := m.cl.Call(&currentNumber, "ctxc_blockNumber"); err != nil {
		log.Error("Call ipc method ctxc_blockNumber failed", "error", err)
		return 0, err
	}
	if m.CurrentNumber() != uint64(currentNumber) {
		m.currentNumber.Store(uint64(currentNumber))
	}

	return uint64(currentNumber), nil
}

func (m *Monitor) skip(i uint64) bool {
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
	currentNumber, err := m.currentBlock()
	if err != nil {
		return 0
	}

	if currentNumber < m.lastNumber {
		log.Warn("Fs sync rollback", "current", currentNumber, "last", m.lastNumber, "offset", m.lastNumber-currentNumber)
		if currentNumber > 65536 {
			m.lastNumber = currentNumber - 65536
		} else {
			m.lastNumber = 0
		}
		m.startNumber = m.lastNumber
	}

	minNumber := m.lastNumber + 1
	maxNumber := uint64(0)
	if currentNumber > delay {
		maxNumber = currentNumber - delay
	}

	if m.lastNumber > currentNumber {
		if m.lastNumber > batch {
			minNumber = m.lastNumber - batch
		}
	}

	if maxNumber > batch*8+minNumber {
		maxNumber = minNumber + batch*8
	}
	if maxNumber < minNumber {
		return 0
	}
	if m.start == 0 {
		m.start = mclock.Now()
	}
	//start := mclock.Now()
	for i := minNumber; i <= maxNumber; { // i++ {
		if m.terminated.Load() {
			log.Warn("Fs scan terminated", "number", i)
			maxNumber = i - 1
			break
		}

		if m.ckp != nil && m.skip(i) {
			//m.lastNumber = i - 1
			i++
			continue
		}

		if maxNumber-i >= m.scope {
			blocks, rpcErr := m.rpcBatchBlockByNumber(i, i+m.scope)
			if rpcErr != nil {
				log.Error("Sync old block failed", "number", i, "error", rpcErr)
				m.lastNumber = i - 1
				return 0
			}
			for _, rpcBlock := range blocks {
				if err := m.solve(rpcBlock); err != nil {
					m.lastNumber = i - 1
					return 0
				}
				i++
				/*if len(m.taskCh) < cap(m.taskCh) {
					m.taskCh <- rpcBlock
					i++
				} else {
					m.lastNumber = i - 1
					if maxNumber-minNumber > delay/2 {
						elapsed := time.Duration(mclock.Now()) - time.Duration(start)
						elapsedA := time.Duration(mclock.Now()) - time.Duration(m.start)
						log.Warn("Chain segment frozen", "from", minNumber, "to", i, "range", uint64(i-minNumber), "current", uint64(m.currentNumber), "progress", float64(i)/float64(m.currentNumber), "last", m.lastNumber, "elapsed", common.PrettyDuration(elapsed), "bps", float64(i-minNumber)*1000*1000*1000/float64(elapsed), "bps_a", float64(maxNumber)*1000*1000*1000/float64(elapsedA), "cap", len(m.taskCh))
					}
					return 0
				}*/
			}
		} else {

			rpcBlock, rpcErr := m.rpcBlockByNumber(i)
			if rpcErr != nil {
				log.Error("Sync old block failed", "number", i, "error", rpcErr)
				m.lastNumber = i - 1
				return 0
			}
			if err := m.solve(rpcBlock); err != nil {
				m.lastNumber = i - 1
				return 0
			}
			i++
			/*if len(m.taskCh) < cap(m.taskCh) {
				m.taskCh <- rpcBlock
				i++
			} else {
				m.lastNumber = i - 1
				if maxNumber-minNumber > delay/2 {
					elapsed := time.Duration(mclock.Now()) - time.Duration(start)
					elapsedA := time.Duration(mclock.Now()) - time.Duration(m.start)
					log.Warn("Chain segment frozen", "from", minNumber, "to", i, "range", uint64(i-minNumber), "current", uint64(m.currentNumber), "progress", float64(i)/float64(m.currentNumber), "last", m.lastNumber, "elapsed", common.PrettyDuration(elapsed), "bps", float64(i-minNumber)*1000*1000*1000/float64(elapsed), "bps_a", float64(maxNumber)*1000*1000*1000/float64(elapsedA), "cap", len(m.taskCh))
				}
				return 0
			}*/
		}
	}
	m.lastNumber = maxNumber
	//if maxNumber-minNumber > batch-1 {
	if maxNumber-minNumber > delay {
		elapsedA := time.Duration(mclock.Now()) - time.Duration(m.start)
		log.Debug("Chain segment frozen", "from", minNumber, "to", maxNumber, "range", uint64(maxNumber-minNumber), "current", uint64(m.CurrentNumber()), "progress", float64(maxNumber)/float64(m.CurrentNumber()), "last", m.lastNumber, "bps", float64(maxNumber)*1000*1000*1000/float64(elapsedA), "elapsed", common.PrettyDuration(elapsedA))
	}
	return uint64(maxNumber - minNumber)
}

func (m *Monitor) solve(block *types.Block) error {
	i := block.Number
	if i%65536 == 0 {
		defer func() {
			elapsedA := time.Duration(mclock.Now()) - time.Duration(m.start)
			log.Info("Nas monitor", "start", m.startNumber, "max", uint64(m.CurrentNumber()), "last", m.lastNumber, "cur", i, "bps", math.Abs(float64(i)-float64(m.startNumber))*1000*1000*1000/float64(elapsedA), "elapsed", common.PrettyDuration(elapsedA), "scope", m.scope, "db", common.PrettyDuration(m.fs.Metrics()), "blocks", len(m.fs.Blocks()), "txs", m.fs.Txs(), "files", len(m.fs.Files()), "root", m.fs.Root())
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
				//m.fs.LastListenBlockNumber = i
				m.fs.Anchor(i)
			}

			log.Trace("Confirm to seal the fs record", "number", i)
		}
		m.blockCache.Add(i, block.Hash.Hex())
	}
	return nil
}
