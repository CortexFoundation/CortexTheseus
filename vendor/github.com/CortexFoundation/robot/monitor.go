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
	"math"
	//"sort"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/types"
	lru "github.com/hashicorp/golang-lru/v2"
	ttl "github.com/hashicorp/golang-lru/v2/expirable"
	"github.com/ucwong/golang-kv"

	"github.com/CortexFoundation/robot/backend"
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
	//blockCache *lru.LRU[uint64, string]
	sizeCache *ttl.LRU[string, uint64]
	ckp       *params.TrustedCheckpoint
	start     mclock.AbsTime

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
		scope: uint64(math.Max(float64(runtime.NumCPU()*2), float64(8))),
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

	m.fs.Init()

	m.lastNumber.Store(0)
	m.currentNumber.Store(0)
	m.startNumber.Store(0)

	m.terminated.Store(false)
	//m.blockCache = lru.NewLRU[uint64, string](delay, nil, time.Second*60)
	m.blockCache, _ = lru.New[uint64, string](delay)
	m.sizeCache = ttl.NewLRU[string, uint64](batch, nil, time.Second*60)
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

/*func (m *Monitor) loadHistory() error {
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
}*/

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
		if err := m.engine.Close(); err != nil {
			return err
		}
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
			//m.fs.Init()
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

	counter := 0

	// Helper function to determine the next delay based on sync progress
	getNextDelay := func(progress uint64) time.Duration {
		if progress >= delay {
			return 0 // Trigger immediately
		}
		if progress > 1 {
			return time.Millisecond * 2000
		}
		if progress == 1 {
			return time.Millisecond * 6750
		}

		// If progress is 0, check for listener and checkpoint status
		if !m.listen && ((m.ckp != nil && m.currentNumber.Load() >= m.ckp.TfsCheckPoint) || (m.ckp == nil && m.currentNumber.Load() > 0)) {
			// This part seems to have a specific termination or pause logic
			// The original code has some commented-out `return`, so I'm assuming it's a "steady state" delay.
			return time.Millisecond * 6750
		}

		return time.Millisecond * 6750 // Default case for other conditions
	}

	for {
		select {
		case sv := <-m.srvCh:
			if err := m.doSwitch(sv); err != nil {
				log.Error("Service switch failed", "srv", sv, "err", err)
			}

		case <-timer.C:
			progress := m.syncLastBlock()

			// Determine the next delay and reset the timer
			nextDelay := getNextDelay(progress)
			timer.Reset(nextDelay)

			// Log status periodically
			counter += int(progress)
			if counter > 65536 {
				log.Info("Monitor status", "blocks", progress, "current", m.CurrentNumber(), "latest", m.lastNumber.Load(), "txs", m.fs.Txs(), "ckp", m.fs.CheckPoint(), "last", m.fs.LastListenBlockNumber(), "progress", progress, "root", m.fs.Root())
				counter = 0
			}
			// Always flush at the end of a timer cycle
			//m.fs.Anchor(m.lastNumber.Load())
			//m.fs.Flush()

		case <-m.exitCh:
			return
		}
	}
}

func (m *Monitor) skip(num uint64) (bool, uint64) {
	if m.srv.Load() != SRV_MODEL {
		return false, 0
	}

	if len(m.ckp.Skips) == 0 || num > m.ckp.Skips[len(m.ckp.Skips)-1].To || num < m.ckp.Skips[0].From {
		return false, 0
	}

	for _, skip := range m.ckp.Skips {
		if num > skip.From && num < skip.To {
			return true, skip.To
		}
	}
	return false, 0
}

func (m *Monitor) syncLastBlock() uint64 {
	currentNumber := m.currentNumber.Load()
	lastNumber := m.lastNumber.Load()

	// Step 1: Handle rollback logic if current block number is less than the last processed number.
	if currentNumber < lastNumber {
		log.Warn("Fs sync rollback detected", "current", currentNumber, "last", lastNumber, "offset", lastNumber-currentNumber)
		rollbackNumber := uint64(0)
		if currentNumber > 65536 {
			rollbackNumber = currentNumber - 65536
		}
		m.lastNumber.Store(rollbackNumber)
		m.startNumber.Store(rollbackNumber)
	}

	// Step 2: Determine the block range for this sync batch.
	minNumber := m.lastNumber.Load() + 1
	maxNumber := currentNumber
	if currentNumber > delay {
		maxNumber = currentNumber - delay
	}

	// If the last processed block is unexpectedly higher than the current block (after rollback check),
	// this indicates a need to sync backward from the last number.
	if m.lastNumber.Load() > currentNumber {
		minNumber = m.lastNumber.Load() - batch
		if m.lastNumber.Load() < batch {
			minNumber = 0 // Avoids underflow if lastNumber is smaller than batch
		}
	}

	// Adjust maxNumber to not exceed the batch size.
	if maxNumber > minNumber+batch {
		maxNumber = minNumber + batch
	}

	if maxNumber < minNumber {
		return 0
	}

	m.start = mclock.Now()

	processedCount := 0
	for i := minNumber; i <= maxNumber; {
		if m.terminated.Load() {
			log.Warn("Fs scan terminated by signal", "lastProcessed", i-1)
			m.lastNumber.Store(i - 1)
			return 0
		}

		if m.ckp != nil {
			if skip, to := m.skip(i); skip {
				i = to
				continue
			}
		}

		// Step 3: Fetch blocks in a batch or individually based on remaining scope.
		remainingScope := maxNumber - i
		if remainingScope >= m.scope {
			// Process a batch of blocks
			blocks, err := m.rpcBatchBlockByNumber(i, i+m.scope)
			if err != nil {
				return m.handleSyncError("Batch sync old block failed", err, i-1)
			}

			// Send blocks to the processing channel
			for _, block := range blocks {
				m.taskCh <- block
			}

			// Wait for the processing results for the entire batch
			for range blocks {
				select {
				case err := <-m.errCh:
					if err != nil {
						return m.handleSyncError("Processing error", err, i-1)
					}
				case <-m.exitCh:
					log.Info("Task checker quit signal received")
					m.lastNumber.Store(i - 1)
					return 0
				}
			}

			i += uint64(len(blocks))
			processedCount += len(blocks)
		} else {
			// Process a single block
			block, err := m.rpcBlockByNumber(i)
			if err != nil {
				return m.handleSyncError("Sync old block failed", err, i-1)
			}

			if err := m.solve(block); err != nil {
				return m.handleSyncError("solve err", err, i-1)
			}

			i++
			processedCount++
		}
	}

	// Step 4: Finalize the sync operation.
	m.lastNumber.Store(maxNumber)
	elapsed := time.Duration(mclock.Now()) - time.Duration(m.start)

	log.Debug("Chain segment frozen",
		"from", minNumber,
		"to", maxNumber,
		"totalBlocks", uint64(maxNumber-minNumber+1),
		"processed", processedCount,
		"scope", m.scope,
		"current", m.CurrentNumber(),
		"last", m.lastNumber.Load(),
		"elapsed", common.PrettyDuration(elapsed),
		"bps", float64(processedCount)/elapsed.Seconds(),
	)

	return uint64(maxNumber - minNumber)
}

// handleSyncError is a helper function to log an error, update the last processed block number, and return 0.
func (m *Monitor) handleSyncError(msg string, err error, lastBlock uint64) uint64 {
	log.Error(msg, "error", err, "lastProcessed", lastBlock)
	m.lastNumber.Store(lastBlock)
	return 0
}
