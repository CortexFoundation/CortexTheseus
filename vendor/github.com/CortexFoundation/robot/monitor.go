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
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/metrics"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"github.com/CortexFoundation/robot/backend"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/types"
	lru "github.com/hashicorp/golang-lru/v2"
	ttl "github.com/hashicorp/golang-lru/v2/expirable"
	"github.com/ucwong/golang-kv"
	"math"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
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
	log.Debug("Chain segment frozen", "from", minNumber, "to", maxNumber, "range", uint64(maxNumber-minNumber+1), "counter", counter, "scope", m.scope, "current", m.CurrentNumber(), "prog", float64(maxNumber)/float64(m.CurrentNumber()), "last", m.lastNumber.Load(), "bps", float64(counter)*1000*1000*1000/float64(elapsedA), "elapsed", common.PrettyDuration(elapsedA))
	return uint64(maxNumber - minNumber)
}
