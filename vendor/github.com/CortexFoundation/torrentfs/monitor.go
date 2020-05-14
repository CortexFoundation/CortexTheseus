// Copyright 2020 The CortexTheseus Authors
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
package torrentfs

import (
	//"encoding/json"
	"errors"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/types"
	"github.com/anacrolix/torrent/metainfo"
	lru "github.com/hashicorp/golang-lru"
	//"net"
	//"net/http"
	//"os"
	"runtime"
	//"sort"
	"strconv"
	//"strings"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

// Errors that are used throughout the Torrent API.
var (
//ErrBuildConn      = errors.New("build internal-rpc connection failed")
//ErrGetLatestBlock = errors.New("get latest block failed")
//ErrNoRPCClient    = errors.New("no rpc client")

//ErrBlockHash = errors.New("block or parent block hash invalid")
//blockCache, _ = lru.New(6)
//healthPeers, _ = lru.New(50)
//sizeCache, _   = lru.New(batch)
)

const (
	batch = SyncBatch
	delay = Delay

//defaultTimerInterval  = 1
//connTryTimes          = 300
//connTryInterval = 2
//fetchBlockTryTimes    = 5
//fetchBlockTryInterval = 3
//fetchBlockLogStep     = 10000
//minBlockNum           = 0

//maxSyncBlocks = 1024
)

// Monitor observes the data changes on the blockchain and synchronizes.
// cl for ipc/rpc communication, dl for download manager, and fs for data storage.
type Monitor struct {
	config *Config
	cl     *rpc.Client
	fs     *ChainIndex
	dl     *TorrentManager

	//listenID rpc.ID

	//uncheckedCh chan uint64

	exitCh        chan struct{}
	terminated    int32
	lastNumber    uint64
	startNumber   uint64
	scope         uint64
	currentNumber uint64
	//dirty      bool

	//closeOnce sync.Once
	wg sync.WaitGroup
	//peersWg sync.WaitGroup
	rpcWg sync.WaitGroup
	//trackerLock sync.Mutex
	//portLock sync.Mutex
	//portsWg  sync.WaitGroup

	taskCh      chan *types.Block
	newTaskHook func(*types.Block)
	blockCache  *lru.Cache
	//healthPeers *lru.Cache
	sizeCache *lru.Cache
	ckp       *params.TrustedCheckpoint
	start     mclock.AbsTime
}

// NewMonitor creates a new instance of monitor.
// Once Ipcpath is settle, this method prefers to build socket connection in order to
// get higher communicating performance.
// IpcPath is unavailable on windows.
func NewMonitor(flag *Config, cache, compress bool) (m *Monitor, e error) {
	log.Info("Initialising FS")
	// File Storage
	fs, fsErr := NewChainIndex(flag)
	if fsErr != nil {
		log.Error("file storage failed", "err", fsErr)
		return nil, fsErr
	}
	log.Info("File storage initialized")

	// Torrent Manager
	tMana, err := NewTorrentManager(flag, fs.ID(), cache, compress)
	if err != nil || tMana == nil {
		log.Error("fs manager failed")
		return nil, errors.New("fs download manager initialise failed")
	}
	log.Info("Fs manager initialized")

	m = &Monitor{
		config: flag,
		cl:     nil,
		fs:     fs,
		dl:     tMana,
		//uncheckedCh: make(chan uint64, 20),
		exitCh:        make(chan struct{}),
		terminated:    0,
		lastNumber:    uint64(0),
		scope:         uint64(math.Min(float64(runtime.NumCPU()*2), float64(8))),
		currentNumber: uint64(0),
		taskCh:        make(chan *types.Block, batch),
		start:         mclock.Now(),
	}
	m.blockCache, _ = lru.New(delay)
	//m.healthPeers, _ = lru.New(0)
	m.sizeCache, _ = lru.New(batch)
	e = nil

	/*log.Info("Loading storage data ... ...", "latest", m.fs.LastListenBlockNumber)

		fileMap := make(map[metainfo.Hash]*FileInfo)
		for _, block := range m.fs.Blocks() {
			if record, parseErr := m.parseBlockTorrentInfo(block); parseErr != nil {
	                        log.Error("Parse new block", "number", block.Number, "block", block, "error", parseErr)
	                        return nil, parseErr
			} else {
				log.Info("Block storage info", "number", block.Number, "record", record)
			}
		}

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
			bytesRequested = 0
			if file.Meta.RawSize > file.LeftSize {
				bytesRequested = file.Meta.RawSize - file.LeftSize
			}
			capcity += bytesRequested
			log.Info("File storage info", "addr", file.ContractAddr, "hash", file.Meta.InfoHash, "remain", common.StorageSize(file.LeftSize), "raw", common.StorageSize(file.Meta.RawSize), "request", common.StorageSize(bytesRequested))
			m.dl.UpdateTorrent(FlowControlMeta{
				InfoHash:       file.Meta.InfoHash,
				BytesRequested: bytesRequested,
				IsCreate:       true,
			})
			if file.LeftSize == 0 {
				seed += 1
			} else if file.Meta.RawSize == file.LeftSize && file.LeftSize > 0 {
				pending += 1
			} else if file.Meta.RawSize > file.LeftSize && file.LeftSize > 0 {
				pause += 1
			}
		}
		log.Info("Storage current state", "total", len(fileMap), "seed", seed, "pause", pause, "pending", pending, "capcity", common.StorageSize(capcity))*/

	return m, e
}

func (m *Monitor) indexInit() error {
	log.Info("Loading storage data ... ...", "latest", m.fs.LastListenBlockNumber, "checkpoint", m.fs.CheckPoint, "root", m.fs.Root(), "version", m.fs.Version(), "current", m.currentNumber)
	genesis, err := m.rpcBlockByNumber(0)
	if err != nil {
		return err
	}

	if checkpoint, ok := params.TrustedCheckpoints[genesis.Hash]; ok {
		//if uint64(len(m.fs.Blocks())) < checkpoint.TfsBlocks || uint64(len(m.fs.Files())) < checkpoint.TfsFiles {
		//	log.Warn("Fs storage version upgrade", "version", m.fs.Version(), "blocks", len(m.fs.Blocks()), "files", len(m.fs.Files()))
		//	m.lastNumber = 0
		//}
		/*if uint64(len(m.fs.Blocks())) < checkpoint.TfsBlocks || uint64(m.fs.CheckPoint) < checkpoint.TfsCheckPoint || uint64(len(m.fs.Files())) < checkpoint.TfsFiles {
			m.lastNumber = m.fs.CheckPoint
			log.Info("Torrent fs block unmatch, reloading ...", "blocks", len(m.fs.Blocks()), "limit", checkpoint.TfsBlocks, "ckp", m.fs.CheckPoint, "checkpoint", checkpoint.TfsCheckPoint, "files", len(m.fs.Files()))
		} else {
			block := m.fs.GetBlockByNumber(checkpoint.TfsCheckPoint)
			if block != nil && checkpoint.TfsCkpHead == block.Hash {
				log.Info("Torrent fs block passed", "blocks", len(m.fs.Blocks()), "limit", checkpoint.TfsBlocks, "ckp", m.fs.CheckPoint, "checkpoint", checkpoint.TfsCheckPoint, "files", len(m.fs.Files()), "head", block.Hash)
			} else {
				log.Info("Torrent fs check point unmatch, reloading ...", "blocks", len(m.fs.Blocks()), "limit", checkpoint.TfsBlocks, "ckp", m.fs.CheckPoint, "checkpoint", checkpoint.TfsCheckPoint, "files", len(m.fs.Files()), "head", block.Hash)
				m.lastNumber = 0
			}
		}*/
		m.ckp = checkpoint

		version := m.fs.GetRootByNumber(checkpoint.TfsCheckPoint)
		if common.BytesToHash(version) != checkpoint.TfsRoot {
			log.Warn("Fs storage is reloading ...", "name", m.ckp.Name, "number", checkpoint.TfsCheckPoint, "version", common.BytesToHash(version), "checkpoint", checkpoint.TfsRoot, "blocks", len(m.fs.Blocks()), "files", len(m.fs.Files()), "txs", m.fs.Txs())
			if m.lastNumber > checkpoint.TfsCheckPoint {
				m.lastNumber = 0
				err := m.fs.Reset()
				if err != nil {
					return err
				}
			}
			//m.fs.LastListenBlockNumber = 0
			//m.fs.CheckPoint = 0
			//	if m.lastNumber > 0 {
			//	m.lastNumber = 0
			//	err := m.fs.Reset()
			//	if err != nil {
			//		return err
			//		}
			//	}
			//return nil
		} else {
			log.Info("Fs storage version check passed", "name", m.ckp.Name, "number", checkpoint.TfsCheckPoint, "version", common.BytesToHash(version), "blocks", len(m.fs.Blocks()), "files", len(m.fs.Files()), "txs", m.fs.Txs())
		}
	}

	/*blocks := m.fs.Blocks()
	sort.Slice(blocks, func(i, j int) bool {
		return blocks[i].Number < blocks[j].Number
	})*/

	/*for i, block := range m.fs.Blocks() {
		if block.Number > m.currentNumber {
			break
		}
		if record, parseErr := m.parseBlockTorrentInfo(block); parseErr != nil {
			log.Error("Parse new block", "number", block.Number, "block", block, "error", parseErr)
			return parseErr
		} else {
			log.Debug("Block storage info", "number", block.Number, "record", record, "i", i)
			if !record {
				log.Warn("Block storage info", "number", block.Number, "record", record, "txs", block.Txs)
			}
		}
	}

	log.Info("Block refresh finished", "len", len(m.fs.Blocks()), "txs", m.fs.Txs())*/

	fileMap := make(map[metainfo.Hash]*types.FileInfo)
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
		m.dl.UpdateTorrent(types.FlowControlMeta{
			InfoHash:       file.Meta.InfoHash,
			BytesRequested: bytesRequested,
			IsCreate:       true,
		})
		if file.LeftSize == 0 {
			seed += 1
		} else if file.Meta.RawSize == file.LeftSize && file.LeftSize > 0 {
			pending += 1
		} else if file.Meta.RawSize > file.LeftSize && file.LeftSize > 0 {
			pause += 1
		}
	}
	log.Info("Storage current state", "total", len(m.fs.Files()), "dis", len(fileMap), "seed", seed, "pause", pause, "pending", pending, "capcity", common.StorageSize(capcity), "blocks", len(m.fs.Blocks()), "txs", m.fs.Txs())
	return nil
}

func (m *Monitor) taskLoop() {
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
			log.Info("Monitor task channel closed")
			return
		}
	}
}

// SetConnection method builds connection to remote or local communicator.
func (m *Monitor) buildConnection(clientURI string) (*rpc.Client, error) {
	for {
		time.Sleep(time.Second * queryTimeInterval)
		cl, err := rpc.Dial(clientURI)
		if err != nil {
			log.Warn("Building internal ipc connection ... ", "uri", clientURI, "error", err)
		} else {
			log.Info("Internal ipc connection established", "uri", clientURI)
			return cl, nil
		}

		if atomic.LoadInt32(&(m.terminated)) == 1 {
			break
		}
	}

	return nil, errors.New("building internal ipc connection failed")
}

func (m *Monitor) rpcBlockByNumber(blockNumber uint64) (*types.Block, error) {
	block := &types.Block{}
	blockNumberHex := "0x" + strconv.FormatUint(blockNumber, 16)

	err := m.cl.Call(block, "ctxc_getBlockByNumber", blockNumberHex, true)
	if err == nil {
		return block, nil
	}

	return nil, err //errors.New("[ Internal IPC Error ] try to get block out of times")
}

func (m *Monitor) rpcBatchBlockByNumber(from, to uint64) (result []*types.Block, err error) {
	batch := to - from
	//log.Info("batch", "from", from, "to", to, "batch", batch)
	//c := make(chan bool)
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
			//c <- true
		}(i)
	}

	m.rpcWg.Wait()

	/*for i := 0; i < int(batch); i++ {
		select {
		case <-c:
		}

	}*/
	return result, err
}

/*func (m *Monitor) rpcBlockByHash(blockHash string) (*Block, error) {
	block := &Block{}

	for i := 0; i < fetchBlockTryTimes; i++ {
		err := m.cl.Call(block, "ctxc_getBlockByHash", blockHash, true)
		if err == nil {
			return block, nil
		}

		time.Sleep(time.Second * fetchBlockTryInterval)
		log.Warn("Torrent Fs Internal IPC ctx_getBlockByHash", "retry", i, "error", err)
	}

	return nil, errors.New("[ Internal IPC Error ] try to get block out of times")
}*/

/*var (
	ports            = params.Tracker_ports //[]string{"5007", "5008", "5009", "5010"}
	TRACKER_PORT     []string               // = append(TRACKER_PORT, ports...)
	UDP_TRACKER_PORT []string
	client           http.Client
	//trackers     []string
)

func (m *Monitor) init() {
	TRACKER_PORT = append(TRACKER_PORT, ports...)
	UDP_TRACKER_PORT = params.UDP_Tracker_ports
	client = http.Client{
		Timeout: time.Duration(3 * time.Second),
	}
}

func (m *Monitor) http_tracker_build(ip, port string) string {
	return "http://" + ip + ":" + port + "/announce"
}

func (m *Monitor) udp_tracker_build(ip, port string) string {
	return "udp://" + ip + ":" + port + "/announce"
}*/

/*func (m *Monitor) ws_tracker_build(ip, port string) string {
	return "ws://" + ip + ":" + port + "/announce"
}*/

/*func (m *Monitor) peers() ([]*p2p.PeerInfo, error) {
	var peers []*p2p.PeerInfo // = make([]*p2p.PeerInfo, 0, 25)
	err := m.cl.Call(&peers, "admin_peers")
	if err == nil && len(peers) > 0 {
		flush := false
		start := mclock.Now()
		for _, peer := range peers {
			m.peersWg.Add(1)
			go func(peer *p2p.PeerInfo) {
				defer m.peersWg.Done()
				ip := strings.Split(peer.Network.RemoteAddress, ":")[0]
				if ps, suc := m.batch_udp_healthy(ip, UDP_TRACKER_PORT); suc && len(ps) > 0 {
					for _, p := range ps {
						tracker := m.udp_tracker_build(ip, p) //"udp://" + ip + ":" + p + "/announce"
						if m.healthPeers.Contains(tracker) {
							//continue
						} else {
							flush = true
						}
						m.healthPeers.Add(tracker, tracker)
					}
				}
			}(peer)
		}
		//log.Info("Waiting dynamic tracker", "size", len(peers))
		m.peersWg.Wait()

		var trackers []string
		for _, data := range m.healthPeers.Keys() {
			if str, ok := data.(string); ok {
				trackers = append(trackers, str)
			}
			//trackers = append(trackers, string(k))
		}
		//log.Info("Waiting dynamic tracker done", "size", len(peers))
		if len(trackers) > 0 && flush {
			//m.fs.CurrentTorrentManager().UpdateDynamicTrackers(trackers)
			m.dl.UpdateDynamicTrackers(trackers)
			for _, t := range trackers {
				log.Trace("Healthy trackers", "tracker", t)
			}
			elapsed := time.Duration(mclock.Now()) - time.Duration(start)
			log.Info("✨ FS SEARCH COMPLETE", "ips", len(peers), "healthy", len(trackers), "nodes", m.healthPeers.Len(), "flush", flush, "elapsed", elapsed)
		}
		return peers, nil
	}

	return nil, errors.New("[ Internal IPC Error ] peers")
}*/

/*func (m *Monitor) getBlockByNumber(blockNumber uint64) (*Block, error) {
	block := m.fs.GetBlockByNumber(blockNumber)
	if block == nil {
		return m.rpcBlockByNumber(blockNumber)
	}

	return block, nil
}

func (m *Monitor) getBlockNumber() (hexutil.Uint64, error) {
	var blockNumber hexutil.Uint64

	for i := 0; i < fetchBlockTryTimes; i++ {
		err := m.cl.Call(&blockNumber, "ctx_blockNumber")
		if err == nil {
			return blockNumber, nil
		}

		time.Sleep(time.Second * fetchBlockTryInterval)
		log.Warn("Torrent Fs Internal IPC ctx_blockNumber", "retry", i, "error", err)
	}

	return 0, errors.New("[ Internal IPC Error ] try to get block number out of times")
}*/

func (m *Monitor) getRemainingSize(address string) (uint64, error) {
	if size, suc := m.sizeCache.Get(address); suc && size.(uint64) == 0 {
		return size.(uint64), nil
	}
	var remainingSize hexutil.Uint64
	if err := m.cl.Call(&remainingSize, "ctxc_getUpload", address, "latest"); err != nil {
		return 0, err
	}
	remain := uint64(remainingSize)
	if remain == 0 {
		m.sizeCache.Add(address, remain)
	}
	return remain, nil
}

func (m *Monitor) getReceipt(tx string) (receipt types.TxReceipt, err error) {
	if err = m.cl.Call(&receipt, "ctxc_getTransactionReceipt", tx); err != nil {
		log.Warn("R is nil", "R", tx, "err", err)
		return receipt, err
	}
	return receipt, nil
}

func (m *Monitor) parseFileMeta(tx *types.Transaction, meta *types.FileMeta, b *types.Block) error {
	log.Debug("Monitor", "FileMeta", meta)

	//var receipt types.TxReceipt
	//if err := m.cl.Call(&receipt, "ctxc_getTransactionReceipt", tx.Hash.String()); err != nil {
	//	log.Warn("R is nil", "R", tx.Hash.String(), "err", err)
	//	return err
	//}
	receipt, err := m.getReceipt(tx.Hash.String())
	if err != nil {
		return err
	}

	if receipt.ContractAddr == nil {
		log.Warn("contract address is nil", "tx.Hash.String()", tx.Hash.String())
		return nil
	}

	log.Debug("Transaction Receipt", "address", receipt.ContractAddr.String(), "gas", receipt.GasUsed, "status", receipt.Status) //, "tx", receipt.TxHash.String())

	if receipt.Status != 1 {
		log.Warn("receipt.Status is wrong", "receipt.Status", receipt.Status)
		return nil
	}

	log.Debug("Meta data", "meta", meta)

	info := m.fs.NewFileInfo(meta)
	//info.TxHash = tx.Hash

	info.LeftSize = meta.RawSize
	info.ContractAddr = receipt.ContractAddr
	info.Relate = append(info.Relate, *info.ContractAddr)
	op, update, err := m.fs.UpdateFile(info)
	if err != nil {
		log.Warn("Create file failed", "err", err)
		return err
	} else {
		if update && op == 1 {
			log.Debug("Create new file", "ih", meta.InfoHash, "op", op)
			m.dl.UpdateTorrent(types.FlowControlMeta{
				InfoHash:       meta.InfoHash,
				BytesRequested: 0,
				IsCreate:       true,
			})
		}
	}
	/*var _remainingSize string
	if err := m.cl.Call(&_remainingSize, "ctxc_getUpload", receipt.ContractAddr.String(), "latest"); err != nil {
		log.Warn("Failed to call get upload", "addr", receipt.ContractAddr.String())
		return err
	}

	remainingSize, err_remainingSize := strconv.ParseUint(_remainingSize[2:], 16, 64)
	log.Debug("Monitor", "remainingSize", remainingSize, "err", err_remainingSize)
	if err_remainingSize != nil {
		return err_remainingSize
	}*/

	//remainingSize, err := m.getRemainingSize(receipt.ContractAddr.String())
	//if err != nil {
	//	return err
	//}

	/*bytesRequested := uint64(0)
	if meta.RawSize >= remainingSize {

		//if remainingSize > params.PER_UPLOAD_BYTES {
		//	remainingSize = remainingSize - params.PER_UPLOAD_BYTES
		//} else {
		//	remainingSize = uint64(0)
		//}

		bytesRequested = meta.RawSize - remainingSize
	} else {
		log.Warn("Invalid raw size", "address", receipt.ContractAddr.String(), "hash", meta.InfoHash, "raw", meta.RawSize, "remain", remainingSize)
	}
	log.Debug("Monitor", "meta", meta, "meta info", meta.InfoHash)
	m.dl.UpdateTorrent(FlowControlMeta{
		InfoHash:       meta.InfoHash,
		BytesRequested: bytesRequested,
		IsCreate:       false,
	})
	log.Info("Parse file meta successfully", "tx", receipt.TxHash.Hex(), "remain", remainingSize, "meta", meta)*/
	return nil
}

func (m *Monitor) parseBlockTorrentInfo(b *types.Block) (bool, error) {
	record := false
	if len(b.Txs) > 0 {
		start := mclock.Now()
		var final []types.Transaction
		for _, tx := range b.Txs {
			if meta := tx.Parse(); meta != nil {
				log.Debug("Data encounter", "ih", meta.InfoHash, "number", b.Number, "meta", meta)
				if err := m.parseFileMeta(&tx, meta, b); err != nil {
					log.Error("Parse file meta error", "err", err, "number", b.Number)
					return false, err
				}
				final = append(final, tx)
				record = true
			} else if tx.IsFlowControl() {
				if tx.Recipient == nil {
					log.Trace("Recipient is nil", "num", b.Number)
					continue
				}
				addr := *tx.Recipient
				file := m.fs.GetFileByAddr(addr)
				if file == nil {
					//log.Warn("Uploading a nonexist file", "addr", addr.String(), "number", b.Number, "len", len(b.Txs))
					continue
				}

				receipt, err := m.getReceipt(tx.Hash.String())
				if err != nil {
					return false, err
				}

				if receipt.Status != 1 {
					continue
				}

				if receipt.GasUsed != params.UploadGas {
					//log.Warn("Receipt", "gas", receipt.GasUsed, "expected", params.UploadGas)
					continue
				}

				remainingSize, err := m.getRemainingSize(addr.String())
				if err != nil {
					log.Error("Get remain failed", "err", err, "addr", addr.String())
					return false, err
				}
				//var progress bool
				//log.Info("....", "hash", file.Meta.InfoHash, "addr", addr.String(), "remain", common.StorageSize(remainingSize), "cur", file.LeftSize, "raw", common.StorageSize(file.Meta.RawSize), "number", b.Number)
				if file.LeftSize > remainingSize {
					file.LeftSize = remainingSize
					///progress = true
					//}
					//if _, update, err := m.fs.UpdateFile(file, b, progress); err != nil {
					if _, progress, err := m.fs.UpdateFile(file); err != nil {
						return false, err
					} else if progress { // && progress {
						log.Debug("Update storage success", "ih", file.Meta.InfoHash, "left", file.LeftSize)
						var bytesRequested uint64
						if file.Meta.RawSize > file.LeftSize {
							bytesRequested = file.Meta.RawSize - file.LeftSize
						}
						if file.LeftSize == 0 {
							log.Debug("Data processing completed !!!", "ih", file.Meta.InfoHash, "addr", addr.String(), "remain", common.StorageSize(remainingSize), "request", common.StorageSize(bytesRequested), "raw", common.StorageSize(file.Meta.RawSize), "number", b.Number)
						} else {
							log.Debug("Data processing ...", "ih", file.Meta.InfoHash, "addr", addr.String(), "remain", common.StorageSize(remainingSize), "request", common.StorageSize(bytesRequested), "raw", common.StorageSize(file.Meta.RawSize), "number", b.Number)
						}

						m.dl.UpdateTorrent(types.FlowControlMeta{
							InfoHash:       file.Meta.InfoHash,
							BytesRequested: bytesRequested,
							IsCreate:       false,
						})
					}
				}
				//} else {
				//	log.Debug("Uploading a file", "addr", addr, "ih", file.Meta.InfoHash.String(), "number", b.Number, "left", file.LeftSize, "remain", remainingSize, "raw", file.Meta.RawSize)
				//}

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

		elapsed := time.Duration(mclock.Now()) - time.Duration(start)
		if len(b.Txs) > 0 {
			log.Trace("Transactions scanning", "count", len(b.Txs), "number", b.Number, "elapsed", common.PrettyDuration(elapsed))
		}
	}

	return record, nil
}

func (m *Monitor) Stop() {
	log.Info("Fs listener closing")
	if atomic.LoadInt32(&(m.terminated)) == 1 {
		return
	}
	atomic.StoreInt32(&(m.terminated), 1)
	close(m.exitCh)
	log.Info("Monitor is waiting to be closed")
	m.wg.Wait()

	m.blockCache.Purge()
	m.sizeCache.Purge()
	/*m.wg.Add(1)
		m.closeOnce.Do(func() {
	                defer m.wg.Done()
			if err := m.dl.Close(); err != nil {
	                      log.Error("Monitor Torrent Manager closed", "error", err)
	                }
	                log.Info("Torrent client listener synchronizing closed")
	        })*/
	log.Info("Fs client listener synchronizing closing")
	if err := m.dl.Close(); err != nil {
		log.Error("Monitor Fs Manager closed", "error", err)
	}
	log.Info("Fs client listener synchronizing closed")

	log.Info("Fs listener synchronizing closing")
	if err := m.fs.Close(); err != nil {
		log.Error("Monitor File Storage closed", "error", err)
	}
	log.Info("Fs listener synchronizing closed")
	/*m.wg.Add(1)
	m.closeOnce.Do(func() {
		defer m.wg.Done()
		log.Info("Torrent client listener synchronizing closing")
		if err := m.dl.Close(); err != nil {
			log.Error("Monitor Torrent Manager closed", "error", err)
		}
		log.Info("Torrent client listener synchronizing closed")

		log.Info("Torrent fs listener synchronizing closing")
		if err := m.fs.Close(); err != nil {
			log.Error("Monitor File Storage closed", "error", err)
		}
		log.Info("Torrent fs listener synchronizing closed")
	})
	m.wg.Wait()*/

	log.Info("Fs listener closed")
}

// Start ... start ListenOn on the rpc port of a blockchain full node
func (m *Monitor) Start() error {
	if err := m.dl.Start(); err != nil {
		log.Warn("Fs start error")
		return err
	}
	m.wg.Add(1)
	go func() {
		defer m.wg.Done()
		if err := m.startWork(); err != nil {
			log.Error("Fs monitor start failed", "err", err)
			panic("Fs monitor start failed")
		}
		/*err := m.startWork()
		if err != nil {
			log.Error("Torrent Fs Internal Error", "error", err)
			p, pErr := os.FindProcess(os.Getpid())
			if pErr != nil {
				log.Error("Torrent Fs Internal Error", "error", pErr)
				panic("boom")
				return
			}

			sigErr := p.Signal(os.Interrupt)
			if sigErr != nil {
				log.Error("Torrent Fs Internal Error", "error", sigErr)
				panic("boom")
				return
			}
		}*/
	}()
	return nil
	//return err
}

//var scope = uint64(runtime.NumCPU())

func (m *Monitor) startWork() error {
	// Wait for ipc start...
	//time.Sleep(time.Second)
	//defer TorrentAPIAvailable.Unlock()
	// Rpc Client
	var clientURI string
	if runtime.GOOS != "windows" && m.config.IpcPath != "" {
		clientURI = m.config.IpcPath
	} else {
		if m.config.RpcURI == "" {
			log.Warn("Fs rpc uri is empty")
			return errors.New("fs RpcURI is empty")
		}
		clientURI = m.config.RpcURI
	}

	rpcClient, rpcErr := m.buildConnection(clientURI)
	if rpcErr != nil {
		log.Error("Fs rpc client is wrong", "uri", clientURI, "error", rpcErr, "config", m.config)
		return rpcErr
	}
	m.cl = rpcClient
	m.lastNumber = m.fs.LastListenBlockNumber
	m.currentBlock()
	m.startNumber = uint64(math.Min(float64(m.fs.LastListenBlockNumber), float64(m.currentNumber))) // ? m.currentNumber:m.fs.LastListenBlockNumber
	//if err := m.validateStorage(); err != nil {
	//	log.Error("Starting torrent fs ... ...", "error", err)
	//	return err
	//}

	//log.Info("Torrent fs validation passed")
	if err := m.indexInit(); err != nil {
		return err
	}
	m.wg.Add(1)
	go m.taskLoop()
	m.wg.Add(1)
	go m.listenLatestBlock()
	m.wg.Add(1)
	go m.syncLatestBlock()
	//m.init()
	//m.wg.Add(1)
	//go m.listenPeers()

	return nil
}

/*func (m *Monitor) validateStorage() error {
m.lastNumber = m.fs.LastListenBlockNumber
end := uint64(0)

log.Info("Validate Torrent FS Storage", "last IPC listen number", m.lastNumber, "end", end, "latest", m.fs.LastListenBlockNumber)

for i := m.lastNumber; i > end; i-- {
	rpcBlock, rpcErr := m.rpcBlockByNumber(uint64(i))
	if rpcErr != nil {
		log.Warn("RPC ERROR", "error", rpcErr)
		return rpcErr
	}

	if rpcBlock == nil || rpcBlock.Hash == common.EmptyHash {
		log.Trace("No block found", "number", i)
		m.lastNumber = uint64(i)
		m.dirty = true
		continue
	} else {
		m.dirty = false
	}

	stBlock := m.fs.GetBlockByNumber(uint64(i))
	if stBlock == nil {
		log.Warn("Vaidate Torrent FS Storage state failed, rescan", "number", m.lastNumber, "error", "LastListenBlockNumber not persistent", "dirty", m.fs.LastListenBlockNumber)
		m.lastNumber = uint64(i)
		m.dirty = true
		continue
	}

	if rpcBlock.Hash.Hex() == stBlock.Hash.Hex() {
		//log.Warn("Validate TFS continue", "number", m.lastNumber, "rpc", rpcBlock.Hash.Hex(), "store", stBlock.Hash.Hex())
		break
	}

	// block in storage invalid
	log.Info("Update invalid block in storage", "old hash", stBlock.Hash, "new hash", rpcBlock.Hash, "latest", m.fs.LastListenBlockNumber)
	m.lastNumber = uint64(i)
}

log.Info("Validate Torrent FS Storage ended", "last IPC listen number", m.lastNumber, "end", end, "latest", m.fs.LastListenBlockNumber)
if m.dirty {
	log.Warn("Torrent fs status", "dirty", m.dirty)
}*/

/*if m.lastNumber > batch {
	m.lastNumber = m.lastNumber - batch
} else {
	m.lastNumber = 0
}*/

/*for i := uint64(0); i < m.fs.LastFileIndex; i++ {
	file := m.fs.GetFileByNumber(i)
	if file == nil {
		continue
	}

	var bytesRequested uint64
	if file.Meta.RawSize > file.LeftSize {
		bytesRequested = file.Meta.RawSize - file.LeftSize
	}
	log.Debug("Data recovery", "request", bytesRequested, "raw", file.Meta.RawSize)

	m.dl.UpdateTorrent(FlowControlMeta{
		InfoHash:       file.Meta.InfoHash,
		BytesRequested: bytesRequested,
		IsCreate:       true,
	})

	//m.fs.AddCachedFile(file)
}

if m.lastNumber > 256 {
	m.lastNumber = m.lastNumber - 256
} else {
	m.lastNumber = 0
}*/
/*
	return nil
}*/

func (m *Monitor) listenLatestBlock() {
	defer m.wg.Done()
	timer := time.NewTimer(time.Second * queryTimeInterval)
	defer timer.Stop()
	for {
		select {
		case <-timer.C:
			m.currentBlock()
			timer.Reset(time.Second * queryTimeInterval)
		case <-m.exitCh:
			log.Info("Block listener stopped")
			return
		}
	}
}

func (m *Monitor) syncLatestBlock() {
	defer m.wg.Done()
	timer := time.NewTimer(time.Second * queryTimeInterval)
	defer timer.Stop()
	progress := uint64(0)
	for {
		select {
		case <-timer.C:
			progress = m.syncLastBlock()
			// Avoid sync in full mode, fresh interval may be less.
			if progress >= delay {
				timer.Reset(0)
			} else if progress > 1 {
				timer.Reset(time.Millisecond * 1000)
			} else {
				timer.Reset(time.Millisecond * 2000)
			}
			m.fs.Flush()
		case <-m.exitCh:
			log.Info("Block syncer stopped")
			return
		}
	}
}

/*func (m *Monitor) listenPeers() {
	defer m.wg.Done()
	m.default_tracker_check()
	timer := time.NewTimer(time.Second * 60)
	defer timer.Stop()
	for {
		select {
		case <-timer.C:
			m.peers()
			timer.Reset(time.Second * 600)
		case <-m.exitCh:
			log.Info("Peers listener stopped")
			return
		}
	}
}*/

/*type tracker_stats struct {
	Torrents              int `json:"torrents"`
	ActiveTorrents        int `json:"activeTorrents"`
	PeersAll              int `json:"peersAll"`
	PeersSeederOnly       int `json:"peersSeederOnly"`
	PeersLeecherOnly      int `json:"peersLeecherOnly"`
	PeersSeederAndLeecher int `json:"peersSeederAndLeecher"`
	PeersIPv4             int `json:"peersIPv4"`
	PeersIPv6             int `json:"peersIPv6"`
}*/

/*func (m *Monitor) default_tracker_check() (r []string, err error) {
	for _, tracker := range params.MainnetTrackers {
		url := tracker[0:len(tracker)-9] + "/stats.json"
		if !strings.HasPrefix(url, "http") {
			continue
		}
		response, err := client.Get(url)
		//start := mclock.Now()
		if err != nil || response == nil || response.StatusCode != 200 {
			//log.Warn("Default tracker status is unhealthy", "name", tracker, "err", err)
			continue
		} else {
			var stats tracker_stats
			if jsErr := json.NewDecoder(response.Body).Decode(&stats); jsErr != nil {
				//log.Warn("Default tracker status is unhealthy", "name", tracker, "url", url, "err", jsErr, "stats", stats)
				continue
			}
			//elapsed := time.Duration(mclock.Now()) - time.Duration(start)
			//log.Info("Default tracker status is healthy", "name", tracker, "url", url, "elapsed", elapsed)
			r = append(r, tracker)
		}
	}
	log.Info("Default storage global status", "result", len(r))
	return r, err
}

func (m *Monitor) batch_http_healthy(ip string, ports []string) ([]string, bool) {
	var res []string
	var status = false
	for _, port := range ports {
		if atomic.LoadInt32(&(m.terminated)) == 1 {
			break
		}
		//m.portsWg.Add(1)
		//go func(port string) {
		//defer m.portsWg.Done()
		url := "http://" + ip + ":" + port + "/stats.json"
		response, err := client.Get(url)
		if err != nil || response == nil || response.StatusCode != 200 {
			//return
			continue
		} else {
			var stats tracker_stats
			if jsErr := json.NewDecoder(response.Body).Decode(&stats); jsErr != nil {
				//return
				continue
			}
			//m.portLock.Lock()
			res = append(res, port)
			status = true
			//m.portLock.Unlock()
			break
		}
		//}(port)
	}

	//m.portsWg.Wait()

	return res, status

}*/

/*func (m *Monitor) batch_udp_healthy(ip string, ports []string) ([]string, bool) {
	var res []string
	var status = false
	//request := make([]byte, 1)
	for _, port := range ports {
		addr, err := net.ResolveUDPAddr("udp", ip+":"+port)
		if err != nil {
			continue
		}
		socket, err := net.DialUDP("udp", nil, addr)
		if err != nil {
			continue
		} else {
			defer socket.Close()
			//m.portLock.Lock()
			res = append(res, port)
			status = true
			break
			//m.portLock.Unlock()
			//defer socket.Close()
		}
	}

	return res, status
}*/

func (m *Monitor) currentBlock() (uint64, error) {
	var currentNumber hexutil.Uint64

	if err := m.cl.Call(&currentNumber, "ctxc_blockNumber"); err != nil {
		log.Error("Call ipc method ctxc_blockNumber failed", "error", err)
		return 0, err
	}
	if m.currentNumber != uint64(currentNumber) {
		m.currentNumber = uint64(currentNumber)
	}
	return uint64(currentNumber), nil
}

func (m *Monitor) syncLastBlock() uint64 {
	//var currentNumber hexutil.Uint64

	//if err := m.cl.Call(&currentNumber, "ctxc_blockNumber"); err != nil {
	//	log.Error("Call ipc method ctx_blockNumber failed", "error", err)
	//	return 0
	//}

	//if uint64(currentNumber) <= 0 {
	//	return 0
	//}
	//currentNumber, err := m.currentBlock()
	currentNumber := m.currentNumber
	//if err != nil {
	//	return 0
	//}

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
	//defer m.fs.Flush()
	start := mclock.Now()
	for i := minNumber; i <= maxNumber; { // i++ {
		if atomic.LoadInt32(&(m.terminated)) == 1 {
			log.Warn("Fs scan terminated", "number", i)
			maxNumber = i - 1
			break
		}
		if maxNumber-i >= m.scope {
			blocks, rpcErr := m.rpcBatchBlockByNumber(i, i+m.scope)
			if rpcErr != nil {
				log.Error("Sync old block failed", "number", i, "error", rpcErr)
				m.lastNumber = i - 1
				return 0
			}
			for _, rpcBlock := range blocks {
				if len(m.taskCh) < cap(m.taskCh) {
					m.taskCh <- rpcBlock
					i++
				} else {
					m.lastNumber = i - 1
					if maxNumber-minNumber > delay/2 {
						elapsed := time.Duration(mclock.Now()) - time.Duration(start)
						elapsed_a := time.Duration(mclock.Now()) - time.Duration(m.start)
						log.Warn("Chain segment frozen", "from", minNumber, "to", i, "range", uint64(i-minNumber), "current", uint64(m.currentNumber), "progress", float64(i)/float64(m.currentNumber), "last", m.lastNumber, "elapsed", common.PrettyDuration(elapsed), "bps", float64(i-minNumber)*1000*1000*1000/float64(elapsed), "bps_a", float64(maxNumber)*1000*1000*1000/float64(elapsed_a), "cap", len(m.taskCh))
					}
					return 0
				}
			}
		} else {

			rpcBlock, rpcErr := m.rpcBlockByNumber(i)
			if rpcErr != nil {
				log.Error("Sync old block failed", "number", i, "error", rpcErr)
				m.lastNumber = i - 1
				return 0
			}
			if len(m.taskCh) < cap(m.taskCh) {
				m.taskCh <- rpcBlock
				i++
			} else {
				m.lastNumber = i - 1
				if maxNumber-minNumber > delay/2 {
					elapsed := time.Duration(mclock.Now()) - time.Duration(start)
					elapsed_a := time.Duration(mclock.Now()) - time.Duration(m.start)
					log.Warn("Chain segment frozen", "from", minNumber, "to", i, "range", uint64(i-minNumber), "current", uint64(m.currentNumber), "progress", float64(i)/float64(m.currentNumber), "last", m.lastNumber, "elapsed", common.PrettyDuration(elapsed), "bps", float64(i-minNumber)*1000*1000*1000/float64(elapsed), "bps_a", float64(maxNumber)*1000*1000*1000/float64(elapsed_a), "cap", len(m.taskCh))
				}
				return 0
			}
		}
	}
	m.lastNumber = maxNumber
	if maxNumber-minNumber > delay/2 {
		elapsed := time.Duration(mclock.Now()) - time.Duration(start)
		elapsed_a := time.Duration(mclock.Now()) - time.Duration(m.start)
		log.Debug("Chain segment frozen", "from", minNumber, "to", maxNumber, "range", uint64(maxNumber-minNumber), "current", uint64(m.currentNumber), "progress", float64(maxNumber)/float64(m.currentNumber), "last", m.lastNumber, "elapsed", common.PrettyDuration(elapsed), "bps", float64(maxNumber-minNumber)*1000*1000*1000/float64(elapsed), "bps_a", float64(maxNumber)*1000*1000*1000/float64(elapsed_a), "cap", len(m.taskCh), "duration", common.PrettyDuration(elapsed_a))
	}
	return uint64(maxNumber - minNumber)
}

func (m *Monitor) solve(block *types.Block) error {
	i := block.Number
	if i%65536 == 0 {
		defer func() {
			elapsed_a := time.Duration(mclock.Now()) - time.Duration(m.start)
			log.Info(ProgressBar(int64(i), int64(m.currentNumber), ""), "max", uint64(m.currentNumber), "last", m.lastNumber, "cur", i, "bps", math.Abs(float64(i)-float64(m.startNumber))*1000*1000*1000/float64(elapsed_a), "elapsed", common.PrettyDuration(elapsed_a), "scope", m.scope, "db", common.PrettyDuration(m.fs.Metrics()), "blocks", len(m.fs.Blocks()), "txs", m.fs.Txs(), "files", len(m.fs.Files()), "root", m.fs.Root())
		}()
	}
	if hash, suc := m.blockCache.Get(i); !suc || hash != block.Hash.Hex() {
		if record, parseErr := m.parseBlockTorrentInfo(block); parseErr != nil {
			log.Error("Parse new block", "number", block.Number, "block", block, "error", parseErr)
			return parseErr
		} else if record {
			elapsed := time.Duration(mclock.Now()) - time.Duration(m.start)

			if m.ckp != nil && m.ckp.TfsCheckPoint > 0 && i == m.ckp.TfsCheckPoint {
				if common.BytesToHash(m.fs.GetRootByNumber(i)) == m.ckp.TfsRoot {
					log.Warn("FIRST MILESTONE PASS", "number", i, "root", m.fs.Root(), "blocks", len(m.fs.Blocks()), "txs", m.fs.Txs(), "files", len(m.fs.Files()), "elapsed", common.PrettyDuration(elapsed))
				} else {
					log.Error("Fs checkpoint failed", "number", i, "root", m.fs.Root(), "blocks", len(m.fs.Blocks()), "files", len(m.fs.Files()), "txs", m.fs.Txs(), "elapsed", common.PrettyDuration(elapsed), "exp", m.ckp.TfsRoot)
					panic("Fs sync fatal error, removedb to solve it")
				}
			}

			log.Debug("Seal fs record", "number", i, "cap", len(m.taskCh), "record", record, "root", m.fs.Root().Hex(), "blocks", len(m.fs.Blocks()), "txs", m.fs.Txs(), "files", len(m.fs.Files()), "ckp", m.fs.CheckPoint)
		} else {
			if m.fs.LastListenBlockNumber < i {
				m.fs.LastListenBlockNumber = i
			}

			log.Trace("Confirm to seal the fs record", "number", i, "cap", len(m.taskCh))
		}
		m.blockCache.Add(i, block.Hash.Hex())
	}
	return nil
}

/*func (m *Monitor) parseAndStore(block *Block) error {
	if parseErr := m.parseBlockTorrentInfo(block); parseErr != nil {
		log.Error("Parse new block", "number", block.Number, "block", block, "error", parseErr)
		return parseErr
	}

	if storeErr := m.fs.WriteBlock(block); storeErr != nil {
		log.Error("Store latest block", "number", block.Number, "error", storeErr)
		return storeErr
	}
	return nil
}*/
