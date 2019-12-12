package torrentfs

import (
	"encoding/json"
	"errors"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"github.com/anacrolix/torrent/metainfo"
	lru "github.com/hashicorp/golang-lru"
	"net"
	"net/http"
	//"os"
	"runtime"
	//"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

//------------------------------------------------------------------------------

// Errors that are used throughout the Torrent API.
var (
	ErrBuildConn      = errors.New("build internal-rpc connection failed")
	ErrGetLatestBlock = errors.New("get latest block failed")
	ErrNoRPCClient    = errors.New("no rpc client")

	ErrBlockHash = errors.New("block or parent block hash invalid")
	//blockCache, _ = lru.New(6)
	//healthPeers, _ = lru.New(50)
	//sizeCache, _   = lru.New(batch)
)

const (
	defaultTimerInterval  = 1
	connTryTimes          = 300
	connTryInterval       = 2
	fetchBlockTryTimes    = 5
	fetchBlockTryInterval = 3
	//fetchBlockLogStep     = 10000
	//minBlockNum           = 0

	//maxSyncBlocks = 1024
)

type TorrentManagerAPI interface {
	Start() error
	Close() error
	//RemoveTorrent(metainfo.Hash) error
	UpdateTorrent(interface{}) error
	UpdateDynamicTrackers(trackers []string)
	GetTorrent(ih metainfo.Hash) *Torrent
}

// Monitor observes the data changes on the blockchain and synchronizes.
// cl for ipc/rpc communication, dl for download manager, and fs for data storage.
type Monitor struct {
	config *Config
	cl     *rpc.Client
	fs     *FileStorage
	dl     TorrentManagerAPI

	//listenID rpc.ID

	//uncheckedCh chan uint64

	exitCh     chan struct{}
	terminated int32
	lastNumber uint64
	dirty      bool

	//closeOnce sync.Once
	wg      sync.WaitGroup
	peersWg sync.WaitGroup
	//trackerLock sync.Mutex
	//portLock sync.Mutex
	//portsWg  sync.WaitGroup

	taskCh      chan *Block
	newTaskHook func(*Block)
	blockCache  *lru.Cache
	healthPeers *lru.Cache
	sizeCache   *lru.Cache
	ckp         *params.TrustedCheckpoint
	start       mclock.AbsTime
}

// NewMonitor creates a new instance of monitor.
// Once Ipcpath is settle, this method prefers to build socket connection in order to
// get higher communicating performance.
// IpcPath is unavailable on windows.
func NewMonitor(flag *Config) (m *Monitor, e error) {
	log.Info("Initialising FS")
	// File Storage
	fs, fsErr := NewFileStorage(flag)
	if fsErr != nil {
		log.Error("file storage failed", "err", fsErr)
		return nil, fsErr
	}
	log.Info("File storage initialized")

	id := fs.ID()

	// Torrent Manager
	tMana := NewTorrentManager(flag, id)
	if tMana == nil {
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
		exitCh:     make(chan struct{}),
		terminated: 0,
		lastNumber: uint64(0),
		dirty:      false,
		taskCh:     make(chan *Block, batch),
		start:      mclock.Now(),
	}
	m.blockCache, _ = lru.New(delay)
	m.healthPeers, _ = lru.New(50)
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

func (m *Monitor) storageInit() error {
	log.Info("Loading storage data ... ...", "latest", m.fs.LastListenBlockNumber, "checkpoint", m.fs.CheckPoint, "root", m.fs.Root(), "version", m.fs.Version())
	genesis, err := m.rpcBlockByNumber(0)
	if err != nil {
		return err
	}

	if checkpoint, ok := params.TrustedCheckpoints[genesis.Hash]; ok {
		if uint64(len(m.fs.Blocks())) < checkpoint.TfsBlocks || uint64(len(m.fs.Files())) < checkpoint.TfsFiles {
			log.Info("Fs storage version upgrade", "version", m.fs.Version(), "blocks", len(m.fs.Blocks()), "files", len(m.fs.Files()))
			m.lastNumber = 0
		}
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
			log.Warn("Fs storage version check failed, reloading ...", "number", checkpoint.TfsCheckPoint, "version", common.BytesToHash(version), "checkpoint", checkpoint.TfsRoot)
			m.lastNumber = 0
			//m.fs.LastListenBlockNumber = 0
			//m.fs.CheckPoint = 0
		} else {
			log.Info("Fs storage version check passed", "number", checkpoint.TfsCheckPoint, "version", common.BytesToHash(version))
		}
	}

	/*blocks := m.fs.Blocks()
	sort.Slice(blocks, func(i, j int) bool {
		return blocks[i].Number < blocks[j].Number
	})*/

	for _, block := range m.fs.Blocks() {
		/*if b, err := m.rpcBlockByNumber(block.Number); err == nil && b.Hash != block.Hash {
			m.lastNumber = 0
		}*/
		if record, parseErr := m.parseBlockTorrentInfo(block); parseErr != nil {
			log.Error("Parse new block", "number", block.Number, "block", block, "error", parseErr)
			return parseErr
		} else {
			log.Debug("Block storage info", "number", block.Number, "record", record)
		}
	}

	fileMap := make(map[metainfo.Hash]*FileInfo)
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
		log.Debug("File storage info", "addr", file.ContractAddr, "hash", file.Meta.InfoHash, "remain", common.StorageSize(file.LeftSize), "raw", common.StorageSize(file.Meta.RawSize), "request", common.StorageSize(bytesRequested))
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
	log.Info("Storage current state", "total", len(fileMap), "seed", seed, "pause", pause, "pending", pending, "capcity", common.StorageSize(capcity), "blocks", len(m.fs.Blocks()))
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

			if err := m.deal(task); err != nil {
				log.Warn("Block dealing failed", "err", err)
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
		time.Sleep(time.Second * connTryInterval)
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

func (m *Monitor) rpcBlockByNumber(blockNumber uint64) (*Block, error) {
	block := &Block{}
	blockNumberHex := "0x" + strconv.FormatUint(blockNumber, 16)

	//for i := 0; i < fetchBlockTryTimes; i++ {
	err := m.cl.Call(block, "ctxc_getBlockByNumber", blockNumberHex, true)
	if err == nil {
		return block, nil
	}

	//	time.Sleep(time.Second * fetchBlockTryInterval)
	//	log.Warn("Torrent Fs Internal IPC ctx_getBlockByNumber", "retry", i, "error", err, "number", blockNumber)
	//}

	return nil, errors.New("[ Internal IPC Error ] try to get block out of times")
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

var (
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
}

/*func (m *Monitor) ws_tracker_build(ip, port string) string {
	return "ws://" + ip + ":" + port + "/announce"
}*/

func (m *Monitor) peers() ([]*p2p.PeerInfo, error) {
	var peers []*p2p.PeerInfo // = make([]*p2p.PeerInfo, 0, 25)
	err := m.cl.Call(&peers, "admin_peers")
	if err == nil && len(peers) > 0 {
		flush := false
		start := mclock.Now()
		//m.peersWg.Add(len(peers))
		for _, peer := range peers {
			m.peersWg.Add(1)
			go func(peer *p2p.PeerInfo) {
				defer m.peersWg.Done()
				ip := strings.Split(peer.Network.RemoteAddress, ":")[0]
				//if unhealthPeers.Contains(ip) {
				//continue
				//}
				/*				if ps, suc := m.batch_http_healthy(ip, TRACKER_PORT); suc && len(ps) > 0 {
									for _, p := range ps {
										tracker := m.http_tracker_build(ip, p) //"http://" + ip + ":" + p + "/announce"
										if m.healthPeers.Contains(tracker) {
											//continue
										} else {
											flush = true
										}
										//m.trackerLock.Lock()
										//trackers = append(trackers, tracker)
										//trackers = append(trackers, m.udp_tracker_build(ip, p)) //"udp://" + ip + ":" + p + "/announce")
										//trackers = append(trackers, m.ws_tracker_build(ip, p))  //"ws://" + ip + ":" + p + "/announce")
										//m.trackerLock.Unlock()
										//flush = true
										m.healthPeers.Add(tracker, tracker)
										//if unhealthPeers.Contains(ip) {
										//	unhealthPeers.Remove(ip)
										//}
									}
								} // else {
								//unhealthPeers.Add(ip, peer)
				*/
				if ps, suc := m.batch_udp_healthy(ip, UDP_TRACKER_PORT); suc && len(ps) > 0 {
					for _, p := range ps {
						tracker := m.udp_tracker_build(ip, p) //"udp://" + ip + ":" + p + "/announce"
						if m.healthPeers.Contains(tracker) {
							//continue
						} else {
							flush = true
						}
						//m.trackerLock.Lock()
						//trackers = append(trackers, tracker)
						//m.trackerLock.Unlock()
						//flush = true
						m.healthPeers.Add(tracker, tracker)
						//if unhealthPeers.Contains(ip) {
						//	unhealthPeers.Remove(ip)
						//}
					}
				}
				//}
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
}

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

func (m *Monitor) parseFileMeta(tx *Transaction, meta *FileMeta) error {
	log.Debug("Monitor", "FileMeta", meta)

	var receipt TxReceipt
	if err := m.cl.Call(&receipt, "ctxc_getTransactionReceipt", tx.Hash.String()); err != nil {
		return err
	}

	if receipt.ContractAddr == nil {
		return nil
	}

	log.Debug("Transaction Receipt", "address", receipt.ContractAddr.String(), "gas", receipt.GasUsed, "status", receipt.Status, "tx", receipt.TxHash.String())

	if receipt.Status != 1 {
		return nil
	}

	info := m.fs.NewFileInfo(meta)
	info.TxHash = tx.Hash

	info.LeftSize = meta.RawSize
	info.ContractAddr = receipt.ContractAddr
	index, err := m.fs.AddFile(info)
	if err != nil {
		return err
	} else {
		if index > 0 {
			log.Debug("Create new file", "hash", meta.InfoHash, "index", index)
			m.dl.UpdateTorrent(FlowControlMeta{
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

func (m *Monitor) parseBlockTorrentInfo(b *Block) (bool, error) {
	record := false
	if len(b.Txs) > 0 {
		start := mclock.Now()
		for _, tx := range b.Txs {
			if meta := tx.Parse(); meta != nil {
				log.Debug("Try to create a file", "meta", meta, "number", b.Number, "infohash", meta.InfoHash)
				if err := m.parseFileMeta(&tx, meta); err != nil {
					log.Error("Parse file meta error", "err", err, "number", b.Number)
					return false, err
				}
				record = true
			} else if tx.IsFlowControl() {
				if tx.Recipient == nil {
					continue
				}
				addr := *tx.Recipient
				file := m.fs.GetFileByAddr(addr)
				if file == nil {
					//log.Warn("Uploading a nonexist file", "addr", addr.String(), "number", b.Number)
					continue
				}

				remainingSize, err := m.getRemainingSize(addr.String())
				if err != nil {
					return false, err
				}

				if file.LeftSize > remainingSize {
					file.LeftSize = remainingSize
					if update, err := m.fs.WriteFile(file); err != nil {
						return false, err
					} else if update {
						log.Debug("Update storage success", "hash", file.Meta.InfoHash, "left", file.LeftSize)
						var bytesRequested uint64
						if file.Meta.RawSize > file.LeftSize {
							bytesRequested = file.Meta.RawSize - file.LeftSize
						}
						log.Info("Data processing", "hash", file.Meta.InfoHash, "addr", addr.String(), "remain", common.StorageSize(remainingSize), "request", common.StorageSize(bytesRequested), "raw", common.StorageSize(file.Meta.RawSize), "number", b.Number)

						m.dl.UpdateTorrent(FlowControlMeta{
							InfoHash:       file.Meta.InfoHash,
							BytesRequested: bytesRequested,
							IsCreate:       false,
						})
					}
				} else {
					log.Debug("Uploading a file", "addr", addr, "hash", file.Meta.InfoHash.String(), "number", b.Number, "left", file.LeftSize, "remain", remainingSize, "raw", file.Meta.RawSize)
				}

				record = true
			}
		}
		elapsed := time.Duration(mclock.Now()) - time.Duration(start)
		if len(b.Txs) > 0 {
			log.Debug("Transactions scanning", "count", len(b.Txs), "number", b.Number, "elapsed", common.PrettyDuration(elapsed))
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
		m.startWork()
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
	//if err := m.validateStorage(); err != nil {
	//	log.Error("Starting torrent fs ... ...", "error", err)
	//	return err
	//}

	//log.Info("Torrent fs validation passed")
	if err := m.storageInit(); err != nil {
		return err
	}
	m.wg.Add(1)
	go m.taskLoop()
	m.wg.Add(1)
	go m.listenLatestBlock()
	m.init()
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
	timer := time.NewTimer(time.Second * defaultTimerInterval)
	defer timer.Stop()
	progress := uint64(0)
	for {
		select {
		case <-timer.C:
			progress = m.syncLastBlock()
			// Aviod sync in full mode, fresh interval may be less.
			if progress >= batch {
				timer.Reset(time.Millisecond * 100)
			} else if progress > 6 {
				timer.Reset(time.Millisecond * 1000)
			} else {
				timer.Reset(time.Millisecond * 3000)
			}

			//timer.Reset(time.Second * defaultTimerInterval)
		case <-m.exitCh:
			log.Info("Block listener stopped")
			return
		}
	}
}

func (m *Monitor) listenPeers() {
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
}

type tracker_stats struct {
	Torrents              int `json:"torrents"`
	ActiveTorrents        int `json:"activeTorrents"`
	PeersAll              int `json:"peersAll"`
	PeersSeederOnly       int `json:"peersSeederOnly"`
	PeersLeecherOnly      int `json:"peersLeecherOnly"`
	PeersSeederAndLeecher int `json:"peersSeederAndLeecher"`
	PeersIPv4             int `json:"peersIPv4"`
	PeersIPv6             int `json:"peersIPv6"`
}

func (m *Monitor) default_tracker_check() (r []string, err error) {
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

}

func (m *Monitor) batch_udp_healthy(ip string, ports []string) ([]string, bool) {
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
}

const (
	batch = 4096
	delay = 12
)

func (m *Monitor) syncLastBlock() uint64 {
	var currentNumber hexutil.Uint64

	if err := m.cl.Call(&currentNumber, "ctxc_blockNumber"); err != nil {
		log.Error("Call ipc method ctx_blockNumber failed", "error", err)
		return 0
	}

	//if uint64(currentNumber) <= 0 {
	//	return 0
	//}

	if uint64(currentNumber) < m.lastNumber {
		log.Warn("Fs sync rollback", "current", uint64(currentNumber), "last", m.lastNumber)
		m.lastNumber = 0
	}

	minNumber := m.lastNumber + 1
	maxNumber := uint64(0)
	if uint64(currentNumber) > delay {
		maxNumber = uint64(currentNumber) - delay
	}

	if m.lastNumber > uint64(currentNumber) {
		if m.lastNumber > batch {
			minNumber = m.lastNumber - batch
		}
	}

	if maxNumber > batch+minNumber {
		maxNumber = minNumber + batch
	}
	if maxNumber >= minNumber {
		/*if minNumber > delay {
			minNumber = minNumber - delay
		}*/
		log.Debug("Fs scanning ... ...", "from", minNumber, "to", maxNumber, "current", uint64(currentNumber), "range", uint64(maxNumber-minNumber), "behind", uint64(currentNumber)-maxNumber, "progress", float64(maxNumber)/float64(currentNumber))
	} else {
		return 0
	}

	start := mclock.Now()
	for i := minNumber; i <= maxNumber; i++ {
		if atomic.LoadInt32(&(m.terminated)) == 1 {
			log.Warn("Fs scan terminated", "number", i)
			maxNumber = i - 1
			//close(m.exitCh)
			break
		}

		rpcBlock, rpcErr := m.rpcBlockByNumber(i)
		if rpcErr != nil {
			log.Error("Sync old block failed", "number", i, "error", rpcErr)
			return 0
		}
		if len(m.taskCh) < cap(m.taskCh) {
			m.taskCh <- rpcBlock
		} else {
			m.lastNumber = i - 1
			elapsed := time.Duration(mclock.Now()) - time.Duration(start)
			elapsed_a := time.Duration(mclock.Now()) - time.Duration(m.start)
			log.Info("Blocks scan finished", "from", minNumber, "to", i, "range", uint64(i-minNumber), "current", uint64(currentNumber), "progress", float64(i)/float64(currentNumber), "last", m.lastNumber, "elasped", elapsed, "bps", float64(i-minNumber)*1000*1000*1000/float64(elapsed), "bps_a", float64(maxNumber)*1000*1000*1000/float64(elapsed_a), "cap", len(m.taskCh))
			//return m.lastNumber - minNumber
			return 0
		}
		//if err := m.deal(rpcBlock); err != nil {
		//	return 0
		//}
	}
	elapsed := time.Duration(mclock.Now()) - time.Duration(start)
	m.lastNumber = maxNumber
	elapsed_a := time.Duration(mclock.Now()) - time.Duration(m.start)
	log.Info("Blocks scan finished", "from", minNumber, "to", maxNumber, "range", uint64(maxNumber-minNumber), "current", uint64(currentNumber), "progress", float64(maxNumber)/float64(currentNumber), "last", m.lastNumber, "elasped", elapsed, "bps", float64(maxNumber-minNumber)*1000*1000*1000/float64(elapsed), "bps_a", float64(maxNumber)*1000*1000*1000/float64(elapsed_a), "cap", len(m.taskCh))
	return uint64(maxNumber - minNumber)
}

func (m *Monitor) deal(block *Block) error {
	i := block.Number
	if hash, suc := m.blockCache.Get(i); !suc || hash != block.Hash.Hex() {
		if record, parseErr := m.parseBlockTorrentInfo(block); parseErr != nil {
			log.Error("Parse new block", "number", block.Number, "block", block, "error", parseErr)
			return parseErr
		} else if record {
			if storeErr := m.fs.WriteBlock(block, true); storeErr != nil {
				log.Error("Store latest block", "number", block.Number, "error", storeErr)
				return storeErr
			}

			if i == m.ckp.TfsCheckPoint && m.fs.Root() == m.ckp.TfsRoot {
				elapsed := time.Duration(mclock.Now()) - time.Duration(m.start)
				log.Info("Fs checkpoint goal ❄️ ", "number", i, "root", m.fs.Root(), "blocks", len(m.fs.Blocks()), "files", len(m.fs.Files()), "elapsed", elapsed)
			}

			log.Debug("Confirm to seal the fs record", "number", i, "cap", len(m.taskCh), "record", record)
		} else {
			if i%(batch/8) == 0 {
				if storeErr := m.fs.WriteBlock(block, false); storeErr != nil {
					log.Error("Store latest block", "number", block.Number, "error", storeErr)
					return storeErr
				}

				log.Debug("Confirm to seal the fs record", "number", i, "cap", len(m.taskCh))
			}
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
