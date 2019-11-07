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
	"os"
	"runtime"
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

	ErrBlockHash  = errors.New("block or parent block hash invalid")
	blockCache, _ = lru.New(6)
	//unhealthPeers, _ = lru.New(256)
	healthPeers, _ = lru.New(50)
)

const (
	defaultTimerInterval  = 2
	connTryTimes          = 300
	connTryInterval       = 2
	fetchBlockTryTimes    = 5
	fetchBlockTryInterval = 3
	fetchBlockLogStep     = 10000
	minBlockNum           = 0

	maxSyncBlocks = 1024
)

type TorrentManagerAPI interface {
	Start() error
	Close() error
	RemoveTorrent(metainfo.Hash) error
	UpdateTorrent(interface{}) error
}

// Monitor observes the data changes on the blockchain and synchronizes.
// cl for ipc/rpc communication, dl for download manager, and fs for data storage.
type Monitor struct {
	config *Config
	cl     *rpc.Client
	fs     *FileStorage
	dl     TorrentManagerAPI

	listenID rpc.ID

	uncheckedCh chan uint64

	exitCh     chan struct{}
	terminated int32
	lastNumber uint64
	dirty      bool

	closeOnce sync.Once
	wg        sync.WaitGroup
	peersWg   sync.WaitGroup
	//trackerLock sync.Mutex
	//portLock sync.Mutex
	//portsWg  sync.WaitGroup
}

// NewMonitor creates a new instance of monitor.
// Once Ipcpath is settle, this method prefers to build socket connection in order to
// get higher communicating performance.
// IpcPath is unavailable on windows.
func NewMonitor(flag *Config) (m *Monitor, e error) {
	log.Info("Initialising Torrent FS")
	// File Storage
	fs, fsErr := NewFileStorage(flag)
	if fsErr != nil {
		return nil, fsErr
	}
	log.Info("Torrent file storage initialized")

	// Torrent Manager
	tMana := NewTorrentManager(flag)
	if tMana == nil {
		return nil, errors.New("torrent download manager initialise failed")
	}
	log.Info("Torrent manager initialized")

	m = &Monitor{
		config:      flag,
		cl:          nil,
		fs:          fs,
		dl:          tMana,
		uncheckedCh: make(chan uint64, 20),
		exitCh:      make(chan struct{}),
		terminated:  0,
		lastNumber:  uint64(0),
		dirty:       false,
	}
	e = nil

	log.Info("Loading storage data ... ...")

	fileMap := make(map[metainfo.Hash]*FileInfo)
	for _, file := range m.fs.Files() {
		fileMap[file.Meta.InfoHash] = file
	}
	seed := 0
	pause := 0
	pending := 0

	for _, file := range fileMap {
		var bytesRequested uint64
		bytesRequested = 0
		if file.Meta.RawSize > file.LeftSize {
			bytesRequested = file.Meta.RawSize - file.LeftSize
		}
		log.Info("File storage info", "addr", file.ContractAddr, "hash", file.Meta.InfoHash, "remain", file.LeftSize, "raw", file.Meta.RawSize, "request", bytesRequested)
		m.dl.UpdateTorrent(FlowControlMeta{
			InfoHash:       file.Meta.InfoHash,
			BytesRequested: bytesRequested,
			IsCreate:       false,
		})
		if file.LeftSize == 0 {
			seed += 1
		} else if file.Meta.RawSize == file.LeftSize {
			pending += 1
		} else if file.Meta.RawSize > file.LeftSize && file.LeftSize > 0 {
			pause += 1
		}
	}
	log.Info("Storage current state", "total", len(fileMap), "seed", seed, "pause", pause, "pending", pending)
	return m, e
}

// SetConnection method builds connection to remote or local communicator.
func SetConnection(clientURI string) (*rpc.Client, error) {
	for i := 0; i < connTryTimes; i++ {
		time.Sleep(time.Second * connTryInterval)
		cl, err := rpc.Dial(clientURI)
		if err != nil {
			log.Warn("Building internal-ipc connection ... ", "URI", clientURI, "times", i, "error", err)
		} else {
			log.Debug("Internal-IPC connection established", "URI", clientURI)
			return cl, nil
		}
	}

	return nil, errors.New("Building Internal-IPC Connection Failed")
}

func (m *Monitor) rpcBlockByNumber(blockNumber uint64) (*Block, error) {
	block := &Block{}
	blockNumberHex := "0x" + strconv.FormatUint(blockNumber, 16)

	for i := 0; i < fetchBlockTryTimes; i++ {
		err := m.cl.Call(block, "ctxc_getBlockByNumber", blockNumberHex, true)
		if err == nil {
			return block, nil
		}

		time.Sleep(time.Second * fetchBlockTryInterval)
		log.Warn("Torrent Fs Internal IPC ctx_getBlockByNumber", "retry", i, "error", err, "number", blockNumber)
	}

	return nil, errors.New("[ Internal IPC Error ] try to get block out of times")
}

func (m *Monitor) rpcBlockByHash(blockHash string) (*Block, error) {
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
}

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
		Timeout: time.Duration(5 * time.Second),
	}
}

func (m *Monitor) http_tracker_build(ip, port string) string {
	return "http://" + ip + ":" + port + "/announce"
}

func (m *Monitor) udp_tracker_build(ip, port string) string {
	return "udp://" + ip + ":" + port + "/announce"
}

func (m *Monitor) ws_tracker_build(ip, port string) string {
	return "ws://" + ip + ":" + port + "/announce"
}

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
				if ps, suc := m.batch_http_healthy(ip, TRACKER_PORT); suc && len(ps) > 0 {
					for _, p := range ps {
						tracker := m.http_tracker_build(ip, p) //"http://" + ip + ":" + p + "/announce"
						if healthPeers.Contains(tracker) {
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
						healthPeers.Add(tracker, tracker)
						//if unhealthPeers.Contains(ip) {
						//	unhealthPeers.Remove(ip)
						//}
					}
				} else {
					//unhealthPeers.Add(ip, peer)

					if ps, suc := m.batch_udp_healthy(ip, UDP_TRACKER_PORT); suc && len(ps) > 0 {
						for _, p := range ps {
							tracker := m.udp_tracker_build(ip, p) //"udp://" + ip + ":" + p + "/announce"
							if healthPeers.Contains(tracker) {
								//continue
							} else {
								flush = true
							}
							//m.trackerLock.Lock()
							//trackers = append(trackers, tracker)
							//m.trackerLock.Unlock()
							//flush = true
							healthPeers.Add(tracker, tracker)
							//if unhealthPeers.Contains(ip) {
							//	unhealthPeers.Remove(ip)
							//}
						}
					}
				}
			}(peer)
		}
		//log.Info("Waiting dynamic tracker", "size", len(peers))
		m.peersWg.Wait()

		var trackers []string
		for _, data := range healthPeers.Keys() {
			if str, ok := data.(string); ok {
				trackers = append(trackers, str)
			}
			//trackers = append(trackers, string(k))
		}
		//log.Info("Waiting dynamic tracker done", "size", len(peers))
		if len(trackers) > 0 && flush {
			m.fs.CurrentTorrentManager().UpdateDynamicTrackers(trackers)
			for _, t := range trackers {
				log.Trace("Healthy trackers", "tracker", t)
			}
			elapsed := time.Duration(mclock.Now()) - time.Duration(start)
			log.Info("✨ TORRENT SEARCH COMPLETE", "ips", len(peers), "healthy", len(trackers), "nodes", healthPeers.Len(), "flush", flush, "elapsed", elapsed)
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
	var remainingSize hexutil.Uint64
	if err := m.cl.Call(&remainingSize, "ctxc_getUpload", address, "latest"); err != nil {
		return 0, err
	}
	return uint64(remainingSize), nil
}

func (m *Monitor) parseFileMeta(tx *Transaction, meta *FileMeta) error {
	log.Debug("Monitor", "FileMeta", meta)

	var receipt TxReceipt
	if err := m.cl.Call(&receipt, "ctxc_getTransactionReceipt", tx.Hash.String()); err != nil {
		return err
	}

	if receipt.ContractAddr == nil {
		//log.Warn("Contract address is nil", "receipt", receipt.TxHash)
		return nil
	}

	log.Debug("Transaction Receipt", "address", receipt.ContractAddr.String(), "gas", receipt.GasUsed, "status", receipt.Status, "tx", receipt.TxHash.String())
	//if receipt.GasUsed != params.UploadGas {
	//	log.Warn("Upload gas error", "gas", receipt.GasUsed, "ugas", params.UploadGas)
	//	return nil
	//}

	if receipt.Status != 1 {
		//log.Warn("Upload status error", "status", receipt.Status)
		return nil
	}

	info := m.fs.NewFileInfo(meta)
	info.TxHash = tx.Hash

	info.LeftSize = meta.RawSize
	info.ContractAddr = receipt.ContractAddr
	err := m.fs.AddFile(info)
	if err != nil {
		//return err
	}

	m.dl.UpdateTorrent(FlowControlMeta{
		InfoHash:       meta.InfoHash,
		BytesRequested: 0,
		IsCreate:       true,
	})
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

func (m *Monitor) parseBlockTorrentInfo(b *Block, flowCtrl bool) error {
	if len(b.Txs) > 0 {
		start := mclock.Now()
		for _, tx := range b.Txs {
			if meta := tx.Parse(); meta != nil {
				log.Debug("Try to create a file", "meta", meta, "number", b.Number, "infohash", meta.InfoHash)
				if err := m.parseFileMeta(&tx, meta); err != nil {
					log.Error("Parse file meta error", "err", err, "number", b.Number)
					return err
				}
			} else if flowCtrl && tx.IsFlowControl() {
				addr := *tx.Recipient
				file := m.fs.GetFileByAddr(addr)
				if file == nil {
					log.Warn("Uploading a not exist torrent file", "addr", addr, "tx", tx.Hash.Hex(), "gas", tx.GasLimit, "number", b.Number)
					continue
				}

				log.Debug("Try to upload a file", "addr", addr, "infohash", file.Meta.InfoHash.String(), "number", b.Number)

				remainingSize, err := m.getRemainingSize(addr.String())
				if err != nil {
					return err
				}

				if file.LeftSize > remainingSize {
					file.LeftSize = remainingSize
					err := m.fs.WriteFile(file)
					if err != nil {
						return err
					}

					log.Debug("Update storage success", "hash", file.Meta.InfoHash, "left", file.LeftSize)
					var bytesRequested uint64
					if file.Meta.RawSize > file.LeftSize {
						bytesRequested = file.Meta.RawSize - file.LeftSize
					}

					log.Info("Data processing", "addr", addr.String(), "hash", file.Meta.InfoHash, "remain", remainingSize, "request", bytesRequested, "raw", file.Meta.RawSize, "tx", tx.Hash.Hex(), "number", b.Number)

					m.dl.UpdateTorrent(FlowControlMeta{
						InfoHash:       file.Meta.InfoHash,
						BytesRequested: bytesRequested,
						IsCreate:       false,
					})
				}
			}
		}
		elapsed := time.Duration(mclock.Now()) - time.Duration(start)
		log.Debug("Transactions scanning", "count", len(b.Txs), "number", b.Number, "limit", flowCtrl, "elapsed", common.PrettyDuration(elapsed))
	}

	return nil
}

func (m *Monitor) Stop() {
	log.Info("Torrent listener closing")
	atomic.StoreInt32(&(m.terminated), 1)
	m.closeOnce.Do(func() {
		close(m.exitCh)
		if err := m.fs.Close(); err != nil {
			log.Error("Monitor File Storage closed", "error", err)
		}
		if err := m.dl.Close(); err != nil {
			log.Error("Monitor Torrent Manager closed", "error", err)
		}
		log.Info("Torrent fs listener synchronizing close")
		//m.wg.Wait()
	})
	m.wg.Wait()
	log.Info("Torrent listener closed")
}

// Start ... start ListenOn on the rpc port of a blockchain full node
func (m *Monitor) Start() error {
	if err := m.dl.Start(); err != nil {
		log.Warn("Torrent start error")
		return err
	}

	go func() {
		err := m.startWork()
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
		}
	}()
	return nil
}

func (m *Monitor) startWork() error {
	// Wait for ipc start...
	time.Sleep(time.Second)
	defer TorrentAPIAvailable.Unlock()
	// Rpc Client
	var clientURI string
	if runtime.GOOS != "windows" && m.config.IpcPath != "" {
		clientURI = m.config.IpcPath
	} else {
		if m.config.RpcURI == "" {
			log.Warn("Torrent rpc uri is empty")
			return errors.New("Torrent RpcURI is empty")
		}
		clientURI = m.config.RpcURI
	}

	rpcClient, rpcErr := SetConnection(clientURI)
	if rpcErr != nil {
		log.Error("Torrent rpc client is wrong", "uri", clientURI, "error", rpcErr, "config", m.config)
		return rpcErr
	}
	m.cl = rpcClient

	if err := m.validateStorage(); err != nil {
		log.Error("Starting torrent fs ... ...", "error", err)
		return err
	}

	log.Info("Torrent fs validation passed")
	m.wg.Add(2)
	go m.listenLatestBlock()
	m.init()
	go m.listenPeers()

	return nil
}

func (m *Monitor) validateStorage() error {
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
	}

	for i := uint64(0); i < m.fs.LastFileIndex; i++ {
		file := m.fs.GetFileByNumber(i)

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

		m.fs.AddCachedFile(file)
	}

	if m.lastNumber > 256 {
		m.lastNumber = m.lastNumber - 256
	} else {
		m.lastNumber = 0
	}

	return nil
}

func (m *Monitor) listenLatestBlock() {
	defer m.wg.Done()
	timer := time.NewTimer(time.Second * defaultTimerInterval)
	progress := uint64(0)
	for {
		select {
		case <-timer.C:
			progress = m.syncLastBlock()
			// Aviod sync in full mode, fresh interval may be less.
			if progress > 4096 {
				timer.Reset(time.Millisecond * 100)

			} else if progress > 2048 {
				timer.Reset(time.Millisecond * 500)
			} else if progress > 1024 {
				timer.Reset(time.Millisecond * 1000)
			} else if progress > 6 {
				timer.Reset(time.Millisecond * 2000)
			} else {
				timer.Reset(time.Millisecond * 5000)
			}

		case <-m.exitCh:
			log.Info("Block listener stopped")
			return
		}
	}
}

func (m *Monitor) listenPeers() {
	defer m.wg.Done()
	timer := time.NewTimer(time.Second * 15)

	for {
		select {
		case <-timer.C:
			m.peers()
			if healthPeers.Len() == 0 {
				timer.Reset(time.Second * 5)
			} else if healthPeers.Len() < 6 {
				timer.Reset(time.Second * 60)
			} else {
				timer.Reset(time.Second * 300)
			}
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

func (m *Monitor) batch_http_healthy(ip string, ports []string) ([]string, bool) {
	var res []string
	var status = false
	for _, port := range ports {
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
			/*if _, err = socket.Write(request); err != nil {
				continue
			}
			socket.SetDeadline(time.Now().Add(5 * time.Second))

			reply := make([]byte, 48)
			if _, err = socket.Read(reply); err != nil {
				continue
			}*/
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
	batch = 2048
)

func (m *Monitor) syncLastBlock() uint64 {
	// Latest block number
	var currentNumber hexutil.Uint64

	if err := m.cl.Call(&currentNumber, "ctxc_blockNumber"); err != nil {
		log.Error("Sync old block | IPC ctx_blockNumber", "error", err)
		return 0
	}

	if uint64(currentNumber) <= 0 {
		return 0
	}

	if uint64(currentNumber) < m.lastNumber {
		log.Warn("Torrent fs sync rollback", "current", uint64(currentNumber), "last", m.lastNumber)
		//if m.lastNumber > batch {
		//	m.lastNumber = m.lastNumber - batch
		//}
		m.lastNumber = 0
	}

	//minNumber := uint64(0)
	//if m.lastNumber > 6 {
	//	minNumber = m.lastNumber - 6
	//}
	minNumber := m.lastNumber + 1
	maxNumber := uint64(0)
	if uint64(currentNumber) > 3 {
		//maxNumber = uint64(currentNumber) - 2
		maxNumber = uint64(currentNumber)
	}

	if m.lastNumber > uint64(currentNumber) {
		//block chain rollback
		if m.lastNumber > batch {
			minNumber = m.lastNumber - batch
		}
	}

	if maxNumber > batch+minNumber {
		maxNumber = minNumber + batch
	}
	if maxNumber >= minNumber {
		if minNumber > 5 {
			minNumber = minNumber - 5
		}
		log.Info("Torrent scanning ... ...", "from", minNumber, "to", maxNumber, "current", uint64(currentNumber), "range", uint64(maxNumber-minNumber), "behind", uint64(currentNumber)-maxNumber, "progress", float64(maxNumber)/float64(currentNumber))
	} else {
		return 0
	}

	for i := minNumber; i <= maxNumber; i++ {
		if atomic.LoadInt32(&(m.terminated)) == 1 {
			log.Warn("Torrent scan terminated", "number", i)
			maxNumber = i - 1
			break
		}

		rpcBlock, rpcErr := m.rpcBlockByNumber(i)
		if rpcErr != nil {
			log.Error("Sync old block", "number", i, "error", rpcErr)
			return 0
		}

		if hash, suc := blockCache.Get(i); !suc || hash != rpcBlock.Hash.Hex() {

			block := m.fs.GetBlockByNumber(i)
			if block == nil {
				block = rpcBlock

				if err := m.parseAndStore(block, true); err != nil {
					log.Error("Fail to parse and storge latest block", "number", i, "error", err)
					return 0
				}

			} else {
				if block.Hash.Hex() == rpcBlock.Hash.Hex() {

					if parseErr := m.parseBlockTorrentInfo(block, true); parseErr != nil { //dirty to do
						log.Error("Parse old block", "number", i, "block", block, "error", parseErr)
						return 0
					}
				} else {
					//dirty tfs
					if err := m.parseAndStore(rpcBlock, true); err != nil {
						log.Error("Dirty tfs fail to parse and storge latest block", "number", i, "error", err)
						return 0
					}
				}
			}
			blockCache.Add(i, rpcBlock.Hash.Hex())
		}
	}
	m.lastNumber = maxNumber
	log.Debug("Torrent scan finished", "from", minNumber, "to", maxNumber, "current", uint64(currentNumber), "progress", float64(maxNumber)/float64(currentNumber), "last", m.lastNumber)
	return uint64(maxNumber - minNumber)
}

func (m *Monitor) parseAndStore(block *Block, flow bool) error {
	if parseErr := m.parseBlockTorrentInfo(block, flow); parseErr != nil {
		log.Error("Parse new block", "number", block.Number, "block", block, "error", parseErr)
		return parseErr
	}

	if storeErr := m.fs.WriteBlock(block); storeErr != nil {
		log.Error("Store latest block", "number", block.Number, "error", storeErr)
		return storeErr
	}
	return nil
}
