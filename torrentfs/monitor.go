package torrentfs

import (
	"errors"
	"os"
	"runtime"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/rpc"
)

//------------------------------------------------------------------------------

// Errors that are used throughout the Torrent API.
var (
	ErrBuildConn      = errors.New("build internal-rpc connection failed")
	ErrGetLatestBlock = errors.New("get latest block failed")
	ErrNoRPCClient    = errors.New("no rpc client")

	ErrBlockHash = errors.New("block or parent block hash invalid")
)

const (
	defaultTimerInterval  = 2
	connTryTimes          = 300
	connTryInterval       = 10
	fetchBlockTryTimes    = 5
	fetchBlockTryInterval = 3
	fetchBlockLogStep     = 10000
	minBlockNum           = 0

	maxSyncBlocks = 1024
)

type TorrentManagerAPI interface {
	Start() error
	Close() error
	NewTorrent(string) error
	RemoveTorrent(string) error
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
}

// NewMonitor creates a new instance of monitor.
// Once Ipcpath is settle, this method prefers to build socket connection in order to
// get higher communicating performance.
// IpcPath is unavailable on windows.
func NewMonitor(flag *Config) (*Monitor, error) {
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

	return &Monitor{
		config:      flag,
		cl:          nil,
		fs:          fs,
		dl:          tMana,
		uncheckedCh: make(chan uint64, 20),
		exitCh:      make(chan struct{}),
		terminated:  0,
	}, nil
}

// SetConnection method builds connection to remote or local communicator.
func SetConnection(clientURI string) (*rpc.Client, error) {
	for i := 0; i < connTryTimes; i++ {
		cl, err := rpc.Dial(clientURI)
		if err != nil {
			log.Warn("Building internal-rpc connection failed", "URI", clientURI, "times", i, "error", err)
		} else {
			log.Debug("Internal-IPC connection established", "URI", clientURI)
			return cl, nil
		}

		time.Sleep(time.Second * connTryInterval)
	}

	return nil, errors.New("Building Internal-IPC Connection Failed")
}

func (m *Monitor) rpcBlockByNumber(blockNumber uint64) (*Block, error) {
	block := &Block{}
	blockNumberHex := "0x" + strconv.FormatUint(blockNumber, 16)

	for i := 0; i < fetchBlockTryTimes; i++ {
		err := m.cl.Call(block, "eth_getBlockByNumber", blockNumberHex, true)
		if err == nil {
			return block, nil
		}

		time.Sleep(time.Second * fetchBlockTryInterval)
		log.Warn("Torrent Fs Internal IPC ctx_getBlockByNumber", "retry", i, "error", err)
	}

	return nil, errors.New("[ Internal IPC Error ] try to get block out of times")
}

func (m *Monitor) rpcBlockByHash(blockHash string) (*Block, error) {
	block := &Block{}

	for i := 0; i < fetchBlockTryTimes; i++ {
		err := m.cl.Call(block, "eth_getBlockByHash", blockHash, true)
		if err == nil {
			return block, nil
		}

		time.Sleep(time.Second * fetchBlockTryInterval)
		log.Warn("Torrent Fs Internal IPC ctx_getBlockByHash", "retry", i, "error", err)
	}

	return nil, errors.New("[ Internal IPC Error ] try to get block out of times")
}

func (m *Monitor) getBlockByNumber(blockNumber uint64) (*Block, error) {
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
}

func (m *Monitor) parseFileMeta(tx *Transaction, meta *FileMeta) error {
	m.dl.NewTorrent(meta.URI)

	info := NewFileInfo(meta)
	info.TxHash = tx.Hash

	var receipt TxReceipt
	if err := m.cl.Call(&receipt, "eth_getTransactionReceipt", tx.Hash.String()); err != nil {
		return err
	}

	var _remainingSize string
	if err := m.cl.Call(&_remainingSize, "eth_getUpload", receipt.ContractAddr.String(), "latest"); err != nil {
		return err
	}

	remainingSize, _ := strconv.ParseUint(_remainingSize[2:], 16, 64)
	info.LeftSize = remainingSize
	info.ContractAddr = receipt.ContractAddr
	m.fs.AddFile(info)
	var bytesRequested uint64
	if meta.RawSize > remainingSize {
		bytesRequested = meta.RawSize - remainingSize
	}
	m.dl.UpdateTorrent(FlowControlMeta{
		InfoHash:       *meta.InfoHash(),
		BytesRequested: bytesRequested,
	})
	return nil
}

func (m *Monitor) parseBlockTorrentInfo(b *Block, flowCtrl bool) error {
	if len(b.Txs) > 0 {
		for _, tx := range b.Txs {
			if meta := tx.Parse(); meta != nil {
				if err := m.parseFileMeta(&tx, meta); err != nil {
					return err
				}
			} else if flowCtrl && tx.IsFlowControl() {
				addr := *tx.Recipient
				file := m.fs.GetFileByAddr(addr)
				if file == nil {
					continue
				}

				var remainingSize hexutil.Uint64
				if err := m.cl.Call(&remainingSize, "eth_getUpload", addr.String(), "latest"); err != nil {
					return err
				}

				var bytesRequested uint64
				file.LeftSize = uint64(remainingSize)
				if file.Meta.RawSize > file.LeftSize {
					bytesRequested = file.Meta.RawSize - file.LeftSize
				}
				m.dl.UpdateTorrent(FlowControlMeta{
					InfoHash:       *file.Meta.InfoHash(),
					BytesRequested: bytesRequested,
				})
			}
		}
	}

	return nil
}

func (m *Monitor) Stop() {
	atomic.StoreInt32(&(m.terminated), 1)
	close(m.exitCh)

	if err := m.fs.Close(); err != nil {
		log.Error("Monitor File Storage Closed", "error", err)
	}
	if err := m.dl.Close(); err != nil {
		log.Error("Monitor Torrent Manager Closed", "error", err)
	}
}

// Start ... start ListenOn on the rpc port of a blockchain full node
func (m *Monitor) Start() error {
	if err := m.dl.Start(); err != nil {
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

	// Rpc Client
	var clientURI string
	if runtime.GOOS != "windows" && m.config.IpcPath != "" {
		clientURI = m.config.IpcPath
	} else {
		if m.config.RpcURI == "" {
			return errors.New("Torrent RpcURI is empty")
		}
		clientURI = m.config.RpcURI
	}

	rpcClient, rpcErr := SetConnection(clientURI)
	if rpcErr != nil {
		return rpcErr
	}
	m.cl = rpcClient

	if vaErr := m.validateStorage(); vaErr != nil {
		return vaErr
	}

	// Used for listen latest block
	if blockFilterErr := m.cl.Call(&m.listenID, "eth_newBlockFilter"); blockFilterErr != nil {
		log.Error("Start listen block filter | IPC eth_newBlockFilter", "error", blockFilterErr)
		return blockFilterErr
	}

	go m.syncLastBlock()
	go m.listenLatestBlock()

	return nil
}

func (m *Monitor) validateStorage() error {
	lastNumber := m.fs.LastListenBlockNumber
	log.Info("Validate Torrent FS Storage", "last IPC listen number", lastNumber)
	for i := lastNumber; i > 0; i-- {
		rpcBlock, rpcErr := m.rpcBlockByNumber(uint64(i))
		if rpcErr != nil {
			return rpcErr
		}

		stBlock := m.fs.GetBlockByNumber(uint64(i))
		if stBlock == nil {
			log.Warn("Vaidate Torrent FS Storage state invalid", "number", lastNumber, "error", "LastListenBlockNumber not persistent")
			return nil
		}

		if rpcBlock.Hash.Hex() == stBlock.Hash.Hex() {
			return nil
		}

		// block in storage invalid
		log.Debug("Update invalid block in storage", "old hash", stBlock.Hash, "new hash", rpcBlock.Hash)
		m.fs.WriteBlock(rpcBlock)
	}

	return nil
}

func (m *Monitor) listenLatestBlock() {
	timer := time.NewTimer(time.Second * defaultTimerInterval)

	blockFilter := func() {
		// If blockchain rolled back, the `eth_getFilterChanges` function will send rewritable
		// block hash. It's a simple operator to write block to storage.
		var blockHashes []string
		if changeErr := m.cl.Call(&blockHashes, "eth_getFilterChanges", m.listenID); changeErr != nil {
			log.Error("Listen latest block | IPC ctx_getFilterChanges", "error", changeErr)
			return
		}

		if len(blockHashes) > 0 {
			log.Debug("Torrent FS IPC blocks range", "piece", len(blockHashes))
		}

		for _, hash := range blockHashes {
			block, rpcErr := m.rpcBlockByHash(hash)
			if rpcErr != nil {
				log.Error("Listen latest block", "hash", hash, "error", rpcErr)
				return
			}

			log.Debug("Torrent FS IPC block", "number", block.Number, "hash", hash)

			if parseErr := m.parseBlockTorrentInfo(block, true); parseErr != nil {
				log.Error("Parse latest block", "hash", hash, "block", block, "error", parseErr)
				return
			}

			if storeErr := m.fs.WriteBlock(block); storeErr != nil {
				log.Error("Store latest block", "hash", hash, "error", storeErr)
				return
			}
		}
	}

	for {
		select {
		case <-timer.C:
			go blockFilter()

			// Aviod sync in full mode, fresh interval may be less.
			timer.Reset(time.Second * 3)

		case <-m.exitCh:
			return
		}
	}
}

func (m *Monitor) syncLastBlock() {
	// Latest block number
	var currentNumber hexutil.Uint64

	if err := m.cl.Call(&currentNumber, "ctx_blockNumber"); err != nil {
		log.Error("Sync old block | IPC ctx_blockNumber", "error", err)
		return
	}

	minNumber := uint64(minBlockNum)
	maxNumber := uint64(currentNumber)
	log.Info("Fetch Block Range", "min", minNumber, "max", maxNumber)

	lastBlock := minNumber
	for i := minNumber; i <= maxNumber; i++ {
		if atomic.LoadInt32(&(m.terminated)) == 1 {
			break
		}

		block := m.fs.GetBlockByNumber(i)
		if block == nil {
			rpcBlock, rpcErr := m.rpcBlockByNumber(i)
			block = rpcBlock
			if rpcErr != nil {
				log.Error("Sync old block", "number", i, "error", rpcErr)
				return
			}

			if storeErr := m.fs.WriteBlock(block); storeErr != nil {
				log.Error("Store latest block", "number", i, "error", storeErr)
				return
			}
		}

		if parseErr := m.parseBlockTorrentInfo(block, false); parseErr != nil {
			log.Error("Parse old block", "number", i, "block", block, "error", parseErr)
			return
		}

		if (i-minNumber)%fetchBlockLogStep == 0 || i == maxNumber {
			log.Debug("Blocks have been checked", "from", lastBlock, "to", i)
			lastBlock = i + uint64(1)
		}
	}
}
