package torrentfs

import (
	"errors"
	"os"
	"runtime"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/CortexTheseus/rpc"
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
	lastNumber uint64
	dirty      bool
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
		lastNumber:  uint64(0),
		dirty:       false,
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

func (m *Monitor) parseFileMeta(tx *Transaction, meta *FileMeta) error {
	m.dl.NewTorrent(meta.URI)

	var receipt TxReceipt
	if err := m.cl.Call(&receipt, "ctxc_getTransactionReceipt", tx.Hash.String()); err != nil {
		return err
	}

	if receipt.ContractAddr == nil {
		log.Warn("Contract address is nil", "receipt", receipt.TxHash)
		return nil
	}

	log.Debug("Transaction Receipt", "address", receipt.ContractAddr.String(), "gas", receipt.GasUsed, "status", receipt.Status, "tx", receipt.TxHash.String())
	//if receipt.GasUsed != params.UploadGas {
	//	log.Warn("Upload gas error", "gas", receipt.GasUsed, "ugas", params.UploadGas)
	//	return nil
	//}

	if receipt.Status != 1 {
		log.Warn("Upload status error", "status", receipt.Status)
		return nil
	}

	var _remainingSize string
	if err := m.cl.Call(&_remainingSize, "ctxc_getUpload", receipt.ContractAddr.String(), "latest"); err != nil {
		log.Warn("Failed to call get upload", "addr", receipt.ContractAddr.String())
		return err
	}

	info := NewFileInfo(meta)
	info.TxHash = tx.Hash

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
	log.Info("Parse file meta successfully", "tx", receipt.TxHash.Hex(), "remain", remainingSize, "meta", meta)
	return nil
}

func (m *Monitor) parseBlockTorrentInfo(b *Block, flowCtrl bool) error {
	if len(b.Txs) > 0 {
		start := mclock.Now()
		for _, tx := range b.Txs {
			if meta := tx.Parse(); meta != nil {
				log.Info("Try to create a file", "meta", meta, "number", b.Number)
				if err := m.parseFileMeta(&tx, meta); err != nil {
					log.Error("Parse file meta error", "err", err, "number", b.Number)
					return err
				}
			} else if flowCtrl && tx.IsFlowControl() {
				addr := *tx.Recipient
				file := m.fs.GetFileByAddr(addr)
				log.Info("Try to upload a file", "addr", addr, "tx", tx.Hash.Hex(), "number", b.Number)
				if file == nil {
					log.Warn("Uploading a not exist torrent file", "addr", addr, "tx", tx.Hash.Hex(), "gas", tx.GasLimit, "number", b.Number)
					continue
				}

				var remainingSize hexutil.Uint64
				if err := m.cl.Call(&remainingSize, "ctxc_getUpload", addr.String(), "latest"); err != nil {
					log.Warn("Failed call get upload", "addr", addr.String(), "tx", tx.Hash.Hex(), "number", b.Number)
					return err
				}

				var bytesRequested uint64
				file.LeftSize = uint64(remainingSize)
				if file.Meta.RawSize > file.LeftSize {
					bytesRequested = file.Meta.RawSize - file.LeftSize
				}
				log.Info("Data downloading", "remain", remainingSize, "request", bytesRequested, "raw", file.Meta.RawSize, "tx", tx.Hash.Hex(), "number", b.Number)
				m.dl.UpdateTorrent(FlowControlMeta{
					InfoHash:       *file.Meta.InfoHash(),
					BytesRequested: bytesRequested,
				})
			}
		}
		elapsed := time.Duration(mclock.Now()) - time.Duration(start)
		log.Info("Transactions scanning", "count", len(b.Txs), "number", b.Number, "limit", flowCtrl, "elapsed", common.PrettyDuration(elapsed))
	}

	return nil
}

func (m *Monitor) Stop() {
	atomic.StoreInt32(&(m.terminated), 1)
	close(m.exitCh)

	// var stopFilterFlag bool
	//if blockFilterErr := m.cl.Call(&stopFilterFlag, "ctxc_uninstallFilter", m.listenID); blockFilterErr != nil {
	// log.Error("Block Filter closed | IPC ctxc_uninstallFilter", "error", blockFilterErr)
	//}

	if err := m.fs.Close(); err != nil {
		log.Error("Monitor File Storage closed", "error", err)
	}
	if err := m.dl.Close(); err != nil {
		log.Error("Monitor Torrent Manager closed", "error", err)
	}
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
		log.Error("Torrent rpc client is wrong", "uri", clientURI, "error", rpcErr)
		return rpcErr
	}
	m.cl = rpcClient

	errCh := make(chan error)

	/*if vaErr := m.validateStorage(); vaErr != nil {
		log.Error("Torrent invalid storage", "error", vaErr)
		return vaErr
	}*/
	go m.validateStorage(errCh)

	for {
		select {
		case err := <-errCh:
			if err != nil {
				//return err
				log.Warn("Starting torrent fs ... ...", "error", err)
				go m.validateStorage(errCh)
			} else {
				log.Info("Torrent fs validation passed")
				//break
				go m.listenLatestBlock()
				return nil
			}
		}
	}

	// Used for listen latest block
	//if blockFilterErr := m.cl.Call(&m.listenID, "ctxc_newBlockFilter"); blockFilterErr != nil {
	//	log.Error("Start listen block filter | IPC ctxc_newBlockFilter", "error", blockFilterErr)
	//	return blockFilterErr
	//}

	//go m.syncLastBlock()
	//go m.listenLatestBlock()
	log.Warn("... ... ...")

	return nil
}

func (m *Monitor) validateStorage(errCh chan error) error {
	m.lastNumber = m.fs.LastListenBlockNumber
	end := uint64(0)

	if m.lastNumber > 4096 {
		end = m.lastNumber - 4096
	}

	log.Info("Validate Torrent FS Storage", "last IPC listen number", m.lastNumber, "end", end, "lastest", m.fs.LastListenBlockNumber)

	for i := m.lastNumber; i > end; i-- {
		rpcBlock, rpcErr := m.rpcBlockByNumber(uint64(i))
		if rpcErr != nil {
			log.Warn("RPC ERROR", "error", rpcErr)
			errCh <- rpcErr
			return rpcErr
		}

		if rpcBlock == nil || rpcBlock.Hash == common.EmptyHash {
			log.Warn("No block found", "number", i)
			m.lastNumber = uint64(i)
			//errCh <- nil
			//return nil
			m.dirty = true
			continue
		} else {
			m.dirty = false
		}

		stBlock := m.fs.GetBlockByNumber(uint64(i))
		if stBlock == nil {
			log.Warn("Vaidate Torrent FS Storage state failed, rescan", "number", m.lastNumber, "error", "LastListenBlockNumber not persistent", "dirty", m.fs.LastListenBlockNumber)
			m.lastNumber = uint64(0)
			errCh <- nil
			return nil
		}

		if rpcBlock.Hash.Hex() == stBlock.Hash.Hex() {
			//log.Warn("Validate TFS continue", "number", m.lastNumber, "rpc", rpcBlock.Hash.Hex(), "store", stBlock.Hash.Hex())
			continue
		}

		// block in storage invalid
		log.Info("Update invalid block in storage", "old hash", stBlock.Hash, "new hash", rpcBlock.Hash, "latest", m.fs.LastListenBlockNumber)
		m.lastNumber = uint64(0)
		errCh <- nil
		return nil
	}
	log.Info("Validate Torrent FS Storage ended", "last IPC listen number", m.lastNumber, "end", end, "latest", m.fs.LastListenBlockNumber)

	if m.dirty {
		log.Warn("Torrent fs status", "dirty", m.dirty)
		m.lastNumber = uint64(0)
	}

	errCh <- nil
	return nil
}

func (m *Monitor) listenLatestBlock() {
	timer := time.NewTimer(time.Second * defaultTimerInterval)

	/*blockFilter := func() {
		log.Info("Torrent listen latest block status")
		var blockHashes []string
		if changeErr := m.cl.Call(&blockHashes, "ctxc_getFilterChanges", m.listenID); changeErr != nil {
			log.Error("Listen latest block | IPC ctx_getFilterChanges", "error", changeErr)
			return
		}

		if len(blockHashes) > 0 {
			log.Trace("Torrent FS IPC blocks range", "piece", len(blockHashes))
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
	}*/

	for {
		select {
		case <-timer.C:
			//go blockFilter()
			//go m.syncLastBlock()
			m.syncLastBlock()
			// Aviod sync in full mode, fresh interval may be less.
			timer.Reset(time.Millisecond * 1000)

		case <-m.exitCh:
			return
		}
	}
}

//var lastBlock uint64 = m.fs.LastListenBlockNumber

const (
	batch = 2048
)

func (m *Monitor) syncLastBlock() {
	//log.Info("Torrent sync latest block", "latest", m.lastNumber)
	// Latest block number
	var currentNumber hexutil.Uint64

	if err := m.cl.Call(&currentNumber, "ctxc_blockNumber"); err != nil {
		log.Error("Sync old block | IPC ctx_blockNumber", "error", err)
		return
	}

	if uint64(currentNumber) <= 0 {
		return
	}

	//minNumber := uint64(minBlockNum)
	minNumber := m.lastNumber + 1
	maxNumber := uint64(0)
	if uint64(currentNumber) > params.SeedingBlks/2 {
		maxNumber = uint64(currentNumber) - 3
	}

	if m.lastNumber > uint64(currentNumber) {
		//block chain rollback
		if m.lastNumber > 2048 {
			minNumber = m.lastNumber - 2048
		}
	}

	if maxNumber > batch+minNumber {
		maxNumber = minNumber + batch
	}
	if maxNumber > minNumber {
		log.Info("Torrent scanning ... ...", "from", minNumber, "to", maxNumber, "current", uint64(currentNumber), "progress", float64(maxNumber)/float64(currentNumber))
	} else {
		return
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
			return
		}

		block := m.fs.GetBlockByNumber(i)
		if block == nil {
			//rpcBlock, rpcErr := m.rpcBlockByNumber(i)
			//if rpcErr != nil {
			//	log.Error("Sync old block", "number", i, "error", rpcErr)
			//	return
			//}

			block = rpcBlock

			/*if parseErr := m.parseBlockTorrentInfo(block, true); parseErr != nil {
				log.Error("Parse new block", "number", i, "block", block, "error", parseErr)
				return
			}

			if storeErr := m.fs.WriteBlock(block); storeErr != nil {
				log.Error("Store latest block", "number", i, "error", storeErr)
				return
			}*/

			if err := m.parseAndStore(block, true); err != nil {
				log.Error("Fail to parse and storge latest block", "number", i, "error", err)
				return
			}

		} else {
			if block.Hash.Hex() == rpcBlock.Hash.Hex() {

				if parseErr := m.parseBlockTorrentInfo(block, false); parseErr != nil { //dirty to do
					log.Error("Parse old block", "number", i, "block", block, "error", parseErr)
					return
				}
			} else {
				//dirty tfs
				if err := m.parseAndStore(rpcBlock, true); err != nil {
					log.Error("Dirty tfs fail to parse and storge latest block", "number", i, "error", err)
					return
				}
			}
		}

		//if (i-minNumber)%fetchBlockLogStep == 0 || i == maxNumber {
		//	log.Debug("Blocks have been checked", "from", lastBlock, "to", i)
		//	lastBlock = i + uint64(1)
		//}
	}
	m.lastNumber = maxNumber
	log.Debug("Torrent scan finished", "from", minNumber, "to", maxNumber, "current", uint64(currentNumber), "progress", float64(maxNumber)/float64(currentNumber), "last", m.lastNumber)
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
