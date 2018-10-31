package torrentfs

import (
	"context"
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
)

const (
	defaultTimerInterval  = 2
	connTryTimes          = 300
	connTryInterval       = 10
	fetchBlockTryTimes    = 5
	fetchBlockTryInterval = 3
	fetchBlockLogStep     = 1000
	minBlockNum           = 0
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
	config     *Config
	cl         *rpc.Client
	fs         *FileStorage
	dl         TorrentManagerAPI
	exitCh     chan struct{}
	terminated int32
}

// NewMonitor creates a new instance of monitor.
// Once Ipcpath is settle, this method prefers to build socket connection in order to
// get higher communicating performance.
// IpcPath is unavailable on windows.
func NewMonitor(flag *Config) (*Monitor, error) {
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
		config:     flag,
		cl:         nil,
		fs:         fs,
		dl:         tMana,
		exitCh:     make(chan struct{}),
		terminated: 0,
	}, nil
}

func (m *Monitor) Call(result interface{}, method string, args ...interface{}) error {
	if m.config.TestMode {
		return nil
	} else {
		ctx := context.Background()
		return m.cl.CallContext(ctx, result, method, args...)
	}
}

// SetConnection method builds connection to remote or local communicator.
func SetConnection(clientURI string) (*rpc.Client, error) {
	for i := 0; i < connTryTimes; i++ {
		cl, err := rpc.Dial(clientURI)
		if err != nil {
			log.Warn("Building internal-rpc connection failed", "URI", clientURI, "times", i, "error", err)
		} else {
			log.Debug("Internal-RPC connection established", "URI", clientURI)
			return cl, nil
		}

		time.Sleep(time.Second * connTryInterval)
	}

	return nil, errors.New("Building Internal-RPC Connection Failed")
}

func (m *Monitor) getBlockByNumber(blockNumber uint64) (block *Block, e error) {
	block = m.fs.GetBlockByNumber(blockNumber)
	if block == nil {
		block = &Block{}
		blockNumberHex := "0x" + strconv.FormatUint(blockNumber, 16)

		for i := 0; i < fetchBlockTryTimes; i++ {
			err := m.cl.Call(block, "eth_getBlockByNumber", blockNumberHex, true)
			if err != nil {
				e = err
			} else {
				e = nil
				break
			}
			time.Sleep(time.Second * fetchBlockTryInterval)
		}
	}
	return
}

func (m *Monitor) getBlockNumber() (hexutil.Uint64, error) {
	var blockNumber hexutil.Uint64

	for i := 0; i < fetchBlockTryTimes; i++ {
		if err := m.cl.Call(&blockNumber, "ctx_blockNumber"); err == nil {
			return blockNumber, nil
		}

		time.Sleep(time.Second * fetchBlockTryInterval)
		log.Warn("Torrent Fs Internal JSON-RPC ctx_blockNumber", "retry", i)
	}

	return 0, errors.New("[ Internal JSON-RPC Error ] try to get block number out of times")
}

func (m *Monitor) parseBlockByNumber(blockNumber uint64) error {
	if m.cl == nil {
		return ErrNoRPCClient
	}
	block, err := m.getBlockByNumber(blockNumber)
	if err != nil {
		return err
	}
	if err := m.parseBlock(block); err != nil {
		return err
	}
	return nil
}

func (m *Monitor) parseBlockByHash(hash string) error {
	if m.cl == nil {
		return ErrNoRPCClient
	}
	block := &Block{}
	if err := m.cl.Call(block, "eth_getBlockByHash", hash, true); err != nil {
		return err
	}
	m.parseBlock(block)
	log.Debug("Fetch block", "Hash", hash, "Txs", len(block.Txs))
	return nil
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

func (m *Monitor) parseNewBlock(b *Block) error {
	m.fs.AddBlock(b)
	if len(b.Txs) > 0 {
		for _, tx := range b.Txs {
			if meta := tx.Parse(); meta != nil {
				if err := m.parseFileMeta(&tx, meta); err != nil {
					return err
				}
			} else if tx.IsFlowControl() {
				addr := *tx.Recipient
				file := m.fs.GetFileByAddr(addr)
				if file == nil {
					continue
				}
				var _remainingSize string
				if err := m.cl.Call(&_remainingSize, "eth_getUpload", addr.String(), "latest"); err != nil {
					return err
				}
				remainingSize, _ := strconv.ParseUint(_remainingSize[2:], 16, 64)

				var bytesRequested uint64
				file.LeftSize = remainingSize
				if file.Meta.RawSize > remainingSize {
					bytesRequested = file.Meta.RawSize - remainingSize
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

//
func (m *Monitor) parseBlock(b *Block) error {
	m.fs.AddBlock(b)
	if len(b.Txs) > 0 {
		for _, tx := range b.Txs {
			if meta := tx.Parse(); meta != nil {
				if err := m.parseFileMeta(&tx, meta); err != nil {
					return err
				}
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

	// Latest block number
	number, err := m.getBlockNumber()
	if err != nil {
		log.Error("Torrent Fs Internal JSON-RPC Error")
		return err
	}

	m.fs.LatestBlockNumber.Store(&number)

	go m.syncLastBlock()
	go m.listenLatestBlock()

	return nil
}

func (m *Monitor) listenLatestBlock() {
	timer := time.NewTimer(time.Second * defaultTimerInterval)
	counter := 0

	for {
		select {
		case <-timer.C:
			counter += 1

			bnum, err := m.getBlockNumber()
			if err != nil {
				log.Warn("Torrent Fs Internal JSON-RPC Error", "error", err)
				timer.Reset(time.Second * 2)
				continue
			}

			if counter > 10 {
				counter = 0
				log.Info("Try to fetch blocks", "number", uint64(bnum))
			}

			oldNumber := *(m.fs.CurrentBlockNumber())
			if bnum > oldNumber {
				// Update latest block number
				m.fs.LatestBlockNumber.Store(&bnum)

				for i := bnum; i >= oldNumber; i-- {
					if m.fs.HasBlock(uint64(i)) {
						break
					}
					m.parseBlockByNumber(uint64(i))
					log.Debug("Fetch block", "Number", uint64(i))
				}
			}
			timer.Reset(time.Second * 3)
		case <-m.exitCh:
			return
		}
	}
}

func (m *Monitor) syncLastBlock() {
	reverse := m.config.SyncMode != "full"

	blockChecked := 0
	minNumber := uint64(minBlockNum)
	maxNumber := uint64(*(m.fs.CurrentBlockNumber()))
	log.Info("Fetch Block Range", "min", minNumber, "max", maxNumber)

	if reverse {
		lastBlock := maxNumber
		for i := maxNumber; i >= minNumber; i-- {
			if atomic.LoadInt32(&(m.terminated)) == 1 {
				break
			}

			blockChecked++
			m.parseBlockByNumber(uint64(i))
			if blockChecked%fetchBlockLogStep == 0 || i == 0 {
				log.Debug("Blocks have been checked", "from", i, "to", lastBlock)
				lastBlock = i - uint64(1)
			}
		}
	} else {
		lastBlock := minNumber
		for i := uint64(minNumber); i <= maxNumber; i++ {
			if atomic.LoadInt32(&(m.terminated)) == 1 {
				break
			}

			blockChecked++
			m.parseBlockByNumber(i)
			if blockChecked%fetchBlockLogStep == 0 || i == maxNumber {
				log.Debug("Blocks have been checked", "from", lastBlock, "to", i)
				lastBlock = i + uint64(1)
			}
		}
	}
}
