package torrentfs

import (
	"errors"
	"runtime"
	"strconv"
	"time"

	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/rpc"
)

//------------------------------------------------------------------------------

// Errors that are used throughout the Torrent API.
var (
	ErrBuildConn         = errors.New("build internal-rpc connection failed")
	ErrGetLatestBlock    = errors.New("get latest block failed")
	ErrNoRPCClient       = errors.New("no rpc client")
	ErrNoDownloadManager = errors.New("no download manager")
)

const (
	defaultTimerInterval  = 2
	connTryTimes          = 300
	connTryInterval       = 10
	fetchBlockTryTimes    = 5
	fetchBlockTryInterval = 3
	fetchBlockLogStep     = 500
	minBlockNum           = 36000
)


type TorrentManagerAPI interface {
	CloseAll(struct{})         error
	NewTorrent(string)         error
	RemoveTorrent(string)      error
	UpdateTorrent(interface{}) error
}

// Monitor observes the data changes on the blockchain and synchronizes.
// cl for ipc/rpc communication, dl for download manager, and fs for data storage.
type Monitor struct {
	cl *rpc.Client
	dl TorrentManagerAPI
	fs *FileStorage
}

// NewMonitor creates a new instance of monitor.
// Once Ipcpath is settle, this method perfers to build socket connection in order to
// get higher communicating performance.
// IpcPath is unaviliable on windows.
func NewMonitor(flag *Config) *Monitor {
	m := &Monitor{
		nil,
		nil,
		NewFileStorage(flag),
	}
	if runtime.GOOS != "windows" && flag.IpcPath != "" {
		m.SetConnection(flag.IpcPath)
	} else {
		if flag.RpcURI == "" {
			return nil
		}
		m.SetConnection(flag.RpcURI)
	}
	return m
}

// SetConnection method builds connection to remote or local communicator.
func (m *Monitor) SetConnection(clientURI string) (e error) {
	for i := 0; i < connTryTimes; i++ {
		cl, err := rpc.Dial(clientURI)
		if err != nil {
			log.Info("Building internal-rpc connection failed", "URI", clientURI, "times", i)
			e = err
		} else {
			e = nil
			log.Info("Internal-RPC connection established", "URI", clientURI)
			m.cl = cl
			break
		}
		time.Sleep(time.Second * connTryInterval)
	}
	return
}

// SetDownloader ...
func (m *Monitor) SetDownloader(dl TorrentManagerAPI) error {
	if dl == nil {
		return ErrNoDownloadManager
	}
	log.Info("Torrent manager initialized")
	m.dl = dl
	return nil
}

func (m *Monitor) getBlockByNumber(blockNumber uint64) (block *Block, e error) {
	block = m.fs.GetBlock(blockNumber)
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

func (m *Monitor) parseNewBlockByNumber(blockNumber uint64) error {
	if m.cl == nil {
		return ErrNoRPCClient
	}
	block, err := m.getBlockByNumber(blockNumber)
	if err != nil {
		return err
	}
	if err := m.parseNewBlock(block); err != nil {
		return err
	}
	log.Info("Fetch block", "Number", block.Number, "Txs", len(block.Txs))
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
	log.Info("Fetch block", "Hash", hash, "Txs", len(block.Txs))
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
	// log.Info(receipt.ContractAddr.String())
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

func (m *Monitor) initialCheck() {
	blockChecked := 0
	lastblock := m.fs.LatestBlockNumber
	log.Info("Fetch Block from", "Number", lastblock)
	for i := lastblock; i >= minBlockNum; i-- {
		if m.fs.HasBlock(i) {
			continue
		}
		m.parseBlockByNumber(i)
		blockChecked++
		if blockChecked%fetchBlockLogStep == 0 || i == 0 {
			log.Info("Blocks have been checked", "from", i, "to", lastblock)
			lastblock = i - 1
		}
	}
}

func (m *Monitor) getLatestBlock() (b *Block, e error) {
	b = &Block{}
	for i := 0; i < fetchBlockTryTimes; i++ {
		err := m.cl.Call(b, "eth_getBlockByNumber", "latest", true)
		if err != nil {
			e = err
		} else {
			e = nil
			break
		}
		time.Sleep(time.Second * fetchBlockTryInterval)
	}
	return
}

// Start ... start ListenOn on the rpc port of a blockchain full node
func (m *Monitor) Start() error {
	b, err := m.getLatestBlock()
	if err != nil {
		log.Info("Fetch latest block failed")
		return err
	}
	m.parseNewBlock(b)
	go m.initialCheck()

	timer := time.NewTimer(time.Second * defaultTimerInterval)
	for {
		select {
		case <-timer.C:
			// Try to get the latest b
			b := &Block{}
			if err := m.cl.Call(b, "eth_getBlockByNumber", "latest", true); err != nil {
				timer.Reset(time.Second * 2)
				continue
			}
			bnum := b.Number
			log.Info("try to fetch new block", "number", bnum)
			if bnum > m.fs.LatestBlockNumber {
				m.parseBlock(b)
				log.Info("Fetch block", "Number", bnum, "Txs", len(b.Txs))
				for i := m.fs.LatestBlockNumber - 1; i >= minBlockNum; i-- {
					if m.fs.HasBlock(i) {
						break
					}
					m.parseNewBlockByNumber(i)
				}
			}
			timer.Reset(time.Second * 3)
		}
	}
}
