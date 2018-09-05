package monitor

import (
	"errors"
	"log"
	"strconv"
	"time"

	download "../manager"
	"../types"
	"github.com/CortexFoundation/CortexTheseus/rpc"
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
	defaultTimerInterval = 2
	fetchBlockLogStep    = 500
	minBlockNum          = 36000
)

// Monitor ...
type Monitor struct {
	cl                *rpc.Client
	dl                *download.TorrentManager
	files             map[string]*types.FileInfo
	blocks            map[uint64]*types.Block
	latestBlockNumber uint64
}

// NewMonitor ...
func NewMonitor() *Monitor {
	m := &Monitor{
		nil,
		nil,
		make(map[string]*types.FileInfo),
		make(map[uint64]*types.Block),
		0,
	}
	return m
}

// SetRPCServer ...
func (m *Monitor) SetRPCServer(clientURI *string) error {
	cl, err := rpc.Dial(*clientURI)
	if err != nil {
		return ErrBuildConn
	}
	log.Println("Internal-RPC connection established.")
	m.cl = cl
	return nil
}

// SetDownloader ...
func (m *Monitor) SetDownloader(dl *download.TorrentManager) error {
	if dl == nil {
		return ErrNoDownloadManager
	}
	log.Println("Torrent manager initialized.")
	m.dl = dl
	return nil
}

func (m *Monitor) verifyBlock(b *types.Block) error {
	return nil
}

func (m *Monitor) parseBlockByNumber(blockNumber uint64) error {
	if m.cl == nil {
		return ErrNoRPCClient
	}
	block := &types.Block{}
	m.blocks[blockNumber] = block
	blockNumberHex := "0x" + strconv.FormatUint(blockNumber, 16)
	if err := m.cl.Call(block, "eth_getBlockByNumber", blockNumberHex, true); err != nil {
		return err
	}
	if err := m.parseBlock(block); err != nil {
		return err
	}
	if err := m.verifyBlock(block); err != nil {
		return err
	}
	return nil
}

func (m *Monitor) parseNewBlockByNumber(blockNumber uint64) error {
	if m.cl == nil {
		return ErrNoRPCClient
	}
	block := &types.Block{}
	m.blocks[blockNumber] = block
	blockNumberHex := "0x" + strconv.FormatUint(blockNumber, 16)
	if err := m.cl.Call(block, "eth_getBlockByNumber", blockNumberHex, true); err != nil {
		return err
	}
	if err := m.parseNewBlock(block); err != nil {
		return err
	}
	log.Printf("Fetch block #%s with %d Txs", blockNumberHex, len(block.Txs))
	if err := m.verifyBlock(block); err != nil {
		return err
	}
	return nil
}

func (m *Monitor) parseBlockByHash(hash string) error {
	if m.cl == nil {
		return ErrNoRPCClient
	}
	block := &types.Block{}
	if err := m.cl.Call(block, "eth_getBlockByHash", hash, true); err != nil {
		return err
	}
	blockNumber := block.Number
	m.blocks[blockNumber] = block
	m.parseBlock(block)
	log.Printf("Fetch block #%s with %d Txs", hash, len(block.Txs))
	if err := m.verifyBlock(block); err != nil {
		return err
	}
	return nil
}

func (m *Monitor) parseNewBlock(b *types.Block) error {
	blockNumber := b.Number
	m.blocks[blockNumber] = b
	if len(b.Txs) > 0 {
		for _, tx := range b.Txs {
			if meta := tx.Parse(); meta != nil {
				m.dl.NewTorrent <- meta.URI

				info := types.NewFileInfo(meta)
				info.TxHash = tx.Hash

				var receipt types.Receipt
				// TODO: Fix string format
				log.Println(tx.Hash.String())
				if err := m.cl.Call(&receipt, "eth_getTransactionReceipt", tx.Hash.String()); err != nil {
					return err
				}

				var _remainingSize string
				// TODO: Fix string format
				log.Println(receipt.ContractAddr.String())
				if err := m.cl.Call(&_remainingSize, "eth_getUpload", receipt.ContractAddr.String(), "latest"); err != nil {
					return err
				}

				remainingSize, _ := strconv.ParseUint(_remainingSize[2:], 16, 64)
				info.LeftSize = remainingSize
				info.ContractAddr = receipt.ContractAddr
				var bytesRequested uint64
				if meta.RawSize > remainingSize {
					bytesRequested = meta.RawSize - remainingSize
				}
				m.dl.UpdateTorrent <- types.FlowControlMeta{
					URI:            meta.URI,
					BytesRequested: bytesRequested,
				}
			} else if tx.IsFlowControl() {
				/*

						addr := tx.Recipient.String()
						if file, ok := m.files[addr]; ok {
							var _remainingSize string
							if err := m.cl.Call(&_remainingSize, "eth_getUpload", addr, "latest"); err != nil {
								return err
							}
							remainingSize, _ := strconv.ParseUint(_remainingSize[2:], 16, 64)

							var bytesRequested uint64
							if file.RawSize > remainingSize {
								bytesRequested = file.RawSize - remainingSize
							}
							m.dl.UpdateTorrent <- types.FlowControlMeta{
								URI:            file.URI,
								BytesRequested: bytesRequested,
							}
						}
					}
				*/
			}
		}
	}
	return nil
}

//
func (m *Monitor) parseBlock(b *types.Block) error {
	blockNumber := b.Number
	m.blocks[blockNumber] = b
	if len(b.Txs) > 0 {
		for _, tx := range b.Txs {
			if meta := tx.Parse(); meta != nil {
				m.dl.NewTorrent <- meta.URI

				info := types.NewFileInfo(meta)
				info.TxHash = tx.Hash

				var receipt types.Receipt
				if err := m.cl.Call(&receipt, "eth_getTransactionReceipt", tx.Hash.String()); err != nil {
					return err
				} else if receipt.ContractAddr == nil || receipt.TxHash == nil {
					continue
				}

				var _remainingSize string
				if err := m.cl.Call(&_remainingSize, "eth_getUpload", receipt.ContractAddr.String(), "latest"); err != nil {
					return err
				}

				remainingSize, _ := strconv.ParseUint(_remainingSize[2:], 16, 64)
				info.LeftSize = remainingSize
				info.ContractAddr = receipt.ContractAddr
				var bytesRequested uint64
				if meta.RawSize > remainingSize {
					bytesRequested = meta.RawSize - remainingSize
				}
				m.dl.UpdateTorrent <- types.FlowControlMeta{
					URI:            meta.URI,
					BytesRequested: bytesRequested,
				}
			}
		}
	}
	return nil
}

func (m *Monitor) initialCheck() {
	blockChecked := 0
	lastblock := m.latestBlockNumber
	log.Println("lastblock: ", lastblock)
	for i := m.latestBlockNumber; i >= minBlockNum; i-- {
		if m.blocks[i] != nil {
			continue
		}
		m.parseBlockByNumber(i)
		blockChecked++
		if blockChecked%fetchBlockLogStep == 0 || i == 0 {
			log.Printf("Block #%d-%d have been checked.", i, lastblock)
			lastblock = i - 1
		}
	}
}

// Start ... start ListenOn on the rpc port of a blockchain full node
func (m *Monitor) Start() error {
	b := &types.Block{}
	if err := m.cl.Call(b, "eth_getBlockByNumber", "latest", true); err != nil {
		log.Println(err)
		return err
	}
	m.latestBlockNumber = b.Number
	m.parseNewBlock(b)
	go m.initialCheck()

	timer := time.NewTimer(time.Second * defaultTimerInterval)
	for {
		select {
		case <-timer.C:
			// Try to get the latest b
			b := &types.Block{}
			if err := m.cl.Call(b, "eth_getBlockByNumber", "latest", true); err != nil {
				return err
			}
			bnum := b.Number
			if bnum > m.latestBlockNumber {
				m.latestBlockNumber = bnum
				m.parseBlock(b)
				log.Printf("Block #%d: %d Txs.", bnum, len(b.Txs))
				for i := m.latestBlockNumber - 1; i >= minBlockNum; i-- {
					if m.blocks[i] != nil {
						break
					}
					m.parseNewBlockByNumber(i)
				}
			}
			timer.Reset(time.Second * 2)
		}
	}
}
