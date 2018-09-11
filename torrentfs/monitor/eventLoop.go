package monitor

import (
	"bytes"
	"encoding/hex"
	"errors"
	"log"
	"strconv"
	"time"

	"../common"
	download "../manager"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"github.com/ethereum/go-ethereum/rlp"
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
	opCommon             = "0x0000"
	opCreateModel        = "0x0001"
	opCreateInput        = "0x0002"
	opNoInput            = "0x0003"
)

// Monitor ...
type Monitor struct {
	cl                *rpc.Client
	dl                *download.TorrentManager
	files             map[string]*common.FileMeta
	blocks            map[uint64]*common.Block
	latestBlockNumber uint64
}

// NewMonitor ...
func NewMonitor() *Monitor {
	m := &Monitor{
		nil,
		nil,
		make(map[string]*common.FileMeta),
		make(map[uint64]*common.Block),
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

func (m *Monitor) verifyBlock(b *common.Block) error {
	return nil
}

func (m *Monitor) parseBlockByNumber(blockNumber uint64) error {
	if m.cl == nil {
		return ErrNoRPCClient
	}
	block := &common.Block{}
	m.blocks[blockNumber] = block
	blockNumberHex := "0x" + strconv.FormatUint(blockNumber, 16)
	if err := m.cl.Call(block, "eth_getBlockByNumber", blockNumberHex, true); err != nil {
		return err
	}
	m.parseBlock(block)
	if err := m.verifyBlock(block); err != nil {
		return err
	}
	return nil
}

func (m *Monitor) parseNewBlockByNumber(blockNumber uint64) error {
	if m.cl == nil {
		return ErrNoRPCClient
	}
	block := &common.Block{}
	m.blocks[blockNumber] = block
	blockNumberHex := "0x" + strconv.FormatUint(blockNumber, 16)
	if err := m.cl.Call(block, "eth_getBlockByNumber", blockNumberHex, true); err != nil {
		return err
	}
	m.parseNewBlock(block)
	log.Printf("fetch b #%s with %d Txs", blockNumberHex, len(block.Txs))
	if err := m.verifyBlock(block); err != nil {
		return err
	}
	return nil
}

func (m *Monitor) parseBlockByHash(hash string) error {
	if m.cl == nil {
		return ErrNoRPCClient
	}
	block := &common.Block{}
	if err := m.cl.Call(block, "eth_getBlockByHash", hash, true); err != nil {
		return err
	}
	blockNumber, _ := strconv.ParseUint(block.Number[2:], 16, 64)
	m.blocks[blockNumber] = block
	m.parseBlock(block)
	log.Printf("fetch b #%s with %d Txs", hash, len(block.Txs))
	if err := m.verifyBlock(block); err != nil {
		return err
	}
	return nil
}

func parseInputData(rawInput string) (string, string, uint64, uint64, error) {
	var meta types.InputMeta
	if input, err := hex.DecodeString(rawInput); err != nil {
		return "", "", 0, 0, err
	} else {
		rlp.Decode(bytes.NewReader(input), &meta)
	}
	return meta.AuthorAddress.Hex(), meta.URI, meta.RawSize, uint64(meta.BlockNum.Int64()), nil
}

func parseModelData(rawInput string) (string, string, uint64, uint64, error) {
	var meta types.ModelMeta
	if input, err := hex.DecodeString(rawInput); err != nil {
		return "", "", 0, 0, err
	} else {
		rlp.Decode(bytes.NewReader(input), &meta)
	}
	return meta.AuthorAddress.Hex(), meta.URI, meta.RawSize, uint64(meta.BlockNum.Int64()), nil
}

func (m *Monitor) parseNewBlock(b *common.Block) error {
	blockNumber, _ := strconv.ParseUint(b.Number[2:], 16, 64)
	m.blocks[blockNumber] = b
	if len(b.Txs) > 0 {
		for _, tx := range b.Txs {
			var op string
			var input = tx["input"]
			var value = tx["value"]
			var hash = tx["hash"]
			if len(input) >= 6 {
				op = input[:6]
			} else if len(input) == 0 {
				op = opNoInput
			}

			if op == opCreateInput || op == opCreateModel {
				rawInput := tx["input"][6:]
				var _AuthorAddress string
				var _URI string
				var _RawSize uint64
				var _BlockNum uint64

				if op == opCreateModel {
					_AuthorAddress, _URI, _RawSize, _BlockNum, _ = parseModelData(rawInput)
				} else {
					_AuthorAddress, _URI, _RawSize, _BlockNum, _ = parseInputData(rawInput)
				}
				m.dl.NewTorrent <- _URI

				var receipt common.TransactionReceipt
				if err := m.cl.Call(&receipt, "eth_getTransactionReceipt", hash); err != nil {
					return err
				}

				var _remainingSize string
				if err := m.cl.Call(&_remainingSize, "eth_getUpload", receipt.ContractAddr, "latest"); err != nil {
					return err
				}
				file := &common.FileMeta{
					TxHash:     hash,
					TxAddr:     "0x" + hash[26:],
					AuthorAddr: _AuthorAddress,
					URI:        _URI,
					RawSize:    _RawSize,
					BlockNum:   _BlockNum,
				}
				m.files[receipt.ContractAddr] = file
				remainingSize, _ := strconv.ParseUint(_remainingSize[2:], 16, 64)

				var bytesRequested uint64
				if file.RawSize > remainingSize {
					bytesRequested = file.RawSize - remainingSize
				}
				m.dl.UpdateTorrent <- common.FlowControlMeta{
					URI:            file.URI,
					BytesRequested: bytesRequested,
				}
			} else if input == "" && value == "0x0" {
				ContractAddress := tx["to"]
				if file, ok := m.files[ContractAddress]; ok {
					var _remainingSize string
					if err := m.cl.Call(&_remainingSize, "eth_getUpload", ContractAddress, "latest"); err != nil {
						return err
					}
					remainingSize, _ := strconv.ParseUint(_remainingSize[2:], 16, 64)

					var bytesRequested uint64
					if file.RawSize > remainingSize {
						bytesRequested = file.RawSize - remainingSize
					}
					m.dl.UpdateTorrent <- common.FlowControlMeta{
						URI:            file.URI,
						BytesRequested: bytesRequested,
					}
				}
			}
		}
	}
	return nil
}

//
func (m *Monitor) parseBlock(b *common.Block) error {
	blockNumber, _ := strconv.ParseUint(b.Number[2:], 16, 64)
	m.blocks[blockNumber] = b
	if len(b.Txs) > 0 {
		for _, tx := range b.Txs {
			var op string
			var input = tx["input"]
			var hash = tx["hash"]
			if len(input) >= 6 {
				op = input[:6]
			} else if len(input) == 0 {
				op = opNoInput
			}

			if op == opCreateInput || op == opCreateModel {
				rawInput := tx["input"][6:]
				var _AuthorAddress string
				var _URI string
				var _RawSize uint64
				var _BlockNum uint64

				if op == opCreateModel {
					_AuthorAddress, _URI, _RawSize, _BlockNum, _ = parseModelData(rawInput)
				} else {
					_AuthorAddress, _URI, _RawSize, _BlockNum, _ = parseInputData(rawInput)
				}
				m.dl.NewTorrent <- _URI

				var receipt common.TransactionReceipt
				if err := m.cl.Call(&receipt, "eth_getTransactionReceipt", hash); err != nil {
					return err
				}

				var _remainingSize string
				if err := m.cl.Call(&_remainingSize, "eth_getUpload", receipt.ContractAddr, "latest"); err != nil {
					return err
				}
				file := &common.FileMeta{
					TxHash:     hash,
					TxAddr:     "0x" + hash[26:],
					AuthorAddr: _AuthorAddress,
					URI:        _URI,
					RawSize:    _RawSize,
					BlockNum:   _BlockNum,
				}
				m.files[receipt.ContractAddr] = file
				remainingSize, _ := strconv.ParseUint(_remainingSize[2:], 16, 64)

				var bytesRequested uint64
				if file.RawSize > remainingSize {
					bytesRequested = file.RawSize - remainingSize
				}
				m.dl.UpdateTorrent <- common.FlowControlMeta{
					URI:            file.URI,
					BytesRequested: bytesRequested,
				}
			}
		}
	}
	return nil
}

// Start ... start ListenOn on the rpc port of a blockchain full node
func (m *Monitor) Start() error {
	b := &common.Block{}
	if err := m.cl.Call(b, "eth_getBlockByNumber", "latest", true); err != nil {
		return err
	}
	m.latestBlockNumber, _ = strconv.ParseUint(b.Number[2:], 16, 64)
	m.parseNewBlock(b)
	go func() {
		blockChecked := 0
		lastblock := m.latestBlockNumber
		for i := m.latestBlockNumber - 1; i >= 0; i-- {
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
	}()

	timer := time.NewTimer(time.Second * defaultTimerInterval)
	for {
		select {
		case <-timer.C:
			// Try to get the latest b
			b := &common.Block{}
			if err := m.cl.Call(b, "eth_getBlockByNumber", "latest", true); err != nil {
				return err
			}
			bnum, _ := strconv.ParseUint(b.Number[2:], 16, 64)
			if bnum > m.latestBlockNumber {
				m.latestBlockNumber = bnum
				m.parseBlock(b)
				log.Printf("Block #%d: %d Txs.", bnum, len(b.Txs))
				for i := m.latestBlockNumber - 1; i >= 0; i-- {
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
