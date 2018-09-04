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
	ErrWrongOpCode       = errors.New("unexpected opCode")
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
	block := &common.Block{}
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
	block := &common.Block{}
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

func (m *Monitor) parseNewBlock(b *common.Block) error {
	blockNumber := b.Number
	m.blocks[blockNumber] = b
	if len(b.Txs) > 0 {
		for _, tx := range b.Txs {
			op, input, value, hash := extractFromTx(&tx)

			if op == opCreateInput || op == opCreateModel {
				AuthorAddress, URI, RawSize, BlockNum, _ := parseData(input, op)
				m.dl.NewTorrent <- URI

				var receipt common.Receipt
				if err := m.cl.Call(&receipt, "eth_getTransactionReceipt", hash); err != nil {
					return err
				}

				var _remainingSize string
				if err := m.cl.Call(&_remainingSize, "eth_getUpload", receipt.ContractAddr, "latest"); err != nil {
					return err
				}

				file := &common.FileMeta{
					TxHash:       hash,
					ContractAddr: receipt.ContractAddr,
					AuthorAddr:   AuthorAddress,
					URI:          URI,
					RawSize:      RawSize,
					BlockNum:     BlockNum,
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
	blockNumber := b.Number
	m.blocks[blockNumber] = b
	if len(b.Txs) > 0 {
		for _, tx := range b.Txs {
			log.Println("tx", tx)
			op, input, _, hash := extractFromTx(&tx)

			if op == opCreateInput || op == opCreateModel {
				AuthorAddress, URI, RawSize, BlockNum, _ := parseData(input, op)
				m.dl.NewTorrent <- URI

				var receipt common.Receipt
				if err := m.cl.Call(&receipt, "eth_getTransactionReceipt", hash); err != nil {
					return err
				}

				var _remainingSize string
				if err := m.cl.Call(&_remainingSize, "eth_getUpload", receipt.ContractAddr, "latest"); err != nil {
					return err
				}
				file := &common.FileMeta{
					TxHash:       hash,
					ContractAddr: receipt.ContractAddr,
					AuthorAddr:   AuthorAddress,
					URI:          URI,
					RawSize:      RawSize,
					BlockNum:     BlockNum,
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
	b := &common.Block{}
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
			b := &common.Block{}
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

func parseInputData(rawInput string) (string, string, uint64, uint64, error) {
	var meta types.InputMeta
	input, err := hex.DecodeString(rawInput)
	if err != nil {
		return "", "", 0, 0, err
	}
	rlp.Decode(bytes.NewReader(input), &meta)
	return meta.AuthorAddress.Hex(), meta.URI, meta.RawSize, uint64(meta.BlockNum.Int64()), nil
}

func parseModelData(rawInput string) (string, string, uint64, uint64, error) {
	var meta types.ModelMeta
	input, err := hex.DecodeString(rawInput)
	if err != nil {
		return "", "", 0, 0, err
	}
	rlp.Decode(bytes.NewReader(input), &meta)
	return meta.AuthorAddress.Hex(), meta.URI, meta.RawSize, uint64(meta.BlockNum.Int64()), nil
}

func parseData(rawInput string, op string) (string, string, uint64, uint64, error) {
	if op == opCreateInput {
		return parseInputData(rawInput)
	} else if op == opCreateModel {
		return parseModelData(rawInput)
	} else {
		return "", "", 0, 0, ErrWrongOpCode
	}
}

func extractFromTx(tx *common.Transaction) (op, input, value, hash string) {
	op = opCommon
	input = string(tx.Payload)
	value = tx.Amount.String()
	hash = tx.Hash.String()
	if len(input) >= 6 {
		op = input[:6]
		input = input[6:]
	} else if len(input) == 0 {
		op = opNoInput
	} else {
		input = ""
	}
	return
}
