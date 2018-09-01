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
	"github.com/CortexFoundation/CortexTheseus/rlp"
	"github.com/CortexFoundation/CortexTheseus/rpc"
)

//------------------------------------------------------------------------------

// Errors that are used throughout the Torrent API.
var (
	ErrBuildConn      = errors.New("build internal-rpc connection failed")
	ErrGetLatestBlock = errors.New("get latest block failed")
)

const (
	defaultTimerInterval = 2
	minBlockNum          = 36000
)

// ListenOn ... start ListenOn on the rpc port of a blockchain full node
func ListenOn(clientURI string, manager *download.TorrentManager) error {
	client, err := rpc.Dial(clientURI)
	if err != nil {
		return ErrBuildConn
	}
	log.Println("Internal-RPC connection established.")

	timer := time.NewTimer(time.Second * defaultTimerInterval)
	blockCounts := make(map[int]int)
	var b common.Block
	var blockTxCount int
	maxBlock := 0
	files := make(map[string]*common.FileMeta)

	for {
		select {
		case <-timer.C:
			// Try to get the latest b
			if err := client.Call(&b, "eth_getBlockByNumber", "latest", true); err != nil {
				return err
			}

			blockTxCount = len(b.Txs)
			bnum64, _ := strconv.ParseInt(b.Number[2:], 16, 64)
			bnum := int(bnum64)
			blockChecked := 0
			recoverMode := false

			for {
				_, ok := blockCounts[bnum]
				if ok {
					break
				}
				blockCounts[bnum] = len(b.Txs)
				blockChecked++
				if bnum > maxBlock {
					maxBlock = bnum
					recoverMode = false
					log.Printf("Block #%d: %d Txs.", maxBlock, len(b.Txs))
				} else {
					recoverMode = true
					if blockChecked%200 == 0 {
						log.Printf("Block #%d-%d was synced.", maxBlock-blockChecked+1, maxBlock-blockChecked+200)
					}
				}

				if len(b.Txs) > 0 {
					// blockNumHex := "0x" + strconv.FormatInt(int64(bnum), 16)
					/*
						var blockRaw ctypes.Block
						if err := client.Call(&blockRaw, "eth_getBlockByNumber", blockNumHex, false); err != nil {
							return err
						}
						var txraw types.Transaction
						if err := client.Call(&txraw, "eth_getTransactionByBlockNumberAndIndex", blockNumHex, "0x0"); err != nil {
							return err
						}
						log.Println(txraw)
					*/

					for _, tx := range b.Txs {
						// tx is an input data or a model data.
						var input = tx["input"]
						var op = "0x0000"
						var value = tx["value"]
						if len(input) >= 6 {
							op = input[:6]
						} else if len(input) == 0 {
							op = "0x0003"
						}

						if op == "0x0001" || op == "0x0002" {
							rawInput := tx["input"][6:]
							var _AuthorAddress string
							var _URI string
							var _RawSize uint64
							var _BlockNum int64

							if op == "0x0001" {
								// create model
								var meta types.ModelMeta
								if input, err := hex.DecodeString(rawInput); err != nil {
									continue
								} else {
									rlp.Decode(bytes.NewReader(input), &meta)
								}
								manager.NewTorrent <- meta.URI
								_AuthorAddress = meta.AuthorAddress.Hex()
								_URI = meta.URI
								_RawSize = meta.RawSize
								_BlockNum = meta.BlockNum.Int64()
							} else {
								// create input
								var meta types.InputMeta
								if input, err := hex.DecodeString(rawInput); err != nil {
									continue
								} else {
									rlp.Decode(bytes.NewReader(input), &meta)
								}
								manager.NewTorrent <- meta.URI
								_AuthorAddress = meta.AuthorAddress.Hex()
								_URI = meta.URI
								_RawSize = meta.RawSize
								_BlockNum = meta.BlockNum.Int64()
							}

							var receipt common.TransactionReceipt
							if err := client.Call(&receipt, "eth_getTransactionReceipt", tx["hash"]); err != nil {
								return err
							}

							var _remainingSize string
							if err := client.Call(&_remainingSize, "eth_getUpload", receipt.ContractAddress, "latest"); err != nil {
								return err
							}
							file := &common.FileMeta{
								TxHash:        tx["hash"],
								TxAddress:     "0x" + tx["hash"][26:],
								AuthorAddress: _AuthorAddress,
								URI:           _URI,
								RawSize:       _RawSize,
								BlockNum:      uint64(_BlockNum),
							}
							files[receipt.ContractAddress] = file
							remainingSize, _ := strconv.ParseInt(_remainingSize[2:], 16, 64)

							manager.UpdateTorrent <- common.FlowControlMeta{
								URI:            file.URI,
								BytesRequested: int64(file.RawSize) - remainingSize,
							}
						} else if !recoverMode && input == "" && value == "0x0" {
							ContractAddress := tx["to"]
							if file, ok := files[ContractAddress]; ok {
								var _remainingSize string
								if err := client.Call(&_remainingSize, "eth_getUpload", ContractAddress, "latest"); err != nil {
									return err
								}
								remainingSize, _ := strconv.ParseInt(_remainingSize[2:], 16, 64)

								manager.UpdateTorrent <- common.FlowControlMeta{
									URI:            file.URI,
									BytesRequested: int64(file.RawSize) - remainingSize,
								}
							}
						}
					}
				}

				if bnum <= minBlockNum {
					break
				} else {
					bnum--
					blockNumHex := "0x" + strconv.FormatInt(int64(bnum), 16)
					var _blockTxCount string
					if err := client.Call(&_blockTxCount, "eth_getBlockTransactionCountByNumber", blockNumHex); err != nil {
						return err
					}
					blockTxCountInt32, _ := strconv.ParseInt(_blockTxCount[2:], 16, 32)
					blockTxCount = int(blockTxCountInt32)
					if blockTxCount > 0 {
						if err := client.Call(&b, "eth_getBlockByNumber", blockNumHex, true); err != nil {
							return err
						}
						log.Printf("fetch b #%s with %d Txs", blockNumHex, blockTxCount)
					}
				}
			}
			timer.Reset(time.Second * 2)
		}
	}
}
