package monitor

import (
	"bytes"
	"errors"
	"log"
	"strconv"
	"time"

	"encoding/hex"

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

// Block ... block struct
type Block struct {
	Number       string
	Hash         string
	ParentHash   string
	Transactions []map[string]string
}

// FileMeta ...
type FileMeta struct {
	// Transaction hash
	TxHash string

	// transaction address
	TxAddress     string
	AuthorAddress string
	URI           string
	RawSize       uint64
	BlockNum      uint32
}

// ListenOn ... start ListenOn on the rpc port of a blockchain full node
func ListenOn(clientURI string, manager *download.TorrentManager) error {
	client, err := rpc.Dial(clientURI)
	if err != nil {
		return ErrBuildConn
	}
	log.Println("Internal-RPC connection established.")

	timer := time.NewTimer(time.Second * defaultTimerInterval)
	blockCounts := make(map[int]int)
	var block Block
	var blockTxCount int
	maxBlock := 0

	for {
		select {
		case <-timer.C:
			if err := client.Call(&block, "eth_getBlockByNumber", "latest", true); err != nil {
				return err
			}

			blockTxCount = len(block.Transactions)
			blockNum, _ := strconv.ParseInt(block.Number[2:], 16, 32)
			blockChecked := 0
			recoverMode := false

			for {
				_, ok := blockCounts[int(blockNum)]
				if ok {
					break
				}
				blockCounts[int(blockNum)] = blockTxCount
				blockChecked++
				if int(blockNum) > maxBlock {
					maxBlock = int(blockNum)
					recoverMode = false
					log.Printf("Block #%d: %d Transactions.", maxBlock, blockTxCount)
				} else {
					recoverMode = true
					if blockChecked%200 == 0 {
						log.Printf("Block #%d-%d was synced.", maxBlock-blockChecked+1, maxBlock-blockChecked+200)
					}
				}

				if blockTxCount > 0 {
					for _, tx := range block.Transactions {
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

							if op == "0x0001" {
								// create model
								var meta types.ModelMeta
								if input, err := hex.DecodeString(rawInput); err != nil {
									continue
								} else {
									rlp.Decode(bytes.NewReader(input), &meta)
								}
								manager.NewTorrent <- meta.URI
							} else {
								// create input
								var meta types.InputMeta
								if input, err := hex.DecodeString(rawInput); err != nil {
									continue
								} else {
									rlp.Decode(bytes.NewReader(input), &meta)
								}
								manager.NewTorrent <- meta.URI
							}

						} else if !recoverMode && input == "" && value == "0x0" {
							msg := struct {
								mURI     string
								progress int
							}{
								mURI:     "",
								progress: 0,
							}
							manager.UpdateTorrent <- msg
						}
					}
				}

				if blockNum <= minBlockNum {
					break
				} else {
					blockNum--
					blockNumHex := "0x" + strconv.FormatInt(blockNum, 16)
					var _blockTxCount string
					if err := client.Call(&_blockTxCount, "eth_getBlockTransactionCountByNumber", blockNumHex); err != nil {
						return err
					}
					blockTxCountInt32, _ := strconv.ParseInt(_blockTxCount[2:], 16, 32)
					blockTxCount = int(blockTxCountInt32)
					if blockTxCount > 0 {
						if err := client.Call(&block, "eth_getBlockByNumber", blockNumHex, true); err != nil {
							return err
						}
						log.Printf("fetch block #%s with %d transactions", blockNumHex, blockTxCount)
					}
				}
			}
			timer.Reset(time.Second * 2)
		}
	}
}
