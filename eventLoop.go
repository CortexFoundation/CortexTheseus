package main

import (
	"bytes"
	"log"
	"strconv"
	"time"

	"encoding/hex"

	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/rlp"
	"github.com/CortexFoundation/CortexTheseus/rpc"
)

// Block ... block struct
type Block struct {
	Number       string
	Hash         string
	ParentHash   string
	Transactions []map[string]string
}

// ListenOn ... start ListenOn on the rpc port of a blockchain full node
func ListenOn(clientURI string, torrentFiles chan string) {
	client, _ := rpc.Dial(clientURI)

	timer := time.NewTimer(time.Second * 2)
	blocks := make(map[int]int)
	var block Block
	var blockTxCount int

	for {
		select {
		case <-timer.C:

			if err := client.Call(&block, "eth_getBlockByNumber", "latest", true); err != nil {
				log.Println("can't get latest block:", err)
				return
			}
			blockTxCount = len(block.Transactions)
			blockNum, _ := strconv.ParseInt(block.Number[2:], 16, 32)
			blockChecked := 0

			for {
				_, ok := blocks[int(blockNum)]
				if ok {
					break
				}
				blocks[int(blockNum)] = blockTxCount
				blockChecked++

				if blockChecked%100 == 0 {
					log.Printf("%d blocks have been checked", blockChecked)
				}

				if blockTxCount > 0 {
					for _, tx := range block.Transactions {
						if len(tx["input"]) < 6 {
							continue
						}
						if tx["input"][:6] == "0x0001" {
							// model
							var meta types.ModelMeta
							input, _ := hex.DecodeString(tx["input"][6:])
							rlp.Decode(bytes.NewReader(input), &meta)
							log.Println(tx["hash"], meta.URI)
							torrentFiles <- meta.URI
						} else if tx["input"][:6] == "0x0002" {
							// input
							var meta types.InputMeta
							input, _ := hex.DecodeString(tx["input"][6:])
							rlp.Decode(bytes.NewReader(input), &meta)
							log.Println(tx["hash"], meta.URI)
							torrentFiles <- meta.URI
						}
					}
				}

				if blockNum == 0 {
					break
				} else {
					blockNum--
					blockNumHex := "0x" + strconv.FormatInt(blockNum, 16)
					var _blockTxCount string
					if err := client.Call(&_blockTxCount, "eth_getBlockTransactionCountByNumber", blockNumHex); err != nil {
						log.Printf("can't get block %s: %s", blockNumHex, err)
						return
					}
					blockTxCountInt32, _ := strconv.ParseInt(_blockTxCount[2:], 16, 32)
					blockTxCount = int(blockTxCountInt32)
					if blockTxCount > 0 {
						if err := client.Call(&block, "eth_getBlockByNumber", blockNumHex, true); err != nil {
							log.Printf("can't get block %s: %s", blockNumHex, err)
							return
						}
						log.Printf("fetch block #%s with %d transactions", blockNumHex, blockTxCount)
					}
				}
			}
			timer.Reset(time.Second * 2)
		}
	}
}
