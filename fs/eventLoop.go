package fs

import (
	"bytes"
	"log"
	"math"
	"strconv"
	"time"
	torrent "torrent/libtorrent"

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

/*
func torrentList(chan string magnetlink) {

}
*/

func eventLoop() {
	clientURI := "http://192.168.5.11:28888"
	client, _ := rpc.Dial(clientURI)

	timer := time.NewTimer(time.Second * 2)
	blocks := make(map[int]int)
	var block Block
	var blockTxCount int
	torrentList := make([]string, 0, 1024)

	for {
		select {
		case <-timer.C:

			if err := client.Call(&block, "eth_getBlockByNumber", "latest", true); err != nil {
				log.Println("can't get latest block:", err)
				return
			}
			blockTxCount = len(block.Transactions)
			blockNum, _ := strconv.ParseInt(block.Number[2:], 16, 32)

			for {
				_, ok := blocks[int(blockNum)]
				if ok {
					break
				}
				blocks[int(blockNum)] = blockTxCount

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
							torrentList = append(torrentList, meta.URI)
						} else if tx["input"][:6] == "0x0002" {
							// input
							var meta types.InputMeta
							input, _ := hex.DecodeString(tx["input"][6:])
							rlp.Decode(bytes.NewReader(input), &meta)
							log.Println(tx["hash"], meta.URI)
							torrentList = append(torrentList, meta.URI)
						}
					}
				}

				if blockNum == 0 {
					flags := &torrent.TorrentFlags{
						Dial:                nil,
						Port:                7777,
						FileDir:             "/home/lizhen/storage",
						SeedRatio:           math.Inf(0),
						UseDeadlockDetector: true,
						UseLPD:              true,
						UseDHT:              true,
						UseUPnP:             false,
						UseNATPMP:           false,
						TrackerlessMode:     false,
						// IP address of gateway
						Gateway:            "",
						InitialCheck:       true,
						FileSystemProvider: torrent.OsFsProvider{},
						Cacher:             torrent.NewRamCacheProvider(2048),
						ExecOnSeeding:      "",
						QuickResume:        true,
						MaxActive:          128,
						MemoryPerTorrent:   -1,
					}
					torrent.RunTorrents(flags, torrentList)
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

// StartListening ... start listening on the rpc port of a blockchain full node
func StartListening() {
	eventLoop()
}
