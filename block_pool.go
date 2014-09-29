package eth

import (
	"bytes"
	"container/list"
	"math"
	"math/big"
	"sync"
	"time"

	"github.com/ethereum/eth-go/ethchain"
	"github.com/ethereum/eth-go/ethlog"
	"github.com/ethereum/eth-go/ethutil"
	"github.com/ethereum/eth-go/ethwire"
)

var poollogger = ethlog.NewLogger("BPOOL")

type block struct {
	from      *Peer
	peer      *Peer
	block     *ethchain.Block
	reqAt     time.Time
	requested int
}

type BlockPool struct {
	mut sync.Mutex

	eth *Ethereum

	hashPool [][]byte
	pool     map[string]*block

	td   *big.Int
	quit chan bool

	fetchingHashes    bool
	downloadStartedAt time.Time

	ChainLength, BlocksProcessed int
}

func NewBlockPool(eth *Ethereum) *BlockPool {
	return &BlockPool{
		eth:  eth,
		pool: make(map[string]*block),
		td:   ethutil.Big0,
		quit: make(chan bool),
	}
}

func (self *BlockPool) Len() int {
	return len(self.hashPool)
}

func (self *BlockPool) HasLatestHash() bool {
	self.mut.Lock()
	defer self.mut.Unlock()

	return self.pool[string(self.eth.BlockChain().CurrentBlock.Hash())] != nil
}

func (self *BlockPool) HasCommonHash(hash []byte) bool {
	return self.eth.BlockChain().GetBlock(hash) != nil
}

func (self *BlockPool) Blocks() (blocks ethchain.Blocks) {
	for _, item := range self.pool {
		if item.block != nil {
			blocks = append(blocks, item.block)
		}
	}

	return
}

func (self *BlockPool) AddHash(hash []byte, peer *Peer) {
	self.mut.Lock()
	defer self.mut.Unlock()

	if self.pool[string(hash)] == nil {
		self.pool[string(hash)] = &block{peer, nil, nil, time.Now(), 0}

		self.hashPool = append([][]byte{hash}, self.hashPool...)
	}
}

func (self *BlockPool) Add(b *ethchain.Block, peer *Peer) {
	self.mut.Lock()
	defer self.mut.Unlock()

	hash := string(b.Hash())

	if self.pool[hash] == nil && !self.eth.BlockChain().HasBlock(b.Hash()) {
		poollogger.Infof("Got unrequested block (%x...)\n", hash[0:4])

		self.hashPool = append(self.hashPool, b.Hash())
		self.pool[hash] = &block{peer, peer, b, time.Now(), 0}

		if !self.eth.BlockChain().HasBlock(b.PrevHash) && !self.fetchingHashes {
			poollogger.Infof("Unknown block, requesting parent (%x...)\n", b.PrevHash[0:4])
			peer.QueueMessage(ethwire.NewMessage(ethwire.MsgGetBlockHashesTy, []interface{}{b.PrevHash, uint32(256)}))
		}
	} else if self.pool[hash] != nil {
		self.pool[hash].block = b
	}

	self.BlocksProcessed++
}

func (self *BlockPool) Remove(hash []byte) {
	self.mut.Lock()
	defer self.mut.Unlock()

	self.hashPool = ethutil.DeleteFromByteSlice(self.hashPool, hash)
	delete(self.pool, string(hash))
}

func (self *BlockPool) ProcessCanonical(f func(block *ethchain.Block)) (procAmount int) {
	blocks := self.Blocks()

	ethchain.BlockBy(ethchain.Number).Sort(blocks)
	for _, block := range blocks {
		if self.eth.BlockChain().HasBlock(block.PrevHash) {
			procAmount++

			f(block)

			self.Remove(block.Hash())
		}

	}

	return
}

func (self *BlockPool) DistributeHashes() {
	self.mut.Lock()
	defer self.mut.Unlock()

	var (
		peerLen = self.eth.peers.Len()
		amount  = 256 * peerLen
		dist    = make(map[*Peer][][]byte)
	)

	num := int(math.Min(float64(amount), float64(len(self.pool))))
	for i, j := 0, 0; i < len(self.hashPool) && j < num; i++ {
		hash := self.hashPool[i]
		item := self.pool[string(hash)]

		if item != nil && item.block == nil {
			var peer *Peer
			lastFetchFailed := time.Since(item.reqAt) > 5*time.Second

			// Handle failed requests
			if lastFetchFailed && item.requested > 5 && item.peer != nil {
				if item.requested < 100 {
					// Select peer the hash was retrieved off
					peer = item.from
				} else {
					// Remove it
					self.hashPool = ethutil.DeleteFromByteSlice(self.hashPool, hash)
					delete(self.pool, string(hash))
				}
			} else if lastFetchFailed || item.peer == nil {
				// Find a suitable, available peer
				eachPeer(self.eth.peers, func(p *Peer, v *list.Element) {
					if peer == nil && len(dist[p]) < amount/peerLen {
						peer = p
					}
				})
			}

			if peer != nil {
				item.reqAt = time.Now()
				item.peer = peer
				item.requested++

				dist[peer] = append(dist[peer], hash)
			}
		}
	}

	for peer, hashes := range dist {
		peer.FetchBlocks(hashes)
	}

	if len(dist) > 0 {
		self.downloadStartedAt = time.Now()
	}
}

func (self *BlockPool) Start() {
	go self.downloadThread()
	go self.chainThread()
}

func (self *BlockPool) Stop() {
	close(self.quit)
}

func (self *BlockPool) downloadThread() {
	serviceTimer := time.NewTicker(100 * time.Millisecond)
out:
	for {
		select {
		case <-self.quit:
			break out
		case <-serviceTimer.C:
			// Check if we're catching up. If not distribute the hashes to
			// the peers and download the blockchain
			self.fetchingHashes = false
			eachPeer(self.eth.peers, func(p *Peer, v *list.Element) {
				if p.statusKnown && p.FetchingHashes() {
					self.fetchingHashes = true
				}
			})

			if !self.fetchingHashes && len(self.hashPool) > 0 {
				self.DistributeHashes()
			}

			if self.ChainLength < len(self.hashPool) {
				self.ChainLength = len(self.hashPool)
			}
		}
	}
}

func (self *BlockPool) chainThread() {
	procTimer := time.NewTicker(500 * time.Millisecond)
out:
	for {
		select {
		case <-self.quit:
			break out
		case <-procTimer.C:
			blocks := self.Blocks()
			ethchain.BlockBy(ethchain.Number).Sort(blocks)

			if len(blocks) > 0 {
				if self.eth.BlockChain().HasBlock(blocks[0].PrevHash) {
					for i, block := range blocks[1:] {
						// NOTE: The Ith element in this loop refers to the previous block in
						// outer "blocks"
						if bytes.Compare(block.PrevHash, blocks[i].Hash()) != 0 {
							blocks = blocks[:i]

							break
						}
					}
				} else {
					blocks = nil
				}
			}

			// Handle in batches of 4k
			max := int(math.Min(4000, float64(len(blocks))))
			for _, block := range blocks[:max] {
				self.eth.Eventer().Post("block", block)

				self.Remove(block.Hash())
			}
		}
	}
}
