package blockpool

import (
	"fmt"
	"math/big"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/errs"
	ethlogger "github.com/ethereum/go-ethereum/logger"
	"github.com/ethereum/go-ethereum/pow"
)

var plog = ethlogger.NewLogger("Blockpool")

var (
	// max number of block hashes sent in one request
	blockHashesBatchSize = 256
	// max number of blocks sent in one request
	blockBatchSize = 64
	// interval between two consecutive block checks (and requests)
	blocksRequestInterval = 3 * time.Second
	// level of redundancy in block requests sent
	blocksRequestRepetition = 1
	// interval between two consecutive block hash checks (and requests)
	blockHashesRequestInterval = 3 * time.Second
	// max number of idle iterations, ie., check through a section without new blocks coming in
	blocksRequestMaxIdleRounds = 20
	// timeout interval: max time allowed for peer without sending a block hash
	blockHashesTimeout = 60 * time.Second
	// timeout interval: max time allowed for peer without sending a block
	blocksTimeout = 60 * time.Second
	//
	idleBestPeerTimeout    = 120 * time.Second
	peerSuspensionInterval = 300 * time.Second
)

// config embedded in components, by default fall back to constants
// by default all resolved to local
type Config struct {
	BlockHashesBatchSize       int
	BlockBatchSize             int
	BlocksRequestRepetition    int
	BlocksRequestMaxIdleRounds int
	BlockHashesRequestInterval time.Duration
	BlocksRequestInterval      time.Duration
	BlockHashesTimeout         time.Duration
	BlocksTimeout              time.Duration
	IdleBestPeerTimeout        time.Duration
	PeerSuspensionInterval     time.Duration
}

// blockpool errors
const (
	ErrInvalidBlock = iota
	ErrInvalidPoW
	ErrUnrequestedBlock
	ErrInsufficientChainInfo
	ErrIdleTooLong
)

var errorToString = map[int]string{
	ErrInvalidBlock:          "Invalid block",
	ErrInvalidPoW:            "Invalid PoW",
	ErrUnrequestedBlock:      "Unrequested block",
	ErrInsufficientChainInfo: "Insufficient chain info",
	ErrIdleTooLong:           "Idle too long",
}

// init initialises all your laundry
func (self *Config) init() {
	if self.BlockHashesBatchSize == 0 {
		self.BlockHashesBatchSize = blockHashesBatchSize
	}
	if self.BlockBatchSize == 0 {
		self.BlockBatchSize = blockBatchSize
	}
	if self.BlocksRequestRepetition == 0 {
		self.BlocksRequestRepetition = blocksRequestRepetition
	}
	if self.BlocksRequestMaxIdleRounds == 0 {
		self.BlocksRequestMaxIdleRounds = blocksRequestMaxIdleRounds
	}
	if self.BlockHashesRequestInterval == 0 {
		self.BlockHashesRequestInterval = blockHashesRequestInterval
	}
	if self.BlocksRequestInterval == 0 {
		self.BlocksRequestInterval = blocksRequestInterval
	}
	if self.BlockHashesTimeout == 0 {
		self.BlockHashesTimeout = blockHashesTimeout
	}
	if self.BlocksTimeout == 0 {
		self.BlocksTimeout = blocksTimeout
	}
	if self.IdleBestPeerTimeout == 0 {
		self.IdleBestPeerTimeout = idleBestPeerTimeout
	}
	if self.PeerSuspensionInterval == 0 {
		self.PeerSuspensionInterval = peerSuspensionInterval
	}
}

// node is the basic unit of the internal model of block chain/tree in the blockpool
type node struct {
	lock    sync.RWMutex
	hash    common.Hash
	block   *types.Block
	hashBy  string
	blockBy string
	td      *big.Int
}

type index struct {
	int
}

// entry is the struct kept and indexed in the pool
type entry struct {
	node    *node
	section *section
	index   *index
}

type BlockPool struct {
	Config *Config

	// the minimal interface with blockchain
	hasBlock    func(hash common.Hash) bool
	insertChain func(types.Blocks) error
	verifyPoW   func(pow.Block) bool

	pool  map[string]*entry
	peers *peers

	lock      sync.RWMutex
	chainLock sync.RWMutex
	// alloc-easy pool of hash slices
	hashSlicePool chan []common.Hash

	status *status

	quit    chan bool
	wg      sync.WaitGroup
	running bool
}

// public constructor
func New(
	hasBlock func(hash common.Hash) bool,
	insertChain func(types.Blocks) error,
	verifyPoW func(pow.Block) bool,
) *BlockPool {

	return &BlockPool{
		Config:      &Config{},
		hasBlock:    hasBlock,
		insertChain: insertChain,
		verifyPoW:   verifyPoW,
	}
}

func severity(code int) ethlogger.LogLevel {
	switch code {
	case ErrUnrequestedBlock:
		return ethlogger.WarnLevel
	default:
		return ethlogger.ErrorLevel
	}
}

// allows restart
func (self *BlockPool) Start() {
	self.lock.Lock()
	defer self.lock.Unlock()

	if self.running {
		return
	}

	self.Config.init()
	self.hashSlicePool = make(chan []common.Hash, 150)
	self.status = newStatus()
	self.quit = make(chan bool)
	self.pool = make(map[string]*entry)
	self.running = true

	self.peers = &peers{
		errors: &errs.Errors{
			Package: "Blockpool",
			Errors:  errorToString,
			Level:   severity,
		},
		peers:     make(map[string]*peer),
		blacklist: make(map[string]time.Time),
		status:    self.status,
		bp:        self,
	}
	timer := time.NewTicker(3 * time.Second)
	go func() {
		for {
			select {
			case <-self.quit:
				return
			case <-timer.C:
				plog.DebugDetailf("status:\n%v", self.Status())
			}
		}
	}()
	plog.Infoln("Started")
}

func (self *BlockPool) Stop() {
	self.lock.Lock()
	if !self.running {
		self.lock.Unlock()
		return
	}
	self.running = false

	self.lock.Unlock()

	plog.Infoln("Stopping...")

	close(self.quit)

	self.lock.Lock()
	self.peers = nil
	self.pool = nil
	self.lock.Unlock()

	plog.Infoln("Stopped")
}

// Wait blocks until active processes finish
func (self *BlockPool) Wait(t time.Duration) {
	self.lock.Lock()
	if !self.running {
		self.lock.Unlock()
		return
	}
	self.lock.Unlock()

	plog.Infoln("Waiting for processes to complete...")
	w := make(chan bool)
	go func() {
		self.wg.Wait()
		close(w)
	}()

	select {
	case <-w:
		plog.Infoln("Processes complete")
	case <-time.After(t):
		plog.Warnf("Timeout")
	}
}

/*
AddPeer is called by the eth protocol instance running on the peer after
the status message has been received with total difficulty and current block hash
Called a second time with the same peer id, it is used to update chain info for a peer. This is used when a new (mined) block message is received.
RemovePeer needs to be called when the peer disconnects.
Peer info is currently not persisted across disconnects (or sessions)
*/
func (self *BlockPool) AddPeer(

	td *big.Int, currentBlockHash common.Hash,
	peerId string,
	requestBlockHashes func(common.Hash) error,
	requestBlocks func([]common.Hash) error,
	peerError func(*errs.Error),

) (best bool, suspended bool) {

	return self.peers.addPeer(td, currentBlockHash, peerId, requestBlockHashes, requestBlocks, peerError)
}

// RemovePeer needs to be called when the peer disconnects
func (self *BlockPool) RemovePeer(peerId string) {
	self.peers.removePeer(peerId)
}

/*
AddBlockHashes

Entry point for eth protocol to add block hashes received via BlockHashesMsg

only hashes from the best peer are handled

initiates further hash requests until a known parent is reached (unless cancelled by a peerSwitch event, i.e., when a better peer becomes best peer)
launches all block request processes on each chain section

the first argument is an iterator function. Using this block hashes are decoded from the rlp message payload on demand. As a result, AddBlockHashes needs to run synchronously for one peer since the message is discarded if the caller thread returns.
*/
func (self *BlockPool) AddBlockHashes(next func() (common.Hash, bool), peerId string) {

	bestpeer, best := self.peers.getPeer(peerId)
	if !best {
		return
	}
	// bestpeer is still the best peer

	self.wg.Add(1)

	defer func() { self.wg.Done() }()

	self.status.lock.Lock()
	self.status.activePeers[bestpeer.id]++
	self.status.lock.Unlock()

	var n int
	var hash common.Hash
	var ok, headSection, peerswitch bool
	var sec, child, parent *section
	var entry *entry
	var nodes []*node

	hash, ok = next()
	bestpeer.lock.Lock()

	plog.Debugf("AddBlockHashes: peer <%s> starting from [%s] (peer head: %s)", peerId, hex(bestpeer.parentHash), hex(bestpeer.currentBlockHash))

	// first check if we are building the head section of a peer's chain
	if bestpeer.parentHash == hash {
		if self.hasBlock(bestpeer.currentBlockHash) {
			return
		}
		/*
		 when peer is promoted in switchPeer, a new header section process is launched
		 as the head section skeleton is actually created here, it is signaled to the process
		 so that it can quit
		 in the special case that the node for parent of the head block is found in the blockpool
		 (with or without fetched block)
		*/
		headSection = true
		if entry := self.get(bestpeer.currentBlockHash); entry == nil {
			plog.DebugDetailf("AddBlockHashes: peer <%s> (head: %s) head section starting from [%s] ", peerId, hex(bestpeer.currentBlockHash), hex(bestpeer.parentHash))
			// if head block is not yet in the pool, create entry and start node list for section
			node := &node{
				hash:    bestpeer.currentBlockHash,
				block:   bestpeer.currentBlock,
				hashBy:  peerId,
				blockBy: peerId,
			}
			// nodes is a list of nodes in one section ordered top-bottom (old to young)
			nodes = append(nodes, node)
			n++
		} else {
			// otherwise set child section iff found node is the root of a section
			// this is a possible scenario when a singleton head section was created
			// on an earlier occasion this peer or another with the same block was best peer
			if entry.node == entry.section.bottom {
				child = entry.section
				plog.DebugDetailf("AddBlockHashes: peer <%s>: connects to child section root %s", peerId, hex(bestpeer.currentBlockHash))
			}
		}
	} else {
		// otherwise : we are not building the head section of the peer
		plog.DebugDetailf("AddBlockHashes: peer <%s> (head: %s) section starting from [%s] ", peerId, hex(bestpeer.currentBlockHash), hex(hash))
	}
	// the switch channel signals peerswitch event
	switchC := bestpeer.switchC
	bestpeer.lock.Unlock()

	// iterate over hashes coming from peer (first round we have hash set above)
LOOP:
	for ; ok; hash, ok = next() {

		select {
		case <-self.quit:
			// global quit for blockpool
			return

		case <-switchC:
			// if the peer is demoted, no more hashes read
			plog.DebugDetailf("AddBlockHashes: demoted peer <%s> (head: %s)", peerId, hex(bestpeer.currentBlockHash), hex(hash))
			peerswitch = true
			break LOOP
		default:
		}

		// if we reach the blockchain we stop reading more
		if self.hasBlock(hash) {
			// check if known block connecting the downloaded chain to our blockchain
			plog.DebugDetailf("AddBlockHashes: peer <%s> (head: %s) found block %s in the blockchain", peerId, hex(bestpeer.currentBlockHash), hex(hash))
			if len(nodes) == 1 {
				plog.DebugDetailf("AddBlockHashes: singleton section pushed to blockchain peer <%s> (head: %s) found block %s in the blockchain", peerId, hex(bestpeer.currentBlockHash), hex(hash))

				// create new section if needed and push it to the blockchain
				sec = self.newSection(nodes)
				sec.addSectionToBlockChain(bestpeer)
			} else {

				/*
					 not added hash yet but according to peer child section built
					earlier chain connects with blockchain
					this maybe a potential vulnarability
					the root block arrives (or already there but its parenthash was not pointing to known block in the blockchain)
					we start inserting -> error -> remove the entire chain
					instead of punishing this peer
					solution: when switching peers always make sure best peers own head block
					and td together with blockBy are recorded on the node
				*/
				if len(nodes) == 0 && child != nil {
					plog.DebugDetailf("AddBlockHashes: child section [%s] pushed to blockchain peer <%s> (head: %s) found block %s in the blockchain", sectionhex(child), peerId, hex(bestpeer.currentBlockHash), hex(hash))

					child.addSectionToBlockChain(bestpeer)
				}
			}
			break LOOP
		}

		// look up node in the pool
		entry = self.get(hash)
		if entry != nil {
			// reached a known chain in the pool
			if entry.node == entry.section.bottom && n == 1 {
				/*
					the first block hash received is an orphan in the pool
					this also supports clients that (despite the spec) include <from> hash in their
					response to hashes request. Note that by providing <from> we can link sections
					without having to wait for the root block of the child section to arrive, so it allows for superior performance
				*/
				plog.DebugDetailf("AddBlockHashes: peer <%s> (head: %s) found head block [%s] as root of connecting child section [%s] skipping", peerId, hex(bestpeer.currentBlockHash), hex(hash), sectionhex(entry.section))
				// record the entry's chain section as child section
				child = entry.section
				continue LOOP
			}
			// otherwise record entry's chain section as parent connecting it to the pool
			plog.DebugDetailf("AddBlockHashes: peer <%s> (head: %s) found block [%s] in section [%s]. Connected to pool.", peerId, hex(bestpeer.currentBlockHash), hex(hash), sectionhex(entry.section))
			parent = entry.section
			break LOOP
		}

		// finally if node for block hash does not exist, create it and append node to section nodes
		node := &node{
			hash:   hash,
			hashBy: peerId,
		}
		nodes = append(nodes, node)
	} //for

	/*
		we got here if
		- run out of hashes (parent = nil) sent by our best peer
		- our peer is demoted (peerswitch = true)
		- reached blockchain or blockpool
		- quitting
	*/
	self.chainLock.Lock()

	plog.DebugDetailf("AddBlockHashes: peer <%s> (head: %s): %v nodes in new section", peerId, hex(bestpeer.currentBlockHash), len(nodes))
	/*
	  handle forks where connecting node is mid-section
	  by splitting section at fork
	  no splitting needed if connecting node is head of a section
	*/
	if parent != nil && entry != nil && entry.node != parent.top && len(nodes) > 0 {
		plog.DebugDetailf("AddBlockHashes: peer <%s> (head: %s): fork after %s", peerId, hex(bestpeer.currentBlockHash), hex(hash))

		self.splitSection(parent, entry)

		self.status.lock.Lock()
		self.status.values.Forks++
		self.status.lock.Unlock()
	}

	/*
	  if new section is created, link it to parent/child sections
	  and launch section process fetching blocks and further hashes
	*/
	sec = self.linkSections(nodes, parent, child)

	if sec != nil {
		self.status.lock.Lock()
		self.status.values.BlockHashes += len(nodes)
		self.status.lock.Unlock()
		plog.DebugDetailf("AddBlockHashes: peer <%s> (head: %s): section [%s] created", peerId, hex(bestpeer.currentBlockHash), sectionhex(sec))
	}

	self.chainLock.Unlock()

	/*
		if a blockpool node is reached (parent section is not nil),
		activate section (unless our peer is demoted by now).
		this can be the bottom half of a newly split section in case of a fork.
		bestPeer is nil if we got here after our peer got demoted while processing.
		in this case no activation should happen
	*/
	if parent != nil && !peerswitch {
		self.activateChain(parent, bestpeer, nil)
		plog.DebugDetailf("AddBlockHashes: peer <%s> (head: %s): parent section [%s]", peerId, hex(bestpeer.currentBlockHash), sectionhex(parent))
	}

	/*
	  if a new section was created,
	  register section iff head section or no child known
	  activate it with this peer
	*/
	if sec != nil {
		// switch on section process (it is paused by switchC)
		if !peerswitch {
			if headSection || child == nil {
				bestpeer.lock.Lock()
				bestpeer.sections = append(bestpeer.sections, sec.top.hash)
				bestpeer.lock.Unlock()
			}
			/*
			  request next block hashes for parent section here.
			  but only once, repeating only when bottom block arrives,
			  otherwise no way to check if it arrived
			*/
			bestpeer.requestBlockHashes(sec.bottom.hash)
			plog.DebugDetailf("AddBlockHashes: peer <%s> (head: %s): start requesting blocks for section [%s]", peerId, hex(bestpeer.currentBlockHash), sectionhex(sec))
			sec.activate(bestpeer)
		} else {
			plog.DebugDetailf("AddBlockHashes: peer <%s> (head: %s) no longer best: delay requesting blocks for section [%s]", peerId, hex(bestpeer.currentBlockHash), sectionhex(sec))
			sec.deactivate()
		}
	}

	// if we are processing peer's head section, signal it to headSection process that it is created

	if headSection {
		plog.DebugDetailf("AddBlockHashes: peer <%s> (head: %s) head section registered on head section process", peerId, hex(bestpeer.currentBlockHash))

		var headSec *section
		switch {
		case sec != nil:
			headSec = sec
		case child != nil:
			headSec = child
		default:
			headSec = parent
		}
		if !peerswitch {
			plog.DebugDetailf("AddBlockHashes: peer <%s> (head: %s) head section [%s] created signalled to head section process", peerId, hex(bestpeer.currentBlockHash), sectionhex(headSec))
			bestpeer.headSectionC <- headSec
		}
	}
}

/*
	AddBlock is the entry point for the eth protocol to call when blockMsg is received.

	It has a strict interpretation of the protocol in that if the block received has not been requested, it results in an error

	At the same time it is opportunistic in that if a requested block may be provided by any peer.

	The received block is checked for PoW. Only the first PoW-valid block for a hash is considered legit.
*/
func (self *BlockPool) AddBlock(block *types.Block, peerId string) {
	hash := block.Hash()

	sender, _ := self.peers.getPeer(peerId)
	if sender == nil {
		return
	}

	self.status.lock.Lock()
	self.status.activePeers[peerId]++
	self.status.lock.Unlock()

	entry := self.get(hash)

	// a peer's current head block is appearing the first time
	if hash == sender.currentBlockHash {
		if sender.currentBlock == nil {
			plog.Debugf("AddBlock: add head block %s for peer <%s> (head: %s)", hex(hash), peerId, hex(sender.currentBlockHash))
			sender.setChainInfoFromBlock(block)
			// sender.currentBlockC <- block

			self.status.lock.Lock()
			self.status.values.BlockHashes++
			self.status.values.Blocks++
			self.status.values.BlocksInPool++
			self.status.lock.Unlock()
		} else {
			plog.DebugDetailf("AddBlock: head block %s for peer <%s> (head: %s) already known", hex(hash), peerId, hex(sender.currentBlockHash))
			sender.currentBlockC <- block
		}
	} else {

		plog.DebugDetailf("AddBlock: block %s received from peer <%s> (head: %s)", hex(hash), peerId, hex(sender.currentBlockHash))

		sender.lock.Lock()
		// update peer chain info if more recent than what we registered
		if block.Td != nil && block.Td.Cmp(sender.td) > 0 {
			sender.td = block.Td
			sender.currentBlockHash = block.Hash()
			sender.parentHash = block.ParentHash()
			sender.currentBlock = block
			sender.headSection = nil
		}
		sender.lock.Unlock()

		if entry == nil {
			// penalise peer for sending what we have not asked
			plog.DebugDetailf("AddBlock: unrequested block %s received from peer <%s> (head: %s)", hex(hash), peerId, hex(sender.currentBlockHash))
			sender.addError(ErrUnrequestedBlock, "%x", hash)

			self.status.lock.Lock()
			self.status.badPeers[peerId]++
			self.status.lock.Unlock()
			return
		}
	}
	if entry == nil {
		return
	}

	node := entry.node
	node.lock.Lock()
	defer node.lock.Unlock()

	// check if block already present
	if node.block != nil {
		plog.DebugDetailf("AddBlock: block %s from peer <%s> (head: %s) already sent by <%s> ", hex(hash), peerId, hex(sender.currentBlockHash), node.blockBy)
		return
	}

	// check if block is already inserted in the blockchain
	if self.hasBlock(hash) {
		plog.DebugDetailf("AddBlock: block %s from peer <%s> (head: %s) already in the blockchain", hex(hash), peerId, hex(sender.currentBlockHash))
		return
	}

	// validate block for PoW
	if !self.verifyPoW(block) {
		plog.Warnf("AddBlock: invalid PoW on block %s from peer  <%s> (head: %s)", hex(hash), peerId, hex(sender.currentBlockHash))
		sender.addError(ErrInvalidPoW, "%x", hash)

		self.status.lock.Lock()
		self.status.badPeers[peerId]++
		self.status.lock.Unlock()

		return
	}

	node.block = block
	node.blockBy = peerId
	node.td = block.Td // optional field

	self.status.lock.Lock()
	self.status.values.Blocks++
	self.status.values.BlocksInPool++
	self.status.lock.Unlock()

}

/*
  iterates down a chain section by section
  activating section process on incomplete sections with peer
  relinking orphaned sections with their parent if root block (and its parent hash) is known)
*/
func (self *BlockPool) activateChain(sec *section, p *peer, connected map[string]*section) {

	p.lock.RLock()
	switchC := p.switchC
	p.lock.RUnlock()

	var i int

LOOP:
	for sec != nil {
		parent := self.getParent(sec)
		plog.DebugDetailf("activateChain: section [%s] activated by peer <%s>", sectionhex(sec), p.id)
		sec.activate(p)
		if i > 0 && connected != nil {
			connected[sec.top.hash.Str()] = sec
		}
		/*
		  we need to relink both complete and incomplete sections
		  the latter could have been blockHashesRequestsComplete before being delinked from its parent
		*/
		if parent == nil {
			if sec.bottom.block != nil {
				if entry := self.get(sec.bottom.block.ParentHash()); entry != nil {
					parent = entry.section
					plog.DebugDetailf("activateChain: [%s]-[%s] link", sectionhex(parent), sectionhex(sec))
					link(parent, sec)
				}
			} else {
				plog.DebugDetailf("activateChain: section [%s] activated by peer <%s> has missing root block", sectionhex(sec), p.id)
			}
		}
		sec = parent

		// stop if peer got demoted
		select {
		case <-switchC:
			break LOOP
		case <-self.quit:
			break LOOP
		default:
		}
	}
}

// must run in separate go routine, otherwise
// switchpeer -> activateChain -> activate deadlocks on section process select and peers.lock
func (self *BlockPool) requestBlocks(attempts int, hashes []common.Hash) {
	self.wg.Add(1)
	go func() {
		self.peers.requestBlocks(attempts, hashes)
		self.wg.Done()
	}()
}

// convenience methods to access adjacent sections
func (self *BlockPool) getParent(sec *section) *section {
	self.chainLock.RLock()
	defer self.chainLock.RUnlock()
	return sec.parent
}

func (self *BlockPool) getChild(sec *section) *section {
	self.chainLock.RLock()
	defer self.chainLock.RUnlock()
	return sec.child
}

// accessor and setter for entries in the pool
func (self *BlockPool) get(hash common.Hash) *entry {
	self.lock.RLock()
	defer self.lock.RUnlock()
	return self.pool[hash.Str()]
}

func (self *BlockPool) set(hash common.Hash, e *entry) {
	self.lock.Lock()
	defer self.lock.Unlock()
	self.pool[hash.Str()] = e
}

func (self *BlockPool) remove(sec *section) {
	// delete node entries from pool index under pool lock
	self.lock.Lock()
	defer self.lock.Unlock()

	for _, node := range sec.nodes {
		delete(self.pool, node.hash.Str())
	}
	if sec.initialised && sec.poolRootIndex != 0 {
		self.status.lock.Lock()
		self.status.values.BlocksInPool -= len(sec.nodes) - sec.missing
		self.status.lock.Unlock()
	}
}

func (self *BlockPool) getHashSlice() (s []common.Hash) {
	select {
	case s = <-self.hashSlicePool:
	default:
		s = make([]common.Hash, self.Config.BlockBatchSize)
	}
	return
}

// Return returns a Client to the pool.
func (self *BlockPool) putHashSlice(s []common.Hash) {
	if len(s) == self.Config.BlockBatchSize {
		select {
		case self.hashSlicePool <- s:
		default:
		}
	}
}

// pretty prints hash (byte array) with first 4 bytes in hex
func hex(hash common.Hash) (name string) {
	if (hash == common.Hash{}) {
		name = ""
	} else {
		name = fmt.Sprintf("%x", hash[:4])
	}
	return
}

// pretty prints a section using first 4 bytes in hex of bottom and top blockhash of the section
func sectionhex(section *section) (name string) {
	if section == nil {
		name = ""
	} else {
		name = fmt.Sprintf("%x-%x", section.bottom.hash[:4], section.top.hash[:4])
	}
	return
}
