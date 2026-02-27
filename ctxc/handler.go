// Copyright 2019 The go-ethereum Authors
// This file is part of the CortexFoundation library.
//
// The CortexFoundation library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexFoundation library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the CortexFoundation library. If not, see <http://www.gnu.org/licenses/>.

package ctxc

import (
	"cmp"
	crand "crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/big"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/core"
	"github.com/CortexFoundation/CortexTheseus/core/forkid"
	"github.com/CortexFoundation/CortexTheseus/core/txpool"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/ctxc/downloader"
	"github.com/CortexFoundation/CortexTheseus/ctxc/fetcher"
	"github.com/CortexFoundation/CortexTheseus/ctxc/protocols/ctxc"
	"github.com/CortexFoundation/CortexTheseus/ctxcdb"
	"github.com/CortexFoundation/CortexTheseus/event"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/CortexFoundation/CortexTheseus/p2p/enode"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/CortexTheseus/rlp"
	"github.com/CortexFoundation/CortexTheseus/trie"
	"github.com/dchest/siphash"
)

const (
	softResponseLimit = 2 * 1024 * 1024 // Target maximum size of returned blocks, headers or node data.
	estHeaderRlpSize  = 500             // Approximate size of an RLP encoded block header
	// maxHeadersServe is the maximum number of block headers to serve. This number
	// is there to limit the number of disk lookups.
	maxHeadersServe = 1024
	// maxBodiesServe is the maximum number of block bodies to serve. This number
	// is mostly there to limit the number of disk lookups. With 24KB block sizes
	// nowadays, the practical limit will always be softResponseLimit.
	maxBodiesServe = 1024

	// maxReceiptsServe is the maximum number of block receipts to serve. This
	// number is mostly there to limit the number of disk lookups. With block
	// containing 200+ transactions nowadays, the practical limit will always
	// be softResponseLimit.
	maxReceiptsServe = 1024

	// txChanSize is the size of channel listening to NewTxsEvent.
	// The number is referenced from the size of tx pool.
	txChanSize = 4096

	// txMaxBroadcastSize is the max size of a transaction that will be broadcasted.
	// All transactions with a higher size will be announced and need to be fetched
	// by the peer.
	txMaxBroadcastSize = 4096
)

var (
	syncChallengeTimeout = 15 * time.Second
)

func errResp(code errCode, format string, v ...any) error {
	return fmt.Errorf("%v - %v", code, fmt.Sprintf(format, v...))
}

// txPool defines the methods needed from a transaction pool implementation to
// support all the operations needed by the Cortex chain protocols.
type txPool interface {
	// Has returns an indicator whether txpool has a transaction
	// cached with the given hash.
	Has(hash common.Hash) bool

	// Get retrieves the transaction from local txpool with given
	// tx hash.
	Get(hash common.Hash) *types.Transaction

	// AddRemotes should add the given transactions to the pool.
	AddRemotes([]*types.Transaction) []error

	// Pending should return pending transactions.
	// The slice should be modifiable by the caller.
	Pending(filter txpool.PendingFilter) map[common.Address]types.Transactions

	// SubscribeNewTxsEvent should return an event subscription of
	// NewTxsEvent and send events to the given channel.
	SubscribeNewTxsEvent(chan<- core.NewTxsEvent) event.Subscription
}

// handlerConfig is the collection of initialization parameters to create a full
// node network handler.
type handlerConfig struct {
	NodeID     enode.ID                  // P2P node ID used for tx propagation topology
	Database   ctxcdb.Database           // Database for direct sync insertions
	Chain      *core.BlockChain          // Blockchain to serve data from
	TxPool     txPool                    // Transaction pool to propagate from
	Network    uint64                    // Network identifier to adfvertise
	Sync       downloader.SyncMode       // Whether to fast or full sync
	BloomCache uint64                    // Megabytes to alloc for fast sync bloom
	EventMux   *event.TypeMux            // Legacy event mux, deprecate for `feed`
	Checkpoint *params.TrustedCheckpoint // Hard coded checkpoint for sync challenges
	Whitelist  map[uint64]common.Hash    // Hard coded whitelist for sync challenged
}

type ProtocolManager struct {
	nodeID     enode.ID
	networkID  uint64
	forkFilter forkid.Filter // Fork ID filter, constant across the lifetime of the node

	fastSync  atomic.Bool // Flag whether fast sync is enabled (gets disabled if we already have blocks)
	acceptTxs atomic.Bool // Flag whether we're considered synchronised (enables transaction processing)

	checkpointNumber uint64      // Block number for the sync progress validator to cross reference
	checkpointHash   common.Hash // Block hash for the sync progress validator to cross reference
	checkpointName   string

	txpool      txPool
	blockchain  *core.BlockChain
	chaindb     ctxcdb.Database
	chainconfig *params.ChainConfig
	maxPeers    int

	downloader     *downloader.Downloader
	fetcher        *fetcher.Fetcher
	txFetcher      *fetcher.TxFetcher
	peers          *peerSet
	txBroadcastKey [16]byte

	eventMux      *event.TypeMux
	txsCh         chan core.NewTxsEvent
	txsSub        event.Subscription
	minedBlockSub *event.TypeMuxSubscription

	whitelist map[uint64]common.Hash

	// channels for fetcher, syncer, txsyncLoop
	txsyncCh chan *txsync
	quitSync chan struct{}

	chainSync                *chainSyncer
	wg                       sync.WaitGroup
	handlerStartCh           chan struct{}
	handlerDoneCh            chan struct{}
	broadcastTxAnnouncesOnly bool // Testing field, disable transaction propagation
}

// NewProtocolManager returns a new Cortex sub protocol manager. The Cortex sub protocol manages peers capable
// with the Cortex network.
func NewProtocolManager(config *handlerConfig) (*ProtocolManager, error) {
	// Create the protocol manager with the base fields
	manager := &ProtocolManager{
		nodeID:         config.NodeID,
		networkID:      config.Network,
		forkFilter:     forkid.NewFilter(config.Chain),
		eventMux:       config.EventMux,
		txpool:         config.TxPool,
		blockchain:     config.Chain,
		chaindb:        config.Database,
		peers:          newPeerSet(),
		whitelist:      config.Whitelist,
		txsyncCh:       make(chan *txsync),
		txBroadcastKey: newBroadcastChoiceKey(),
		quitSync:       make(chan struct{}),
		handlerDoneCh:  make(chan struct{}),
		handlerStartCh: make(chan struct{}),
	}
	// Figure out whether to allow fast sync or not
	//if mode == downloader.FastSync && blockchain.CurrentBlock().NumberU64() == 0 {
	//	manager.fastSync = uint32(1)
	//} else {
	//	mode = downloader.FullSync
	//}

	if config.Sync == downloader.FullSync {
		// The database seems empty as the current block is the genesis. Yet the fast
		// block is ahead, so fast sync was enabled for this node at a certain point.
		// The scenarios where this can happen is
		// * if the user manually (or via a bad block) rolled back a fast sync node
		//   below the sync point.
		// * the last fast sync is not finished while user specifies a full sync this
		//   time. But we don't have any recent state for full sync.
		// In these cases however it's safe to reenable fast sync.
		fullBlock, fastBlock := config.Chain.CurrentBlock(), config.Chain.CurrentFastBlock()
		if fullBlock.NumberU64() == 0 && fastBlock.NumberU64() > 0 {
			manager.fastSync.Store(true)
			log.Warn("Switch sync mode from full sync to fast sync")
		}
	} else {
		if config.Chain.CurrentBlock().NumberU64() > 0 {
			// Print warning log if database is not empty to run fast sync.
			log.Warn("Switch sync mode from fast sync to full sync")
		} else {
			// If fast sync was requested and our database is empty, grant it
			manager.fastSync.Store(true)
		}
	}

	// If we have trusted checkpoints, enforce them on the chain
	if checkpoint, ok := params.TrustedCheckpoints[config.Chain.Genesis().Hash()]; ok {
		manager.checkpointNumber = (checkpoint.SectionIndex+1)*params.CHTFrequency - 1
		manager.checkpointHash = checkpoint.SectionHead
		manager.checkpointName = checkpoint.Name
		log.Info("Check point", "section", checkpoint.SectionIndex, "number", manager.checkpointNumber, "hash", manager.checkpointHash, "genesis", config.Chain.Genesis().Hash(), "ok", ok)
	} else {
		log.Warn("No check point found", "genesis", config.Chain.Genesis().Hash())
	}
	// Initiate a sub-protocol for every implemented version we can handle
	var stateBloom *trie.SyncBloom
	if manager.fastSync.Load() {
		stateBloom = trie.NewSyncBloom(config.BloomCache, config.Database)
	}
	manager.downloader = downloader.New(manager.checkpointNumber, config.Database, stateBloom, manager.eventMux, config.Chain, manager.removePeer)

	// Construct the fetcher (short sync)
	validator := func(header *types.Header) error {
		return config.Chain.Engine().VerifyHeader(config.Chain, header, true)
	}
	heighter := func() uint64 {
		return config.Chain.CurrentBlock().NumberU64()
	}
	inserter := func(blocks types.Blocks) (int, error) {
		// If sync hasn't reached the checkpoint yet, deny importing weird blocks.
		//
		// Ideally we would also compare the head block's timestamp and similarly reject
		// the propagated block if the head is too old. Unfortunately there is a corner
		// case when starting new networks, where the genesis might be ancient (0 unix)
		// which would prevent full nodes from accepting it.
		if manager.blockchain.CurrentBlock().NumberU64() < manager.checkpointNumber {
			log.Warn("Unsynced yet, discarded propagated block", "number", blocks[0].Number(), "hash", blocks[0].Hash())
			return 0, nil
		}
		// If fast sync is running, deny importing weird blocks. This is a problematic
		// clause when starting up a new network, because fast-syncing miners might not
		// accept each others' blocks until a restart. Unfortunately we haven't figured
		// out a way yet where nodes can decide unilaterally whether the network is new
		// or not. This should be fixed if we figure out a solution.
		if manager.fastSync.Load() {
			log.Warn("Fast syncing, discarded propagated block", "number", blocks[0].Number(), "hash", blocks[0].Hash())
			return 0, nil
		}
		n, err := manager.blockchain.InsertChain(blocks)
		if err == nil {
			manager.acceptTxs.Store(true)
		}
		return n, err
	}
	manager.fetcher = fetcher.New(config.Chain.GetBlockByHash, validator, manager.BroadcastBlock, heighter, inserter, manager.removePeer)

	fetchTx := func(peer string, hashes []common.Hash) error {
		p := manager.peers.Peer(peer)
		if p == nil {
			return errors.New("unknown peer")
		}
		return p.RequestTxs(hashes)
	}
	manager.txFetcher = fetcher.NewTxFetcher(manager.blockchain, config.TxPool.Has, config.TxPool.AddRemotes, fetchTx)
	manager.chainSync = newChainSyncer(manager)

	return manager, nil
}

// protoTracker tracks the number of active protocol handlers.
func (h *ProtocolManager) protoTracker() {
	defer h.wg.Done()
	var active int
	for {
		select {
		case <-h.handlerStartCh:
			active++
		case <-h.handlerDoneCh:
			active--
		case <-h.quitSync:
			// Wait for all active handlers to finish.
			for ; active > 0; active-- {
				<-h.handlerDoneCh
			}
			return
		}
	}
}

// incHandlers signals to increment the number of active handlers if not
// quitting.
func (h *ProtocolManager) incHandlers() bool {
	select {
	case h.handlerStartCh <- struct{}{}:
		return true
	case <-h.quitSync:
		return false
	}
}

// decHandlers signals to decrement the number of active handlers.
func (h *ProtocolManager) decHandlers() {
	h.handlerDoneCh <- struct{}{}
}

func (pm *ProtocolManager) makeProtocol(version uint) p2p.Protocol {
	length, ok := protocolLengths[version]
	if !ok {
		panic("makeProtocol for unknown version")
	}

	return p2p.Protocol{
		Name:    protocolName,
		Version: version,
		Length:  length,
		Run: func(p *p2p.Peer, rw p2p.MsgReadWriter) error {
			return pm.runPeer(pm.newPeer(version, p, rw, pm.txpool.Get))
		},
		NodeInfo: func() any {
			return pm.NodeInfo()
		},
		PeerInfo: func(id enode.ID) any {
			if p := pm.peers.Peer(fmt.Sprintf("%x", id[:8])); p != nil {
				return p.Info()
			}
			return nil
		},
	}
}

func (pm *ProtocolManager) removePeer(id string) {
	peer := pm.peers.Peer(id)
	if peer != nil {
		peer.Peer.Disconnect(p2p.DiscUselessPeer)
	}
}

// unregisterPeer removes a peer from the downloader, fetchers and main peer set.
func (pm *ProtocolManager) unregisterPeer(id string) {
	// Short circuit if the peer was already removed
	peer := pm.peers.Peer(id)
	if peer == nil {
		return
	}
	log.Debug("Removing Cortex peer", "peer", id)

	// Unregister the peer from the downloader and Cortex peer set
	pm.downloader.UnregisterPeer(id)
	pm.txFetcher.Drop(id)

	if err := pm.peers.Unregister(id); err != nil {
		log.Error("Peer removal failed", "peer", id, "err", err)
	}
	// Hard disconnect at the networking layer
	if peer != nil {
		//peer.Peer.Disconnect(p2p.DiscUselessPeer)
	}
}

func (pm *ProtocolManager) Start(maxPeers int) {
	pm.maxPeers = maxPeers

	// broadcast transactions
	pm.wg.Add(1)
	pm.txsCh = make(chan core.NewTxsEvent, txChanSize)
	pm.txsSub = pm.txpool.SubscribeNewTxsEvent(pm.txsCh)
	go pm.txBroadcastLoop()

	// broadcast mined blocks
	pm.wg.Add(1)
	pm.minedBlockSub = pm.eventMux.Subscribe(core.NewMinedBlockEvent{})
	go pm.minedBroadcastLoop()

	// start sync handlers
	pm.wg.Add(2)
	go pm.chainSync.loop()
	go pm.txsyncLoop64()

	// start peer handler tracker
	pm.wg.Add(1)
	go pm.protoTracker()
}

func (pm *ProtocolManager) Stop() {
	pm.txsSub.Unsubscribe()        // quits txBroadcastLoop
	pm.minedBlockSub.Unsubscribe() // quits blockBroadcastLoop

	// Quit chainSync and txsync64.
	// After this is done, no new peers will be accepted.
	close(pm.quitSync)
	log.Warn("Cortex peers stopping ... ...")

	// Disconnect existing sessions.
	// This also closes the gate for any new registrations on the peer set.
	// sessions which are already established but not added to pm.peers yet
	// will exit when they try to register.
	pm.peers.Close()
	log.Info("Cortex protocol stopping ... ...")
	pm.wg.Wait()

	log.Info("Cortex protocol stopped")
}

func (pm *ProtocolManager) newPeer(pv uint, p *p2p.Peer, rw p2p.MsgReadWriter, getPooledTx func(hash common.Hash) *types.Transaction) *peer {
	return newPeer(pv, p, rw, getPooledTx)
}

func (pm *ProtocolManager) runPeer(p *peer) error {
	if !pm.incHandlers() {
		return p2p.DiscQuitting
	}
	defer pm.decHandlers()

	return pm.handle(p)
}

// handle is the callback invoked to manage the life cycle of an ctxc peer. When
// this function terminates, the peer is disconnected.
func (pm *ProtocolManager) handle(p *peer) error {
	// Ignore maxPeers if this is a trusted peer
	if pm.peers.Len() >= pm.maxPeers && !p.Peer.Info().Network.Trusted {
		return p2p.DiscTooManyPeers
	}
	p.Log().Debug("Cortex peer connected", "name", p.Name())

	// Execute the Cortex handshake
	var (
		genesis = pm.blockchain.Genesis()
		head    = pm.blockchain.CurrentHeader()
		hash    = head.Hash()
		number  = head.Number.Uint64()
		td      = pm.blockchain.GetTd(hash, number)
	)
	if err := p.Handshake(pm.networkID, td, hash, genesis.Hash(), forkid.NewID(pm.blockchain.Config(), genesis, number, head.Time), pm.forkFilter); err != nil {
		p.Log().Debug("Cortex handshake failed", "err", err)
		return err
	}
	// Register the peer locally
	if err := pm.peers.Register(p, pm.removePeer); err != nil {
		p.Log().Error("Cortex peer registration failed", "err", err)
		return err
	}
	defer pm.unregisterPeer(p.id)

	if pm.peers.Peer(p.id) == nil {
		return errors.New("peer dropped during handling")
	}

	// Register the peer in the downloader. If the downloader considers it banned, we disconnect
	if err := pm.downloader.RegisterPeer(p.id, p.version, p); err != nil {
		p.Log().Error("Failed to register peer in syncer", "err", err)
		return err
	}
	pm.chainSync.handlePeerEvent()

	// Propagate existing transactions. new transactions appearing
	// after this will be sent via broadcasts.
	pm.syncTransactions(p)

	// If we have a trusted CHT, reject all peers below that (avoid fast sync eclipse)
	if pm.checkpointHash != (common.Hash{}) {
		// Request the peer's checkpoint header for chain height/weight validation
		if err := p.RequestHeadersByNumber(pm.checkpointNumber, 1, 0, false); err != nil {
			return err
		}

		p.syncDrop = time.AfterFunc(syncChallengeTimeout, func() {
			p.Log().Warn("Checkpoint challenge timed out, dropping", "addr", p.RemoteAddr(), "type", p.Name())
			pm.removePeer(p.id)
		})

		defer func() {
			if p.syncDrop != nil {
				p.syncDrop.Stop()
				p.syncDrop = nil
			}
		}()
	}
	// If we have any explicit whitelist block hashes, request them
	for number := range pm.whitelist {
		if err := p.RequestHeadersByNumber(number, 1, 0, false); err != nil {
			return err
		}
	}
	// main loop. handle incoming messages.
	for {
		if err := pm.handleMsg(p); err != nil {
			p.Log().Debug("Cortex message handling failed", "err", err)
			return err
		}
	}
}

func serviceNonContiguousBlockHeaderQuery(chain *core.BlockChain, query *getBlockHeadersData, peer *peer) []rlp.RawValue {
	hashMode := query.Origin.Hash != (common.Hash{})
	first := true
	maxNonCanonical := uint64(100)
	// Gather headers until the fetch or network limits is reached
	var (
		bytes   common.StorageSize
		headers []rlp.RawValue
		unknown bool
		lookups int
	)
	for !unknown && len(headers) < int(query.Amount) && bytes < softResponseLimit &&
		len(headers) < maxHeadersServe && lookups < 2*maxHeadersServe {
		lookups++
		// Retrieve the next header satisfying the query
		var origin *types.Header
		if hashMode {
			if first {
				first = false
				origin = chain.GetHeaderByHash(query.Origin.Hash)
				if origin != nil {
					query.Origin.Number = origin.Number.Uint64()
				}
			} else {
				origin = chain.GetHeader(query.Origin.Hash, query.Origin.Number)
			}
		} else {
			origin = chain.GetHeaderByNumber(query.Origin.Number)
		}
		if origin == nil {
			break
		}
		if rlpData, err := rlp.EncodeToBytes(origin); err != nil {
			log.Crit("Unable to encode our own headers", "err", err)
		} else {
			headers = append(headers, rlp.RawValue(rlpData))
			bytes += common.StorageSize(len(rlpData))
		}
		// Advance to the next header of the query
		switch {
		case hashMode && query.Reverse:
			// Hash based traversal towards the genesis block
			ancestor := query.Skip + 1
			if ancestor == 0 {
				unknown = true
			} else {
				query.Origin.Hash, query.Origin.Number = chain.GetAncestor(query.Origin.Hash, query.Origin.Number, ancestor, &maxNonCanonical)
				unknown = (query.Origin.Hash == common.Hash{})
			}
		case hashMode && !query.Reverse:
			// Hash based traversal towards the leaf block
			var (
				current = origin.Number.Uint64()
				next    = current + query.Skip + 1
			)
			if next <= current {
				infos, _ := json.MarshalIndent(peer.Peer.Info(), "", "  ")
				peer.Log().Warn("GetBlockHeaders skip overflow attack", "current", current, "skip", query.Skip, "next", next, "attacker", infos)
				unknown = true
			} else {
				if header := chain.GetHeaderByNumber(next); header != nil {
					nextHash := header.Hash()
					expOldHash, _ := chain.GetAncestor(nextHash, next, query.Skip+1, &maxNonCanonical)
					if expOldHash == query.Origin.Hash {
						query.Origin.Hash, query.Origin.Number = nextHash, next
					} else {
						unknown = true
					}
				} else {
					unknown = true
				}
			}
		case query.Reverse:
			// Number based traversal towards the genesis block
			current := query.Origin.Number
			ancestor := current - (query.Skip + 1)
			if ancestor >= current { // check for underflow
				unknown = true
			} else {
				query.Origin.Number = ancestor
			}
		case !query.Reverse:
			current := query.Origin.Number
			next := current + query.Skip + 1
			if next <= current { // check for overflow
				unknown = true
			} else {
				query.Origin.Number = next
			}
		}
	}
	return headers
}

func serviceContiguousBlockHeaderQuery(chain *core.BlockChain, query *getBlockHeadersData) []rlp.RawValue {
	count := query.Amount
	if count > maxHeadersServe {
		count = maxHeadersServe
	}
	if query.Origin.Hash == (common.Hash{}) {
		// Number mode, just return the canon chain segment. The backend
		// delivers in [N, N-1, N-2..] descending order, so we need to
		// accommodate for that.
		from := query.Origin.Number
		if !query.Reverse {
			from = from + count - 1
		}
		headers := chain.GetHeadersFrom(from, count)
		if !query.Reverse {
			for i, j := 0, len(headers)-1; i < j; i, j = i+1, j-1 {
				headers[i], headers[j] = headers[j], headers[i]
			}
		}
		return headers
	}
	// Hash mode.
	var (
		headers []rlp.RawValue
		hash    = query.Origin.Hash
		header  = chain.GetHeaderByHash(hash)
	)
	if header != nil {
		rlpData, _ := rlp.EncodeToBytes(header)
		headers = append(headers, rlpData)
	} else {
		// We don't even have the origin header
		return headers
	}
	num := header.Number.Uint64()
	if !query.Reverse {
		// Theoretically, we are tasked to deliver header by hash H, and onwards.
		// However, if H is not canon, we will be unable to deliver any descendants of
		// H.
		if canonHash := chain.GetCanonicalHash(num); canonHash != hash {
			// Not canon, we can't deliver descendants
			return headers
		}
		descendants := chain.GetHeadersFrom(num+count-1, count-1)
		for i, j := 0, len(descendants)-1; i < j; i, j = i+1, j-1 {
			descendants[i], descendants[j] = descendants[j], descendants[i]
		}
		headers = append(headers, descendants...)
		return headers
	}
	{ // Last mode: deliver ancestors of H
		for i := uint64(1); i < count; i++ {
			header = chain.GetHeaderByHash(header.ParentHash)
			if header == nil {
				break
			}
			rlpData, _ := rlp.EncodeToBytes(header)
			headers = append(headers, rlpData)
		}
		return headers
	}
}

// ServiceGetBlockHeadersQuery assembles the response to a header query. It is
// exposed to allow external packages to test protocol behavior.
func ServiceGetBlockHeadersQuery(chain *core.BlockChain, query *getBlockHeadersData, peer *peer) []rlp.RawValue {
	if query.Amount == 0 {
		return nil
	}
	if query.Skip == 0 {
		// The fast path: when the request is for a contiguous segment of headers.
		return serviceContiguousBlockHeaderQuery(chain, query)
	} else {
		return serviceNonContiguousBlockHeaderQuery(chain, query, peer)
	}
}

// handleMsg is invoked whenever an inbound message is received from a remote
// peer. The remote connection is torn down upon returning any error.
func (pm *ProtocolManager) handleMsg(p *peer) error {
	// Read the next message from the remote peer, and ensure it's fully consumed
	msg, err := p.rw.ReadMsg()
	if err != nil {
		return err
	}
	if msg.Size > protocolMaxMsgSize {
		return errResp(ErrMsgTooLarge, "%v > %v", msg.Size, protocolMaxMsgSize)
	}
	defer msg.Discard()

	// Handle the message depending on its contents
	switch {
	case msg.Code == ctxc.StatusMsg:
		// Status messages should never arrive after the handshake
		return errResp(ErrExtraStatusMsg, "uncontrolled status message")

	// Block header query, collect the requested headers and reply
	case msg.Code == ctxc.GetBlockHeadersMsg:
		// Decode the complex header query
		var query getBlockHeadersData
		if err := msg.Decode(&query); err != nil {
			return errResp(ErrDecode, "%v: %v", msg, err)
		}

		response := ServiceGetBlockHeadersQuery(pm.blockchain, &query, p)
		return p.SendBlockHeaders(response)

	case msg.Code == ctxc.BlockHeadersMsg:
		// A batch of headers arrived to one of our previous requests
		var headers []*types.Header
		if err := msg.Decode(&headers); err != nil {
			return errResp(ErrDecode, "msg %v: %v", msg, err)
		}

		if len(headers) == 0 && p.syncDrop != nil {
			// Stop the timer either way, decide later to drop or not
			p.syncDrop.Stop()
			p.syncDrop = nil

			// If we're doing a fast sync, we must enforce the checkpoint block to avoid
			// eclipse attacks. Unsynced nodes are welcome to connect after we're done
			// joining the network
			if pm.fastSync.Load() {
				p.Log().Warn("Dropping unsynced node during fast sync", "addr", p.RemoteAddr(), "type", p.Name())
				return errors.New("unsynced node cannot serve fast sync")
			}
		}
		// Filter out any explicitly requested headers, deliver the rest to the downloader
		filter := len(headers) == 1
		if filter {
			// If it's a potential sync progress check, validate the content and advertised chain weight
			if p.syncDrop != nil && headers[0].Number.Uint64() == pm.checkpointNumber {
				// Disable the sync drop timer
				p.syncDrop.Stop()
				p.syncDrop = nil

				// Validate the header and either drop the peer or continue
				if headers[0].Hash() != pm.checkpointHash {
					return errors.New("checkpoint hash mismatch")
				}
				return nil
			}
			// Otherwise if it's a whitelisted block, validate against the set
			if want, ok := pm.whitelist[headers[0].Number.Uint64()]; ok {
				if hash := headers[0].Hash(); want != hash {
					p.Log().Info("Whitelist mismatch, dropping peer", "number", headers[0].Number.Uint64(), "hash", hash, "want", want)
					return errors.New("whitelist block mismatch")
				}
				p.Log().Debug("Whitelist block verified", "number", headers[0].Number.Uint64(), "hash", want)
			}
			// Irrelevant of the fork checks, send the header to the fetcher just in case
			headers = pm.fetcher.FilterHeaders(p.id, headers, time.Now())
		}
		if len(headers) > 0 || !filter {
			err := pm.downloader.DeliverHeaders(p.id, headers)
			if err != nil {
				log.Debug("Failed to deliver headers", "err", err)
			}
		}

	case msg.Code == ctxc.GetBlockBodiesMsg:
		// Decode the retrieval message
		msgStream := rlp.NewStream(msg.Payload, uint64(msg.Size))
		if _, err := msgStream.List(); err != nil {
			return err
		}
		// Gather blocks until the fetch or network limits is reached
		var (
			hash    common.Hash
			bytes   int
			bodies  []rlp.RawValue
			lookups int
		)
		for bytes < softResponseLimit && len(bodies) < downloader.MaxBlockFetch && lookups < 2*maxBodiesServe {
			lookups++
			// Retrieve the hash of the next block
			if err := msgStream.Decode(&hash); err == rlp.EOL {
				break
			} else if err != nil {
				return errResp(ErrDecode, "msg %v: %v", msg, err)
			}
			// Retrieve the requested block body, stopping if enough was found
			if data := pm.blockchain.GetBodyRLP(hash); len(data) != 0 {
				bodies = append(bodies, data)
				bytes += len(data)
			}
		}
		return p.SendBlockBodiesRLP(bodies)

	case msg.Code == ctxc.BlockBodiesMsg:
		// A batch of block bodies arrived to one of our previous requests
		var request blockBodiesData
		if err := msg.Decode(&request); err != nil {
			return errResp(ErrDecode, "msg %v: %v", msg, err)
		}
		// Deliver them all to the downloader for queuing
		transactions := make([][]*types.Transaction, len(request))
		uncles := make([][]*types.Header, len(request))

		for i, body := range request {
			transactions[i] = body.Transactions
			uncles[i] = body.Uncles
		}
		// Filter out any explicitly requested bodies, deliver the rest to the downloader
		filter := len(transactions) > 0 || len(uncles) > 0
		if filter {
			transactions, uncles = pm.fetcher.FilterBodies(p.id, transactions, uncles, time.Now())
		}
		if len(transactions) > 0 || len(uncles) > 0 || !filter {
			err := pm.downloader.DeliverBodies(p.id, transactions, uncles)
			if err != nil {
				log.Debug("Failed to deliver bodies", "err", err)
			}
		}

	case msg.Code == ctxc.GetNodeDataMsg:
		// Decode the retrieval message
		msgStream := rlp.NewStream(msg.Payload, uint64(msg.Size))
		if _, err := msgStream.List(); err != nil {
			return err
		}
		// Gather state data until the fetch or network limits is reached
		var (
			hash  common.Hash
			bytes int
			data  [][]byte
		)
		for bytes < softResponseLimit && len(data) < downloader.MaxStateFetch {
			// Retrieve the hash of the next state entry
			if err := msgStream.Decode(&hash); err == rlp.EOL {
				break
			} else if err != nil {
				return errResp(ErrDecode, "msg %v: %v", msg, err)
			}
			// Retrieve the requested state entry, stopping if enough was found
			// todo now the code and trienode is mixed in the protocol level,
			// separate these two types.
			if !pm.downloader.SyncBloomContains(hash[:]) {
				// Only lookup the trie node if there's chance that we actually have it
				continue
			}
			entry, err := pm.blockchain.TrieNode(hash)
			if len(entry) == 0 || err != nil {
				// Read the contract code with prefix only to save unnecessary lookups.
				entry, err = pm.blockchain.ContractCodeWithPrefix(hash)
			}
			if err == nil && len(entry) > 0 {
				data = append(data, entry)
				bytes += len(entry)
			}
		}
		return p.SendNodeData(data)

	case msg.Code == ctxc.NodeDataMsg:
		// A batch of node state data arrived to one of our previous requests
		var data [][]byte
		if err := msg.Decode(&data); err != nil {
			return errResp(ErrDecode, "msg %v: %v", msg, err)
		}
		// Deliver all to the downloader
		if err := pm.downloader.DeliverNodeData(p.id, data); err != nil {
			log.Debug("Failed to deliver node state data", "err", err)
		}

	case msg.Code == ctxc.GetReceiptsMsg:
		// Decode the retrieval message
		msgStream := rlp.NewStream(msg.Payload, uint64(msg.Size))
		if _, err := msgStream.List(); err != nil {
			return err
		}
		// Gather state data until the fetch or network limits is reached
		var (
			hash     common.Hash
			bytes    int
			receipts []rlp.RawValue
			lookups  int
		)
		for bytes < softResponseLimit && len(receipts) < maxReceiptsServe && lookups < 2*maxReceiptsServe {
			lookups++
			// Retrieve the hash of the next block
			if err := msgStream.Decode(&hash); err == rlp.EOL {
				break
			} else if err != nil {
				return errResp(ErrDecode, "msg %v: %v", msg, err)
			}
			// Retrieve the requested block's receipts, skipping if unknown to us
			results := pm.blockchain.GetReceiptsByHash(hash)
			if results == nil {
				if header := pm.blockchain.GetHeaderByHash(hash); header == nil || header.ReceiptHash != types.EmptyRootHash {
					continue
				}
			}
			// If known, encode and queue for response packet
			if encoded, err := rlp.EncodeToBytes(results); err != nil {
				log.Error("Failed to encode receipt", "err", err)
			} else {
				receipts = append(receipts, encoded)
				bytes += len(encoded)
			}
		}
		return p.SendReceiptsRLP(receipts)

	case msg.Code == ctxc.ReceiptsMsg:
		// A batch of receipts arrived to one of our previous requests
		var receipts [][]*types.Receipt
		if err := msg.Decode(&receipts); err != nil {
			return errResp(ErrDecode, "msg %v: %v", msg, err)
		}
		// Deliver all to the downloader
		if err := pm.downloader.DeliverReceipts(p.id, receipts); err != nil {
			log.Debug("Failed to deliver receipts", "err", err)
		}

	case msg.Code == ctxc.NewBlockHashesMsg:
		var announces newBlockHashesData
		if err := msg.Decode(&announces); err != nil {
			return errResp(ErrDecode, "%v: %v", msg, err)
		}
		// Mark the hashes as present at the remote node
		for _, block := range announces {
			p.MarkBlock(block.Hash)
		}
		//propAnnounceAllInMeter.Mark(int64(len(announces)))
		// Schedule all the unknown hashes for retrieval
		//unknown := make(newBlockHashesData, 0, len(announces))
		for _, block := range announces {
			//	if !pm.blockchain.HasBlock(block.Hash, block.Number) {
			//		unknown = append(unknown, block)
			//	}
			//}
			//for _, block := range unknown {
			pm.fetcher.Notify(p.id, block.Hash, block.Number, time.Now(), p.RequestOneHeader, p.RequestBodies)
		}

	case msg.Code == ctxc.NewBlockMsg:
		// Retrieve and decode the propagated block
		var request newBlockData
		if err := msg.Decode(&request); err != nil {
			return errResp(ErrDecode, "%v: %v", msg, err)
		}
		//propBroadcastAllInMeter.Mark(1)
		if hash := types.CalcUncleHash(request.Block.Uncles()); hash != request.Block.UncleHash() {
			log.Warn("Propagated block has invalid uncles", "have", hash, "exp", request.Block.UncleHash())
			break // TODO return error eventually, but wait a few releases
		}
		if hash := types.DeriveSha(request.Block.Transactions(), trie.NewStackTrie(nil)); hash != request.Block.TxHash() {
			log.Warn("Propagated block has invalid body", "have", hash, "exp", request.Block.TxHash())
			break // TODO return error eventually, but wait a few releases
		}
		if err := request.sanityCheck(); err != nil {
			return err
		}
		request.Block.ReceivedAt = msg.ReceivedAt
		request.Block.ReceivedFrom = p

		// Mark the peer as owning the block and schedule it for import
		p.MarkBlock(request.Block.Hash())
		pm.fetcher.Enqueue(p.id, request.Block)

		// Assuming the block is importable by the peer, but possibly not yet done so,
		// calculate the head hash and TD that the peer truly must have.
		var (
			trueHead = request.Block.ParentHash()
			trueTD   = new(big.Int).Sub(request.TD, request.Block.Difficulty())
		)
		// Update the peers total difficulty if better than the previous
		if _, td := p.Head(); trueTD.Cmp(td) > 0 {
			p.SetHead(trueHead, trueTD)
			pm.chainSync.handlePeerEvent()
		}

	case msg.Code == ctxc.NewPooledTransactionHashesMsg && p.version >= ctxc65:
		// New transaction announcement arrived, make sure we have
		// a valid and fresh chain to handle them
		if !pm.acceptTxs.Load() {
			break
		}
		var hashes []common.Hash
		if err := msg.Decode(&hashes); err != nil {
			return errResp(ErrDecode, "msg %v: %v", msg, err)
		}
		// Schedule all the unknown hashes for retrieval
		for _, hash := range hashes {
			p.MarkTransaction(hash)
		}
		pm.txFetcher.Notify(p.id, hashes)

	case msg.Code == ctxc.GetPooledTransactionsMsg && p.version >= ctxc65:
		// Decode the retrieval message
		msgStream := rlp.NewStream(msg.Payload, uint64(msg.Size))
		if _, err := msgStream.List(); err != nil {
			return err
		}
		// Gather transactions until the fetch or network limits is reached
		var (
			hash   common.Hash
			bytes  int
			hashes []common.Hash
			txs    []rlp.RawValue
		)
		for bytes < softResponseLimit {
			// Retrieve the hash of the next block
			if err := msgStream.Decode(&hash); err == rlp.EOL {
				break
			} else if err != nil {
				return errResp(ErrDecode, "msg %v: %v", msg, err)
			}
			// Retrieve the requested transaction, skipping if unknown to us
			tx := pm.txpool.Get(hash)
			if tx == nil {
				continue
			}
			// If known, encode and queue for response packet
			if encoded, err := rlp.EncodeToBytes(tx); err != nil {
				log.Error("Failed to encode transaction", "err", err)
			} else {
				hashes = append(hashes, hash)
				txs = append(txs, encoded)
				bytes += len(encoded)
			}
		}
		return p.SendPooledTransactionsRLP(hashes, txs)

	case msg.Code == ctxc.TransactionMsg || (msg.Code == ctxc.PooledTransactionsMsg && p.version >= ctxc65):
		// Transactions arrived, make sure we have a valid and fresh chain to handle them
		if !pm.acceptTxs.Load() {
			break
		}
		// Transactions can be processed, parse all of them and deliver to the pool
		var txs []*types.Transaction
		if err := msg.Decode(&txs); err != nil {
			return errResp(ErrDecode, "msg %v: %v", msg, err)
		}
		// Duplicate transactions are not allowed
		seen := make(map[common.Hash]struct{})
		for i, tx := range txs {
			// Validate and mark the remote transaction
			if tx == nil {
				return errResp(ErrDecode, "transaction %d is nil", i)
			}
			hash := tx.Hash()
			if _, exists := seen[hash]; exists {
				return fmt.Errorf("Transactions: multiple copies of the same hash %v", hash)
			}
			seen[hash] = struct{}{}
			p.MarkTransaction(hash)
		}
		pm.txFetcher.Enqueue(p.id, txs, msg.Code == ctxc.PooledTransactionsMsg)

	default:
		return errResp(ErrInvalidMsgCode, "%v", msg.Code)
	}
	return nil
}

// BroadcastBlock will either propagate a block to a subset of it's peers, or
// will only announce it's availability (depending what's requested).
func (pm *ProtocolManager) BroadcastBlock(block *types.Block, propagate bool) {
	hash := block.Hash()
	peers := pm.peers.PeersWithoutBlock(hash)

	// If propagation is requested, send to a subset of the peer
	if propagate {
		// Calculate the TD of the block (it's not imported yet, so block.Td is not valid)
		var td *big.Int
		if parent := pm.blockchain.GetBlock(block.ParentHash(), block.NumberU64()-1); parent != nil {
			td = new(big.Int).Add(block.Difficulty(), pm.blockchain.GetTd(block.ParentHash(), block.NumberU64()-1))
		} else {
			log.Error("Propagating dangling block", "number", block.Number(), "hash", hash)
			return
		}
		// Send the block to a subset of our peers
		transfer := peers[:int(math.Sqrt(float64(len(peers))))]
		for _, peer := range transfer {
			peer.AsyncSendNewBlock(block, td)
		}
		log.Trace("Propagated block", "hash", hash, "recipients", len(transfer), "duration", common.PrettyDuration(time.Since(block.ReceivedAt)))
		return
	}
	// Otherwise if the block is indeed in out own chain, announce it
	if pm.blockchain.HasBlock(hash, block.NumberU64()) {
		for _, peer := range peers {
			peer.AsyncSendNewBlockHash(block)
		}
		log.Trace("Announced block", "hash", hash, "recipients", len(peers), "duration", common.PrettyDuration(time.Since(block.ReceivedAt)))
	}
}

// BroadcastTxs will propagate a batch of transactions to all peers which are not known to
// already have the given transaction.
func (pm *ProtocolManager) BroadcastTransactions(txs types.Transactions) {
	var (
		largeTxs int // Number of large transactions to announce only

		annoCount   int // Count of announcements made
		annoPeers   int
		directCount int // Count of the txs sent directly to peers
		directPeers int // Count of the peers that were sent transactions directly

		txset = make(map[*peer][]common.Hash) // Set peer->hash to transfer directly
		annos = make(map[*peer][]common.Hash) // Set peer->hash to announce

		signer = types.LatestSigner(pm.blockchain.Config())
		choice = newBroadcastChoice(pm.nodeID, pm.txBroadcastKey)
		peers  = pm.peers.all()
	)

	for _, tx := range txs {
		var directSet map[*peer]struct{}
		switch {
		case tx.Size() > txMaxBroadcastSize:
			largeTxs++
		default:
			// Get transaction sender address. Here we can ignore any error
			// since we're just interested in any value.
			txSender, _ := types.Sender(signer, tx)
			directSet = choice.choosePeers(peers, txSender)
		}

		for _, peer := range peers {
			if peer.KnownTransaction(tx.Hash()) {
				continue
			}
			if _, ok := directSet[peer]; ok {
				// Send direct.
				txset[peer] = append(txset[peer], tx.Hash())
			} else {
				// Send announcement.
				annos[peer] = append(annos[peer], tx.Hash())
			}
		}
	}

	for peer, hashes := range txset {
		directPeers++
		directCount += len(hashes)
		peer.AsyncSendTransactions(hashes)
	}
	for peer, hashes := range annos {
		annoPeers++
		annoCount += len(hashes)
		if peer.version >= ctxc65 {
			peer.AsyncSendPooledTransactionHashes(hashes)
		} else {
			peer.AsyncSendTransactions(hashes)
		}
	}
	log.Trace("Transaction broadcast", "txs", len(txs)-largeTxs, "largetxs", largeTxs,
		"announce packs", annoPeers, "announced hashes", annoCount,
		"tx packs", directPeers, "broadcast txs", directCount)
}

// Mined broadcast loop
func (pm *ProtocolManager) minedBroadcastLoop() {
	defer pm.wg.Done()

	for obj := range pm.minedBlockSub.Chan() {
		if ev, ok := obj.Data.(core.NewMinedBlockEvent); ok {
			pm.BroadcastBlock(ev.Block, true)  // First propagate block to peers
			pm.BroadcastBlock(ev.Block, false) // Only then announce to the rest
		}
	}
}

// txBroadcastLoop announces new transactions to connected peers.
func (pm *ProtocolManager) txBroadcastLoop() {
	defer pm.wg.Done()
	for {
		select {
		case event := <-pm.txsCh:
			pm.BroadcastTransactions(event.Txs)
		case <-pm.txsSub.Err():
			return
		}
	}
}

// NodeInfo represents a short summary of the Cortex sub-protocol metadata
// known about the host peer.
type NodeInfo struct {
	Network    uint64              `json:"network"`    // Cortex network ID
	Difficulty *big.Int            `json:"difficulty"` // Total difficulty of the host's blockchain
	Genesis    common.Hash         `json:"genesis"`    // SHA3 hash of the host's genesis block
	Config     *params.ChainConfig `json:"config"`     // Chain configuration for the fork rules
	Head       common.Hash         `json:"head"`       // SHA3 hash of the host's best owned block
}

// NodeInfo retrieves some protocol metadata about the running host node.
func (pm *ProtocolManager) NodeInfo() *NodeInfo {
	currentBlock := pm.blockchain.CurrentBlock()
	return &NodeInfo{
		Network:    pm.networkID,
		Difficulty: pm.blockchain.GetTd(currentBlock.Hash(), currentBlock.NumberU64()),
		Genesis:    pm.blockchain.Genesis().Hash(),
		Config:     pm.blockchain.Config(),
		Head:       currentBlock.Hash(),
	}
}

// broadcastChoice implements a deterministic random choice of peers. This is designed
// specifically for choosing which peer receives a direct broadcast of a transaction.
//
// The choice is made based on the involved p2p node IDs and the transaction sender,
// ensuring that the flow of transactions is grouped by account to (try and) avoid nonce
// gaps.
type broadcastChoice struct {
	self   enode.ID
	key    [16]byte
	buffer map[*peer]struct{}
	tmp    []broadcastPeer
}

type broadcastPeer struct {
	p     *peer
	score uint64
}

func newBroadcastChoiceKey() (k [16]byte) {
	crand.Read(k[:])
	return k
}

func newBroadcastChoice(self enode.ID, key [16]byte) *broadcastChoice {
	return &broadcastChoice{
		self:   self,
		key:    key,
		buffer: make(map[*peer]struct{}),
	}
}

// choosePeers selects the peers that will receive a direct transaction broadcast message.
// Note the return value will only stay valid until the next call to choosePeers.
func (bc *broadcastChoice) choosePeers(peers []*peer, txSender common.Address) map[*peer]struct{} {
	// Compute randomized scores.
	bc.tmp = slices.Grow(bc.tmp[:0], len(peers))[:len(peers)]
	hash := siphash.New(bc.key[:])
	for i, peer := range peers {
		hash.Reset()
		hash.Write(bc.self[:])
		hash.Write(peer.Peer.ID().Bytes())
		hash.Write(txSender[:])
		bc.tmp[i] = broadcastPeer{peer, hash.Sum64()}
	}

	// Sort by score.
	slices.SortFunc(bc.tmp, func(a, b broadcastPeer) int {
		return cmp.Compare(a.score, b.score)
	})

	// Take top n.
	clear(bc.buffer)
	n := int(math.Ceil(math.Sqrt(float64(len(bc.tmp)))))
	for i := range n {
		bc.buffer[bc.tmp[i].p] = struct{}{}
	}
	return bc.buffer
}
