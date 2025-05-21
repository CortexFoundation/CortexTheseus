// Copyright 2014 The go-ethereum Authors
// This file is part of The go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with The go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

// Package core implements the Cortex consensus protocol.
package core

import (
	"errors"
	"fmt"
	"io"
	"math/big"
	mrand "math/rand"
	"sync"
	"sync/atomic"
	"time"

	"slices"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/lru"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/common/prque"
	"github.com/CortexFoundation/CortexTheseus/consensus"
	"github.com/CortexFoundation/CortexTheseus/core/history"
	"github.com/CortexFoundation/CortexTheseus/core/rawdb"
	"github.com/CortexFoundation/CortexTheseus/core/state"
	"github.com/CortexFoundation/CortexTheseus/core/state/snapshot"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/core/vm"
	"github.com/CortexFoundation/CortexTheseus/ctxcdb"
	"github.com/CortexFoundation/CortexTheseus/event"
	"github.com/CortexFoundation/CortexTheseus/internal/syncx"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/metrics"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/CortexTheseus/rlp"
	"github.com/CortexFoundation/CortexTheseus/trie"
)

var (
	headBlockGauge          = metrics.NewRegisteredGauge("chain/head/block", nil)
	headHeaderGauge         = metrics.NewRegisteredGauge("chain/head/header", nil)
	headFastBlockGauge      = metrics.NewRegisteredGauge("chain/head/receipt", nil)
	headFinalizedBlockGauge = metrics.NewRegisteredGauge("chain/head/finalized", nil)
	headSafeBlockGauge      = metrics.NewRegisteredGauge("chain/head/safe", nil)

	chainInfoGauge = metrics.NewRegisteredGaugeInfo("chain/info", nil)

	accountReadTimer   = metrics.NewRegisteredTimer("chain/account/reads", nil)
	accountHashTimer   = metrics.NewRegisteredTimer("chain/account/hashes", nil)
	accountUpdateTimer = metrics.NewRegisteredTimer("chain/account/updates", nil)
	accountCommitTimer = metrics.NewRegisteredTimer("chain/account/commits", nil)

	storageReadTimer   = metrics.NewRegisteredTimer("chain/storage/reads", nil)
	storageHashTimer   = metrics.NewRegisteredTimer("chain/storage/hashes", nil)
	storageUpdateTimer = metrics.NewRegisteredTimer("chain/storage/updates", nil)
	storageCommitTimer = metrics.NewRegisteredTimer("chain/storage/commits", nil)

	snapshotAccountReadTimer = metrics.NewRegisteredTimer("chain/snapshot/account/reads", nil)
	snapshotStorageReadTimer = metrics.NewRegisteredTimer("chain/snapshot/storage/reads", nil)
	snapshotCommitTimer      = metrics.NewRegisteredTimer("chain/snapshot/commits", nil)

	blockInsertTimer     = metrics.NewRegisteredTimer("chain/inserts", nil)
	blockValidationTimer = metrics.NewRegisteredTimer("chain/validation", nil)
	blockExecutionTimer  = metrics.NewRegisteredTimer("chain/execution", nil)
	blockWriteTimer      = metrics.NewRegisteredTimer("chain/write", nil)

	blockReorgMeter     = metrics.NewRegisteredMeter("chain/reorg/executes", nil)
	blockReorgAddMeter  = metrics.NewRegisteredMeter("chain/reorg/add", nil)
	blockReorgDropMeter = metrics.NewRegisteredMeter("chain/reorg/drop", nil)

	blockPrefetchExecuteTimer   = metrics.NewRegisteredTimer("chain/prefetch/executes", nil)
	blockPrefetchInterruptMeter = metrics.NewRegisteredMeter("chain/prefetch/interrupts", nil)

	errInsertionInterrupted = errors.New("insertion is interrupted")
	errChainStopped         = errors.New("blockchain is stopped")

	errInvalidOldChain = errors.New("invalid old chain")
	errInvalidNewChain = errors.New("invalid new chain")
)

const (
	bodyCacheLimit      = 256
	blockCacheLimit     = 256
	receiptsCacheLimit  = 32
	txLookupCacheLimit  = 1024
	maxFutureBlocks     = 256
	maxTimeFutureBlocks = 30
	badBlockLimit       = 10

	// BlockChainVersion ensures that an incompatible database forces a resync from scratch.
	//
	// Changelog:
	//
	// - Version 4
	//   The following incompatible database changes were added:
	//   * the `BlockNumber`, `TxHash`, `TxIndex`, `BlockHash` and `Index` fields of log are deleted
	//   * the `Bloom` field of receipt is deleted
	//   * the `BlockIndex` and `TxIndex` fields of txlookup are deleted
	//
	// - Version 5
	//  The following incompatible database changes were added:
	//    * the `TxHash`, `GasCost`, and `ContractAddress` fields are no longer stored for a receipt
	//    * the `TxHash`, `GasCost`, and `ContractAddress` fields are computed by looking up the
	//      receipts' corresponding block
	//
	// - Version 6
	//  The following incompatible database changes were added:
	//    * Transaction lookup information stores the corresponding block number instead of block hash
	//
	// - Version 7
	//  The following incompatible database changes were added:
	//    * Use freezer as the ancient database to maintain all ancient data
	//
	// - Version 8
	//  The following incompatible database changes were added:
	//    * New scheme for contract code in order to separate the codes and trie nodes
	//
	// - Version 9
	//  The following incompatible database changes were added:
	//  * Total difficulty has been removed from both the key-value store and the ancient store.
	//  * The metadata structure of freezer is changed by adding 'flushOffset'
	BlockChainVersion uint64 = 9
)

// CacheConfig contains the configuration values for the trie caching/pruning
// that's resident in a blockchain.
type CacheConfig struct {
	TrieCleanLimit      int           // Memory allowance (MB) to use for caching trie nodes in memory
	TrieCleanJournal    string        // Disk journal for saving clean cache entries.
	TrieCleanRejournal  time.Duration // Time interval to dump clean cache to disk periodically
	TrieCleanNoPrefetch bool          // Whether to disable heuristic state prefetching for followup blocks
	TrieDirtyLimit      int           // Memory limit (MB) at which to start flushing dirty trie nodes to disk
	TrieDirtyDisabled   bool          // Whether to disable trie write caching and GC altogether (archive node)
	TrieTimeLimit       time.Duration // Time limit after which to flush the current in-memory trie to disk
	SnapshotLimit       int           // Memory allowance (MB) to use for caching snapshot entries in memory
	Preimages           bool          // Whether to store preimage of trie key to the disk

	SnapshotNoBuild bool // Whether the background generation is allowed
	SnapshotWait    bool // Wait for snapshot construction on startup. TODO(karalabe): This is a dirty hack for testing, nuke it

	// This defines the cutoff block for history expiry.
	// Blocks before this number may be unavailable in the chain database.
	ChainHistoryMode history.HistoryMode
}

// defaultCacheConfig are the default caching values if none are specified by the
// user (also used during testing).
var defaultCacheConfig = &CacheConfig{
	TrieCleanLimit: 256,
	TrieDirtyLimit: 256,
	TrieTimeLimit:  5 * time.Minute,
	SnapshotLimit:  256,
	SnapshotWait:   true,
}

// txLookup is wrapper over transaction lookup along with the corresponding
// transaction object.
type txLookup struct {
	lookup      *rawdb.LegacyTxLookupEntry
	transaction *types.Transaction
}

// BlockChain represents the canonical chain given a database with a genesis
// block. The Blockchain manages chain imports, reverts, chain reorganisations.
//
// Importing blocks in to the block chain happens according to the set of rules
// defined by the two stage Validator. Processing of blocks is done using the
// Processor which processes the included transaction. The validation of the state
// is done in the second part of the Validator. Failing results in aborting of
// the import.
//
// The BlockChain also helps in returning blocks from **any** chain included
// in the database as well as blocks that represents the canonical chain. It's
// important to note that GetBlock can return any block and does not need to be
// included in the canonical one where as GetBlockByNumber always represents the
// canonical chain.
type BlockChain struct {
	chainConfig *params.ChainConfig // Chain & network configuration
	cacheConfig *CacheConfig        // Cache configuration for pruning

	db     ctxcdb.Database                  // Low level persistent database to store final content in
	snaps  *snapshot.Tree                   // Snapshot tree for fast trie leaf access
	triegc *prque.Prque[int64, common.Hash] // Priority queue mapping block numbers to tries to gc
	gcproc time.Duration                    // Accumulates canonical block processing for trie dumping

	lastWrite     uint64
	flushInterval atomic.Int64

	// txLookupLimit is the maximum number of blocks from head whose tx indices
	// are reserved:
	//  * 0:   means no limit and regenerate any missing indexes
	//  * N:   means N block limit [HEAD-N+1, HEAD] and delete extra indexes
	//  * nil: disable tx reindexer/deleter, but still index new blocks
	txLookupLimit uint64

	hc               *HeaderChain
	rmLogsFeed       event.Feed
	chainFeed        event.Feed
	chainSideFeed    event.Feed
	chainHeadFeed    event.Feed
	logsFeed         event.Feed
	blockProcFeed    event.Feed
	blockProcCounter int32
	scope            event.SubscriptionScope
	genesisBlock     *types.Block

	// This mutex synchronizes chain write operations.
	// Readers don't need to take it, they can just read the database.
	chainmu *syncx.ClosableMutex

	currentBlock          atomic.Value // Current head of the block chain
	currentFastBlock      atomic.Value // Current head of the fast-sync chain (may be above the block chain!)
	currentFinalizedBlock atomic.Value // Current finalized head
	currentSafeBlock      atomic.Value // Current safe head
	historyPrunePoint     atomic.Pointer[history.PrunePoint]

	stateCache    *state.CachingDB // State database to reuse between imports (contains state cache)
	txIndexer     *txIndexer
	bodyCache     *lru.Cache[common.Hash, *types.Body]
	bodyRLPCache  *lru.Cache[common.Hash, rlp.RawValue]
	receiptsCache *lru.Cache[common.Hash, []*types.Receipt]
	blockCache    *lru.Cache[common.Hash, *types.Block]
	txLookupCache *lru.Cache[common.Hash, txLookup]

	txLookupLock sync.RWMutex

	// future blocks are blocks added for later processing
	futureBlocks *lru.Cache[common.Hash, *types.Block]

	wg            sync.WaitGroup //
	quit          chan struct{}  // shutdown signal, closed in Stop.
	stopping      atomic.Bool    // false if chain is running, true when stopped
	procInterrupt atomic.Bool    // interrupt signaler for block processing

	engine     consensus.Engine
	validator  Validator // Block and state validator interface
	prefetcher Prefetcher
	processor  Processor // Block transaction processor interface
	vmConfig   vm.Config

	shouldPreserve     func(*types.Header) bool // Function used to determine whether should preserve the given block.
	writeLegacyJournal bool                     // Testing flag used to flush the snapshot journal in legacy format.
	utcNow             int64
	Viper              bool
}

// NewBlockChain returns a fully initialised block chain using information
// available in the database. It initialises the default Cortex Validator and
// Processor.
func NewBlockChain(db ctxcdb.Database, cacheConfig *CacheConfig, chainConfig *params.ChainConfig, engine consensus.Engine, vmConfig vm.Config, shouldPreserve func(block *types.Header) bool, txLookupLimit *uint64) (*BlockChain, error) {
	if cacheConfig == nil {
		cacheConfig = defaultCacheConfig
	}

	// Open trie database with provided config
	stateCache := state.NewDatabaseWithConfig(db, nil, &trie.Config{
		Cache:     cacheConfig.TrieCleanLimit,
		Journal:   cacheConfig.TrieCleanJournal,
		Preimages: cacheConfig.Preimages,
	})

	bc := &BlockChain{
		chainConfig:   chainConfig,
		cacheConfig:   cacheConfig,
		db:            db,
		triegc:        prque.New[int64, common.Hash](nil),
		stateCache:    stateCache,
		quit:          make(chan struct{}),
		chainmu:       syncx.NewClosableMutex(),
		bodyCache:     lru.NewCache[common.Hash, *types.Body](bodyCacheLimit),
		bodyRLPCache:  lru.NewCache[common.Hash, rlp.RawValue](bodyCacheLimit),
		receiptsCache: lru.NewCache[common.Hash, []*types.Receipt](receiptsCacheLimit),
		blockCache:    lru.NewCache[common.Hash, *types.Block](blockCacheLimit),
		txLookupCache: lru.NewCache[common.Hash, txLookup](txLookupCacheLimit),
		futureBlocks:  lru.NewCache[common.Hash, *types.Block](maxFutureBlocks),
		engine:        engine,
		vmConfig:      vmConfig,
	}
	bc.validator = NewBlockValidator(chainConfig, bc, engine)
	bc.prefetcher = newStatePrefetcher(chainConfig, bc, engine)
	bc.processor = NewStateProcessor(chainConfig, bc, engine)

	bc.utcNow = time.Now().Unix()

	var err error
	bc.hc, err = NewHeaderChain(db, chainConfig, engine, bc.insertStopped)
	if err != nil {
		return nil, err
	}
	bc.genesisBlock = bc.GetBlockByNumber(0)
	if bc.genesisBlock == nil {
		return nil, ErrNoGenesis
	}

	var nilBlock *types.Block
	bc.currentBlock.Store(nilBlock)
	bc.currentFastBlock.Store(nilBlock)
	bc.currentFinalizedBlock.Store(nilBlock)
	bc.currentSafeBlock.Store(nilBlock)

	// Update chain info data metrics
	chainInfoGauge.Update(metrics.GaugeInfoValue{"chain_id": bc.chainConfig.ChainID.String()})

	// If Geth is initialized with an external ancient store, re-initialize the
	// missing chain indexes and chain flags. This procedure can survive crash
	// and can be resumed in next restart since chain flags are updated in last step.
	if bc.empty() {
		rawdb.InitDatabaseFromFreezer(bc.db)
	}

	if err := bc.loadLastState(); err != nil {
		return nil, err
	}

	// Make sure the state associated with the block is available
	head := bc.CurrentBlock()
	if !bc.HasState(head.Root()) {
		if head.NumberU64() == 0 {
			// The genesis state is missing, which is only possible in the path-based
			// scheme. This situation occurs when the initial state sync is not finished
			// yet, or the chain head is rewound below the pivot point. In both scenarios,
			// there is no possible recovery approach except for rerunning a snap sync.
			// Do nothing here until the state syncer picks it up.
			log.Info("Genesis state is missing, wait state sync")
		} else {
			// Head state is missing, before the state recovery, find out the
			// disk layer point of snapshot(if it's enabled). Make sure the
			// rewound point is lower than disk layer.
			var diskRoot common.Hash
			if bc.cacheConfig.SnapshotLimit > 0 {
				diskRoot = rawdb.ReadSnapshotRoot(bc.db)
			}
			if diskRoot != (common.Hash{}) {
				log.Warn("Head state missing, repairing", "number", head.Number(), "hash", head.Hash(), "snaproot", diskRoot)

				snapDisk, err := bc.setHeadBeyondRoot(head.NumberU64(), 0, diskRoot, true)
				if err != nil {
					return nil, err
				}
				// Chain rewound, persist old snapshot number to indicate recovery procedure
				if snapDisk != 0 {
					rawdb.WriteSnapshotRecoveryNumber(bc.db, snapDisk)
				}
			} else {
				log.Warn("Head state missing, repairing", "number", head.Number(), "hash", head.Hash())
				if _, err := bc.setHeadBeyondRoot(head.NumberU64(), 0, common.Hash{}, true); err != nil {
					return nil, err
				}
			}
		}
	}

	// Ensure that a previous crash in SetHead doesn't leave extra ancients
	if frozen, err := bc.db.Ancients(); err == nil && frozen > 0 {
		var (
			needRewind bool
			low        uint64
		)
		// The head full block may be rolled back to a very low height due to
		// blockchain repair. If the head full block is even lower than the ancient
		// chain, truncate the ancient store.
		fullBlock := bc.CurrentBlock()
		if fullBlock != nil && fullBlock.Hash() != bc.genesisBlock.Hash() && fullBlock.NumberU64() < frozen-1 {
			needRewind = true
			low = fullBlock.NumberU64()
		}
		// In fast sync, it may happen that ancient data has been written to the
		// ancient store, but the LastFastBlock has not been updated, truncate the
		// extra data here.
		fastBlock := bc.CurrentFastBlock()
		if fastBlock != nil && fastBlock.NumberU64() < frozen-1 {
			needRewind = true
			if fastBlock.NumberU64() < low || low == 0 {
				low = fastBlock.NumberU64()
			}
		}
		if needRewind {
			log.Error("Truncating ancient chain", "from", bc.CurrentHeader().Number.Uint64(), "to", low)
			if err := bc.SetHead(low); err != nil {
				return nil, err
			}
		}
	}
	// The first thing the node will do is reconstruct the verification data for
	// the head block (cuckoo cache or clique voting snapshot). Might as well do
	// it in advance.
	bc.engine.VerifyHeader(bc, bc.CurrentHeader(), true)

	// Check the current state of the block hashes and make sure that we do not have any of the bad blocks in our chain
	for hash := range BadHashes {
		if header := bc.GetHeaderByHash(hash); header != nil {
			// get the canonical block corresponding to the offending header's number
			headerByNumber := bc.GetHeaderByNumber(header.Number.Uint64())
			// make sure the headerByNumber (if present) is in our current canonical chain
			if headerByNumber != nil && headerByNumber.Hash() == header.Hash() {
				log.Error("Found bad hash, rewinding chain", "number", header.Number, "hash", header.ParentHash)
				if err := bc.SetHead(header.Number.Uint64() - 1); err != nil {
					return nil, err
				}
				log.Error("Chain rewind was successful, resuming normal operation")
			}
		}
	}

	// Load any existing snapshot, regenerating it if loading failed
	if bc.cacheConfig.SnapshotLimit > 0 {
		// If the chain was rewound past the snapshot persistent layer (causing
		// a recovery block number to be persisted to disk), check if we're still
		// in recovery mode and in that case, don't invalidate the snapshot on a
		// head mismatch.
		var recover bool

		head := bc.CurrentBlock()
		if layer := rawdb.ReadSnapshotRecoveryNumber(bc.db); layer != nil && *layer > head.NumberU64() {
			log.Warn("Enabling snapshot recovery", "chainhead", head.NumberU64(), "diskbase", *layer)
			recover = true
		}
		snapconfig := snapshot.Config{
			CacheSize:  bc.cacheConfig.SnapshotLimit,
			Recovery:   recover,
			NoBuild:    bc.cacheConfig.SnapshotNoBuild,
			AsyncBuild: !bc.cacheConfig.SnapshotWait,
		}
		bc.snaps, _ = snapshot.New(snapconfig, bc.db, bc.stateCache.TrieDB(), head.Root())

		bc.stateCache.SetSnapshot(bc.snaps)
	}

	// Start future block processor.
	bc.wg.Add(1)
	go bc.updateFutureBlocks()

	// If periodic cache journal is required, spin it up.
	if bc.cacheConfig.TrieCleanRejournal > 0 {
		if bc.cacheConfig.TrieCleanRejournal < time.Minute {
			log.Warn("Sanitizing invalid trie cache journal time", "provided", bc.cacheConfig.TrieCleanRejournal, "updated", time.Minute)
			bc.cacheConfig.TrieCleanRejournal = time.Minute
		}
		triedb := bc.stateCache.TrieDB()
		bc.wg.Add(1)
		go func() {
			defer bc.wg.Done()
			triedb.SaveCachePeriodically(bc.cacheConfig.TrieCleanJournal, bc.cacheConfig.TrieCleanRejournal, bc.quit)
		}()
	}
	// Start tx indexer/unindexer if required.
	if txLookupLimit != nil {
		bc.txIndexer = newTxIndexer(*txLookupLimit, bc)
	}
	return bc, nil
}

// empty returns an indicator whether the blockchain is empty.
// Note, it's a special case that we connect a non-empty ancient
// database with an empty node, so that we can plugin the ancient
// into node seamlessly.
func (bc *BlockChain) empty() bool {
	genesis := bc.genesisBlock.Hash()
	for _, hash := range []common.Hash{rawdb.ReadHeadBlockHash(bc.db), rawdb.ReadHeadHeaderHash(bc.db), rawdb.ReadHeadFastBlockHash(bc.db)} {
		if hash != genesis {
			return false
		}
	}
	return true
}

// loadLastState loads the last known chain state from the database. This method
// assumes that the chain manager mutex is held.
func (bc *BlockChain) loadLastState() error {
	// Restore the last known head block
	head := rawdb.ReadHeadBlockHash(bc.db)
	if head == (common.Hash{}) {
		// Corrupt or empty database, init from scratch
		log.Warn("Empty database, resetting chain")
		return bc.Reset()
	}
	// Make sure the entire head block is available
	currentHeader := bc.GetHeaderByHash(head)
	if currentHeader == nil {
		// Corrupt or empty database, init from scratch
		log.Warn("Head header missing, resetting chain", "hash", head)
		return bc.Reset()
	}

	var currentBlock *types.Block
	if cmp := currentHeader.Number.Cmp(new(big.Int)); cmp == 1 {
		// Make sure the entire head block is available.
		currentBlock = bc.GetBlockByHash(head)
	} else if cmp == 0 {
		// On a pruned node the block body might not be available. But a pruned
		// block should never be the head block. The only exception is when, as
		// a last resort, chain is reset to genesis.
		currentBlock = bc.genesisBlock
	}
	if currentBlock == nil {
		// Corrupt or empty database, init from scratch
		log.Warn("Head block missing, resetting chain", "hash", head)
		return bc.Reset()
	}
	// Everything seems to be fine, set as the head block
	bc.currentBlock.Store(currentBlock)
	headBlockGauge.Update(int64(currentBlock.NumberU64()))

	// Restore the last known head header
	if head := rawdb.ReadHeadHeaderHash(bc.db); head != (common.Hash{}) {
		if header := bc.GetHeaderByHash(head); header != nil {
			currentHeader = header
		}
	}
	bc.hc.SetCurrentHeader(currentHeader)

	// Initialize history pruning.
	latest := max(currentBlock.NumberU64(), currentHeader.Number.Uint64())
	if err := bc.initializeHistoryPruning(latest); err != nil {
		return err
	}

	// Restore the last known head fast block
	bc.currentFastBlock.Store(currentBlock)
	headFastBlockGauge.Update(int64(currentBlock.NumberU64()))

	if head := rawdb.ReadHeadFastBlockHash(bc.db); head != (common.Hash{}) {
		if block := bc.GetBlockByHash(head); block != nil {
			bc.currentFastBlock.Store(block)
			headFastBlockGauge.Update(int64(block.NumberU64()))
		}
	}

	if head := rawdb.ReadFinalizedBlockHash(bc.db); head != (common.Hash{}) {
		if block := bc.GetBlockByHash(head); block != nil {
			bc.currentFinalizedBlock.Store(block)
			headFinalizedBlockGauge.Update(int64(block.NumberU64()))
			bc.currentSafeBlock.Store(block)
			headSafeBlockGauge.Update(int64(block.NumberU64()))
		}
	}
	// Issue a status log for the user
	currentFastBlock := bc.CurrentFastBlock()
	currentFinalizedBlock := bc.CurrentFinalizedBlock()

	headerTd := bc.GetTd(currentHeader.Hash(), currentHeader.Number.Uint64())
	blockTd := bc.GetTd(currentBlock.Hash(), currentBlock.NumberU64())
	fastTd := bc.GetTd(currentFastBlock.Hash(), currentFastBlock.NumberU64())

	log.Info("Loaded most recent local header", "number", currentHeader.Number, "hash", currentHeader.Hash(), "td", headerTd, "age", common.PrettyAge(time.Unix(int64(currentHeader.Time), 0)))
	log.Info("Loaded most recent local full block", "number", currentBlock.Number(), "hash", currentBlock.Hash(), "td", blockTd, "age", common.PrettyAge(time.Unix(int64(currentBlock.Time()), 0)))
	log.Info("Loaded most recent local fast block", "number", currentFastBlock.Number(), "hash", currentFastBlock.Hash(), "td", fastTd, "age", common.PrettyAge(time.Unix(int64(currentFastBlock.Time()), 0)))
	if currentFinalizedBlock != nil {
		finalTd := bc.GetTd(currentFinalizedBlock.Hash(), currentFinalizedBlock.NumberU64())
		log.Info("Loaded most recent local finalized block", "number", currentFinalizedBlock.Number(), "hash", currentFinalizedBlock.Hash(), "td", finalTd, "age", common.PrettyAge(time.Unix(int64(currentFinalizedBlock.Time()), 0)))
	}

	if pivot := rawdb.ReadLastPivotNumber(bc.db); pivot != nil {
		log.Info("Loaded last fast-sync pivot marker", "number", *pivot)
	}
	if pruning := bc.historyPrunePoint.Load(); pruning != nil {
		log.Info("Chain history is pruned", "earliest", pruning.BlockNumber, "hash", pruning.BlockHash)
	}
	return nil
}

// initializeHistoryPruning sets bc.historyPrunePoint.
func (bc *BlockChain) initializeHistoryPruning(latest uint64) error {
	freezerTail, _ := bc.db.Tail()

	switch bc.cacheConfig.ChainHistoryMode {
	case history.KeepAll:
		if freezerTail == 0 {
			return nil
		}
		// The database was pruned somehow, so we need to figure out if it's a known
		// configuration or an error.
		predefinedPoint := history.PrunePoints[bc.genesisBlock.Hash()]
		if predefinedPoint == nil || freezerTail != predefinedPoint.BlockNumber {
			log.Error("Chain history database is pruned with unknown configuration", "tail", freezerTail)
			return fmt.Errorf("unexpected database tail")
		}
		bc.historyPrunePoint.Store(predefinedPoint)
		return nil

	case history.KeepPostMerge:
		if freezerTail == 0 && latest != 0 {
			// This is the case where a user is trying to run with --history.chain
			// postmerge directly on an existing DB. We could just trigger the pruning
			// here, but it'd be a bit dangerous since they may not have intended this
			// action to happen. So just tell them how to do it.
			log.Error(fmt.Sprintf("Chain history mode is configured as %q, but database is not pruned.", bc.cacheConfig.ChainHistoryMode.String()))
			log.Error(fmt.Sprintf("Run 'geth prune-history' to prune pre-merge history."))
			return fmt.Errorf("history pruning requested via configuration")
		}
		predefinedPoint := history.PrunePoints[bc.genesisBlock.Hash()]
		if predefinedPoint == nil {
			log.Error("Chain history pruning is not supported for this network", "genesis", bc.genesisBlock.Hash())
			return fmt.Errorf("history pruning requested for unknown network")
		} else if freezerTail > 0 && freezerTail != predefinedPoint.BlockNumber {
			log.Error("Chain history database is pruned to unknown block", "tail", freezerTail)
			return fmt.Errorf("unexpected database tail")
		}
		bc.historyPrunePoint.Store(predefinedPoint)
		return nil

	default:
		return fmt.Errorf("invalid history mode: %d", bc.cacheConfig.ChainHistoryMode)
	}
}

// SetHead rewinds the local chain to a new head. Depending on whether the node
// was fast synced or full synced and in which state, the method will try to
// delete minimal data from disk whilst retaining chain consistency.
func (bc *BlockChain) SetHead(head uint64) error {
	if _, err := bc.setHeadBeyondRoot(head, 0, common.Hash{}, false); err != nil {
		return err
	}
	// Send chain head event to update the transaction pool
	if block := bc.CurrentBlock(); block == nil {
		if block.Number().Uint64() > 0 {
			// This should never happen. In practice, previously currentBlock
			// contained the entire block whereas now only a "marker", so there
			// is an ever so slight chance for a race we should handle.
			log.Error("Current block not found in database", "block", block.Number(), "hash", block.Hash())
			return fmt.Errorf("current block missing: #%d [%x..]", block.Number(), block.Hash().Bytes()[:4])
		}
	} else {
		bc.chainHeadFeed.Send(ChainHeadEvent{Header: block.Header()})
	}
	return nil
}

// SetHeadWithTimestamp rewinds the local chain to a new head that has at max
// the given timestamp. Depending on whether the node was fast synced or full
// synced and in which state, the method will try to delete minimal data from
// disk whilst retaining chain consistency.
func (bc *BlockChain) SetHeadWithTimestamp(timestamp uint64) error {
	if _, err := bc.setHeadBeyondRoot(0, timestamp, common.Hash{}, false); err != nil {
		return err
	}
	// Send chain head event to update the transaction pool
	if block := bc.CurrentBlock(); block == nil {
		if block.Number().Uint64() > 0 {
			// This should never happen. In practice, previously currentBlock
			// contained the entire block whereas now only a "marker", so there
			// is an ever so slight chance for a race we should handle.
			log.Error("Current block not found in database", "block", block.Number(), "hash", block.Hash())
			return fmt.Errorf("current block missing: #%d [%x..]", block.Number(), block.Hash().Bytes()[:4])
		}
	} else {
		bc.chainHeadFeed.Send(ChainHeadEvent{Header: block.Header()})
	}
	return nil
}

// SetFinalized sets the finalized block.
func (bc *BlockChain) SetFinalized(block *types.Block) {
	bc.currentFinalizedBlock.Store(block)
	if block != nil {
		rawdb.WriteFinalizedBlockHash(bc.db, block.Hash())
		headFinalizedBlockGauge.Update(int64(block.NumberU64()))
	} else {
		rawdb.WriteFinalizedBlockHash(bc.db, common.Hash{})
		headFinalizedBlockGauge.Update(0)
	}
}

// SetSafe sets the safe block.
func (bc *BlockChain) SetSafe(block *types.Block) {
	bc.currentSafeBlock.Store(block)
	if block != nil {
		headSafeBlockGauge.Update(int64(block.NumberU64()))
	} else {
		headSafeBlockGauge.Update(0)
	}
}

// setHeadBeyondRoot rewinds the local chain to a new head with the extra condition
// that the rewind must pass the specified state root. This method is meant to be
// used when rewiding with snapshots enabled to ensure that we go back further than
// persistent disk layer. Depending on whether the node was fast synced or full, and
// in which state, the method will try to delete minimal data from disk whilst
// retaining chain consistency.
//
// The method returns the block number where the requested root cap was found.
func (bc *BlockChain) setHeadBeyondRoot(head uint64, time uint64, root common.Hash, repair bool) (uint64, error) {
	if !bc.chainmu.TryLock() {
		return 0, errChainStopped
	}
	defer bc.chainmu.Unlock()

	// Track the block number of the requested root hash
	var (
		rootNumber uint64 // (no root == always 0)

		// Retrieve the last pivot block to short circuit rollbacks beyond it and the
		// current freezer limit to start nuking id underflown
		pivot = rawdb.ReadLastPivotNumber(bc.db)
	)

	updateFn := func(db ctxcdb.KeyValueWriter, header *types.Header) (*types.Header, bool) {
		// Rewind the block chain, ensuring we don't end up with a stateless head
		// block. Note, depth equality is permitted to allow using SetHead as a
		// chain reparation mechanism without deleting any data!
		if currentBlock := bc.CurrentBlock(); currentBlock != nil && header.Number.Uint64() <= currentBlock.NumberU64() {
			newHeadBlock := bc.GetBlock(header.Hash(), header.Number.Uint64())
			if newHeadBlock == nil {
				log.Error("Gap in the chain, rewinding to genesis", "number", header.Number, "hash", header.Hash())
				newHeadBlock = bc.genesisBlock
			} else {
				// Block exists, keep rewinding until we find one with state,
				// keeping rewinding until we exceed the optional threshold
				// root hash
				beyondRoot := (root == common.Hash{}) // Flag whether we're beyond the requested root (no root, always true)

				for {
					// If a root threshold was requested but not yet crossed, check
					if root != (common.Hash{}) && !beyondRoot && newHeadBlock.Root() == root {
						beyondRoot, rootNumber = true, newHeadBlock.NumberU64()
					}
					if !bc.HasState(newHeadBlock.Root()) {
						log.Trace("Block state missing, rewinding further", "number", newHeadBlock.NumberU64(), "hash", newHeadBlock.Hash())
						if pivot == nil || newHeadBlock.NumberU64() > *pivot {
							parent := bc.GetBlock(newHeadBlock.ParentHash(), newHeadBlock.NumberU64()-1)
							if parent != nil {
								newHeadBlock = parent
								continue
							}
							log.Error("Missing block in the middle, aiming genesis", "number", newHeadBlock.NumberU64()-1, "hash", newHeadBlock.ParentHash())
							newHeadBlock = bc.genesisBlock
						} else {
							log.Trace("Rewind passed pivot, aiming genesis", "number", newHeadBlock.NumberU64(), "hash", newHeadBlock.Hash(), "pivot", *pivot)
							newHeadBlock = bc.genesisBlock
						}
					}

					if beyondRoot || newHeadBlock.NumberU64() == 0 {
						if newHeadBlock.NumberU64() == 0 {
							// Recommit the genesis state into disk in case the rewinding destination
							// is genesis block and the relevant state is gone. In the future this
							// rewinding destination can be the earliest block stored in the chain
							// if the historical chain pruning is enabled. In that case the logic
							// needs to be improved here.
							if !bc.HasState(bc.genesisBlock.Root()) {
								//if err := CommitGenesisState(bc.db, bc.triedb, bc.genesisBlock.Hash()); err != nil {
								//	log.Crit("Failed to commit genesis state", "err", err)
								//}
								log.Debug("Recommitted genesis state to disk")
							}
						}
						log.Debug("Rewound to block with state", "number", newHeadBlock.NumberU64(), "hash", newHeadBlock.Hash())
						break
					}
					log.Debug("Skipping block with threshold state", "number", newHeadBlock.NumberU64(), "hash", newHeadBlock.Hash(), "root", newHeadBlock.Root())
					newHeadBlock = bc.GetBlock(newHeadBlock.ParentHash(), newHeadBlock.NumberU64()-1) // Keep rewinding
				}
			}
			rawdb.WriteHeadBlockHash(db, newHeadBlock.Hash())

			// Degrade the chain markers if they are explicitly reverted.
			// In theory we should update all in-memory markers in the
			// last step, however the direction of SetHead is from high
			// to low, so it's safe the update in-memory markers directly.
			bc.currentBlock.Store(newHeadBlock)
			headBlockGauge.Update(int64(newHeadBlock.NumberU64()))
		}
		// Rewind the fast block in a simpleton way to the target head
		if currentFastBlock := bc.CurrentFastBlock(); currentFastBlock != nil && header.Number.Uint64() < currentFastBlock.NumberU64() {
			newHeadFastBlock := bc.GetBlock(header.Hash(), header.Number.Uint64())
			// If either blocks reached nil, reset to the genesis state
			if newHeadFastBlock == nil {
				newHeadFastBlock = bc.genesisBlock
			}
			rawdb.WriteHeadFastBlockHash(db, newHeadFastBlock.Hash())

			// Degrade the chain markers if they are explicitly reverted.
			// In theory we should update all in-memory markers in the
			// last step, however the direction of SetHead is from high
			// to low, so it's safe the update in-memory markers directly.
			bc.currentFastBlock.Store(newHeadFastBlock)
			headFastBlockGauge.Update(int64(newHeadFastBlock.NumberU64()))
		}
		var (
			headHeader = bc.CurrentBlock().Header()
			headNumber = headHeader.Number.Uint64()
		)
		// If setHead underflown the freezer threshold and the block processing
		// intent afterwards is full block importing, delete the chain segment
		// between the stateful-block and the sethead target.
		var wipe bool
		frozen, _ := bc.db.Ancients()
		if headNumber+1 < frozen {
			wipe = pivot == nil || headNumber >= *pivot
		}
		return headHeader, wipe // Only force wipe if full synced
	}
	// Rewind the header chain, deleting all block bodies until then
	delFn := func(db ctxcdb.KeyValueWriter, hash common.Hash, num uint64) {
		// Ignore the error here since light client won't hit this path
		frozen, _ := bc.db.Ancients()
		if num+1 <= frozen {
			// The chain segment, such as the block header, canonical hash,
			// body, and receipt, will be removed from the ancient store
			// in one go.
			//
			// The hash-to-number mapping in the key-value store will be
			// removed by the hc.SetHead function.
		} else {
			// Remove the associated body and receipts from the key-value store.
			// The header, hash-to-number mapping, and canonical hash will be
			// removed by the hc.SetHead function.
			rawdb.DeleteBody(db, hash, num)
			rawdb.DeleteReceipts(db, hash, num)
		}
		// Todo(rjl493456442) txlookup, bloombits, etc
	}
	// If SetHead was only called as a chain reparation method, try to skip
	// touching the header chain altogether, unless the freezer is broken
	if repair {
		if target, force := updateFn(bc.db, bc.CurrentBlock().Header()); force {
			bc.hc.SetHead(target.Number.Uint64(), nil, delFn)
		}
	} else {
		// Rewind the chain to the requested head and keep going backwards until a
		// block with a state is found or fast sync pivot is passed
		if time > 0 {
			log.Warn("Rewinding blockchain to timestamp", "target", time)
			bc.hc.SetHeadWithTimestamp(time, updateFn, delFn)
		} else {
			log.Warn("Rewinding blockchain to block", "target", head)
			bc.hc.SetHead(head, updateFn, delFn)
		}
	}
	// Clear out any stale content from the caches
	bc.bodyCache.Purge()
	bc.bodyRLPCache.Purge()
	bc.receiptsCache.Purge()
	bc.blockCache.Purge()
	bc.txLookupCache.Purge()
	bc.futureBlocks.Purge()
	// Clear safe block, finalized block if needed
	if safe := bc.CurrentSafeBlock(); safe != nil && head < safe.NumberU64() {
		log.Warn("SetHead invalidated safe block")
		bc.SetSafe(nil)
	}
	if finalized := bc.CurrentFinalizedBlock(); finalized != nil && head < finalized.NumberU64() {
		log.Error("SetHead invalidated finalized block")
		bc.SetFinalized(nil)
	}

	return rootNumber, bc.loadLastState()
}

// FastSyncCommitHead sets the current head block to the one defined by the hash
// irrelevant what the chain contents were prior.
func (bc *BlockChain) FastSyncCommitHead(hash common.Hash) error {
	// Make sure that both the block as well at its state trie exists
	block := bc.GetBlockByHash(hash)
	if block == nil {
		return fmt.Errorf("non existent block [%x…]", hash[:4])
	}
	if _, err := trie.NewStateTrie(trie.TrieID(block.Root()), bc.stateCache.TrieDB()); err != nil {
		return err
	}

	// If all checks out, manually set the head block.
	if !bc.chainmu.TryLock() {
		return errChainStopped
	}
	bc.currentBlock.Store(block)
	headBlockGauge.Update(int64(block.NumberU64()))
	bc.chainmu.Unlock()

	// Destroy any existing state snapshot and regenerate it in the background
	if bc.snaps != nil {
		bc.snaps.Rebuild(block.Root())
	}
	log.Info("Committed new head block", "number", block.Number(), "hash", hash)
	return nil
}

// Reset purges the entire blockchain, restoring it to its genesis state.
func (bc *BlockChain) Reset() error {
	return bc.ResetWithGenesisBlock(bc.genesisBlock)
}

// ResetWithGenesisBlock purges the entire blockchain, restoring it to the
// specified genesis state.
func (bc *BlockChain) ResetWithGenesisBlock(genesis *types.Block) error {
	// Dump the entire block chain and purge the caches
	if err := bc.SetHead(0); err != nil {
		return err
	}
	if !bc.chainmu.TryLock() {
		return errChainStopped
	}
	defer bc.chainmu.Unlock()

	// Prepare the genesis block and reinitialise the chain
	batch := bc.db.NewBatch()
	rawdb.WriteTd(batch, genesis.Hash(), genesis.NumberU64(), genesis.Difficulty())
	rawdb.WriteBlock(batch, genesis)
	if err := batch.Write(); err != nil {
		log.Crit("Failed to write genesis block", "err", err)
	}
	bc.writeHeadBlock(genesis)

	// Last update all in-memory chain markers
	bc.genesisBlock = genesis
	bc.currentBlock.Store(bc.genesisBlock)
	headBlockGauge.Update(int64(bc.genesisBlock.NumberU64()))
	bc.hc.SetGenesis(bc.genesisBlock.Header())
	bc.hc.SetCurrentHeader(bc.genesisBlock.Header())
	bc.currentFastBlock.Store(bc.genesisBlock)
	headFastBlockGauge.Update(int64(bc.genesisBlock.NumberU64()))

	// Reset history pruning status.
	return bc.initializeHistoryPruning(0)
}

// Export writes the active chain to the given writer.
func (bc *BlockChain) Export(w io.Writer) error {
	return bc.ExportN(w, uint64(0), bc.CurrentBlock().NumberU64())
}

// ExportN writes a subset of the active chain to the given writer.
func (bc *BlockChain) ExportN(w io.Writer, first uint64, last uint64) error {
	if first > last {
		return fmt.Errorf("export failed: first (%d) is greater than last (%d)", first, last)
	}
	log.Info("Exporting batch of blocks", "count", last-first+1)

	var (
		parentHash common.Hash
		start      = time.Now()
		reported   = time.Now()
	)
	for nr := first; nr <= last; nr++ {
		block := bc.GetBlockByNumber(nr)
		if block == nil {
			return fmt.Errorf("export failed on #%d: not found", nr)
		}
		if nr > first && block.ParentHash() != parentHash {
			return fmt.Errorf("export failed: chain reorg during export")
		}
		parentHash = block.Hash()
		if err := block.EncodeRLP(w); err != nil {
			return err
		}
		if time.Since(reported) >= statsReportLimit {
			log.Info("Exporting blocks", "exported", block.NumberU64()-first, "elapsed", common.PrettyDuration(time.Since(start)))
			reported = time.Now()
		}
	}
	return nil
}

// writeHeadBlock injects a new head block into the current block chain. This method
// assumes that the block is indeed a true head. It will also reset the head
// header and the head fast sync block to this very same block if they are older
// or if they are on a different side chain.
//
// Note, this function assumes that the `mu` mutex is held!
func (bc *BlockChain) writeHeadBlock(block *types.Block) {
	// Add the block to the canonical chain number scheme and mark as the head
	batch := bc.db.NewBatch()
	rawdb.WriteHeadHeaderHash(batch, block.Hash())
	rawdb.WriteHeadFastBlockHash(batch, block.Hash())
	rawdb.WriteCanonicalHash(batch, block.Hash(), block.NumberU64())
	rawdb.WriteTxLookupEntriesByBlock(batch, block)
	rawdb.WriteHeadBlockHash(batch, block.Hash())

	// Flush the whole batch into the disk, exit the node if failed
	if err := batch.Write(); err != nil {
		log.Crit("Failed to update chain indexes and markers", "err", err)
	}
	// Update all in-memory chain markers in the last step
	bc.hc.SetCurrentHeader(block.Header())

	bc.currentFastBlock.Store(block)
	headFastBlockGauge.Update(int64(block.NumberU64()))

	bc.currentBlock.Store(block)
	headBlockGauge.Update(int64(block.NumberU64()))
}

// stopWithoutSaving stops the blockchain service. If any imports are currently in progress
// it will abort them using the procInterrupt. This method stops all running
// goroutines, but does not do all the post-stop work of persisting data.
// OBS! It is generally recommended to use the Stop method!
// This method has been exposed to allow tests to stop the blockchain while simulating
// a crash.
func (bc *BlockChain) stopWithoutSaving() {
	if !bc.stopping.CompareAndSwap(false, true) {
		return
	}

	// Signal shutdown tx indexer.
	if bc.txIndexer != nil {
		bc.txIndexer.close()
	}

	// Unsubscribe all subscriptions registered from blockchain.
	bc.scope.Close()

	// Signal shutdown to all goroutines.
	close(bc.quit)
	bc.StopInsert()

	// Now wait for all chain modifications to end and persistent goroutines to exit.
	//
	// Note: Close waits for the mutex to become available, i.e. any running chain
	// modification will have exited when Close returns. Since we also called StopInsert,
	// the mutex should become available quickly. It cannot be taken again after Close has
	// returned.
	bc.chainmu.Close()
	bc.wg.Wait()
}

// Stop stops the blockchain service. If any imports are currently in progress
// it will abort them using the procInterrupt.
func (bc *BlockChain) Stop() {
	bc.stopWithoutSaving()

	// Ensure that the entirety of the state snapshot is journalled to disk.
	var snapBase common.Hash
	if bc.snaps != nil {
		var err error
		if bc.writeLegacyJournal {
			if snapBase, err = bc.snaps.LegacyJournal(bc.CurrentBlock().Root()); err != nil {
				log.Error("Failed to journal state snapshot", "err", err)
			}
		} else {
			if snapBase, err = bc.snaps.Journal(bc.CurrentBlock().Root()); err != nil {
				log.Error("Failed to journal state snapshot", "err", err)
			}
		}
		bc.snaps.Release()
	}

	// Ensure the state of a recent block is also stored to disk before exiting.
	// We're writing three different states to catch different restart scenarios:
	//  - HEAD:     So we don't need to reprocess any blocks in the general case
	//  - HEAD-1:   So we don't do large reorgs if our HEAD becomes an uncle
	//  - HEAD-127: So we have a hard limit on the number of blocks reexecuted
	if !bc.cacheConfig.TrieDirtyDisabled {
		triedb := bc.stateCache.TrieDB()

		for _, offset := range []uint64{0, 1, state.TriesInMemory - 1} {
			if number := bc.CurrentBlock().NumberU64(); number > offset {
				num := number - offset
				recent := bc.GetBlockByNumber(num)
				if recent == nil {
					log.Error("Failed to get block", "num", num)
					continue
				}

				log.Info("Writing cached state to disk", "block", recent.Number(), "hash", recent.Hash(), "root", recent.Root())
				if err := triedb.Commit(recent.Root(), true); err != nil {
					log.Error("Failed to commit recent state trie", "err", err)
				}
			}
		}
		if snapBase != (common.Hash{}) {
			log.Info("Writing snapshot state to disk", "root", snapBase)
			if err := triedb.Commit(snapBase, true); err != nil {
				log.Error("Failed to commit recent state trie", "err", err)
			}
		}
		for !bc.triegc.Empty() {
			triedb.Dereference(bc.triegc.PopItem())
		}
		if size, _ := triedb.Size(); size != 0 {
			log.Error("Dangling trie nodes after full cleanup")
		}
	}
	// Flush the collected preimages to disk
	if err := bc.stateCache.TrieDB().CommitPreimages(); err != nil {
		log.Error("Failed to commit trie preimages", "err", err)
	}
	// Ensure all live cached entries be saved into disk, so that we can skip
	// cache warmup when node restarts.
	if bc.cacheConfig.TrieCleanJournal != "" {
		triedb := bc.stateCache.TrieDB()
		triedb.SaveCache(bc.cacheConfig.TrieCleanJournal)
	}
	log.Info("Blockchain stopped")
}

// StopInsert interrupts all insertion methods, causing them to return
// errInsertionInterrupted as soon as possible. Insertion is permanently disabled after
// calling this method.
func (bc *BlockChain) StopInsert() {
	bc.procInterrupt.Store(true)
}

// insertStopped returns true after StopInsert has been called.
func (bc *BlockChain) insertStopped() bool {
	return bc.procInterrupt.Load()
}

func (bc *BlockChain) procFutureBlocks() {
	blocks := make([]*types.Block, 0, bc.futureBlocks.Len())
	for _, hash := range bc.futureBlocks.Keys() {
		if block, exist := bc.futureBlocks.Peek(hash); exist {
			blocks = append(blocks, block)
		}
	}
	if len(blocks) > 0 {
		slices.SortFunc(blocks, func(a, b *types.Block) int {
			return a.Number().Cmp(b.Number())
		})
		// Insert one by one as chain insertion needs contiguous ancestry between blocks
		for i := range blocks {
			bc.InsertChain(blocks[i : i+1])
		}
	}
}

// WriteStatus status of write
type WriteStatus byte

const (
	NonStatTy WriteStatus = iota
	CanonStatTy
	SideStatTy
)

// numberHash is just a container for a number and a hash, to represent a block
type numberHash struct {
	number uint64
	hash   common.Hash
}

// InsertReceiptChain attempts to complete an already existing header chain with
// transaction and receipt data.
func (bc *BlockChain) InsertReceiptChain(blockChain types.Blocks, receiptChain []types.Receipts, ancientLimit uint64) (int, error) {
	// We don't require the chainMu here since we want to maximize the
	// concurrency of header insertion and receipt insertion.
	bc.wg.Add(1)
	defer bc.wg.Done()

	var (
		ancientBlocks, liveBlocks     types.Blocks
		ancientReceipts, liveReceipts []types.Receipts
	)
	// Do a sanity check that the provided chain is actually ordered and linked
	for i, block := range blockChain {
		if i != 0 {
			prev := blockChain[i-1]
			if block.NumberU64() != prev.NumberU64()+1 || block.ParentHash() != prev.Hash() {
				log.Error("Non contiguous receipt insert",
					"number", block.Number(), "hash", block.Hash(), "parent", block.ParentHash(),
					"prevnumber", prev.Number(), "prevhash", prev.Hash())
				return 0, fmt.Errorf("non contiguous insert: item %d is #%d [%x..], item %d is #%d [%x..] (parent [%x..])",
					i-1, prev.NumberU64(), prev.Hash().Bytes()[:4],
					i, block.NumberU64(), block.Hash().Bytes()[:4], block.ParentHash().Bytes()[:4])
			}
		}
		if block.NumberU64() <= ancientLimit {
			ancientBlocks, ancientReceipts = append(ancientBlocks, block), append(ancientReceipts, receiptChain[i])
		} else {
			liveBlocks, liveReceipts = append(liveBlocks, block), append(liveReceipts, receiptChain[i])
		}
	}

	var (
		stats = struct{ processed, ignored int32 }{}
		start = time.Now()
		size  = int64(0)
	)

	// updateHead updates the head fast sync block if the inserted blocks are better
	// and returns an indicator whether the inserted blocks are canonical.
	updateHead := func(head *types.Block) bool {
		if !bc.chainmu.TryLock() {
			return false
		}
		defer bc.chainmu.Unlock()

		// Rewind may have occurred, skip in that case.
		if bc.CurrentHeader().Number.Cmp(head.Number()) >= 0 {
			currentFastBlock, td := bc.CurrentFastBlock(), bc.GetTd(head.Hash(), head.NumberU64())
			if bc.GetTd(currentFastBlock.Hash(), currentFastBlock.NumberU64()).Cmp(td) < 0 {
				rawdb.WriteHeadFastBlockHash(bc.db, head.Hash())
				bc.currentFastBlock.Store(head)
				headFastBlockGauge.Update(int64(head.NumberU64()))
				return true
			}
		}
		return false
	}

	// writeAncient writes blockchain and corresponding receipt chain into ancient store.
	//
	// this function only accepts canonical chain data. All side chain will be reverted
	// eventually.
	writeAncient := func(blockChain types.Blocks, receiptChain []types.Receipts) (int, error) {
		first := blockChain[0]
		last := blockChain[len(blockChain)-1]

		// Ensure genesis is in ancients.
		if first.NumberU64() == 1 {
			if frozen, _ := bc.db.Ancients(); frozen == 0 {
				td := bc.genesisBlock.Difficulty()
				writeSize, err := rawdb.WriteAncientBlocks(bc.db, []*types.Block{bc.genesisBlock}, []types.Receipts{nil}, td)
				if err != nil {
					log.Error("Error writing genesis to ancients", "err", err)
					return 0, err
				}
				size += writeSize
				log.Info("Wrote genesis to ancients")
			}
		}
		// Before writing the blocks to the ancients, we need to ensure that
		// they correspond to the what the headerchain 'expects'.
		// We only check the last block/header, since it's a contiguous chain.
		if !bc.HasHeader(last.Hash(), last.NumberU64()) {
			return 0, fmt.Errorf("containing header #%d [%x..] unknown", last.Number(), last.Hash().Bytes()[:4])
		}

		// Write all chain data to ancients.
		td := bc.GetTd(first.Hash(), first.NumberU64())
		writeSize, err := rawdb.WriteAncientBlocks(bc.db, blockChain, receiptChain, td)
		if err != nil {
			log.Error("Error importing chain data to ancients", "err", err)
			return 0, err
		}
		size += writeSize

		// Sync the ancient store explicitly to ensure all data has been flushed to disk.
		if err := bc.db.SyncAncient(); err != nil {
			return 0, err
		}
		// Update the current fast block because all block data is now present in DB.
		previousFastBlock := bc.CurrentFastBlock().NumberU64()
		if !updateHead(blockChain[len(blockChain)-1]) {
			// We end up here if the header chain has reorg'ed, and the blocks/receipts
			// don't match the canonical chain.
			if _, err := bc.db.TruncateHead(previousFastBlock + 1); err != nil {
				log.Error("Can't truncate ancient store after failed insert", "err", err)
			}
			return 0, errSideChainReceipts
		}

		// Delete block data from the main database.
		var (
			batch       = bc.db.NewBatch()
			canonHashes = make(map[common.Hash]struct{}, len(blockChain))
		)
		for _, block := range blockChain {
			canonHashes[block.Hash()] = struct{}{}
			if block.NumberU64() == 0 {
				continue
			}
			rawdb.DeleteCanonicalHash(batch, block.NumberU64())
			rawdb.DeleteBlockWithoutNumber(batch, block.Hash(), block.NumberU64())
		}
		// Delete side chain hash-to-number mappings.
		for _, nh := range rawdb.ReadAllHashesInRange(bc.db, first.NumberU64(), last.NumberU64()) {
			if _, canon := canonHashes[nh.Hash]; !canon {
				rawdb.DeleteHeader(batch, nh.Hash, nh.Number)
			}
		}
		if err := batch.Write(); err != nil {
			return 0, err
		}
		stats.processed += int32(len(blockChain))
		return 0, nil
	}

	// writeLive writes blockchain and corresponding receipt chain into active store.
	writeLive := func(blockChain types.Blocks, receiptChain []types.Receipts) (int, error) {
		var (
			skipPresenceCheck = false
			batch             = bc.db.NewBatch()
		)
		for i, block := range blockChain {
			// Short circuit insertion if shutting down or processing failed
			if bc.insertStopped() {
				return 0, errInsertionInterrupted
			}
			// Short circuit if the owner header is unknown
			if !bc.HasHeader(block.Hash(), block.NumberU64()) {
				return i, fmt.Errorf("containing header #%d [%x…] unknown", block.Number(), block.Hash().Bytes()[:4])
			}
			if !skipPresenceCheck {
				// Ignore if the entire data is already known
				if bc.HasBlock(block.Hash(), block.NumberU64()) {
					stats.ignored++
					continue
				} else {
					// If block N is not present, neither are the later blocks.
					// This should be true, but if we are mistaken, the shortcut
					// here will only cause overwriting of some existing data
					skipPresenceCheck = true
				}
			}
			// Write all the data out into the database
			rawdb.WriteBody(batch, block.Hash(), block.NumberU64(), block.Body())
			rawdb.WriteReceipts(batch, block.Hash(), block.NumberU64(), receiptChain[i])

			// Write everything belongs to the blocks into the database. So that
			// we can ensure all components of body is completed(body, receipts,
			// tx indexes)
			if batch.ValueSize() >= ctxcdb.IdealBatchSize {
				if err := batch.Write(); err != nil {
					return 0, err
				}
				size += int64(batch.ValueSize())
				batch.Reset()
			}
			stats.processed++
		}
		// Write everything belongs to the blocks into the database. So that
		// we can ensure all components of body is completed(body, receipts,
		// tx indexes)
		if batch.ValueSize() > 0 {
			size += int64(batch.ValueSize())
			if err := batch.Write(); err != nil {
				return 0, err
			}
		}
		updateHead(blockChain[len(blockChain)-1])
		return 0, nil
	}

	// Write downloaded chain data and corresponding receipt chain data
	if len(ancientBlocks) > 0 {
		if n, err := writeAncient(ancientBlocks, ancientReceipts); err != nil {
			if err == errInsertionInterrupted {
				return 0, nil
			}
			return n, err
		}
	}
	if len(liveBlocks) > 0 {
		if n, err := writeLive(liveBlocks, liveReceipts); err != nil {
			if err == errInsertionInterrupted {
				return 0, nil
			}
			return n, err
		}
	}
	var (
		head    = blockChain[len(blockChain)-1]
		context = []interface{}{
			"count", stats.processed, "elapsed", common.PrettyDuration(time.Since(start)),
			"number", head.Number(), "hash", head.Hash(), "age", common.PrettyAge(time.Unix(int64(head.Time()), 0)),
			"size", common.StorageSize(size),
		}
	)
	if stats.ignored > 0 {
		context = append(context, []any{"ignored", stats.ignored}...)
	}
	log.Info("Imported new block receipts", context...)

	return 0, nil
}

// writeBlockWithoutState writes only the block and its metadata to the database,
// but does not write any state. This is used to construct competing side forks
// up to the point where they exceed the canonical total difficulty.
func (bc *BlockChain) writeBlockWithoutState(block *types.Block, td *big.Int) (err error) {
	if bc.insertStopped() {
		return errInsertionInterrupted
	}

	batch := bc.db.NewBatch()
	rawdb.WriteTd(batch, block.Hash(), block.NumberU64(), td)
	rawdb.WriteBlock(batch, block)
	if err := batch.Write(); err != nil {
		log.Crit("Failed to write block into disk", "err", err)
	}
	return nil
}

// writeKnownBlock updates the head block flag with a known block
// and introduces chain reorg if necessary.
func (bc *BlockChain) writeKnownBlock(block *types.Block) error {
	current := bc.CurrentBlock()
	if block.ParentHash() != current.Hash() {
		if err := bc.reorg(current.Header(), block.Header()); err != nil {
			return err
		}
	}
	bc.writeHeadBlock(block)
	return nil
}

// WriteBlockWithState writes the block and all associated state to the database.
func (bc *BlockChain) WriteBlockWithState(block *types.Block, receipts []*types.Receipt, logs []*types.Log, state *state.StateDB, emitHeadEvent bool) (status WriteStatus, err error) {
	if !bc.chainmu.TryLock() {
		return NonStatTy, errInsertionInterrupted
	}
	defer bc.chainmu.Unlock()
	return bc.writeBlockWithState(block, receipts, logs, state, emitHeadEvent)
}

// writeBlockWithState writes the block and all associated state to the database,
// but is expects the chain mutex to be held.
func (bc *BlockChain) writeBlockWithState(block *types.Block, receipts []*types.Receipt, logs []*types.Log, statedb *state.StateDB, emitHeadEvent bool) (status WriteStatus, err error) {
	if bc.insertStopped() {
		return NonStatTy, errInsertionInterrupted
	}

	// Calculate the total difficulty of the block
	ptd := bc.GetTd(block.ParentHash(), block.NumberU64()-1)
	if ptd == nil {
		return NonStatTy, consensus.ErrUnknownAncestor
	}
	// Make sure no inconsistent state is leaked during insertion
	currentBlock := bc.CurrentBlock()
	localTd := bc.GetTd(currentBlock.Hash(), currentBlock.NumberU64())
	externTd := new(big.Int).Add(block.Difficulty(), ptd)

	// Irrelevant of the canonical status, write the block itself to the database.
	//
	// Note all the components of block(td, hash->number map, header, body, receipts)
	// should be written atomically. BlockBatch is used for containing all components.
	blockBatch := bc.db.NewBatch()
	rawdb.WriteTd(blockBatch, block.Hash(), block.NumberU64(), externTd)
	rawdb.WriteBlock(blockBatch, block)
	rawdb.WriteReceipts(blockBatch, block.Hash(), block.NumberU64(), receipts)
	rawdb.WritePreimages(blockBatch, statedb.Preimages())
	if err := blockBatch.Write(); err != nil {
		log.Crit("Failed to write block into disk", "err", err)
	}
	// Commit all cached state changes into underlying memory database.
	root, err := statedb.Commit(block.NumberU64(), bc.chainConfig.IsEIP158(block.Number()))
	if err != nil {
		return NonStatTy, err
	}
	triedb := bc.stateCache.TrieDB()

	// If we're running an archive node, always flush
	if bc.cacheConfig.TrieDirtyDisabled {
		if err := triedb.Commit(root, false); err != nil {
			return NonStatTy, err
		}
	} else {
		// Full but not archive node, do proper garbage collection
		triedb.Reference(root, common.Hash{}) // metadata reference to keep trie alive
		bc.triegc.Push(root, -int64(block.NumberU64()))

		if current := block.NumberU64(); current > state.TriesInMemory {
			// If we exceeded our memory allowance, flush matured singleton nodes to disk
			var (
				nodes, imgs = triedb.Size()
				limit       = common.StorageSize(bc.cacheConfig.TrieDirtyLimit) * 1024 * 1024
			)
			if nodes > limit || imgs > 4*1024*1024 {
				triedb.Cap(limit - ctxcdb.IdealBatchSize)
			}
			// Find the next state trie we need to commit
			chosen := current - state.TriesInMemory

			// If we exceeded out time allowance, flush an entire trie to disk
			if bc.gcproc > bc.cacheConfig.TrieTimeLimit {
				// If the header is missing (canonical chain behind), we're reorging a low
				// diff sidechain. Suspend committing until this operation is completed.
				header := bc.GetHeaderByNumber(chosen)
				if header == nil {
					log.Warn("Reorg in progress, trie commit postponed", "number", chosen)
				} else {
					// If we're exceeding limits but haven't reached a large enough memory gap,
					// warn the user that the system is becoming unstable.
					if chosen < bc.lastWrite+state.TriesInMemory && bc.gcproc >= 2*bc.cacheConfig.TrieTimeLimit {
						log.Info("State in memory for too long, committing", "time", bc.gcproc, "allowance", bc.cacheConfig.TrieTimeLimit, "optimum", float64(chosen-bc.lastWrite)/state.TriesInMemory)
					}
					// Flush an entire trie and restart the counters
					triedb.Commit(header.Root, true)
					bc.lastWrite = chosen
					bc.gcproc = 0
				}
			}
			// Garbage collect anything below our required write retention
			for !bc.triegc.Empty() {
				root, number := bc.triegc.Pop()
				if uint64(-number) > chosen {
					bc.triegc.Push(root, number)
					break
				}
				triedb.Dereference(root)
			}
		}
	}
	// If the total difficulty is higher than our known, add it to the canonical chain
	// Second clause in the if statement reduces the vulnerability to selfish mining.
	// Please refer to http://www.cs.cornell.edu/~ie53/publications/btcProcFC.pdf
	reorg := externTd.Cmp(localTd) > 0
	currentBlock = bc.CurrentBlock()
	if !reorg && externTd.Cmp(localTd) == 0 {
		// Split same-difficulty blocks by number, then preferentially select
		// the block generated by the local miner as the canonical block.
		if block.NumberU64() < currentBlock.NumberU64() {
			reorg = true
		} else if block.NumberU64() == currentBlock.NumberU64() {
			var currentPreserve, blockPreserve bool
			if bc.shouldPreserve != nil {
				currentPreserve, blockPreserve = bc.shouldPreserve(currentBlock.Header()), bc.shouldPreserve(block.Header())
			}
			reorg = !currentPreserve && (blockPreserve || mrand.Float64() < 0.5)
		}
	}
	if reorg {
		// Reorganise the chain if the parent is not the head block
		if block.ParentHash() != currentBlock.Hash() {
			if err := bc.reorg(currentBlock.Header(), block.Header()); err != nil {
				return NonStatTy, err
			}
		}
		status = CanonStatTy
	} else {
		status = SideStatTy
	}
	// Set new head.
	if status == CanonStatTy {
		bc.writeHeadBlock(block)
	}
	bc.futureBlocks.Remove(block.Hash())

	if status == CanonStatTy {
		logs := bc.collectLogs(block, false)
		//bc.chainFeed.Send(ChainEvent{Header: block.Header(), Hash: block.Hash(), Logs: logs})
		bc.chainFeed.Send(ChainEvent{Header: block.Header()})
		if len(logs) > 0 {
			bc.logsFeed.Send(logs)
		}
		// In theory we should fire a ChainHeadEvent when we inject
		// a canonical block, but sometimes we can insert a batch of
		// canonicial blocks. Avoid firing too much ChainHeadEvents,
		// we will fire an accumulated ChainHeadEvent and disable fire
		// event here.
		if emitHeadEvent {
			bc.chainHeadFeed.Send(ChainHeadEvent{Header: block.Header()})
		}
	} else {
		bc.chainSideFeed.Send(ChainSideEvent{Header: block.Header()})
	}
	return status, nil
}

// addFutureBlock checks if the block is within the max allowed window to get
// accepted for future processing, and returns an error if the block is too far
// ahead and was not added.
func (bc *BlockChain) addFutureBlock(block *types.Block) error {
	max := uint64(time.Now().Unix() + maxTimeFutureBlocks)
	if block.Time() > max {
		return fmt.Errorf("future block timestamp %v > allowed %v", block.Time(), max)
	}
	if block.Difficulty().Sign() == 0 {
		// Never add PoS blocks into the future queue
		return nil
	}
	bc.futureBlocks.Add(block.Hash(), block)
	return nil
}

// InsertChain attempts to insert the given batch of blocks in to the canonical
// chain or, otherwise, create a fork. If an error is returned it will return
// the index number of the failing block as well an error describing what went
// wrong.
//
// After insertion is done, all accumulated events will be fired.
func (bc *BlockChain) InsertChain(chain types.Blocks) (int, error) {
	// Sanity check that we have something meaningful to import
	if len(chain) == 0 {
		return 0, nil
	}

	// Do a sanity check that the provided chain is actually ordered and linked.
	for i := 1; i < len(chain); i++ {
		block, prev := chain[i], chain[i-1]
		if block.NumberU64() != prev.NumberU64()+1 || block.ParentHash() != prev.Hash() {
			log.Error("Non contiguous block insert",
				"number", block.Number(),
				"hash", block.Hash(),
				"parent", block.ParentHash(),
				"prevnumber", prev.Number(),
				"prevhash", prev.Hash(),
			)
			return 0, fmt.Errorf("non contiguous insert: item %d is #%d [%x..], item %d is #%d [%x..] (parent [%x..])", i-1, prev.NumberU64(),
				prev.Hash().Bytes()[:4], i, block.NumberU64(), block.Hash().Bytes()[:4], block.ParentHash().Bytes()[:4])
		}
	}

	// Pre-check passed, start the full block imports.
	if !bc.chainmu.TryLock() {
		return 0, errChainStopped
	}
	defer bc.chainmu.Unlock()
	return bc.insertChain(chain, true)
}

// InsertChainWithoutSealVerification works exactly the same
// except for seal verification, seal verification is omitted
func (bc *BlockChain) InsertChainWithoutSealVerification(block *types.Block) (int, error) {
	if !bc.chainmu.TryLock() {
		return 0, errChainStopped
	}
	defer bc.chainmu.Unlock()
	return bc.insertChain(types.Blocks([]*types.Block{block}), false)
}

// insertChain is the internal implementation of InsertChain, which assumes that
// 1) chains are contiguous, and 2) The chain mutex is held.
//
// This method is split out so that import batches that require re-injecting
// historical blocks can do so without releasing the lock, which could lead to
// racey behaviour. If a sidechain import is in progress, and the historic state
// is imported, but then new canon-head is added before the actual sidechain
// completes, then the historic state could be pruned again
func (bc *BlockChain) insertChain(chain types.Blocks, verifySeals bool) (int, error) {
	// If the chain is terminating, don't even bother starting up.
	if bc.insertStopped() {
		return 0, nil
	}

	if atomic.AddInt32(&bc.blockProcCounter, 1) == 1 {
		bc.blockProcFeed.Send(true)
	}
	defer func() {
		if atomic.AddInt32(&bc.blockProcCounter, -1) == 0 {
			bc.blockProcFeed.Send(false)
		}
	}()

	// Start a parallel signature recovery (signer will fluke on fork transition, minimal perf loss)
	SenderCacher.RecoverFromBlocks(types.MakeSigner(bc.chainConfig, chain[0].Number(), chain[0].Time()), chain)

	var (
		stats     = insertStats{startTime: mclock.Now()}
		lastCanon *types.Block
	)
	// Fire a single chain head event if we've progressed the chain
	defer func() {
		if lastCanon != nil && bc.CurrentBlock().Hash() == lastCanon.Hash() {
			bc.chainHeadFeed.Send(ChainHeadEvent{Header: lastCanon.Header()})
		}
	}()
	// Start the parallel header verifier
	headers := make([]*types.Header, len(chain))
	seals := make([]bool, len(chain))

	for i, block := range chain {
		headers[i] = block.Header()
		seals[i] = verifySeals
	}
	abort, results := bc.engine.VerifyHeaders(bc, headers, seals)
	defer close(abort)

	// Peek the error for the first block to decide the directing import logic
	it := newInsertIterator(chain, results, bc.validator)
	block, err := it.next()

	// Left-trim all the known blocks that don't need to build snapshot
	if bc.skipBlock(err, it) {
		// First block (and state) is known
		//   1. We did a roll-back, and should now do a re-import
		//   2. The block is stored as a sidechain, and is lying about it's stateroot, and passes a stateroot
		//      from the canonical chain, which has not been verified.
		// Skip all known blocks that are behind us.
		var (
			current  = bc.CurrentBlock()
			localTd  = bc.GetTd(current.Hash(), current.NumberU64())
			externTd = bc.GetTd(block.ParentHash(), block.NumberU64()-1) // The first block can't be nil
		)
		for block != nil && bc.skipBlock(err, it) {
			externTd = new(big.Int).Add(externTd, block.Difficulty())
			if localTd.Cmp(externTd) < 0 {
				break
			}
			log.Debug("Ignoring already known block", "number", block.Number(), "hash", block.Hash())
			stats.ignored++

			block, err = it.next()
		}
		// The remaining blocks are still known blocks, the only scenario here is:
		// During the fast sync, the pivot point is already submitted but rollback
		// happens. Then node resets the head full block to a lower height via `rollback`
		// and leaves a few known blocks in the database.
		//
		// When node runs a fast sync again, it can re-import a batch of known blocks via
		// `insertChain` while a part of them have higher total difficulty than current
		// head full block(new pivot point).
		for block != nil && bc.skipBlock(err, it) {
			log.Debug("Writing previously known block", "number", block.Number(), "hash", block.Hash())
			if err := bc.writeKnownBlock(block); err != nil {
				return it.index, err
			}
			lastCanon = block

			block, err = it.next()
		}
		// Falls through to the block import
	}
	switch {
	// First block is pruned, insert as sidechain and reorg only if TD grows enough
	case errors.Is(err, consensus.ErrPrunedAncestor):
		log.Debug("Pruned ancestor, inserting as sidechain", "number", block.Number(), "hash", block.Hash())
		return bc.insertSideChain(block, it)

	// First block is future, shove it (and all children) to the future queue (unknown ancestor)
	case errors.Is(err, consensus.ErrFutureBlock) || (errors.Is(err, consensus.ErrUnknownAncestor) && bc.futureBlocks.Contains(it.first().ParentHash())):
		for block != nil && (it.index == 0 || errors.Is(err, consensus.ErrUnknownAncestor)) {
			log.Debug("Future block, postponing import", "number", block.Number(), "hash", block.Hash())
			if err := bc.addFutureBlock(block); err != nil {
				return it.index, err
			}
			block, err = it.next()
		}
		stats.queued += it.processed()
		stats.ignored += it.remaining()

		// If there are any still remaining, mark as ignored
		return it.index, err

	// Some other error(except ErrKnownBlock) occurred, abort.
	// ErrKnownBlock is allowed here since some known blocks
	// still need re-execution to generate snapshots that are missing
	case err != nil && !errors.Is(err, ErrKnownBlock):
		bc.futureBlocks.Remove(block.Hash())
		stats.ignored += len(it.chain)
		bc.reportBlock(block, nil, err)
		return it.index, err
	}
	// No validation errors for the first block (or chain prefix skipped)
	var activeState *state.StateDB
	defer func() {
		// The chain importer is starting and stopping trie prefetchers. If a bad
		// block or other error is hit however, an early return may not properly
		// terminate the background threads. This defer ensures that we clean up
		// and dangling prefetcher, without defering each and holding on live refs.
		if activeState != nil {
			activeState.StopPrefetcher()
		}
	}()

	for ; block != nil && err == nil || errors.Is(err, ErrKnownBlock); block, err = it.next() {
		// If the chain is terminating, stop processing blocks
		if bc.insertStopped() {
			log.Debug("Abort during block processing")
			break
		}
		// If the header is a banned one, straight out abort
		if BadHashes[block.Hash()] {
			bc.reportBlock(block, nil, ErrBlacklistedHash)
			return it.index, ErrBlacklistedHash
		}
		// If the block is known (in the middle of the chain), it's a special case for
		// Clique blocks where they can share state among each other, so importing an
		// older block might complete the state of the subsequent one. In this case,
		// just skip the block (we already validated it once fully (and crashed), since
		// its header and body was already in the database). But if the corresponding
		// snapshot layer is missing, forcibly rerun the execution to build it.
		if bc.skipBlock(err, it) {
			logger := log.Debug
			if bc.chainConfig.Clique == nil {
				logger = log.Warn
			}
			logger("Inserted known block", "number", block.Number(), "hash", block.Hash(),
				"uncles", len(block.Uncles()), "txs", len(block.Transactions()), "gas", block.GasUsed(),
				"root", block.Root())

			// Special case. Commit the empty receipt slice if we meet the known
			// block in the middle. It can only happen in the clique chain. Whenever
			// we insert blocks via `insertSideChain`, we only commit `td`, `header`
			// and `body` if it's non-existent. Since we don't have receipts without
			// reexecution, so nothing to commit. But if the sidechain will be adpoted
			// as the canonical chain eventually, it needs to be reexecuted for missing
			// state, but if it's this special case here(skip reexecution) we will lose
			// the empty receipt entry.
			if len(block.Transactions()) == 0 {
				rawdb.WriteReceipts(bc.db, block.Hash(), block.NumberU64(), nil)
			} else {
				log.Error("Please file an issue, skip known block execution without receipt",
					"hash", block.Hash(), "number", block.NumberU64())
			}
			if err := bc.writeKnownBlock(block); err != nil {
				return it.index, err
			}
			stats.processed++

			// We can assume that logs are empty here, since the only way for consecutive
			// Clique blocks to have the same state is if there are no transactions.
			lastCanon = block
			continue
		}

		// Retrieve the parent block and it's state to execute on top
		start := time.Now()
		parent := it.previous()
		if parent == nil {
			parent = bc.GetHeader(block.ParentHash(), block.NumberU64()-1)
		}
		statedb, err := state.New(parent.Root, bc.stateCache)
		if err != nil {
			return it.index, err
		}

		// If we are past Byzantium, enable prefetching to pull in trie node paths
		// while processing transactions. Before Byzantium the prefetcher is mostly
		// useless due to the intermediate root hashing after each transaction.
		if bc.chainConfig.IsByzantium(block.Number()) {
			statedb.StartPrefetcher("chain", !bc.vmConfig.EnableWitnessCollection)
		}
		activeState = statedb

		// If we have a followup block, run that against the current state to pre-cache
		// transactions and probabilistically some of the account/storage trie nodes.
		var followupInterrupt atomic.Bool
		if !bc.cacheConfig.TrieCleanNoPrefetch {
			if followup, err := it.peek(); followup != nil && err == nil {
				throwaway, _ := state.New(parent.Root, bc.stateCache)

				go func(start time.Time, followup *types.Block, throwaway *state.StateDB) {
					bc.prefetcher.Prefetch(followup, throwaway, bc.vmConfig, &followupInterrupt)

					blockPrefetchExecuteTimer.Update(time.Since(start))
					if followupInterrupt.Load() {
						blockPrefetchInterruptMeter.Mark(1)
					}
				}(time.Now(), followup, throwaway)
			}
		}

		block.Header().Quota = parent.Quota + bc.chainConfig.GetBlockQuota(block.Number())
		block.Header().QuotaUsed = parent.QuotaUsed
		// Process block using the parent state as reference point
		substart := time.Now()
		receipts, logs, usedGas, err := bc.processor.Process(block, statedb, bc.vmConfig)
		if err != nil {
			bc.reportBlock(block, receipts, err)
			followupInterrupt.Store(true)
			return it.index, err
		}

		// Update the metrics touched during block processing
		accountReadTimer.Update(statedb.AccountReads)                 // Account reads are complete, we can mark them
		storageReadTimer.Update(statedb.StorageReads)                 // Storage reads are complete, we can mark them
		accountUpdateTimer.Update(statedb.AccountUpdates)             // Account updates are complete, we can mark them
		storageUpdateTimer.Update(statedb.StorageUpdates)             // Storage updates are complete, we can mark them
		snapshotAccountReadTimer.Update(statedb.SnapshotAccountReads) // Account reads are complete, we can mark them
		snapshotStorageReadTimer.Update(statedb.SnapshotStorageReads) // Storage reads are complete, we can mark them
		triehash := statedb.AccountHashes + statedb.StorageHashes     // Save to not double count in validation
		trieproc := statedb.SnapshotAccountReads + statedb.AccountReads + statedb.AccountUpdates
		trieproc += statedb.SnapshotStorageReads + statedb.StorageReads + statedb.StorageUpdates

		blockExecutionTimer.Update(time.Since(substart) - trieproc - triehash)

		// Validate the state using the default validator
		substart = time.Now()
		if err := bc.validator.ValidateState(block, statedb, receipts, usedGas); err != nil {
			bc.reportBlock(block, receipts, err)
			followupInterrupt.Store(true)
			return it.index, err
		}
		proctime := time.Since(start)

		// Update the metrics touched during block validation
		accountHashTimer.Update(statedb.AccountHashes) // Account hashes are complete, we can mark them
		storageHashTimer.Update(statedb.StorageHashes) // Storage hashes are complete, we can mark them

		blockValidationTimer.Update(time.Since(substart) - (statedb.AccountHashes + statedb.StorageHashes - triehash))

		// Write the block to the chain and get the status.
		substart = time.Now()
		status, err := bc.writeBlockWithState(block, receipts, logs, statedb, false)
		followupInterrupt.Store(true)
		if err != nil {
			return it.index, err
		}
		// Update the metrics touched during block commit
		accountCommitTimer.Update(statedb.AccountCommits)   // Account commits are complete, we can mark them
		storageCommitTimer.Update(statedb.StorageCommits)   // Storage commits are complete, we can mark them
		snapshotCommitTimer.Update(statedb.SnapshotCommits) // Snapshot commits are complete, we can mark them

		blockWriteTimer.Update(time.Since(substart) - statedb.AccountCommits - statedb.StorageCommits - statedb.SnapshotCommits)
		blockInsertTimer.UpdateSince(start)

		// Report the import stats before returning the various results
		stats.processed++
		stats.usedGas += usedGas

		dirty, _ := bc.stateCache.TrieDB().Size()
		stats.report(chain, it.index, dirty)

		switch status {
		case CanonStatTy:
			log.Debug("Inserted new block", "number", block.Number(), "hash", block.Hash(),
				"uncles", len(block.Uncles()), "txs", len(block.Transactions()), "gas", block.GasUsed(),
				"elapsed", common.PrettyDuration(time.Since(start)),
				"root", block.Root())

			lastCanon = block

			// Only count canonical blocks for GC processing time
			bc.gcproc += proctime

		case SideStatTy:
			log.Debug("Inserted forked block", "number", block.Number(), "hash", block.Hash(),
				"diff", block.Difficulty(), "elapsed", common.PrettyDuration(time.Since(start)),
				"txs", len(block.Transactions()), "gas", block.GasUsed(), "uncles", len(block.Uncles()),
				"root", block.Root())

		default:
			// This in theory is impossible, but lets be nice to our future selves and leave
			// a log, instead of trying to track down blocks imports that don't emit logs.
			log.Warn("Inserted block with unknown status", "number", block.Number(), "hash", block.Hash(),
				"diff", block.Difficulty(), "elapsed", common.PrettyDuration(time.Since(start)),
				"txs", len(block.Transactions()), "gas", block.GasUsed(), "uncles", len(block.Uncles()),
				"root", block.Root())
		}
	}

	offset := time.Now().Unix() - bc.utcNow
	if offset <= params.SPROUT_TIME && bc.Viper {
		if time.Now().Unix()%3 == 0 {
			log.Info("Blockchain sprout time", "left", params.SPROUT_TIME-offset, "viper", bc.Viper)
		}
	}

	// Any blocks remaining here? The only ones we care about are the future ones
	if block != nil && errors.Is(err, consensus.ErrFutureBlock) {
		if err := bc.addFutureBlock(block); err != nil {
			return it.index, err
		}
		block, err = it.next()

		for ; block != nil && errors.Is(err, consensus.ErrUnknownAncestor); block, err = it.next() {
			if err := bc.addFutureBlock(block); err != nil {
				return it.index, err
			}
			stats.queued++
		}
	}
	stats.ignored += it.remaining()

	return it.index, err
}

// insertSideChain is called when an import batch hits upon a pruned ancestor
// error, which happens when a sidechain with a sufficiently old fork-block is
// found.
//
// The method writes all (header-and-body-valid) blocks to disk, then tries to
// switch over to the new chain if the TD exceeded the current chain.
func (bc *BlockChain) insertSideChain(block *types.Block, it *insertIterator) (int, error) {
	var (
		externTd *big.Int
		current  = bc.CurrentBlock()
	)
	// The first sidechain block error is already verified to be ErrPrunedAncestor.
	// Since we don't import them here, we expect ErrUnknownAncestor for the remaining
	// ones. Any other errors means that the block is invalid, and should not be written
	// to disk.
	err := consensus.ErrPrunedAncestor
	for ; block != nil && errors.Is(err, consensus.ErrPrunedAncestor); block, err = it.next() {
		// Check the canonical state root for that number
		if number := block.NumberU64(); current.NumberU64() >= number {
			canonical := bc.GetBlockByNumber(number)
			if canonical != nil && canonical.Hash() == block.Hash() {
				// Not a sidechain block, this is a re-import of a canon block which has it's state pruned

				// Collect the TD of the block. Since we know it's a canon one,
				// we can get it directly, and not (like further below) use
				// the parent and then add the block on top
				externTd = bc.GetTd(block.Hash(), block.NumberU64())
				continue
			}
			if canonical != nil && canonical.Root() == block.Root() {
				// This is most likely a shadow-state attack. When a fork is imported into the
				// database, and it eventually reaches a block height which is not pruned, we
				// just found that the state already exist! This means that the sidechain block
				// refers to a state which already exists in our canon chain.
				//
				// If left unchecked, we would now proceed importing the blocks, without actually
				// having verified the state of the previous blocks.
				log.Warn("Sidechain ghost-state attack detected", "number", block.NumberU64(), "sideroot", block.Root(), "canonroot", canonical.Root())

				// If someone legitimately side-mines blocks, they would still be imported as usual. However,
				// we cannot risk writing unverified blocks to disk when they obviously target the pruning
				// mechanism.
				return it.index, errors.New("sidechain ghost-state attack")
			}
		}
		if externTd == nil {
			externTd = bc.GetTd(block.ParentHash(), block.NumberU64()-1)
		}
		externTd = new(big.Int).Add(externTd, block.Difficulty())

		if !bc.HasBlock(block.Hash(), block.NumberU64()) {
			start := time.Now()
			if err := bc.writeBlockWithoutState(block, externTd); err != nil {
				return it.index, err
			}
			log.Debug("Injected sidechain block", "number", block.Number(), "hash", block.Hash(),
				"diff", block.Difficulty(), "elapsed", common.PrettyDuration(time.Since(start)),
				"txs", len(block.Transactions()), "gas", block.GasUsed(), "uncles", len(block.Uncles()),
				"root", block.Root())
		}
	}
	// At this point, we've written all sidechain blocks to database. Loop ended
	// either on some other error or all were processed. If there was some other
	// error, we can ignore the rest of those blocks.
	//
	// If the externTd was larger than our local TD, we now need to reimport the previous
	// blocks to regenerate the required state
	localTd := bc.GetTd(current.Hash(), current.NumberU64())
	if localTd.Cmp(externTd) > 0 {
		log.Info("Sidechain written to disk", "start", it.first().NumberU64(), "end", it.previous().Number, "sidetd", externTd, "localtd", localTd)
		return it.index, err
	}
	// Gather all the sidechain hashes (full blocks may be memory heavy)
	var (
		hashes  []common.Hash
		numbers []uint64
	)
	parent := it.previous()
	for parent != nil && !bc.HasState(parent.Root) {
		hashes = append(hashes, parent.Hash())
		numbers = append(numbers, parent.Number.Uint64())

		parent = bc.GetHeader(parent.ParentHash, parent.Number.Uint64()-1)
	}

	if parent == nil {
		return it.index, errors.New("missing parent")
	}

	if len(hashes) > params.HEAVY_CHAIN_LIMIT {
		if !bc.Viper || time.Now().Unix() > bc.utcNow+params.SPROUT_TIME {
			return it.index, errors.New(fmt.Sprintf("Heavy side chain detected %v, use --viper to fix it", len(hashes)))
		} else {
			log.Info("Heavey chain import for blockchain protection", "offset", time.Now().Unix()-bc.utcNow, "size", len(hashes), "start", numbers[len(numbers)-1], "end", numbers[0], "viper", bc.Viper)
		}
	}

	if bc.Viper {
		//TODO
	}

	// Import all the pruned blocks to make the state available
	var (
		blocks []*types.Block
		memory uint64
	)
	for i := len(hashes) - 1; i >= 0; i-- {
		// Append the next block to our batch
		block := bc.GetBlock(hashes[i], numbers[i])

		blocks = append(blocks, block)
		memory += block.Size()

		// If memory use grew too large, import and continue. Sadly we need to discard
		// all raised events and logs from notifications since we're too heavy on the
		// memory here.
		if len(blocks) >= 2048 || memory > 64*1024*1024 {
			log.Info("Importing heavy sidechain segment", "blocks", len(blocks), "start", blocks[0].NumberU64(), "end", block.NumberU64())
			if _, err := bc.insertChain(blocks, false); err != nil {
				return 0, err
			}
			blocks, memory = blocks[:0], 0

			// If the chain is terminating, stop processing blocks
			if bc.insertStopped() {
				log.Debug("Abort during blocks processing")
				return 0, nil
			}
		}
	}
	if len(blocks) > 0 {
		log.Info("Importing sidechain segment", "start", blocks[0].NumberU64(), "end", blocks[len(blocks)-1].NumberU64())
		return bc.insertChain(blocks, false)
	}
	return 0, nil
}

// collectLogs collects the logs that were generated or removed during
// the processing of the block that corresponds with the given hash.
// These logs are later announced as deleted or reborn.
func (bc *BlockChain) collectLogs(b *types.Block, removed bool) []*types.Log {
	receipts := rawdb.ReadRawReceipts(bc.db, b.Hash(), b.NumberU64())
	if err := receipts.DeriveFields(bc.chainConfig, b.Hash(), b.NumberU64(), b.Time(), b.Transactions()); err != nil {
		log.Error("Failed to derive block receipts fields", "hash", b.Hash(), "number", b.NumberU64(), "err", err)
	}

	var logs []*types.Log
	for _, receipt := range receipts {
		for _, log := range receipt.Logs {
			if removed {
				log.Removed = true
			}
			logs = append(logs, log)
		}
	}
	return logs
}

// reorg takes two blocks, an old chain and a new chain and will reconstruct the
// blocks and inserts them to be part of the new canonical chain and accumulates
// potential missing transactions and post an event about them.
func (bc *BlockChain) reorg(oldBlock, newBlock *types.Header) error {
	var (
		newChain    []*types.Header
		oldChain    []*types.Header
		commonBlock *types.Header
	)
	// Reduce the longer chain to the same number as the shorter one
	if oldBlock.Number.Uint64() > newBlock.Number.Uint64() {
		// Old chain is longer, gather all transactions and logs as deleted ones
		for ; oldBlock != nil && oldBlock.Number.Uint64() != newBlock.Number.Uint64(); oldBlock = bc.GetHeader(oldBlock.ParentHash, oldBlock.Number.Uint64()-1) {
			oldChain = append(oldChain, oldBlock)
		}
	} else {
		// New chain is longer, stash all blocks away for subsequent insertion
		for ; newBlock != nil && newBlock.Number.Uint64() != oldBlock.Number.Uint64(); newBlock = bc.GetHeader(newBlock.ParentHash, newBlock.Number.Uint64()-1) {
			newChain = append(newChain, newBlock)
		}
	}
	if oldBlock == nil {
		return fmt.Errorf("invalid old chain")
	}
	if newBlock == nil {
		return fmt.Errorf("invalid new chain")
	}
	// Both sides of the reorg are at the same number, reduce both until the common
	// ancestor is found
	for {
		// If the common ancestor was found, bail out
		if oldBlock.Hash() == newBlock.Hash() {
			commonBlock = oldBlock
			break
		}
		// Remove an old block as well as stash away a new block
		oldChain = append(oldChain, oldBlock)
		newChain = append(newChain, newBlock)

		// Step back with both chains
		oldBlock = bc.GetHeader(oldBlock.ParentHash, oldBlock.Number.Uint64()-1)
		if oldBlock == nil {
			return fmt.Errorf("invalid old chain")
		}
		newBlock = bc.GetHeader(newBlock.ParentHash, newBlock.Number.Uint64()-1)
		if newBlock == nil {
			return fmt.Errorf("invalid new chain")
		}
	}

	// Ensure the user sees large reorgs
	if len(oldChain) > 0 && len(newChain) > 0 {
		logFn := log.Info
		msg := "Chain reorg detected"
		if len(oldChain) > 63 {
			msg = "Large chain reorg detected"
			logFn = log.Warn
		}
		logFn(msg, "number", commonBlock.Number, "hash", commonBlock.Hash(),
			"drop", len(oldChain), "dropfrom", oldChain[0].Hash(), "add", len(newChain), "addfrom", newChain[0].Hash())
		blockReorgAddMeter.Mark(int64(len(newChain)))
		blockReorgDropMeter.Mark(int64(len(oldChain)))
		blockReorgMeter.Mark(1)
	} else if len(newChain) > 0 {
		// Special case happens in the post merge stage that current head is
		// the ancestor of new head while these two blocks are not consecutive
		log.Info("Extend chain", "add", len(newChain), "number", newChain[0].Number, "hash", newChain[0].Hash())
		blockReorgAddMeter.Mark(int64(len(newChain)))
	} else {
		log.Error("Impossible reorg, please file an issue", "oldnum", oldBlock.Number, "oldhash", oldBlock.Hash(), "newnum", newBlock.Number, "newhash", newBlock.Hash())
	}
	// Acquire the tx-lookup lock before mutation. This step is essential
	// as the txlookups should be changed atomically, and all subsequent
	// reads should be blocked until the mutation is complete.
	bc.txLookupLock.Lock()

	var (
		deletedTxs []common.Hash
		rebirthTxs []common.Hash

		deletedLogs []*types.Log
		rebirthLogs []*types.Log
	)

	// Send out events for logs from the old canon chain, and 'reborn'
	// logs from the new canon chain. The number of logs can be very
	// high, so the events are sent in batches of size around 512.

	// Deleted logs + blocks:
	{
		for i := len(oldChain) - 1; i >= 0; i-- {
			block := bc.GetBlock(oldChain[i].Hash(), oldChain[i].Number.Uint64())
			if block == nil {
				return errInvalidOldChain // Corrupt database, mostly here to avoid weird panics
			}
			// Also send event for blocks removed from the canon chain.
			bc.chainSideFeed.Send(ChainSideEvent{Header: block.Header()})

			// Collect deleted logs for notification
			if logs := bc.collectLogs(block, true); len(logs) > 0 {
				deletedLogs = append(deletedLogs, logs...)
			}
			if len(deletedLogs) > 512 {
				bc.rmLogsFeed.Send(RemovedLogsEvent{deletedLogs})
				deletedLogs = nil
			}
		}

		if len(deletedLogs) > 0 {
			bc.rmLogsFeed.Send(RemovedLogsEvent{deletedLogs})
		}
	}

	// Undo old blocks in reverse order
	for i := 0; i < len(oldChain); i++ {
		// Collect all the deleted transactions
		block := bc.GetBlock(oldChain[i].Hash(), oldChain[i].Number.Uint64())
		if block == nil {
			return errInvalidOldChain // Corrupt database, mostly here to avoid weird panics
		}
		for _, tx := range block.Transactions() {
			deletedTxs = append(deletedTxs, tx.Hash())
		}
		// Collect deleted logs and emit them for new integrations
		if logs := bc.collectLogs(block, true); len(logs) > 0 {
			// Emit revertals latest first, older then
			slices.Reverse(logs)
		}
	}

	// Insert the new chain segment in incremental order, from the old
	// to the new. The new chain head (newChain[0]) is not inserted here,
	// as it will be handled separately outside of this function
	for i := len(newChain) - 1; i >= 1; i-- {
		block := bc.GetBlock(newChain[i].Hash(), newChain[i].Number.Uint64())
		if block == nil {
			return errInvalidNewChain // Corrupt database, mostly here to avoid weird panics
		}
		for _, tx := range block.Transactions() {
			rebirthTxs = append(rebirthTxs, tx.Hash())
		}
		// Collect inserted logs and emit them
		if logs := bc.collectLogs(block, false); len(logs) > 0 {
			rebirthLogs = append(rebirthLogs, logs...)
		}
		if len(rebirthLogs) > 512 {
			bc.logsFeed.Send(rebirthLogs)
			rebirthLogs = nil
		}
		// Insert the block in the canonical way, re-writing history
		bc.writeHeadBlock(block)
	}

	if len(rebirthLogs) > 0 {
		bc.logsFeed.Send(rebirthLogs)
	}

	// Delete useless indexes right now which includes the non-canonical
	// transaction indexes, canonical chain indexes which above the head.
	batch := bc.db.NewBatch()
	for _, tx := range types.HashDifference(deletedTxs, rebirthTxs) {
		rawdb.DeleteTxLookupEntry(batch, tx)
	}
	// Delete all hash markers that are not part of the new canonical chain.
	// Because the reorg function does not handle new chain head, all hash
	// markers greater than or equal to new chain head should be deleted.
	number := commonBlock.Number.Uint64()
	if len(newChain) > 1 {
		number = newChain[1].Number.Uint64()
	}
	for i := number + 1; ; i++ {
		hash := rawdb.ReadCanonicalHash(bc.db, i)
		if hash == (common.Hash{}) {
			break
		}
		rawdb.DeleteCanonicalHash(batch, i)
	}
	if err := batch.Write(); err != nil {
		log.Crit("Failed to delete useless indexes", "err", err)
	}
	// Reset the tx lookup cache to clear stale txlookup cache.
	bc.txLookupCache.Purge()

	// Release the tx-lookup lock after mutation.
	bc.txLookupLock.Unlock()

	return nil
}

// futureBlocksLoop processes the 'future block' queue.
func (bc *BlockChain) updateFutureBlocks() {
	defer bc.wg.Done()

	futureTimer := time.NewTicker(5 * time.Second)
	defer futureTimer.Stop()
	for {
		select {
		case <-futureTimer.C:
			bc.procFutureBlocks()
		case <-bc.quit:
			return
		}
	}
}

// skipBlock returns 'true', if the block being imported can be skipped over, meaning
// that the block does not need to be processed but can be considered already fully 'done'.
func (bc *BlockChain) skipBlock(err error, it *insertIterator) bool {
	// We can only ever bypass processing if the only error returned by the validator
	// is ErrKnownBlock, which means all checks passed, but we already have the block
	// and state.
	if !errors.Is(err, ErrKnownBlock) {
		return false
	}
	// If we're not using snapshots, we can skip this, since we have both block
	// and (trie-) state
	if bc.snaps == nil {
		return true
	}
	var (
		header     = it.current() // header can't be nil
		parentRoot common.Hash
	)
	// If we also have the snapshot-state, we can skip the processing.
	if bc.snaps.Snapshot(header.Root) != nil {
		return true
	}
	// In this case, we have the trie-state but not snapshot-state. If the parent
	// snapshot-state exists, we need to process this in order to not get a gap
	// in the snapshot layers.
	// Resolve parent block
	if parent := it.previous(); parent != nil {
		parentRoot = parent.Root
	} else if parent = bc.GetHeaderByHash(header.ParentHash); parent != nil {
		parentRoot = parent.Root
	}
	if parentRoot == (common.Hash{}) {
		return false // Theoretically impossible case
	}
	// Parent is also missing snapshot: we can skip this. Otherwise process.
	if bc.snaps.Snapshot(parentRoot) == nil {
		return true
	}
	return false
}

// reportBlock logs a bad block error.
func (bc *BlockChain) reportBlock(block *types.Block, receipts types.Receipts, err error) {
	if errors.Is(err, vm.ErrRuntime) {
		return
	}
	rawdb.WriteBadBlock(bc.db, block)

	var receiptString string
	for i, receipt := range receipts {
		receiptString += fmt.Sprintf("\t %d: cumulative: %v gas: %v contract: %v status: %v tx: %v logs: %v bloom: %x state: %x\n",
			i, receipt.CumulativeGasUsed, receipt.GasUsed, receipt.ContractAddress.Hex(),
			receipt.Status, receipt.TxHash.Hex(), receipt.Logs, receipt.Bloom, receipt.PostState)
	}
	log.Error(fmt.Sprintf(`
########## BAD BLOCK #########
Chain config: %v
Number: %v
Hash: 0x%x
%v
Error: %v
##############################
`, bc.chainConfig, block.Number(), block.Hash(), receiptString, err))
}

// InsertHeaderChain attempts to insert the given header chain in to the local
// chain, possibly creating a reorg. If an error is returned, it will return the
// index number of the failing header as well an error describing what went wrong.
//
// The verify parameter can be used to fine tune whether nonce verification
// should be done or not. The reason behind the optional check is because some
// of the header retrieval mechanisms already need to verify nonces, as well as
// because nonces can be verified sparsely, not needing to check each.
func (bc *BlockChain) InsertHeaderChain(chain []*types.Header, checkFreq int) (int, error) {
	if len(chain) == 0 {
		return 0, nil
	}
	start := time.Now()
	if i, err := bc.hc.ValidateHeaderChain(chain, checkFreq); err != nil {
		return i, err
	}

	if !bc.chainmu.TryLock() {
		return 0, errChainStopped
	}
	defer bc.chainmu.Unlock()
	_, err := bc.hc.InsertHeaderChain(chain, start)
	return 0, err
}

// InsertHeadersBeforeCutoff inserts the given headers into the ancient store
// as they are claimed older than the configured chain cutoff point. All the
// inserted headers are regarded as canonical and chain reorg is not supported.
func (bc *BlockChain) InsertHeadersBeforeCutoff(headers []*types.Header) (int, error) {
	if len(headers) == 0 {
		return 0, nil
	}
	// TODO(rjl493456442): Headers before the configured cutoff have already
	// been verified by the hash of cutoff header. Theoretically, header validation
	// could be skipped here.
	if n, err := bc.hc.ValidateHeaderChain(headers, 0); err != nil {
		return n, err
	}
	if !bc.chainmu.TryLock() {
		return 0, errChainStopped
	}
	defer bc.chainmu.Unlock()

	// Initialize the ancient store with genesis block if it's empty.
	var (
		frozen, _ = bc.db.Ancients()
		first     = headers[0].Number.Uint64()
	)
	if first == 1 && frozen == 0 {
		_, err := rawdb.WriteAncientBlocks(bc.db, []*types.Block{bc.genesisBlock}, []types.Receipts{nil}, big.NewInt(0))
		if err != nil {
			log.Error("Error writing genesis to ancients", "err", err)
			return 0, err
		}
		log.Info("Wrote genesis to ancient store")
	} else if frozen != first {
		return 0, fmt.Errorf("headers are gapped with the ancient store, first: %d, ancient: %d", first, frozen)
	}

	// Write headers to the ancient store, with block bodies and receipts set to nil
	// to ensure consistency across tables in the freezer.
	_, err := rawdb.WriteAncientHeaderChain(bc.db, headers)
	if err != nil {
		return 0, err
	}
	// Sync the ancient store explicitly to ensure all data has been flushed to disk.
	if err := bc.db.SyncAncient(); err != nil {
		return 0, err
	}
	// Write hash to number mappings
	batch := bc.db.NewBatch()
	for _, header := range headers {
		rawdb.WriteHeaderNumber(batch, header.Hash(), header.Number.Uint64())
	}
	// Write head header and head snap block flags
	last := headers[len(headers)-1]
	rawdb.WriteHeadHeaderHash(batch, last.Hash())
	rawdb.WriteHeadFastBlockHash(batch, last.Hash())
	if err := batch.Write(); err != nil {
		return 0, err
	}
	// Truncate the useless chain segment (zero bodies and receipts) in the
	// ancient store.
	if _, err := bc.db.TruncateTail(last.Number.Uint64() + 1); err != nil {
		return 0, err
	}
	// Last step update all in-memory markers
	bc.hc.currentHeader.Store(last)
	bc.currentFastBlock.Store(last)
	headHeaderGauge.Update(last.Number.Int64())
	headFastBlockGauge.Update(last.Number.Int64())
	return 0, nil
}

// SetBlockValidatorAndProcessorForTesting sets the current validator and processor.
// This method can be used to force an invalid blockchain to be verified for tests.
// This method is unsafe and should only be used before block import starts.
func (bc *BlockChain) SetBlockValidatorAndProcessorForTesting(v Validator, p Processor) {
	bc.validator = v
	bc.processor = p
}

// SetTrieFlushInterval configures how often in-memory tries are persisted to disk.
// The interval is in terms of block processing time, not wall clock.
// It is thread-safe and can be called repeatedly without side effects.
func (bc *BlockChain) SetTrieFlushInterval(interval time.Duration) {
	bc.flushInterval.Store(int64(interval))
}

// GetTrieFlushInterval gets the in-memory tries flushAlloc interval
func (bc *BlockChain) GetTrieFlushInterval() time.Duration {
	return time.Duration(bc.flushInterval.Load())
}
