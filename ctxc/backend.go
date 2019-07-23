// Copyright 2018 The CortexTheseus Authors
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

// Package ctxc implements the Cortex protocol.
package ctxc

import (
	"fmt"
	"math/big"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/CortexFoundation/CortexTheseus/accounts"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/consensus"
	"github.com/CortexFoundation/CortexTheseus/consensus/clique"
	"github.com/CortexFoundation/CortexTheseus/consensus/cuckoo"
	"github.com/CortexFoundation/CortexTheseus/core"
	"github.com/CortexFoundation/CortexTheseus/core/bloombits"
	"github.com/CortexFoundation/CortexTheseus/core/rawdb"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/core/vm"
	"github.com/CortexFoundation/CortexTheseus/ctxc/downloader"
	"github.com/CortexFoundation/CortexTheseus/ctxc/filters"
	"github.com/CortexFoundation/CortexTheseus/ctxc/gasprice"
	"github.com/CortexFoundation/CortexTheseus/db"
	"github.com/CortexFoundation/CortexTheseus/event"
	"github.com/CortexFoundation/CortexTheseus/inference/synapse"
	"github.com/CortexFoundation/CortexTheseus/internal/ctxcapi"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/miner"
	"github.com/CortexFoundation/CortexTheseus/node"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/CortexTheseus/rlp"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"github.com/CortexFoundation/CortexTheseus/torrentfs"
)

// Cortex implements the Cortex full node service.
type Cortex struct {
	config      *Config
	chainConfig *params.ChainConfig

	// Channel for shutting down the service
	shutdownChan chan bool // Channel for shutting down the Cortex

	// Handlers
	txPool          *core.TxPool
	blockchain      *core.BlockChain
	protocolManager *ProtocolManager

	// DB interfaces
	chainDb ctxcdb.Database // Block chain database

	eventMux       *event.TypeMux
	engine         consensus.Engine
	accountManager *accounts.Manager

	bloomRequests chan chan *bloombits.Retrieval // Channel receiving bloom data retrieval requests
	bloomIndexer  *core.ChainIndexer             // Bloom indexer operating during block imports

	APIBackend *CortexAPIBackend

	miner    *miner.Miner
	synapse  *synapse.Synapse
	gasPrice *big.Int
	coinbase common.Address

	networkID     uint64
	netRPCService *ctxcapi.PublicNetAPI

	lock sync.RWMutex // Protects the variadic fields (e.g. gas price and coinbase)
}

// New creates a new Cortex object (including the
// initialisation of the common Cortex object)
func New(ctx *node.ServiceContext, config *Config) (*Cortex, error) {
	// Ensure configuration values are compatible and sane
	if !config.SyncMode.IsValid() {
		return nil, fmt.Errorf("invalid sync mode %d", config.SyncMode)
	}
	if config.MinerGasPrice == nil || config.MinerGasPrice.Cmp(common.Big0) <= 0 {
		log.Warn("Sanitizing invalid miner gas price", "provided", config.MinerGasPrice, "updated", DefaultConfig.MinerGasPrice)
		config.MinerGasPrice = new(big.Int).Set(DefaultConfig.MinerGasPrice)
	}
	// Assemble the Cortex object
	chainDb, err := CreateDB(ctx, config, "chaindata")
	if err != nil {
		return nil, err
	}
	chainConfig, genesisHash, genesisErr := core.SetupGenesisBlock(chainDb, config.Genesis)
	if _, ok := genesisErr.(*params.ConfigCompatError); genesisErr != nil && !ok {
		return nil, genesisErr
	}
	log.Info("Initialised chain configuration", "config", chainConfig)

	ctxc := &Cortex{
		config:         config,
		chainDb:        chainDb,
		chainConfig:    chainConfig,
		eventMux:       ctx.EventMux,
		accountManager: ctx.AccountManager,
		engine:         CreateConsensusEngine(ctx, chainConfig, &config.Cuckoo, config.MinerNotify, config.MinerNoverify, chainDb),
		shutdownChan:   make(chan bool),
		networkID:      config.NetworkId,
		gasPrice:       config.MinerGasPrice,
		coinbase:       config.Coinbase,
		bloomRequests:  make(chan chan *bloombits.Retrieval),
		bloomIndexer:   NewBloomIndexer(chainDb, params.BloomBitsBlocks, params.BloomConfirms),
	}

	log.Info("Initialising Cortex protocol", "versions", ProtocolVersions, "network", config.NetworkId)

	if !config.SkipBcVersionCheck {
		bcVersion := rawdb.ReadDatabaseVersion(chainDb)
		//if bcVersion != core.BlockChainVersion && bcVersion != 0 {
		//	return nil, fmt.Errorf("Blockchain DB version mismatch (%d / %d).\n", bcVersion, core.BlockChainVersion)
		//}
		if bcVersion != nil && *bcVersion > core.BlockChainVersion {
			return nil, fmt.Errorf("database version is v%d, Ctxc %s only supports v%d", *bcVersion, params.VersionWithMeta, core.BlockChainVersion)
		} else if bcVersion != nil && *bcVersion < core.BlockChainVersion {
			log.Warn("Upgrade blockchain database version", "from", *bcVersion, "to", core.BlockChainVersion)
		}
		rawdb.WriteDatabaseVersion(chainDb, core.BlockChainVersion)
	}

	ctxc.synapse = synapse.New(&synapse.Config{
		DeviceType    : config.InferDeviceType,
		DeviceId      : config.InferDeviceId,
		MaxMemoryUsage: config.InferMemoryUsage,
		IsRemoteInfer : config.InferURI != "",
		InferURI      : config.InferURI,
		IsNotCache    : false,
		Storagefs     : torrentfs.Torrentfs_handle,
	})

	var (
		vmConfig = vm.Config{
			EnablePreimageRecording: config.EnablePreimageRecording,
			StorageDir:              config.StorageDir,
		}
		cacheConfig = &core.CacheConfig{Disabled: config.NoPruning, TrieNodeLimit: config.TrieCache, TrieTimeLimit: config.TrieTimeout}
	)
	ctxc.blockchain, err = core.NewBlockChain(chainDb, cacheConfig, ctxc.chainConfig, ctxc.engine, vmConfig, ctxc.shouldPreserve)
	if err != nil {
		return nil, err
	}
	// Rewind the chain in case of an incompatible config upgrade.
	if compat, ok := genesisErr.(*params.ConfigCompatError); ok {
		log.Warn("Rewinding chain to upgrade configuration", "err", compat)
		ctxc.blockchain.SetHead(compat.RewindTo)
		rawdb.WriteChainConfig(chainDb, genesisHash, chainConfig)
	}
	ctxc.bloomIndexer.Start(ctxc.blockchain)

	if config.TxPool.Journal != "" {
		config.TxPool.Journal = ctx.ResolvePath(config.TxPool.Journal)
	}
	ctxc.txPool = core.NewTxPool(config.TxPool, ctxc.chainConfig, ctxc.blockchain)

	if ctxc.protocolManager, err = NewProtocolManager(ctxc.chainConfig, config.SyncMode, config.NetworkId, ctxc.eventMux, ctxc.txPool, ctxc.engine, ctxc.blockchain, chainDb, config.Whitelist); err != nil {
		return nil, err
	}

	ctxc.miner = miner.New(ctxc, ctxc.chainConfig, ctxc.EventMux(), ctxc.engine, config.MinerRecommit, config.MinerGasFloor, config.MinerGasCeil)
	ctxc.miner.SetExtra(makeExtraData(config.MinerExtraData))

	ctxc.APIBackend = &CortexAPIBackend{ctxc, nil}
	gpoParams := config.GPO
	if gpoParams.Default == nil {
		gpoParams.Default = config.MinerGasPrice
	}
	ctxc.APIBackend.gpo = gasprice.NewOracle(ctxc.APIBackend, gpoParams)

	return ctxc, nil
}

func makeExtraData(extra []byte) []byte {
	if len(extra) == 0 {
		// create default extradata
		extra, _ = rlp.EncodeToBytes([]interface{}{
			uint(params.VersionMajor<<16 | params.VersionMinor<<8 | params.VersionPatch),
			"cortex",
			runtime.Version(),
			runtime.GOOS,
		})
	}
	if uint64(len(extra)) > params.MaximumExtraDataSize {
		log.Warn("Miner extra data exceed limit", "extra", hexutil.Bytes(extra), "limit", params.MaximumExtraDataSize)
		extra = nil
	}
	return extra
}

// CreateDB creates the chain database.
func CreateDB(ctx *node.ServiceContext, config *Config, name string) (ctxcdb.Database, error) {
	db, err := ctx.OpenDatabase(name, config.DatabaseCache, config.DatabaseHandles)
	if err != nil {
		return nil, err
	}
	if db, ok := db.(*ctxcdb.LDBDatabase); ok {
		db.Meter("ctxc/db/chaindata/")
	}
	return db, nil
}

// CreateConsensusEngine creates the required type of consensus engine instance for an Cortex service
// func CreateConsensusEngine(ctx *node.ServiceContext, chainConfig *params.ChainConfig, config *cuckoo.Config, notify []string, db ctxcdb.Database) consensus.Engine {
func CreateConsensusEngine(ctx *node.ServiceContext, chainConfig *params.ChainConfig, config *cuckoo.Config, notify []string, noverify bool, db ctxcdb.Database) consensus.Engine {
	// If proof-of-authority is requested, set it up
	if chainConfig.Clique != nil {
		return clique.New(chainConfig.Clique, db)
	}
	// Otherwise assume proof-of-work
	switch config.PowMode {
	case cuckoo.ModeFake:
		log.Warn("Cuckoo used in fake mode")
		return cuckoo.NewFaker()
	case cuckoo.ModeTest:
		log.Warn("Cuckoo used in test mode")
		return cuckoo.NewTester()
	case cuckoo.ModeShared:
		log.Warn("Cuckoo used in shared mode")
		return cuckoo.NewShared()
	default:
		//	engine := cuckoo.New(cuckoo.Config{
		/* CacheDir:       ctx.ResolvePath(config.CacheDir),
			CachesInMem:    config.CachesInMem,
			CachesOnDisk:   config.CachesOnDisk,
			DatasetDir:     config.DatasetDir,
			DatasetsInMem:  config.DatasetsInMem,
			DatasetsOnDisk: config.DatasetsOnDisk,
		}, notify, noverify) */
		//	})
		engine := cuckoo.New(*config)
		//engine.SetThreads(-1) // Disable CPU mining
		return engine
	}
}

// APIs return the collection of RPC services the cortex package offers.
// NOTE, some of these services probably need to be moved to somewhere else.
func (s *Cortex) APIs() []rpc.API {
	apis := ctxcapi.GetAPIs(s.APIBackend, vm.Config{})

	// Append any APIs exposed explicitly by the consensus engine
	apis = append(apis, s.engine.APIs(s.BlockChain())...)

	// Append all the local APIs and return
	return append(apis, []rpc.API{
		{
			Namespace: "ctxc",
			Version:   "1.0",
			Service:   NewPublicCortexAPI(s),
			Public:    true,
		}, {
			Namespace: "ctxc",
			Version:   "1.0",
			Service:   NewPublicMinerAPI(s),
			Public:    true,
		}, {
			Namespace: "ctxc",
			Version:   "1.0",
			Service:   downloader.NewPublicDownloaderAPI(s.protocolManager.downloader, s.eventMux),
			Public:    true,
		}, {
			Namespace: "miner",
			Version:   "1.0",
			Service:   NewPrivateMinerAPI(s),
			Public:    false,
		}, {
			Namespace: "ctxc",
			Version:   "1.0",
			Service:   filters.NewPublicFilterAPI(s.APIBackend, false),
			Public:    true,
		}, {
			Namespace: "admin",
			Version:   "1.0",
			Service:   NewPrivateAdminAPI(s),
		}, {
			Namespace: "debug",
			Version:   "1.0",
			Service:   NewPublicDebugAPI(s),
			Public:    true,
		}, {
			Namespace: "debug",
			Version:   "1.0",
			Service:   NewPrivateDebugAPI(s.chainConfig, s),
		}, {
			Namespace: "net",
			Version:   "1.0",
			Service:   s.netRPCService,
			Public:    true,
		},

		// {
		// 	Namespace: "ctx",
		// 	Version:   "1.0",
		// 	Service:   NewPublicCortexAPI(s),
		// 	Public:    true,
		// }, {
		// 	Namespace: "ctx",
		// 	Version:   "1.0",
		// 	Service:   NewPublicMinerAPI(s),
		// 	Public:    true,
		// }, {
		// 	Namespace: "ctx",
		// 	Version:   "1.0",
		// 	Service:   downloader.NewPublicDownloaderAPI(s.protocolManager.downloader, s.eventMux),
		// 	Public:    true,
		// }, {
		// 	Namespace: "ctx",
		// 	Version:   "1.0",
		// 	Service:   filters.NewPublicFilterAPI(s.APIBackend, false),
		// 	Public:    true,
		// },
	}...)
}

func (s *Cortex) ResetWithGenesisBlock(gb *types.Block) {
	s.blockchain.ResetWithGenesisBlock(gb)
}

func (s *Cortex) Coinbase() (eb common.Address, err error) {
	s.lock.RLock()
	coinbase := s.coinbase
	s.lock.RUnlock()

	if coinbase != (common.Address{}) {
		return coinbase, nil
	}
	if wallets := s.AccountManager().Wallets(); len(wallets) > 0 {
		if accounts := wallets[0].Accounts(); len(accounts) > 0 {
			coinbase := accounts[0].Address

			s.lock.Lock()
			s.coinbase = coinbase
			s.lock.Unlock()

			log.Info("Coinbase automatically configured", "address", coinbase)
			return coinbase, nil
		}
	}
	return common.Address{}, fmt.Errorf("coinbase must be explicitly specified")
}

func (s *Cortex) isLocalBlock(block *types.Block) bool {
        author, err := s.engine.Author(block.Header())
        if err != nil {
                log.Warn("Failed to retrieve block author", "number", block.NumberU64(), "hash", block.Hash(), "err", err)
                return false
        }
        // Check whether the given address is etherbase.
        s.lock.RLock()
        coinbase := s.coinbase
        s.lock.RUnlock()
        if author == coinbase {
                return true
        }
        // Check whether the given address is specified by `txpool.local`
        // CLI flag.
        for _, account := range s.config.TxPool.Locals {
                if account == author {
                        return true
                }
        }
        return false
}

// shouldPreserve checks whether we should preserve the given block
// during the chain reorg depending on whether the author of block
// is a local account.
func (s *Cortex) shouldPreserve(block *types.Block) bool {
        // The reason we need to disable the self-reorg preserving for clique
        // is it can be probable to introduce a deadlock.
        //
        // e.g. If there are 7 available signers
        //
        // r1   A
        // r2     B
        // r3       C
        // r4         D
        // r5   A      [X] F G
        // r6    [X]
        //
        // In the round5, the inturn signer E is offline, so the worst case
        // is A, F and G sign the block of round5 and reject the block of opponents
        // and in the round6, the last available signer B is offline, the whole
        // network is stuck.
        if _, ok := s.engine.(*clique.Clique); ok {
                return false
        }
        return s.isLocalBlock(block)
}

// SetCoinbase sets the mining reward address.
func (s *Cortex) SetCoinbase(coinbase common.Address) {
	s.lock.Lock()
	s.coinbase = coinbase
	s.lock.Unlock()

	s.miner.SetCoinbase(coinbase)
}

// StartMining starts the miner with the given number of CPU threads. If mining
// is already running, this method adjust the number of threads allowed to use
// and updates the minimum price required by the transaction pool.
func (s *Cortex) StartMining(threads int) error {
	// Update the thread count within the consensus engine
	type threaded interface {
		SetThreads(threads int)
	}
	if th, ok := s.engine.(threaded); ok {
		log.Info("Updated mining threads", "threads", threads)
		if threads == 0 {
			threads = -1 // Disable the miner from within
		}
		th.SetThreads(threads)
	}
	// If the miner was not running, initialize it
	if !s.IsMining() {
		// Propagate the initial price point to the transaction pool
		s.lock.RLock()
		price := s.gasPrice
		s.lock.RUnlock()
		s.txPool.SetGasPrice(price)

		// Configure the local mining addess
		eb, err := s.Coinbase()
		if err != nil {
			log.Error("Cannot start mining without coinbase", "err", err)
			return fmt.Errorf("coinbase missing: %v", err)
		}
		if clique, ok := s.engine.(*clique.Clique); ok {
			wallet, err := s.accountManager.Find(accounts.Account{Address: eb})
			if wallet == nil || err != nil {
				log.Error("Coinbase account unavailable locally", "err", err)
				return fmt.Errorf("signer missing: %v", err)
			}
			clique.Authorize(eb, wallet.SignHash)
		}
		// If mining is started, we can disable the transaction rejection mechanism
		// introduced to speed sync times.
		atomic.StoreUint32(&s.protocolManager.acceptTxs, 1)

		go s.miner.Start(eb)
	}
	return nil
}

// StopMining terminates the miner, both at the consensus engine level as well as
// at the block creation level.
func (s *Cortex) StopMining() {
	// Update the thread count within the consensus engine
	type threaded interface {
		SetThreads(threads int)
	}
	if th, ok := s.engine.(threaded); ok {
		th.SetThreads(-1)
	}
	// Stop the block creating itself
	s.miner.Stop()
}

func (s *Cortex) IsMining() bool      { return s.miner.Mining() }
func (s *Cortex) Miner() *miner.Miner { return s.miner }

func (s *Cortex) AccountManager() *accounts.Manager  { return s.accountManager }
func (s *Cortex) BlockChain() *core.BlockChain       { return s.blockchain }
func (s *Cortex) TxPool() *core.TxPool               { return s.txPool }
func (s *Cortex) EventMux() *event.TypeMux           { return s.eventMux }
func (s *Cortex) Engine() consensus.Engine           { return s.engine }
func (s *Cortex) ChainDb() ctxcdb.Database           { return s.chainDb }
func (s *Cortex) IsListening() bool                  { return true } // Always listening
func (s *Cortex) CortexVersion() int                 { return int(s.protocolManager.SubProtocols[0].Version) }
func (s *Cortex) NetVersion() uint64                 { return s.networkID }
func (s *Cortex) Downloader() *downloader.Downloader { return s.protocolManager.downloader }

// Protocols implements node.Service, returning all the currently configured
// network protocols to start.
func (s *Cortex) Protocols() []p2p.Protocol {
	return s.protocolManager.SubProtocols
}

// Start implements node.Service, starting all internal goroutines needed by the
// Cortex protocol implementation.
func (s *Cortex) Start(srvr *p2p.Server) error {
	// Start the bloom bits servicing goroutines
	s.startBloomHandlers(params.BloomBitsBlocks)

	// Start the RPC service
	s.netRPCService = ctxcapi.NewPublicNetAPI(srvr, s.NetVersion())

	// Figure out a max peers count based on the server limits
	maxPeers := srvr.MaxPeers
	// Start the networking layer and the light server if requested
	s.protocolManager.Start(maxPeers)
	return nil
}

// Stop implements node.Service, terminating all internal goroutines used by the
// Cortex protocol.
func (s *Cortex) Stop() error {
	s.bloomIndexer.Close()
	s.blockchain.Stop()
	s.engine.Close()
	if s.synapse != nil {
		s.synapse.Close()
	}
	s.protocolManager.Stop()
	s.txPool.Stop()
	s.miner.Stop()
	s.eventMux.Stop()

	s.chainDb.Close()
	close(s.shutdownChan)
	return nil
}
