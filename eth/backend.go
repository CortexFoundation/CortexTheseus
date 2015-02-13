package eth

import (
	"crypto/ecdsa"
	"fmt"
	"strings"

	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/ethdb"
	"github.com/ethereum/go-ethereum/ethutil"
	"github.com/ethereum/go-ethereum/event"
	ethlogger "github.com/ethereum/go-ethereum/logger"
	"github.com/ethereum/go-ethereum/p2p"
	"github.com/ethereum/go-ethereum/p2p/discover"
	"github.com/ethereum/go-ethereum/p2p/nat"
	"github.com/ethereum/go-ethereum/pow/ezp"
	"github.com/ethereum/go-ethereum/rpc"
	"github.com/ethereum/go-ethereum/whisper"
)

var logger = ethlogger.NewLogger("SERV")

type Config struct {
	Name     string
	KeyStore string
	DataDir  string
	LogFile  string
	LogLevel int
	KeyRing  string

	MaxPeers int
	Port     string

	// This should be a space-separated list of
	// discovery node URLs.
	BootNodes string

	// This key is used to identify the node on the network.
	// If nil, an ephemeral key is used.
	NodeKey *ecdsa.PrivateKey

	NAT  nat.Interface
	Shh  bool
	Dial bool

	KeyManager *crypto.KeyManager
}

var logger = ethlogger.NewLogger("SERV")
var jsonlogger = ethlogger.NewJsonLogger()

func (cfg *Config) parseBootNodes() []*discover.Node {
	var ns []*discover.Node
	for _, url := range strings.Split(cfg.BootNodes, " ") {
		if url == "" {
			continue
		}
		n, err := discover.ParseNode(url)
		if err != nil {
			logger.Errorf("Bootstrap URL %s: %v\n", url, err)
			continue
		}
		ns = append(ns, n)
	}
	return ns
}

type Ethereum struct {
	// Channel for shutting down the ethereum
	shutdownChan chan bool
	quit         chan bool

	// DB interface
	db        ethutil.Database
	blacklist p2p.Blacklist

	//*** SERVICES ***
	// State manager for processing new blocks and managing the over all states
	blockProcessor *core.BlockProcessor
	txPool         *core.TxPool
	chainManager   *core.ChainManager
	blockPool      *BlockPool
	whisper        *whisper.Whisper

	net      *p2p.Server
	eventMux *event.TypeMux
	txSub    event.Subscription
	blockSub event.Subscription

	RpcServer  rpc.RpcServer
	WsServer   rpc.RpcServer
	keyManager *crypto.KeyManager

	logger ethlogger.LogSystem

	Mining bool
}

func New(config *Config) (*Ethereum, error) {
	// Boostrap database
	logger := ethlogger.New(config.DataDir, config.LogFile, config.LogLevel, config.LogFormat)
	db, err := ethdb.NewLDBDatabase("blockchain")
	if err != nil {
		return nil, err
	}

	// Perform database sanity checks
	d, _ := db.Get([]byte("ProtocolVersion"))
	protov := ethutil.NewValue(d).Uint()
	if protov != ProtocolVersion && protov != 0 {
		return nil, fmt.Errorf("Database version mismatch. Protocol(%d / %d). `rm -rf %s`", protov, ProtocolVersion, ethutil.Config.ExecPath+"/database")
	}

	// Create new keymanager
	var keyManager *crypto.KeyManager
	switch config.KeyStore {
	case "db":
		keyManager = crypto.NewDBKeyManager(db)
	case "file":
		keyManager = crypto.NewFileKeyManager(config.DataDir)
	default:
		return nil, fmt.Errorf("unknown keystore type: %s", config.KeyStore)
	}
	// Initialise the keyring
	keyManager.Init(config.KeyRing, 0, false)

	saveProtocolVersion(db)
	//ethutil.Config.Db = db

	eth := &Ethereum{
		shutdownChan: make(chan bool),
		quit:         make(chan bool),
		db:           db,
		keyManager:   keyManager,
		blacklist:    p2p.NewBlacklist(),
		eventMux:     &event.TypeMux{},
		logger:       logger,
	}

	eth.chainManager = core.NewChainManager(db, eth.EventMux())
	eth.txPool = core.NewTxPool(eth.EventMux())
	eth.blockProcessor = core.NewBlockProcessor(db, eth.txPool, eth.chainManager, eth.EventMux())
	eth.chainManager.SetProcessor(eth.blockProcessor)
	eth.whisper = whisper.New()

	hasBlock := eth.chainManager.HasBlock
	insertChain := eth.chainManager.InsertChain
	eth.blockPool = NewBlockPool(hasBlock, insertChain, ezp.Verify)

	ethProto := EthProtocol(eth.txPool, eth.chainManager, eth.blockPool)
	protocols := []p2p.Protocol{ethProto, eth.whisper.Protocol()}
	netprv := config.NodeKey
	if netprv == nil {
		if netprv, err = crypto.GenerateKey(); err != nil {
			return nil, fmt.Errorf("could not generate server key: %v", err)
		}
	}
	eth.net = &p2p.Server{
		PrivateKey:     netprv,
		Name:           config.Name,
		MaxPeers:       config.MaxPeers,
		Protocols:      protocols,
		Blacklist:      eth.blacklist,
		NAT:            config.NAT,
		NoDial:         !config.Dial,
		BootstrapNodes: config.parseBootNodes(),
	}
	if len(config.Port) > 0 {
		eth.net.ListenAddr = ":" + config.Port
	}

	return eth, nil
}

func (s *Ethereum) KeyManager() *crypto.KeyManager {
	return s.keyManager
}

func (s *Ethereum) Logger() ethlogger.LogSystem {
	return s.logger
}

func (s *Ethereum) Name() string {
	return s.net.Name
}

func (s *Ethereum) ChainManager() *core.ChainManager {
	return s.chainManager
}

func (s *Ethereum) BlockProcessor() *core.BlockProcessor {
	return s.blockProcessor
}

func (s *Ethereum) TxPool() *core.TxPool {
	return s.txPool
}

func (s *Ethereum) BlockPool() *BlockPool {
	return s.blockPool
}

func (s *Ethereum) Whisper() *whisper.Whisper {
	return s.whisper
}

func (s *Ethereum) EventMux() *event.TypeMux {
	return s.eventMux
}
func (self *Ethereum) Db() ethutil.Database {
	return self.db
}

func (s *Ethereum) IsMining() bool {
	return s.Mining
}

func (s *Ethereum) IsListening() bool {
	// XXX TODO
	return false
}

func (s *Ethereum) PeerCount() int {
	return s.net.PeerCount()
}

func (s *Ethereum) Peers() []*p2p.Peer {
	return s.net.Peers()
}

func (s *Ethereum) MaxPeers() int {
	return s.net.MaxPeers
}

func (s *Ethereum) Coinbase() []byte {
	return nil // TODO
}

// Start the ethereum
func (s *Ethereum) Start() error {
	jsonlogger.LogJson(&ethlogger.LogStarting{
		ClientString:    s.ClientIdentity().String(),
		Coinbase:        ethutil.Bytes2Hex(s.KeyManager().Address()),
		ProtocolVersion: ProtocolVersion,
		LogEvent:        ethlogger.LogEvent{Guid: ethutil.Bytes2Hex(s.ClientIdentity().Pubkey())},
	})

	err := s.net.Start()
	if err != nil {
		return err
	}

	// Start services
	s.txPool.Start()
	s.blockPool.Start()

	if s.whisper != nil {
		s.whisper.Start()
	}

	// broadcast transactions
	s.txSub = s.eventMux.Subscribe(core.TxPreEvent{})
	go s.txBroadcastLoop()

	// broadcast mined blocks
	s.blockSub = s.eventMux.Subscribe(core.NewMinedBlockEvent{})
	go s.blockBroadcastLoop()

	logger.Infoln("Server started")
	return nil
}

func (self *Ethereum) SuggestPeer(nodeURL string) error {
	n, err := discover.ParseNode(nodeURL)
	if err != nil {
		return fmt.Errorf("invalid node URL: %v", err)
	}
	self.net.SuggestPeer(n)
	return nil
}

func (s *Ethereum) Stop() {
	// Close the database
	defer s.db.Close()

	close(s.quit)

	s.txSub.Unsubscribe()    // quits txBroadcastLoop
	s.blockSub.Unsubscribe() // quits blockBroadcastLoop

	if s.RpcServer != nil {
		s.RpcServer.Stop()
	}
	if s.WsServer != nil {
		s.WsServer.Stop()
	}
	s.txPool.Stop()
	s.eventMux.Stop()
	s.blockPool.Stop()
	if s.whisper != nil {
		s.whisper.Stop()
	}

	logger.Infoln("Server stopped")
	close(s.shutdownChan)
}

// This function will wait for a shutdown and resumes main thread execution
func (s *Ethereum) WaitForShutdown() {
	<-s.shutdownChan
}

// now tx broadcasting is taken out of txPool
// handled here via subscription, efficiency?
func (self *Ethereum) txBroadcastLoop() {
	// automatically stops if unsubscribe
	for obj := range self.txSub.Chan() {
		event := obj.(core.TxPreEvent)
		self.net.Broadcast("eth", TxMsg, event.Tx.RlpData())
	}
}

func (self *Ethereum) blockBroadcastLoop() {
	// automatically stops if unsubscribe
	for obj := range self.blockSub.Chan() {
		switch ev := obj.(type) {
		case core.NewMinedBlockEvent:
			self.net.Broadcast("eth", NewBlockMsg, ev.Block.RlpData(), ev.Block.Td)
		}
	}
}

func saveProtocolVersion(db ethutil.Database) {
	d, _ := db.Get([]byte("ProtocolVersion"))
	protocolVersion := ethutil.NewValue(d).Uint()

	if protocolVersion == 0 {
		db.Put([]byte("ProtocolVersion"), ethutil.NewValue(ProtocolVersion).Bytes())
	}
}
