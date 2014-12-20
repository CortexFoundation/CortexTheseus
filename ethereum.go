package eth

import (
	"container/list"
	"encoding/json"
	"fmt"
	"math/big"
	"math/rand"
	"net"
	"path"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/ethutil"
	"github.com/ethereum/go-ethereum/event"
	"github.com/ethereum/go-ethereum/logger"
	"github.com/ethereum/go-ethereum/rpc"
	"github.com/ethereum/go-ethereum/state"
	"github.com/ethereum/go-ethereum/wire"
)

const (
	seedTextFileUri string = "http://www.ethereum.org/servers.poc3.txt"
	seedNodeAddress        = "poc-7.ethdev.com:30303"
)

var loggerger = logger.NewLogger("SERV")

func eachPeer(peers *list.List, callback func(*Peer, *list.Element)) {
	// Loop thru the peers and close them (if we had them)
	for e := peers.Front(); e != nil; e = e.Next() {
		callback(e.Value.(*Peer), e)
	}
}

const (
	processReapingTimeout = 60 // TODO increase
)

type Ethereum struct {
	// Channel for shutting down the ethereum
	shutdownChan chan bool
	quit         chan bool

	// DB interface
	db ethutil.Database
	// State manager for processing new blocks and managing the over all states
	blockManager *core.BlockManager
	// The transaction pool. Transaction can be pushed on this pool
	// for later including in the blocks
	txPool *core.TxPool
	// The canonical chain
	blockChain *core.ChainManager
	// The block pool
	blockPool *BlockPool
	// Eventer
	eventMux event.TypeMux
	// Peers
	peers *list.List
	// Nonce
	Nonce uint64

	Addr net.Addr
	Port string

	blacklist [][]byte

	peerMut sync.Mutex

	// Capabilities for outgoing peers
	serverCaps Caps

	nat NAT

	// Specifies the desired amount of maximum peers
	MaxPeers int

	Mining bool

	listening bool

	RpcServer *rpc.JsonRpcServer

	keyManager *crypto.KeyManager

	clientIdentity wire.ClientIdentity

	isUpToDate bool

	filterMu sync.RWMutex
	filterId int
	filters  map[int]*core.Filter
}

func New(db ethutil.Database, clientIdentity wire.ClientIdentity, keyManager *crypto.KeyManager, caps Caps, usePnp bool) (*Ethereum, error) {
	var err error
	var nat NAT

	if usePnp {
		nat, err = Discover()
		if err != nil {
			loggerger.Debugln("UPnP failed", err)
		}
	}

	bootstrapDb(db)

	ethutil.Config.Db = db

	nonce, _ := ethutil.RandomUint64()
	ethereum := &Ethereum{
		shutdownChan:   make(chan bool),
		quit:           make(chan bool),
		db:             db,
		peers:          list.New(),
		Nonce:          nonce,
		serverCaps:     caps,
		nat:            nat,
		keyManager:     keyManager,
		clientIdentity: clientIdentity,
		isUpToDate:     true,
		filters:        make(map[int]*core.Filter),
	}

	ethereum.blockPool = NewBlockPool(ethereum)
	ethereum.blockChain = core.NewChainManager(ethereum.EventMux())
	ethereum.txPool = core.NewTxPool(ethereum.blockChain, ethereum, ethereum.EventMux())
	ethereum.blockManager = core.NewBlockManager(ethereum.txPool, ethereum.blockChain, ethereum.EventMux())
	ethereum.blockChain.SetProcessor(ethereum.blockManager)

	// Start the tx pool
	ethereum.txPool.Start()

	return ethereum, nil
}

func (s *Ethereum) KeyManager() *crypto.KeyManager {
	return s.keyManager
}

func (s *Ethereum) ClientIdentity() wire.ClientIdentity {
	return s.clientIdentity
}

func (s *Ethereum) ChainManager() *core.ChainManager {
	return s.blockChain
}

func (s *Ethereum) BlockManager() *core.BlockManager {
	return s.blockManager
}

func (s *Ethereum) TxPool() *core.TxPool {
	return s.txPool
}
func (s *Ethereum) BlockPool() *BlockPool {
	return s.blockPool
}
func (s *Ethereum) EventMux() *event.TypeMux {
	return &s.eventMux
}
func (self *Ethereum) Db() ethutil.Database {
	return self.db
}

func (s *Ethereum) ServerCaps() Caps {
	return s.serverCaps
}
func (s *Ethereum) IsMining() bool {
	return s.Mining
}
func (s *Ethereum) PeerCount() int {
	return s.peers.Len()
}
func (s *Ethereum) IsUpToDate() bool {
	upToDate := true
	eachPeer(s.peers, func(peer *Peer, e *list.Element) {
		if atomic.LoadInt32(&peer.connected) == 1 {
			if peer.catchingUp == true && peer.versionKnown {
				upToDate = false
			}
		}
	})
	return upToDate
}
func (s *Ethereum) PushPeer(peer *Peer) {
	s.peers.PushBack(peer)
}
func (s *Ethereum) IsListening() bool {
	return s.listening
}

func (s *Ethereum) HighestTDPeer() (td *big.Int) {
	td = big.NewInt(0)

	eachPeer(s.peers, func(p *Peer, v *list.Element) {
		if p.td.Cmp(td) > 0 {
			td = p.td
		}
	})

	return
}

func (self *Ethereum) BlacklistPeer(peer *Peer) {
	self.blacklist = append(self.blacklist, peer.pubkey)
}

func (s *Ethereum) AddPeer(conn net.Conn) {
	peer := NewPeer(conn, s, true)

	if peer != nil {
		if s.peers.Len() < s.MaxPeers {
			peer.Start()
		} else {
			loggerger.Debugf("Max connected peers reached. Not adding incoming peer.")
		}
	}
}

func (s *Ethereum) ProcessPeerList(addrs []string) {
	for _, addr := range addrs {
		// TODO Probably requires some sanity checks
		s.ConnectToPeer(addr)
	}
}

func (s *Ethereum) ConnectToPeer(addr string) error {
	if s.peers.Len() < s.MaxPeers {
		var alreadyConnected bool

		ahost, aport, _ := net.SplitHostPort(addr)
		var chost string

		ips, err := net.LookupIP(ahost)

		if err != nil {
			return err
		} else {
			// If more then one ip is available try stripping away the ipv6 ones
			if len(ips) > 1 {
				var ipsv4 []net.IP
				// For now remove the ipv6 addresses
				for _, ip := range ips {
					if strings.Contains(ip.String(), "::") {
						continue
					} else {
						ipsv4 = append(ipsv4, ip)
					}
				}
				if len(ipsv4) == 0 {
					return fmt.Errorf("[SERV] No IPV4 addresses available for hostname")
				}

				// Pick a random ipv4 address, simulating round-robin DNS.
				rand.Seed(time.Now().UTC().UnixNano())
				i := rand.Intn(len(ipsv4))
				chost = ipsv4[i].String()
			} else {
				if len(ips) == 0 {
					return fmt.Errorf("[SERV] No IPs resolved for the given hostname")
					return nil
				}
				chost = ips[0].String()
			}
		}

		eachPeer(s.peers, func(p *Peer, v *list.Element) {
			if p.conn == nil {
				return
			}
			phost, pport, _ := net.SplitHostPort(p.conn.RemoteAddr().String())

			if phost == chost && pport == aport {
				alreadyConnected = true
				//loggerger.Debugf("Peer %s already added.\n", chost)
				return
			}
		})

		if alreadyConnected {
			return nil
		}

		NewOutboundPeer(addr, s, s.serverCaps)
	}

	return nil
}

func (s *Ethereum) OutboundPeers() []*Peer {
	// Create a new peer slice with at least the length of the total peers
	outboundPeers := make([]*Peer, s.peers.Len())
	length := 0
	eachPeer(s.peers, func(p *Peer, e *list.Element) {
		if !p.inbound && p.conn != nil {
			outboundPeers[length] = p
			length++
		}
	})

	return outboundPeers[:length]
}

func (s *Ethereum) InboundPeers() []*Peer {
	// Create a new peer slice with at least the length of the total peers
	inboundPeers := make([]*Peer, s.peers.Len())
	length := 0
	eachPeer(s.peers, func(p *Peer, e *list.Element) {
		if p.inbound {
			inboundPeers[length] = p
			length++
		}
	})

	return inboundPeers[:length]
}

func (s *Ethereum) InOutPeers() []*Peer {
	// Reap the dead peers first
	s.reapPeers()

	// Create a new peer slice with at least the length of the total peers
	inboundPeers := make([]*Peer, s.peers.Len())
	length := 0
	eachPeer(s.peers, func(p *Peer, e *list.Element) {
		// Only return peers with an actual ip
		if len(p.host) > 0 {
			inboundPeers[length] = p
			length++
		}
	})

	return inboundPeers[:length]
}

func (s *Ethereum) Broadcast(msgType wire.MsgType, data []interface{}) {
	msg := wire.NewMessage(msgType, data)
	s.BroadcastMsg(msg)
}

func (s *Ethereum) BroadcastMsg(msg *wire.Msg) {
	eachPeer(s.peers, func(p *Peer, e *list.Element) {
		p.QueueMessage(msg)
	})
}

func (s *Ethereum) Peers() *list.List {
	return s.peers
}

func (s *Ethereum) reapPeers() {
	eachPeer(s.peers, func(p *Peer, e *list.Element) {
		if atomic.LoadInt32(&p.disconnect) == 1 || (p.inbound && (time.Now().Unix()-p.lastPong) > int64(5*time.Minute)) {
			s.removePeerElement(e)
		}
	})
}

func (s *Ethereum) removePeerElement(e *list.Element) {
	s.peerMut.Lock()
	defer s.peerMut.Unlock()

	s.peers.Remove(e)

	s.eventMux.Post(PeerListEvent{s.peers})
}

func (s *Ethereum) RemovePeer(p *Peer) {
	eachPeer(s.peers, func(peer *Peer, e *list.Element) {
		if peer == p {
			s.removePeerElement(e)
		}
	})
}

func (s *Ethereum) reapDeadPeerHandler() {
	reapTimer := time.NewTicker(processReapingTimeout * time.Second)

	for {
		select {
		case <-reapTimer.C:
			s.reapPeers()
		}
	}
}

// Start the ethereum
func (s *Ethereum) Start(seed bool) {
	s.blockPool.Start()

	// Bind to addr and port
	ln, err := net.Listen("tcp", ":"+s.Port)
	if err != nil {
		loggerger.Warnf("Port %s in use. Connection listening disabled. Acting as client", s.Port)
		s.listening = false
	} else {
		s.listening = true
		// Starting accepting connections
		loggerger.Infoln("Ready and accepting connections")
		// Start the peer handler
		go s.peerHandler(ln)
	}

	if s.nat != nil {
		go s.upnpUpdateThread()
	}

	// Start the reaping processes
	go s.reapDeadPeerHandler()
	go s.update()
	go s.filterLoop()

	if seed {
		s.Seed()
	}
	s.ConnectToPeer("localhost:40404")
	loggerger.Infoln("Server started")
}

func (s *Ethereum) Seed() {
	// Sorry Py person. I must blacklist. you perform badly
	s.blacklist = append(s.blacklist, ethutil.Hex2Bytes("64656330303561383532336435376331616537643864663236623336313863373537353163636634333530626263396330346237336262623931383064393031"))
	ips := PastPeers()
	if len(ips) > 0 {
		for _, ip := range ips {
			loggerger.Infoln("Connecting to previous peer ", ip)
			s.ConnectToPeer(ip)
		}
	} else {
		loggerger.Debugln("Retrieving seed nodes")

		// Eth-Go Bootstrapping
		ips, er := net.LookupIP("seed.bysh.me")
		if er == nil {
			peers := []string{}
			for _, ip := range ips {
				node := fmt.Sprintf("%s:%d", ip.String(), 30303)
				loggerger.Debugln("Found DNS Go Peer:", node)
				peers = append(peers, node)
			}
			s.ProcessPeerList(peers)
		}

		// Official DNS Bootstrapping
		_, nodes, err := net.LookupSRV("eth", "tcp", "ethereum.org")
		if err == nil {
			peers := []string{}
			// Iterate SRV nodes
			for _, n := range nodes {
				target := n.Target
				port := strconv.Itoa(int(n.Port))
				// Resolve target to ip (Go returns list, so may resolve to multiple ips?)
				addr, err := net.LookupHost(target)
				if err == nil {
					for _, a := range addr {
						// Build string out of SRV port and Resolved IP
						peer := net.JoinHostPort(a, port)
						loggerger.Debugln("Found DNS Bootstrap Peer:", peer)
						peers = append(peers, peer)
					}
				} else {
					loggerger.Debugln("Couldn't resolve :", target)
				}
			}
			// Connect to Peer list
			s.ProcessPeerList(peers)
		}

		s.ConnectToPeer(seedNodeAddress)
	}
}

func (s *Ethereum) peerHandler(listener net.Listener) {
	for {
		conn, err := listener.Accept()
		if err != nil {
			loggerger.Debugln(err)

			continue
		}

		go s.AddPeer(conn)
	}
}

func (s *Ethereum) Stop() {
	// Stop eventMux first, it will close all subscriptions.
	s.eventMux.Stop()

	// Close the database
	defer s.db.Close()

	var ips []string
	eachPeer(s.peers, func(p *Peer, e *list.Element) {
		ips = append(ips, p.conn.RemoteAddr().String())
	})

	if len(ips) > 0 {
		d, _ := json.MarshalIndent(ips, "", "    ")
		ethutil.WriteFile(path.Join(ethutil.Config.ExecPath, "known_peers.json"), d)
	}

	eachPeer(s.peers, func(p *Peer, e *list.Element) {
		p.Stop()
	})

	close(s.quit)

	if s.RpcServer != nil {
		s.RpcServer.Stop()
	}
	s.txPool.Stop()
	s.blockPool.Stop()

	loggerger.Infoln("Server stopped")
	close(s.shutdownChan)
}

// This function will wait for a shutdown and resumes main thread execution
func (s *Ethereum) WaitForShutdown() {
	<-s.shutdownChan
}

func (s *Ethereum) upnpUpdateThread() {
	// Go off immediately to prevent code duplication, thereafter we renew
	// lease every 15 minutes.
	timer := time.NewTimer(5 * time.Minute)
	lport, _ := strconv.ParseInt(s.Port, 10, 16)
	first := true
out:
	for {
		select {
		case <-timer.C:
			var err error
			_, err = s.nat.AddPortMapping("TCP", int(lport), int(lport), "eth listen port", 20*60)
			if err != nil {
				loggerger.Debugln("can't add UPnP port mapping:", err)
				break out
			}
			if first && err == nil {
				_, err = s.nat.GetExternalAddress()
				if err != nil {
					loggerger.Debugln("UPnP can't get external address:", err)
					continue out
				}
				first = false
			}
			timer.Reset(time.Minute * 15)
		case <-s.quit:
			break out
		}
	}

	timer.Stop()

	if err := s.nat.DeletePortMapping("TCP", int(lport), int(lport)); err != nil {
		loggerger.Debugln("unable to remove UPnP port mapping:", err)
	} else {
		loggerger.Debugln("succesfully disestablished UPnP port mapping")
	}
}

func (self *Ethereum) update() {
	upToDateTimer := time.NewTicker(1 * time.Second)

out:
	for {
		select {
		case <-upToDateTimer.C:
			if self.IsUpToDate() && !self.isUpToDate {
				self.eventMux.Post(ChainSyncEvent{false})
				self.isUpToDate = true
			} else if !self.IsUpToDate() && self.isUpToDate {
				self.eventMux.Post(ChainSyncEvent{true})
				self.isUpToDate = false
			}
		case <-self.quit:
			break out
		}
	}
}

// InstallFilter adds filter for blockchain events.
// The filter's callbacks will run for matching blocks and messages.
// The filter should not be modified after it has been installed.
func (self *Ethereum) InstallFilter(filter *core.Filter) (id int) {
	self.filterMu.Lock()
	id = self.filterId
	self.filters[id] = filter
	self.filterId++
	self.filterMu.Unlock()
	return id
}

func (self *Ethereum) UninstallFilter(id int) {
	self.filterMu.Lock()
	delete(self.filters, id)
	self.filterMu.Unlock()
}

// GetFilter retrieves a filter installed using InstallFilter.
// The filter may not be modified.
func (self *Ethereum) GetFilter(id int) *core.Filter {
	self.filterMu.RLock()
	defer self.filterMu.RUnlock()
	return self.filters[id]
}

func (self *Ethereum) filterLoop() {
	// Subscribe to events
	events := self.eventMux.Subscribe(core.NewBlockEvent{}, state.Messages(nil))
	for event := range events.Chan() {
		switch event := event.(type) {
		case core.NewBlockEvent:
			self.filterMu.RLock()
			for _, filter := range self.filters {
				if filter.BlockCallback != nil {
					filter.BlockCallback(event.Block)
				}
			}
			self.filterMu.RUnlock()

		case state.Messages:
			self.filterMu.RLock()
			for _, filter := range self.filters {
				if filter.MessageCallback != nil {
					msgs := filter.FilterMessages(event)
					if len(msgs) > 0 {
						filter.MessageCallback(msgs)
					}
				}
			}
			self.filterMu.RUnlock()
		}
	}
}

func bootstrapDb(db ethutil.Database) {
	d, _ := db.Get([]byte("ProtocolVersion"))
	protov := ethutil.NewValue(d).Uint()

	if protov == 0 {
		db.Put([]byte("ProtocolVersion"), ethutil.NewValue(ProtocolVersion).Bytes())
	}
}

func PastPeers() []string {
	var ips []string
	data, _ := ethutil.ReadAllFile(path.Join(ethutil.Config.ExecPath, "known_peers.json"))
	json.Unmarshal([]byte(data), &ips)

	return ips
}
