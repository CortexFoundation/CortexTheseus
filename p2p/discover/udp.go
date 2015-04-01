package discover

import (
	"bytes"
	"crypto/ecdsa"
	"errors"
	"fmt"
	"net"
	"time"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/logger"
	"github.com/ethereum/go-ethereum/p2p/nat"
	"github.com/ethereum/go-ethereum/rlp"
)

var log = logger.NewLogger("P2P Discovery")

const Version = 3

// Errors
var (
	errPacketTooSmall   = errors.New("too small")
	errBadHash          = errors.New("bad hash")
	errExpired          = errors.New("expired")
	errBadVersion       = errors.New("version mismatch")
	errUnsolicitedReply = errors.New("unsolicited reply")
	errUnknownNode      = errors.New("unknown node")
	errTimeout          = errors.New("RPC timeout")
	errClosed           = errors.New("socket closed")
)

// Timeouts
const (
	respTimeout = 300 * time.Millisecond
	sendTimeout = 300 * time.Millisecond
	expiration  = 20 * time.Second

	refreshInterval = 1 * time.Hour
)

// RPC packet types
const (
	pingPacket = iota + 1 // zero is 'reserved'
	pongPacket
	findnodePacket
	neighborsPacket
)

// RPC request structures
type (
	ping struct {
		Version    uint   // must match Version
		IP         string // our IP
		Port       uint16 // our port
		Expiration uint64
	}

	// reply to Ping
	pong struct {
		ReplyTok   []byte
		Expiration uint64
	}

	findnode struct {
		// Id to look up. The responding node will send back nodes
		// closest to the target.
		Target     NodeID
		Expiration uint64
	}

	// reply to findnode
	neighbors struct {
		Nodes      []*Node
		Expiration uint64
	}
)

type rpcNode struct {
	IP   string
	Port uint16
	ID   NodeID
}

type packet interface {
	handle(t *udp, from *net.UDPAddr, fromID NodeID, mac []byte) error
}

type conn interface {
	ReadFromUDP(b []byte) (n int, addr *net.UDPAddr, err error)
	WriteToUDP(b []byte, addr *net.UDPAddr) (n int, err error)
	Close() error
	LocalAddr() net.Addr
}

// udp implements the RPC protocol.
type udp struct {
	conn conn
	priv *ecdsa.PrivateKey

	addpending chan *pending
	gotreply   chan reply

	closing chan struct{}
	nat     nat.Interface

	*Table
}

// pending represents a pending reply.
//
// some implementations of the protocol wish to send more than one
// reply packet to findnode. in general, any neighbors packet cannot
// be matched up with a specific findnode packet.
//
// our implementation handles this by storing a callback function for
// each pending reply. incoming packets from a node are dispatched
// to all the callback functions for that node.
type pending struct {
	// these fields must match in the reply.
	from  NodeID
	ptype byte

	// time when the request must complete
	deadline time.Time

	// callback is called when a matching reply arrives. if it returns
	// true, the callback is removed from the pending reply queue.
	// if it returns false, the reply is considered incomplete and
	// the callback will be invoked again for the next matching reply.
	callback func(resp interface{}) (done bool)

	// errc receives nil when the callback indicates completion or an
	// error if no further reply is received within the timeout.
	errc chan<- error
}

type reply struct {
	from  NodeID
	ptype byte
	data  interface{}
	// loop indicates whether there was
	// a matching request by sending on this channel.
	matched chan<- bool
}

// ListenUDP returns a new table that listens for UDP packets on laddr.
func ListenUDP(priv *ecdsa.PrivateKey, laddr string, natm nat.Interface) (*Table, error) {
	addr, err := net.ResolveUDPAddr("udp", laddr)
	if err != nil {
		return nil, err
	}
	conn, err := net.ListenUDP("udp", addr)
	if err != nil {
		return nil, err
	}
	tab, _ := newUDP(priv, conn, natm)
	log.Infoln("Listening,", tab.self)
	return tab, nil
}

func newUDP(priv *ecdsa.PrivateKey, c conn, natm nat.Interface) (*Table, *udp) {
	udp := &udp{
		conn:       c,
		priv:       priv,
		closing:    make(chan struct{}),
		gotreply:   make(chan reply),
		addpending: make(chan *pending),
	}
	realaddr := c.LocalAddr().(*net.UDPAddr)
	if natm != nil {
		if !realaddr.IP.IsLoopback() {
			go nat.Map(natm, udp.closing, "udp", realaddr.Port, realaddr.Port, "ethereum discovery")
		}
		// TODO: react to external IP changes over time.
		if ext, err := natm.ExternalIP(); err == nil {
			realaddr = &net.UDPAddr{IP: ext, Port: realaddr.Port}
		}
	}
	udp.Table = newTable(udp, PubkeyID(&priv.PublicKey), realaddr)
	go udp.loop()
	go udp.readLoop()
	return udp.Table, udp
}

func (t *udp) close() {
	close(t.closing)
	t.conn.Close()
	// TODO: wait for the loops to end.
}

// ping sends a ping message to the given node and waits for a reply.
func (t *udp) ping(toid NodeID, toaddr *net.UDPAddr) error {
	// TODO: maybe check for ReplyTo field in callback to measure RTT
	errc := t.pending(toid, pongPacket, func(interface{}) bool { return true })
	t.send(toaddr, pingPacket, ping{
		Version:    Version,
		IP:         t.self.IP.String(),
		Port:       uint16(t.self.TCPPort),
		Expiration: uint64(time.Now().Add(expiration).Unix()),
	})
	return <-errc
}

func (t *udp) waitping(from NodeID) error {
	return <-t.pending(from, pingPacket, func(interface{}) bool { return true })
}

// findnode sends a findnode request to the given node and waits until
// the node has sent up to k neighbors.
func (t *udp) findnode(toid NodeID, toaddr *net.UDPAddr, target NodeID) ([]*Node, error) {
	nodes := make([]*Node, 0, bucketSize)
	nreceived := 0
	errc := t.pending(toid, neighborsPacket, func(r interface{}) bool {
		reply := r.(*neighbors)
		for _, n := range reply.Nodes {
			nreceived++
			if n.isValid() {
				nodes = append(nodes, n)
			}
		}
		return nreceived >= bucketSize
	})
	t.send(toaddr, findnodePacket, findnode{
		Target:     target,
		Expiration: uint64(time.Now().Add(expiration).Unix()),
	})
	err := <-errc
	return nodes, err
}

// pending adds a reply callback to the pending reply queue.
// see the documentation of type pending for a detailed explanation.
func (t *udp) pending(id NodeID, ptype byte, callback func(interface{}) bool) <-chan error {
	ch := make(chan error, 1)
	p := &pending{from: id, ptype: ptype, callback: callback, errc: ch}
	select {
	case t.addpending <- p:
		// loop will handle it
	case <-t.closing:
		ch <- errClosed
	}
	return ch
}

func (t *udp) handleReply(from NodeID, ptype byte, req packet) bool {
	matched := make(chan bool)
	select {
	case t.gotreply <- reply{from, ptype, req, matched}:
		// loop will handle it
		return <-matched
	case <-t.closing:
		return false
	}
}

// loop runs in its own goroutin. it keeps track of
// the refresh timer and the pending reply queue.
func (t *udp) loop() {
	var (
		pending      []*pending
		nextDeadline time.Time
		timeout      = time.NewTimer(0)
		refresh      = time.NewTicker(refreshInterval)
	)
	<-timeout.C // ignore first timeout
	defer refresh.Stop()
	defer timeout.Stop()

	rearmTimeout := func() {
		if len(pending) == 0 || nextDeadline == pending[0].deadline {
			return
		}
		nextDeadline = pending[0].deadline
		timeout.Reset(nextDeadline.Sub(time.Now()))
	}

	for {
		select {
		case <-refresh.C:
			go t.refresh()

		case <-t.closing:
			for _, p := range pending {
				p.errc <- errClosed
			}
			pending = nil
			return

		case p := <-t.addpending:
			p.deadline = time.Now().Add(respTimeout)
			pending = append(pending, p)
			rearmTimeout()

		case r := <-t.gotreply:
			var matched bool
			for i := 0; i < len(pending); i++ {
				if p := pending[i]; p.from == r.from && p.ptype == r.ptype {
					matched = true
					if p.callback(r.data) {
						// callback indicates the request is done, remove it.
						p.errc <- nil
						copy(pending[i:], pending[i+1:])
						pending = pending[:len(pending)-1]
						i--
					}
				}
			}
			r.matched <- matched

		case now := <-timeout.C:
			// notify and remove callbacks whose deadline is in the past.
			i := 0
			for ; i < len(pending) && now.After(pending[i].deadline); i++ {
				pending[i].errc <- errTimeout
			}
			if i > 0 {
				copy(pending, pending[i:])
				pending = pending[:len(pending)-i]
			}
			rearmTimeout()
		}
	}
}

const (
	macSize  = 256 / 8
	sigSize  = 520 / 8
	headSize = macSize + sigSize // space of packet frame data
)

var headSpace = make([]byte, headSize)

func (t *udp) send(toaddr *net.UDPAddr, ptype byte, req interface{}) error {
	packet, err := encodePacket(t.priv, ptype, req)
	if err != nil {
		return err
	}
	log.DebugDetailf(">>> %v %T %v\n", toaddr, req, req)
	if _, err = t.conn.WriteToUDP(packet, toaddr); err != nil {
		log.DebugDetailln("UDP send failed:", err)
	}
	return err
}

func encodePacket(priv *ecdsa.PrivateKey, ptype byte, req interface{}) ([]byte, error) {
	b := new(bytes.Buffer)
	b.Write(headSpace)
	b.WriteByte(ptype)
	if err := rlp.Encode(b, req); err != nil {
		log.Errorln("error encoding packet:", err)
		return nil, err
	}
	packet := b.Bytes()
	sig, err := crypto.Sign(crypto.Sha3(packet[headSize:]), priv)
	if err != nil {
		log.Errorln("could not sign packet:", err)
		return nil, err
	}
	copy(packet[macSize:], sig)
	// add the hash to the front. Note: this doesn't protect the
	// packet in any way. Our public key will be part of this hash in
	// The future.
	copy(packet, crypto.Sha3(packet[macSize:]))
	return packet, nil
}

// readLoop runs in its own goroutine. it handles incoming UDP packets.
func (t *udp) readLoop() {
	defer t.conn.Close()
	buf := make([]byte, 4096) // TODO: good buffer size
	for {
		nbytes, from, err := t.conn.ReadFromUDP(buf)
		if err != nil {
			return
		}
		packet, fromID, hash, err := decodePacket(buf[:nbytes])
		if err != nil {
			log.Debugf("Bad packet from %v: %v\n", from, err)
			continue
		}
		log.DebugDetailf("<<< %v %T %v\n", from, packet, packet)
		go func() {
			if err := packet.handle(t, from, fromID, hash); err != nil {
				log.Debugf("error handling %T from %v: %v", packet, from, err)
			}
		}()
	}
}

func decodePacket(buf []byte) (packet, NodeID, []byte, error) {
	if len(buf) < headSize+1 {
		return nil, NodeID{}, nil, errPacketTooSmall
	}
	hash, sig, sigdata := buf[:macSize], buf[macSize:headSize], buf[headSize:]
	shouldhash := crypto.Sha3(buf[macSize:])
	if !bytes.Equal(hash, shouldhash) {
		return nil, NodeID{}, nil, errBadHash
	}
	fromID, err := recoverNodeID(crypto.Sha3(buf[headSize:]), sig)
	if err != nil {
		return nil, NodeID{}, hash, err
	}
	var req packet
	switch ptype := sigdata[0]; ptype {
	case pingPacket:
		req = new(ping)
	case pongPacket:
		req = new(pong)
	case findnodePacket:
		req = new(findnode)
	case neighborsPacket:
		req = new(neighbors)
	default:
		return nil, fromID, hash, fmt.Errorf("unknown type: %d", ptype)
	}
	err = rlp.Decode(bytes.NewReader(sigdata[1:]), req)
	return req, fromID, hash, err
}

func (req *ping) handle(t *udp, from *net.UDPAddr, fromID NodeID, mac []byte) error {
	if expired(req.Expiration) {
		return errExpired
	}
	if req.Version != Version {
		return errBadVersion
	}
	t.send(from, pongPacket, pong{
		ReplyTok:   mac,
		Expiration: uint64(time.Now().Add(expiration).Unix()),
	})
	if !t.handleReply(fromID, pingPacket, req) {
		// Note: we're ignoring the provided IP address right now
		t.bond(true, fromID, from, req.Port)
	}
	return nil
}

func (req *pong) handle(t *udp, from *net.UDPAddr, fromID NodeID, mac []byte) error {
	if expired(req.Expiration) {
		return errExpired
	}
	if !t.handleReply(fromID, pongPacket, req) {
		return errUnsolicitedReply
	}
	return nil
}

func (req *findnode) handle(t *udp, from *net.UDPAddr, fromID NodeID, mac []byte) error {
	if expired(req.Expiration) {
		return errExpired
	}
	if t.db.get(fromID) == nil {
		// No bond exists, we don't process the packet. This prevents
		// an attack vector where the discovery protocol could be used
		// to amplify traffic in a DDOS attack. A malicious actor
		// would send a findnode request with the IP address and UDP
		// port of the target as the source address. The recipient of
		// the findnode packet would then send a neighbors packet
		// (which is a much bigger packet than findnode) to the victim.
		return errUnknownNode
	}
	t.mutex.Lock()
	closest := t.closest(req.Target, bucketSize).entries
	t.mutex.Unlock()

	t.send(from, neighborsPacket, neighbors{
		Nodes:      closest,
		Expiration: uint64(time.Now().Add(expiration).Unix()),
	})
	return nil
}

func (req *neighbors) handle(t *udp, from *net.UDPAddr, fromID NodeID, mac []byte) error {
	if expired(req.Expiration) {
		return errExpired
	}
	if !t.handleReply(fromID, neighborsPacket, req) {
		return errUnsolicitedReply
	}
	return nil
}

func expired(ts uint64) bool {
	return time.Unix(int64(ts), 0).Before(time.Now())
}
