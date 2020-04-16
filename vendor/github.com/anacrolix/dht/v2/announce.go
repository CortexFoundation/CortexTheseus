package dht

// get_peers and announce_peers.

import (
	"context"
	"fmt"
	"net"
	"sync/atomic"

	"github.com/anacrolix/log"
	"github.com/anacrolix/missinggo/v2/conntrack"
	"github.com/anacrolix/stm"
	"github.com/anacrolix/stm/stmutil"
	"github.com/benbjohnson/immutable"

	"github.com/anacrolix/dht/v2/krpc"
)

// Maintains state for an ongoing Announce operation. An Announce is started by calling
// Server.Announce.
type Announce struct {
	Peers chan PeersValues

	values chan PeersValues // Responses are pushed to this channel.

	// These only exist to support routines relying on channels for synchronization.
	done    <-chan struct{}
	doneVar *stm.Var
	cancel  func()

	pending  *stm.Var // How many transactions are still ongoing (int).
	server   *Server
	infoHash int160 // Target
	// Count of (probably) distinct addresses we've sent get_peers requests to.
	numGetPeers int64
	// The torrent port that we're announcing.
	announcePort int
	// The torrent port should be determined by the receiver in case we're
	// being NATed.
	announcePortImplied bool
	scrape              bool

	// List of pendingAnnouncePeer. TODO: Perhaps this should be sorted by distance to the target,
	// so we can do that sloppy hash stuff ;).
	pendingAnnouncePeers *stm.Var

	traversal traversal
}

func (a *Announce) String() string {
	return fmt.Sprintf("%[1]T %[1]p of %v on %v", a, a.infoHash, a.server)
}

type pendingAnnouncePeer struct {
	addrMaybeId
	token string
}

// Returns the number of distinct remote addresses the announce has queried.
func (a *Announce) NumContacted() int64 {
	return atomic.LoadInt64(&a.numGetPeers)
}

type AnnounceOpt *struct{}

var scrape = AnnounceOpt(&struct{}{})

func Scrape() AnnounceOpt { return scrape }

// Traverses the DHT graph toward nodes that store peers for the infohash, streaming them to the
// caller, and announcing the local node to each responding node if port is non-zero or impliedPort
// is true.
func (s *Server) Announce(infoHash [20]byte, port int, impliedPort bool, opts ...AnnounceOpt) (*Announce, error) {
	startAddrs, err := s.traversalStartingNodes()
	if err != nil {
		return nil, err
	}
	infoHashInt160 := int160FromByteArray(infoHash)
	a := &Announce{
		Peers:                make(chan PeersValues, 100),
		values:               make(chan PeersValues),
		server:               s,
		infoHash:             infoHashInt160,
		announcePort:         port,
		announcePortImplied:  impliedPort,
		pending:              stm.NewVar(0),
		pendingAnnouncePeers: stm.NewVar(newPendingAnnouncePeers(infoHashInt160)),
		traversal:            newTraversal(infoHashInt160),
	}
	for _, opt := range opts {
		if opt == scrape {
			a.scrape = true
		}
	}
	var ctx context.Context
	ctx, a.cancel = context.WithCancel(context.Background())
	a.done = ctx.Done()
	a.doneVar, _ = stmutil.ContextDoneVar(ctx)
	// Function ferries from values to Peers until discovery is halted.
	go func() {
		defer close(a.Peers)
		for {
			select {
			case psv := <-a.values:
				select {
				case a.Peers <- psv:
				case <-a.done:
					return
				}
			case <-a.done:
				return
			}
		}
	}()
	for _, n := range startAddrs {
		stm.Atomically(a.pendContact(n))
	}
	go a.run()
	return a, nil
}

func validNodeAddr(addr net.Addr) bool {
	// At least for UDP addresses, we know what doesn't work.
	ua := addr.(*net.UDPAddr)
	if ua.Port == 0 {
		return false
	}
	if ip4 := ua.IP.To4(); ip4 != nil && ip4[0] == 0 {
		// Why?
		return false
	}
	return true
}

func (a *Announce) shouldContact(addr krpc.NodeAddr, tx *stm.Tx) bool {
	if !validNodeAddr(addr.UDP()) {
		return false
	}
	if a.server.ipBlocked(addr.IP) {
		return false
	}
	return true
}

func (a *Announce) responseNode(node krpc.NodeInfo) {
	i := int160FromByteArray(node.ID)
	stm.Atomically(a.pendContact(addrMaybeId{node.Addr, &i}))
}

// Store a potential peer announce.
func (a *Announce) maybeAnnouncePeer(to Addr, token *string, peerId *krpc.ID) {
	if token == nil {
		return
	}
	if !a.server.config.NoSecurity && (peerId == nil || !NodeIdSecure(*peerId, to.IP())) {
		return
	}
	stm.Atomically(stm.VoidOperation(func(tx *stm.Tx) {
		x := pendingAnnouncePeer{
			token: *token,
		}
		x.Addr = to.KRPC()
		if peerId != nil {
			id := int160FromByteArray(*peerId)
			x.Id = &id
		}
		tx.Set(a.pendingAnnouncePeers, tx.Get(a.pendingAnnouncePeers).(pendingAnnouncePeers).Push(x))
	}))
}

func (a *Announce) announcePeer(peer pendingAnnouncePeer) numWrites {
	_, writes, _ := a.server.announcePeer(NewAddr(peer.Addr.UDP()), a.infoHash, a.announcePort, peer.token, a.announcePortImplied)
	return writes
}

func (a *Announce) beginAnnouncePeer(tx *stm.Tx) interface{} {
	tx.Assert(a.getPendingAnnouncePeers(tx).Len() != 0)
	new, x := tx.Get(a.pendingAnnouncePeers).(pendingAnnouncePeers).Pop(tx)
	tx.Set(a.pendingAnnouncePeers, new)

	return a.beginQuery(NewAddr(x.Addr.UDP()), "dht announce announce_peer", func() numWrites {
		a.server.logger().Printf("announce_peer to %v", x)
		return a.announcePeer(x)
	})(tx).(func())
}

func finalizeCteh(cteh *conntrack.EntryHandle, writes numWrites) {
	if writes == 0 {
		cteh.Forget()
		// TODO: panic("how to reverse rate limit?")
	} else {
		cteh.Done()
	}
}

func (a *Announce) getPeers(addr Addr) numWrites {
	m, writes, _ := a.server.getPeers(context.TODO(), addr, a.infoHash, a.scrape)
	// Register suggested nodes closer to the target info-hash.
	if r := m.R; r != nil {
		id := &r.ID
		expvars.Add("announce get_peers response nodes values", int64(len(r.Nodes)))
		expvars.Add("announce get_peers response nodes6 values", int64(len(r.Nodes6)))
		r.ForAllNodes(a.responseNode)
		select {
		case a.values <- PeersValues{
			Peers: r.Values,
			NodeInfo: krpc.NodeInfo{
				Addr: addr.KRPC(),
				ID:   *id,
			},
			Return: *r,
		}:
		case <-a.done:
		}
		a.maybeAnnouncePeer(addr, r.Token, id)
	}
	return writes
}

// Corresponds to the "values" key in a get_peers KRPC response. A list of
// peers that a node has reported as being in the swarm for a queried info
// hash.
type PeersValues struct {
	Peers         []Peer // Peers given in get_peers response.
	krpc.NodeInfo        // The node that gave the response.
	krpc.Return
}

// Stop the announce.
func (a *Announce) Close() {
	a.close()
}

func (a *Announce) close() {
	a.cancel()
}

func (a *Announce) pendContact(node addrMaybeId) stm.Operation {
	return stm.VoidOperation(func(tx *stm.Tx) {
		if !a.shouldContact(node.Addr, tx) {
			// log.Printf("shouldn't contact (pend): %v", node)
			return
		}
		a.traversal.pendContact(node)(tx)
	})
}

type txResT struct {
	done bool
	run  func()
}

func wrapRun(f stm.Operation) stm.Operation {
	return func(tx *stm.Tx) interface{} {
		return txResT{run: f(tx).(func())}
	}
}

func (a *Announce) getPending(tx *stm.Tx) int {
	return tx.Get(a.pending).(int)
}

func (a *Announce) farthestAnnouncePeer(tx *stm.Tx) (pendingAnnouncePeer, bool) {
	pending := a.getPendingAnnouncePeers(tx)
	if pending.Len() < pending.k {
		return pendingAnnouncePeer{}, false
	} else {
		return pending.Farthest()
	}
}

func (a *Announce) getPendingAnnouncePeers(tx *stm.Tx) pendingAnnouncePeers {
	return tx.Get(a.pendingAnnouncePeers).(pendingAnnouncePeers)
}

func (a *Announce) run() {
	defer a.cancel()
	for {
		txRes := stm.Atomically(func(tx *stm.Tx) interface{} {
			if tx.Get(a.doneVar).(bool) {
				return txResT{done: true}
			}
			p := a.getPendingAnnouncePeers(tx)
			i := 0
			p.Range(func(value interface{}) {
				a.logger().Printf("pending %d: %v", i, value)
				i++
			})
			if next, ok := a.traversal.popNextContact(tx); ok {
				a.logger().Printf("next: %v", next)
				farthest, ok := a.farthestAnnouncePeer(tx)
				a.logger().Printf("farthest: %v %v", ok, farthest)
				if !ok || !farthest.closerThan(next, a.infoHash) {
					a.logger().Printf("farthest: %v %v", ok, farthest)
					tx.Assert(a.getPending(tx) < 3)
					dhtAddr := NewAddr(next.Addr.UDP())
					return wrapRun(a.beginQuery(dhtAddr, "dht announce get_peers", func() numWrites {
						a.server.logger().Printf("doing get_peers to %v", next)
						atomic.AddInt64(&a.numGetPeers, 1)
						return a.getPeers(dhtAddr)
					}))(tx)
				}
			}
			tx.Assert(a.getPending(tx) == 0)
			return txResT{done: true}
		}).(txResT)
		if txRes.done {
			break
		}
		go txRes.run()
	}
	a.logger().Printf("finishing get peers step")
	for {
		txRes := stm.Atomically(stm.Select(
			wrapRun(a.beginAnnouncePeer),
			func(tx *stm.Tx) interface{} {
				if tx.Get(a.doneVar).(bool) || a.getPending(tx) == 0 && a.getPendingAnnouncePeers(tx).Len() == 0 {
					return txResT{done: true}
				}
				panic(stm.Retry)
			},
		)).(txResT)
		if txRes.done {
			break
		}
		go txRes.run()
	}
}

//
//func (a *Announce) beginGetPeers(tx *stm.Tx) interface{} {
//}

func (a *Announce) beginQuery(addr Addr, reason string, f func() numWrites) stm.Operation {
	return func(tx *stm.Tx) interface{} {
		pending := tx.Get(a.pending).(int)
		tx.Set(a.pending, pending+1)
		return a.server.beginQuery(addr, reason, func() numWrites {
			defer stm.Atomically(stm.VoidOperation(func(tx *stm.Tx) { tx.Set(a.pending, tx.Get(a.pending).(int)-1) }))
			return f()
		})(tx)
	}
}

func (a *Announce) logger() log.Logger {
	return a.server.logger()
}

type pendingAnnouncePeers struct {
	inner *immutable.SortedMap
	k     int
}

func newPendingAnnouncePeers(target int160) pendingAnnouncePeers {
	return pendingAnnouncePeers{
		k: 8,
		inner: immutable.NewSortedMap(comparer{less: func(l, r interface{}) bool {
			return l.(pendingAnnouncePeer).addrMaybeId.closerThan(r.(pendingAnnouncePeer).addrMaybeId, target)
		}}),
	}
}

func (me *pendingAnnouncePeers) Range(f func(interface{})) {
	iter := me.inner.Iterator()
	for !iter.Done() {
		key, _ := iter.Next()
		f(key)
	}
}

func (me pendingAnnouncePeers) Len() int {
	return me.inner.Len()
}

func (me pendingAnnouncePeers) Push(x pendingAnnouncePeer) pendingAnnouncePeers {
	me.inner = me.inner.Set(x, nil)
	for me.inner.Len() > me.k {
		iter := me.inner.Iterator()
		iter.Last()
		key, _ := iter.Next()
		me.inner = me.inner.Delete(key)
	}
	return me
}

func (me pendingAnnouncePeers) Pop(tx *stm.Tx) (pendingAnnouncePeers, pendingAnnouncePeer) {
	iter := me.inner.Iterator()
	x, _ := iter.Next()
	me.inner = me.inner.Delete(x)
	return me, x.(pendingAnnouncePeer)
}

func (me pendingAnnouncePeers) Farthest() (value pendingAnnouncePeer, ok bool) {
	iter := me.inner.Iterator()
	iter.Last()
	if iter.Done() {
		return
	}
	key, _ := iter.Next()
	value = key.(pendingAnnouncePeer)
	ok = true
	return
}

type lessFunc func(l, r interface{}) bool

type comparer struct {
	less lessFunc
}

func (me comparer) Compare(i, j interface{}) int {
	if me.less(i, j) {
		return -1
	} else if me.less(j, i) {
		return 1
	} else {
		return 0
	}
}
