package torrentfs

import (
	"github.com/CortexFoundation/CortexTheseus/p2p"
	mapset "github.com/deckarep/golang-set"
	"sync"
)

type Peer struct {
	host *TorrentFS
	peer *p2p.Peer
	ws   p2p.MsgReadWriter

	trusted bool

	known mapset.Set // Messages already known by the peer to avoid wasting bandwidth
	quit  chan struct{}

	wg sync.WaitGroup
}

func newPeer(host *TorrentFS, remote *p2p.Peer, rw p2p.MsgReadWriter) *Peer {
	return &Peer{
		host:    host,
		peer:    remote,
		ws:      rw,
		trusted: false,
		known:   mapset.NewSet(),
		quit:    make(chan struct{}),
	}
}

func (p *Peer) Start() error {
	return nil
}

func (peer *Peer) handshake() error {
	return nil
}

func (p *Peer) Stop() error {
	close(p.quit)
	p.wg.Wait()
	return nil
}
